import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch._inductor.config
from torch import distributed as dist

from fms.models import get_model
from fms.utils import generation, tokenizers
from fms.utils.generation import generate

# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--fp8",
    action="store_true",
    help="Use float8_experimental Float8Linear to swap linear layers",
)
parser.add_argument(
    "--fp8_linear_type",
    type=str,
    default='dasw',
    choices=['dadw', 'dasw', 'sw', 'ns'],
    help="Choose the float8 linear type",
)
parser.add_argument(
    "--attn_algorithm",
    type=str,
    default=None,
    choices=['triton', 'flash', 'mem', 'math'],
    help="Choose fused attention type",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument("--context_file", type=str, default=None, help="File to summarize")
parser.add_argument(
    "--batch_size",
    type=int,
    help="batch size",
    default=16
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="max number of new output tokens",
    default=128
)
parser.add_argument(
    "--dump_tensor",
    type=str,
    default=None,
    help="Turn on tensor dumping for debug/visualization",
)
args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

prefill_model = model
decode_model = model

if args.fp8:
    from float8_experimental.float8_linear import Float8Linear, Float8DASWLinear, Float8SWLinear
    from float8_experimental.float8_linear_utils import (
        swap_linear_with_float8_linear,
    )
    fp8LinearDict = {
        'dadw': Float8Linear,
        'dasw': Float8DASWLinear,
        'sw':   Float8SWLinear,
        'ns':   None,
    }
    skip_fqn_list = None
    if args.architecture == "llama" and args.variant == "7b":
        #skip_fqn_list = [f"layers.{i}.ff_sub_layer.w2" for i in range(32)]
        skip_fqn_list = [f"layers.{i}.ff_sub_layer.w2" for i in [1, 30]]
    print(f"fp8 skipping layers {skip_fqn_list=}")
    model = swap_linear_with_float8_linear(model, fp8LinearDict[args.fp8_linear_type],skip_fqn_list=skip_fqn_list)

print(model)

if args.dump_tensor is not None:
    try:
        from torch_visual_tensors.torch_hooks import TorchForwardHooks
        attn_type = 'torch' if args.attn_algorithm is None else args.attn_algorithm
        
        layer_hooks = TorchForwardHooks(model, output_dir=f'./tensors/{args.dump_tensor}')
    except:
        print('torch_visual_tensors not installed')

if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    model = torch.compile(model, mode=args.compile_mode)


def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    tokens = ["<s>"] + tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids


def pad_prompt(prompt, pad_len, pad_token="<unk>"):
    to_pad = pad_len - len(prompt)
    if to_pad == 0:
        return prompt

    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    pad_ids = [pad_id] * to_pad
    return torch.cat((torch.tensor(pad_ids, device=device), prompt))


if args.context_file is not None:
    # during testing, the context_file used was a copy/paste of the text of:
    # https://arxiv.org/pdf/2306.15595.pdf
    with open(args.context_file) as file:
        long_prompt = file.read()
        prompt1 = (
            long_prompt
            + "\nPlease give me a brief summary of this research paper in a few bullet points."
        )
        # prompt1 = long_prompt + "\nDescribe work that was done concurrently with the research in this paper."
        prompt2 = long_prompt + "\nPlease write me the abstract for this paper."
else:
    #template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

    #prompt1 = template.format(
    #    "Provide a list of instructions for preparing chicken soup."
    #)
    #prompt2 = template.format("Explain some popular greetings in Spanish.")
    prompt1 = "San Franscisco is a "

prompt1 = ids_for_prompt(prompt1)
# prompt2 = ids_for_prompt(prompt2)

# max_len = max([len(prompt) for prompt in [prompt1, prompt2]])
# max_len = len(prompt1)
# prompt1 = pad_prompt(prompt1, max_len)
# LLaMA 7B did better on the spanish prompt vs 13B.
# TODO: add a better english prompt to demonstrate padding/batching.
# prompt2 = pad_prompt(prompt2, max_len)
ids = torch.stack([prompt1] * args.batch_size, dim=0) 
print(f"{ids.shape=}")

def print_result(result):
    if local_rank != 0:
        return
    # stop at EOS token if present
    result = generation.truncate_after_eos(
        result, tokenizer.convert_tokens_to_ids("</s>")
    )
    # print(result)
    # print(tokenizer.convert_ids_to_tokens(result))
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))
    print()


def infer(use_cache, do_sample):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0:
        print("use_cache", use_cache, ";; do_sample", do_sample)
        print("==================")
    if model.config.ntk_scaling:
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    result = generate(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=use_cache,
        do_sample=do_sample,
        max_seq_len=max_seq_len,
        attn_algorithm=args.attn_algorithm,
    )
    for i in range(result.shape[0]):
        print_result(result[i])


print("generating output", local_rank)
do_sample = [False]
use_cache = [
    args.no_use_cache
]  # True/False are identical with greedy iff `torch.use_deterministic_algorithms(True)`
for sample, cache in itertools.product(do_sample, use_cache):
    infer(cache, sample)