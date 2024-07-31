from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, MutableMapping, Optional

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from fms import distributed
from fms.distributed.strategy import (
    TensorParallelStrategy,
    UniformModelParallelStrategy,
)
from fms.utils import serialization


__models: MutableMapping[str, MutableMapping[str, Callable[[], nn.Module]]] = {}


def register_model(architecture: str, variant: str, factory: Callable[[], nn.Module]):
    """
    Registers a model variant to be made available in the registration API.
    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    variant: A reference for a particular configuration of the architecture,
        e.g. '7b'
    factory: A callable that constructs an instance of the model variant.
    """
    variants: MutableMapping[str, Callable[[], nn.Module]] = {}
    if architecture in __models:
        variants = __models[architecture]
    if variant in variants:
        raise KeyError(
            f"Variant {variant} already registered for architecture {architecture}"
        )
    variants[variant] = factory
    __models[architecture] = variants


def list_models():
    """
    Lists registered model architectures.
    """
    return list(__models.keys())


def list_variants(architecture: str):
    """
    Lists available variants (configurations) of a model architecture.
    E.g. `models.list_variants('llama')` -> ['micro', '7b', '13b', '70b']
    Args:
    architecture: one of the registered architectures returned by `list_models()`.
    """
    if architecture not in __models:
        raise KeyError(
            f"{architecture} is not registered. See `models.list_models()` for available architectures"
        )
    return list(__models[architecture].keys())


def _get_model_instance(
    architecture: str, variant: str, *, dtype=None, device=None, extra_args: dict = {}
) -> nn.Module:
    """
    Gets a model by name and variant, e.g. `models.get_model('llama', '7b')`
    Does not load weights.
    See public API `models.get_model()`
    Args:
    architecture: one of the architectures from list_models(). E.g. llama.
    variant: one of the variants from list_variants(architecture). E.g. '7b'
    extra_args: kwargs to be passed to the model factory.
    """
    if architecture not in __models:
        raise KeyError(
            f"{architecture} is not registered. See `models.list_models()` for available architectures"
        )
    if variant not in __models[architecture]:
        raise KeyError(
            f'{variant} is not a registered variant of {architecture}. See `models.list_variants("{architecture}")` for available variants.'
        )

    model_factory = __models[architecture][variant]

    orig = torch.get_default_dtype()

    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        with device if device is not None else nullcontext():
            return model_factory(**extra_args)
    finally:
        torch.set_default_dtype(orig)


def _guess_num_layers(state_dict):
    """
    This function attempts to guess the number of "layers" in a state_dict by
    looking for lists of sub modules. This can be used to setup model-parallel
    when we don't yet have a model instance.
    """
    if state_dict is None or len(state_dict) == 0:
        raise ValueError(
            "Use model parallel with pre-trained models that have a state dict"
        )

    layers = set()
    import re

    for key in state_dict.keys():
        # when there's a list of layers, layers have numeric IDs in the key
        layerid = re.sub("[^.]*\\.([0-9]+)\\..*", "\\1", key)
        if layerid != key:
            layers.add(layerid)
    return len(layers)


def _class_hierarchy(clz):
    if clz == object:
        return {clz}
    bases = clz.__bases__
    all = [_class_hierarchy(c) for c in bases]
    result = {clz}
    for classes in all:
        result = result | classes
    return result


def _fsdp_autowrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int):
    if recurse:
        return True
    classes = _class_hierarchy(module.__class__)
    for clz in classes:
        name = str(clz).lower()
        if ("layer" in name or "block" in name) and "layernorm" not in name:
            return True
    return False


def _activation_checkpoint_check_fn(layer):
    for name in layer.__class__.__bases__:
        name = str(name).lower()
        if "block" in name or "layer" in name:
            return True
    return False


def _fsdp_wrap(
    model: nn.Module,
    distributed_strategy: Optional[str],
    device: torch.device,
    rank0: bool,
) -> nn.Module:
    # initializes parameters that are on meta devices
    def init_fn(x: nn.Module):
        if not rank0:
            return x.to_empty(device=device, recurse=False)
        else:
            return x

    # TODO: enable other policies
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    if distributed_strategy == "fsdp":
        dp_strategy = ShardingStrategy.FULL_SHARD
    elif distributed_strategy == "hsdp":
        dp_strategy = ShardingStrategy.HYBRID_SHARD
    elif distributed_strategy == "ddp":
        dp_strategy = ShardingStrategy.NO_SHARD
    else:
        raise KeyError("distributed strategy should be one of fsdp, dpp, or hsdp")

    model = FSDP(
        model,
        param_init_fn=init_fn,
        sync_module_states=True,
        device_id=device.index,
        limit_all_gathers=True,
        auto_wrap_policy=_fsdp_autowrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=dp_strategy,
    )

    wrapper_fn = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper_fn,
        check_fn=_activation_checkpoint_check_fn,
    )

    return model


def _is_dp(distributed_strategy):
    return distributed_strategy in {"fsdp", "hsdp", "ddp"}


def get_model(
    architecture: str,
    variant: str,
    model_path: Optional[str] = None,
    source: Optional[str] = None,
    device_type: str = "cpu",
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    group: Optional[ProcessGroup] = None,
    **kwargs,
):
    """
    Load an instance of a model with weights.

    Args:
    architecture: the model architecture, e.g. llama. See
                `models.list_models()`.
    variant: the configuration of the model, e.g. 7b. See
                `models.list_variants(architecture)`
    model_path: the path to the state_dict of weights. If None, don't load.
    device_type: where to load the model
    distributed_strategy: None, 'fsdp', 'hsdp', 'tp', or 'mp'.
    checkpoint_sharding: how the checkpoint files are sharded: None, 'tp',
                'fsdp', or 'layer'. If None, guess based on files.
    source: If the weights in the state dict didn't come from an FMS model,
                `source` specifies which conversion function might be needed.
                See `serialization.list_sources(architecture)`
    group: ProcessGroup The PG to use for any model distribution
    """
    rank, world_size = distributed.rank_and_world(group)
    local_rank = distributed.local_rank()

    if distributed_strategy is None or distributed_strategy == "":
        if world_size > 1:
            distributed_strategy = "tp"

    if device_type == "cuda":
        device = torch.device(device_type, local_rank)
    else:
        device = torch.device(device_type)

    hsdp = distributed_strategy == "hsdp"
    fsdp = distributed_strategy == "fsdp"
    ddp = distributed_strategy == "ddp"
    if hsdp or fsdp or ddp:
        if (hsdp and local_rank != 0) or ((fsdp or ddp) and rank != 0):
            initial_device = torch.device("meta")
        else:
            initial_device = torch.device("cpu")
    elif distributed_strategy == "mp":
        initial_device = torch.device("cpu")
    else:
        initial_device = device

    lazy_sd: MutableMapping[str, Any] = {}
    if model_path is not None:
        lazy_sd = serialization.load_state_dict(
            model_path,
            source=source,
            distributed_strategy=distributed_strategy,
            checkpoint_sharding=checkpoint_sharding,
            initial_device=initial_device,
            rank=rank,
            world_size=world_size,
        )

    extra_args = kwargs
    if "distributed_strategy" not in extra_args:
        if distributed_strategy == "tp":
            print("using tensor parallel")
            extra_args["distributed_strategy"] = TensorParallelStrategy()
        elif distributed_strategy == "mp":
            print("using model parallel")
            devices = [i for i in range(torch.cuda.device_count())]
            extra_args["distributed_strategy"] = UniformModelParallelStrategy(
                devices, _guess_num_layers(lazy_sd)
            )

    # Create the model
    # fms_model = _get_model_instance(
    #     architecture, variant, device=initial_device, extra_args=extra_args
    # )
    # HACK: use meta device
    fms_model = _get_model_instance(
        architecture, variant, device=torch.device("meta"), extra_args=extra_args
    )
    
    # HACK: FP8 linear layers
    # For quantized models, replace linear layers with quantized linear
    # layer implemenations
    # NOTE: This doesn't apply to the use case when quantization is applied
    #       to fp16 weights after loading. In that case, a layer by layer
    #       conversion workflow may be needed to keep the memory footprint
    #       smaller
    is_fp8 = True
    if is_fp8:
        from float8_experimental.inference import (ActivationCasting, QuantConfig, Float8InferenceLinear)
        from float8_experimental.float8_linear_utils import swap_linear_layers
        from float8_experimental.float8_tensor import (
            LinearMMConfig,
            ScaledMMConfig,
        )
        use_fast_accum = True
        # SASW only for now
        quant_config = QuantConfig(
                    activation_casting=ActivationCasting.STATIC,
                    static_quantization_scale=torch.tensor(
                        1.0, dtype=torch.float32, device=initial_device
                    ),
                )

        # skip_fqn_list_llama_2 = [f"layers.{i}.ff_sub_layer.w2" for i in [1, 30]] + [
        #     "shared.head"
        # ]
        # skip_fqn_list = skip_fqn_list_llama_2
        skip_fqn_list = ['shared.head']

        # NOTE: QuantizedLinear should be a specialized linear layer e.g. Float8Linear and GPTQLinear
        #       The assumptions are 
        #           (1) only 1 type of quantize linear is used
        #           (2) only exceptions are skip_fqn_list
        #       They should have implementations for 
        #           (1) loading quantized weights and quantization parameters
        #           (2) forward function implementation        
        # for createing a Float8InferenceLinear using meta tensor
        def to_fp8_linear(module, quant_config, use_fast_accum=True):
            forward_config = ScaledMMConfig(
                False, use_fast_accum, pad_inner_dim=quant_config.pad_inner_dim
            )
            linear_mm_config = LinearMMConfig(
                forward_config, forward_config, forward_config
            )
            linear = Float8InferenceLinear(
                quant_config,
                linear_mm_config,
                # NOTE: not sharded sizes
                module.in_features,
                module.out_features,
                False, # bias
                device=initial_device, # FIXME: makes more sense to use meta device here but keep getting issues. Set to cuda for now
            )
            # Need Float8Tensor type placeholder for weight and create scale placeholder
            linear.set_weight_and_bias(module.weight, module.bias)
            linear.quantize_weight()
            # at this point every parameter is still on the meta device
            print(f"{linear.weight=}") # Float8Tensor
            print(f"{linear.bias=}")   # Original bias
            return linear
            
        if quant_config is not None:
            fms_model = swap_linear_layers(
                fms_model,
                lambda m: to_fp8_linear(m, quant_config, use_fast_accum=True), # memory is allocated on device at this point
                module_filter_fn=lambda m, fqn: (not any([(x in fqn) for x in skip_fqn_list])),
            )
        print(f"{fms_model=}")
        for name, param in fms_model.named_parameters():
            print(name)
            

    # At this point the model instance is created on the meta device without weight values

    # Choose when to wrap and load the model weights based on the combination
    # distribution strategy and checkpoint sharding
    pre_load = (
        distributed_strategy in ["fsdp", "hsdp"] and checkpoint_sharding != "fsdp"
    )

    def model_wrap(model):
        if _is_dp(distributed_strategy):
            return _fsdp_wrap(model, distributed_strategy, device, rank == 0)
        return model

    if not pre_load:
        fms_model = model_wrap(fms_model)

    # HACK skip loading weight
    fms_model.to_empty(device=initial_device)

    # if len(lazy_sd):
    #     serialization.load_state_dict_into_model(
    #         fms_model,
    #         lazy_sd,
    #         architecture,
    #         source if source is not None else "fms",
    #         distributed_strategy,
    #         checkpoint_sharding,
    #         initial_device,
    #     )
    # elif hasattr(fms_model, "reset_parameters"):
    #     fms_model.reset_parameters()

    if pre_load:
        fms_model = model_wrap(fms_model)

    return fms_model


from fms.models import gpt_bigcode, llama, mixtral, roberta
