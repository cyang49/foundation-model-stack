"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl

USE_FP8_E5M2 = True
if USE_FP8_E5M2:
    TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
    pt_fp8_type = torch.float8_e5m2
    tl_fp8_type = tl.float8e5
else:
    TORCH_HAS_FP8 = hasattr(torch, 'float8_e4m3fn')
    pt_fp8_type = torch.float8_e4m3fn
    tl_fp8_type = tl.float8e4nv

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl_fp8_type)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


## We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
## the code below and commenting out the equivalent parameters is convenient for
## re-tuning.
#@triton.autotune(
#   configs=[
#       triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
#       triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#       triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
#    #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#    #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
#    #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
#       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=8),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=7, num_warps=8),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=6, num_warps=8),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=8),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=8),
#    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
#   ],
#   key=['N_CTX'],
#)
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0), # row major
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl_fp8_type else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1), # column major
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0), # row major
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl_fp8_type  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl_fp8_type  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

empty = torch.empty(128, device="cuda")

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, out_dtype=torch.float16):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        assert Lq == Lk and (Lk == Lv or v.dtype == pt_fp8_type)
        assert Lk in {16, 32, 64, 128, 256}
        o = torch.empty_like(q, dtype=out_dtype)
        BLOCK_M = 128
        BLOCK_N = 64 if Lk <= 64 else 32
        num_stages = 4 if Lk <= 64 else 3
        num_warps = 4
        stage = 3 if causal else 1
        # Tuning for H100
        if torch.cuda.get_device_capability()[0] == 9:
            num_warps = 8
            num_stages = 7 if Lk >= 64 else 3
            if v.dtype == pt_fp8_type:
                if Lk < 256:
                    BLOCK_M = 64 if not causal else 128
                    BLOCK_N = 128
                    num_stages = 3 if Lk == 128 else 4
                    num_warps = 4
                else:
                    BLOCK_M = 128
                    BLOCK_N = 128
                    num_stages = 3
                    num_warps = 8
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=Lk,  #
            STAGE=stage,  #
            num_warps=num_warps,  #
            num_stages=num_stages  #
        )
        # ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o

attention = _attention.apply

@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(64, 32, 128, 128), (64, 32, 256, 128), (64, 32, 512, 128)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("fp8_inputs", [False, True] if TORCH_HAS_FP8 else [False])
def test_op(Z, H, N_CTX, D_HEAD, causal, fp8_inputs, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # ref_out.backward(dout)
    # ref_dv, v.grad = v.grad.clone(), None
    # ref_dk, k.grad = k.grad.clone(), None
    # ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    if TORCH_HAS_FP8 and fp8_inputs:
        q = q.to(pt_fp8_type) # TODO: adjust for inference mode
        k = k.to(pt_fp8_type)
        # v = v.permute(0, 1, 3, 2)
        v = v.to(pt_fp8_type)
    tri_out = attention(q, k, v, causal, sm_scale)
    # tri_out.backward(dout)
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dq, q.grad = q.grad.clone(), None
    # compare
    # print(f"{tri_out.dtype=}")
    # print(f"{tri_out.shape=}")
    # print(f"{tri_out[0][0][0]=}")
    # print(f"{ref_out.dtype=}")
    # print(f"{ref_out.shape=}")
    # print(f"{ref_out[0][0][0]=}")

    atol = 1e-2 if not fp8_inputs else 5e-1
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=0)
    # rtol = 0.0
    # # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # # For detailss see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    # if torch.version.hip is not None and triton.runtime.driver.active.get_current_target()[1] == "gfx90a":
    #     rtol = 1e-2
    # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

# BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# HIDDEN = 4096 # HIDDEN = N_HEADS * D_HEAD
# BATCH,  N_CTX = 4, 4096
# N_HEADS = 32
# D_HEAD = HIDDEN // N_HEADS # D_HEAD < 64 in fp8 significantly decreases throughput
HIDDEN = 4096 # HIDDEN = N_HEADS * D_HEAD
BATCH,  N_CTX = 16, 256
N_HEADS = 32
D_HEAD = HIDDEN // N_HEADS # D_HEAD < 64 in fp8 significantly decreases throughput
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [True, False]:
        for fp8_inputs in [False, True]:
            if fp8_inputs and ((not TORCH_HAS_FP8) or mode == "bwd"):
                continue
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    # x_vals=[2**i for i in range(10, 15)],
                    x_vals=[N_CTX], # fix N_CTX
                    line_arg="provider",
                    line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
                    line_names=["Triton"] + (["Flash-2"] if HAS_FLASH else []),
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="ms",
                    plot_name=
                    f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}-fp8={fp8_inputs}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "D_HEAD": D_HEAD,
                        "dtype": torch.float16,
                        "mode": mode,
                        "causal": causal,
                        "fp8_inputs": fp8_inputs,
                    },
                ))

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, fp8_inputs, dtype=torch.float16,
                          device="cuda"):
    assert mode in ["fwd"]
    warmup = 1
    rep = 1
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if mode == "fwd" and TORCH_HAS_FP8 and fp8_inputs:
            q = q.to(pt_fp8_type) # TODO: adjust for inference mode
            k = k.to(pt_fp8_type)
            v = v.permute(0, 1, 3, 2)
            v = v.to(pt_fp8_type)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
