import math
import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Forward pass kernel: computes lse[b,h,m] = logsumexp(q@kᵀ * scale) over k
# -----------------------------------------------------------------------------
@triton.heuristics(
    {
        "EVEN_M":  lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N":  lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HD": lambda args: args["headdim"]   == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,              # bf16 [B,H,M,D]
    K,              # bf16 [B,H,N,D]
    Lse,            # bf16 [B,H,M]
    softmax_scale,  # f32 scalar

    # Q/K strides:
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    # Lse strides:
    stride_lse_b, stride_lse_h, stride_lse_m,

    # meta
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M:        tl.constexpr,
    EVEN_N:        tl.constexpr,
    EVEN_HD:       tl.constexpr,
    BLOCK_M:       tl.constexpr,
    BLOCK_N:       tl.constexpr,
):
    # program IDs
    start_m = tl.program_id(0)                # block-index along M
    hb      = tl.program_id(1)                # b*h flattened
    b       = hb // nheads
    h       = hb %  nheads

    # ranges
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                       # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_HEADDIM)                  # [BLOCK_HEADDIM]

    # pointers into Q, K
    q_ptrs = (
        Q + b*stride_qb + h*stride_qh
          + (offs_m[:, None]*stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + b*stride_kb + h*stride_kh
          + (offs_n[:, None]*stride_kn + offs_d[None, :])
    )

    # --- FLOAT32 accumulators for max & lse (so shape stays consistent) ---
    m_i   = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    lse_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)

    # load Q-block once (bf16 → cast to f32 immediately)
    if EVEN_M & EVEN_N:
        q_bf16 = tl.load(
            q_ptrs,
            mask=(EVEN_HD or (offs_d[None, :] < headdim)),
            other=0.0
        )
    else:
        mask_q = (offs_m[:, None] < seqlen_q) & (EVEN_HD or (offs_d[None, :] < headdim))
        q_bf16 = tl.load(q_ptrs, mask=mask_q, other=0.0)
    q = tl.cast(q_bf16, tl.float32)   # [BLOCK_M, D] in f32

    # loop over K-blocks
    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # load K-block
        if EVEN_M & EVEN_N:
            k_bf16 = tl.load(
                k_ptrs + start_n*stride_kn,
                mask=(EVEN_HD or (offs_d[None, :] < headdim)),
                other=0.0
            )
        else:
            mask_k = ((start_n + offs_n)[:, None] < seqlen_k) & (EVEN_HD or (offs_d[None, :] < headdim))
            k_bf16 = tl.load(k_ptrs + start_n*stride_kn, mask=mask_k, other=0.0)
        k = tl.cast(k_bf16, tl.float32)  # [BLOCK_N, D] in f32

        # q @ kᵀ  → bf16 matmul, then cast to f32
        qk_bf16 = tl.dot(q_bf16, k_bf16.trans())
        qk = tl.cast(qk_bf16, tl.float32)  # [BLOCK_M, BLOCK_N]

        # mask out-of-range columns
        if not EVEN_N:
            mask_col = (start_n + offs_n)[None, :] < seqlen_k
            qk = qk + tl.where(mask_col, 0.0, float("-inf"))

        # scaled‐softmax update in f32
        x       = qk * softmax_scale               # [M,N]
        m_new   = tl.maximum(tl.max(x, 1), m_i)     # [M]
        p       = tl.exp(x - m_new[:, None])        # [M,N]
        lsum    = tl.sum(p, 1)                      # [M]

        # update running m_i & lse_i
        m_i     = m_new
        lse_i   = m_new + tl.log(tl.exp(lse_i - m_new) + lsum)

    # store LSE back to bf16 buffer
    ptr_lse = Lse \
        + b*stride_lse_b \
        + h*stride_lse_h \
        + offs_m * stride_lse_m
    out_bf16 = tl.cast(lse_i, tl.bfloat16)
    tl.store(ptr_lse, out_bf16)


# -----------------------------------------------------------------------------
# Score pass kernel: computes Score[b,h,n] = sum_m softmax(q@kᵀ*scale)
# -----------------------------------------------------------------------------
@triton.heuristics(
    {
        "EVEN_M":  lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N":  lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HD": lambda args: args["headdim"]   == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _score_kernel(
    Q,              # bf16 [B,H,M,D]
    K,              # bf16 [B,H,N,D]
    Lse,            # bf16 [B,H,M]
    Score,          # bf16 [B,H,N]
    softmax_scale,  # f32 scalar

    # Q strides
    stride_qb, stride_qh, stride_qm,
    # K strides
    stride_kb, stride_kh, stride_kn,
    # Lse strides
    stride_lse_b, stride_lse_h, stride_lse_m,
    # Score strides
    stride_s_b, stride_s_h, stride_s_n,

    # meta
    nheads, seqlen_q, seqlen_k, seqlen_q_r, seqlen_k_r,
    headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M:        tl.constexpr,
    EVEN_N:        tl.constexpr,
    EVEN_HD:       tl.constexpr,
    BLOCK_M:       tl.constexpr,
    BLOCK_N:       tl.constexpr,
):
    # program IDs
    start_n = tl.program_id(0)
    hb      = tl.program_id(1)
    b       = hb // nheads
    h       = hb %  nheads

    # ranges
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_m = tl.arange(0, BLOCK_M)                       # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_HEADDIM)                 # [BLOCK_HEADDIM]

    # pointers
    q_ptrs   = Q   + b*stride_qb   + h*stride_qh   + (offs_m[:,None]*stride_qm + offs_d[None,:])
    k_ptrs   = K   + b*stride_kb   + h*stride_kh   + (offs_n[:,None]*stride_kn + offs_d[None,:])
    lse_ptrs = Lse + b*stride_lse_b + h*stride_lse_h + offs_m*stride_lse_m

    # f32 accumulator for the final sum
    score_j = tl.zeros([BLOCK_N], dtype=tl.float32)

    # load one K-block (bf16→f32)
    if EVEN_M & EVEN_N:
        k_bf16 = tl.load(
            k_ptrs,
            mask=(EVEN_HD or (offs_d[None,:]<headdim)),
            other=0.0
        )
    else:
        mask_k = (offs_n[:,None]<seqlen_k) & (EVEN_HD or (offs_d[None,:]<headdim))
        k_bf16 = tl.load(k_ptrs, mask=mask_k, other=0.0)
    k = tl.cast(k_bf16, tl.float32)

    # loop over Q-blocks
    for start_m in range(0, seqlen_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        # load Q-subblock & its LSE
        if EVEN_M & EVEN_N:
            q_bf16  = tl.load(
                q_ptrs + start_m*stride_qm,
                mask=(EVEN_HD or (offs_d[None,:]<headdim)),
                other=0.0
            )
            lse_mi  = tl.load(lse_ptrs + start_m)
        else:
            mask_q = ((start_m+offs_m)[:,None]<seqlen_q) \
                     & (EVEN_HD or (offs_d[None,:]<headdim))
            q_bf16  = tl.load(q_ptrs + start_m*stride_qm, mask=mask_q, other=0.0)
            lse_mi  = tl.load(
                lse_ptrs + start_m,
                mask=(start_m+offs_m)<seqlen_q,
                other=0.0
            )
        q      = tl.cast(q_bf16, tl.float32)

        # bf16 matmul → cast to f32
        qk_bf16 = tl.dot(q_bf16, k_bf16.trans())
        qk      = tl.cast(qk_bf16, tl.float32)

        if not EVEN_N:
            mask_col = offs_n[None,:] < seqlen_k
            qk       = qk + tl.where(mask_col, 0.0, float("-inf"))

        # apply scale & exp
        x    = qk * softmax_scale
        p    = tl.exp(x - lse_mi[:,None])
        p    = tl.where(((start_m+offs_m)[:,None]<seqlen_q), p, 0.0)

        # accumulate
        score_j += tl.sum(p, 0)

    # write back Score (cast to bf16)
    base    = Score + b*stride_s_b + h*stride_s_h
    ptrs    = base + offs_n*stride_s_n
    mask_n  = offs_n < seqlen_k
    out_bf16= tl.cast(score_j, tl.bfloat16)
    tl.store(ptrs, out_bf16, mask=mask_n)


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------
def flash_attn_kv_score(q, k, softmax_scale=None):
    """
    Args:
        q (torch.Tensor): Query tensor with shape [B, seqlen_q, num_heads, head_dim]
        k (torch.Tensor): Key tensor with shape [B, seqlen_kv, num_heads, head_dim]
        softmax_scale (float, optional): Scaling factor for attention scores. 
            Defaults to 1/sqrt(head_dim) if None.
    
    Returns:
        torch.Tensor: Attention scores tensor with shape [B, num_heads, seqlen_kv]
    """
    B, M, H, D = q.shape
    _, N, _, _ = k.shape
    assert (D <= 128)
    assert q.dtype == k.dtype in (torch.float16, torch.bfloat16)
    assert q.is_cuda and k.is_cuda

    softmax_scale = softmax_scale or 1.0 / math.sqrt(D)
    M_r = math.ceil(M/128)*128
    N_r = math.ceil(N/128)*128

    # allocate intermediate & output
    lse   = torch.empty((B, H, M_r), device=q.device, dtype=q.dtype)
    score = torch.empty((B, H, N_r), device=q.device, dtype=q.dtype)

    # forward
    BLOCK = 128
    grid_f = (triton.cdiv(M, BLOCK), B*H)
    _fwd_kernel[grid_f](
        q, k, lse, softmax_scale,
        # Q strides
        q.stride(0), q.stride(2), q.stride(1),
        # K strides
        k.stride(0), k.stride(2), k.stride(1),
        # Lse strides
        lse.stride(0), lse.stride(1), lse.stride(2),
        # meta
        H, M, N, M_r,
        D, M//32, N//32,
        BLOCK_HEADDIM=D, BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        num_warps=4 if D<=64 else 8, num_stages=1,
    )

    # score
    grid_s = (triton.cdiv(N, BLOCK), B*H)
    _score_kernel[grid_s](
        q, k, lse, score, softmax_scale,
        # Q strides
        q.stride(0), q.stride(2), q.stride(1),
        # K strides
        k.stride(0), k.stride(2), k.stride(1),
        # Lse strides
        lse.stride(0), lse.stride(1), lse.stride(2),
        # Score strides
        score.stride(0), score.stride(1), score.stride(2),
        # meta
        H, M, N, M_r, N_r,
        D, M//32, N//32,
        BLOCK_HEADDIM=D, BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        num_warps=4 if D<=64 else 8, num_stages=1,
    )

    return score[..., :N]


def torch_ref_score(q, k, scale):
    # q: [B, M, H, D], k: [B, N, H, D]
    # returns score: [B, H, N]
    q1 = q.transpose(1, 2)  # [B,H,M,D]
    k1 = k.transpose(1, 2)  # [B,H,N,D]
    # compute full matmul
    qk = (q1 @ k1.transpose(-1, -2)) * scale  # [B,H,M,N]
    # softmax over last dim then sum over M
    p  = torch.softmax(qk, dim=-1)
    return p.sum(dim=-2)  # [B,H,N]


def benchmark(func, args, num_runs=10):
    """Simple benchmark function that measures average execution time"""
    import time

    # warmup
    for _ in range(3):
        func(*args)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        result = func(*args)
        torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / num_runs


def run_attention_benchmark():
    """
    Run attention benchmark experiments and return results
    
    Returns:
        dict: Contains lists of results for different token counts
    """
    # Parameters
    bsz, num_text_tokens, num_heads, head_dim = 1, 1000, 28, 128
    scale = head_dim ** -0.5
    
    # Test different numbers of image tokens
    num_image_tokens_list = [100, 1000, 10000, 100000]
    
    results = {
        'num_image_tokens': [],
        'triton_time': [],
        'triton_memory': [],
        'pytorch_time': [],
        'pytorch_memory': [],
        'max_abs_error': [],
        'mse_error': []
    }
    
    for num_image_tokens in num_image_tokens_list:
        print(f"Testing with {num_image_tokens} image tokens...")
        
        # Allocate random bf16 inputs
        q = torch.randn(bsz, num_text_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(bsz, num_image_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        
        # Measure Triton memory & speed
        torch.cuda.reset_peak_memory_stats()
        t_triton = benchmark(flash_attn_kv_score, (q, k, scale))
        mem_triton = torch.cuda.max_memory_allocated() / (1024**2)  # MiB
        
        # Measure torch reference memory & speed
        torch.cuda.reset_peak_memory_stats()
        t_ref = benchmark(torch_ref_score, (q, k, scale))
        mem_ref = torch.cuda.max_memory_allocated() / (1024**2)  # MiB
        
        # Run once for error measurement
        score_triton = flash_attn_kv_score(q, k, scale)
        score_ref = torch_ref_score(q, k, scale)
        
        max_abs = (score_triton - score_ref).abs().max().item()
        mse = torch.mean((score_triton - score_ref).float().pow(2)).item()
        
        # Store results
        results['num_image_tokens'].append(num_image_tokens)
        results['triton_time'].append(t_triton * 1e6)  # Convert to microseconds
        results['triton_memory'].append(mem_triton)
        results['pytorch_time'].append(t_ref * 1e6)  # Convert to microseconds
        results['pytorch_memory'].append(mem_ref)
        results['max_abs_error'].append(max_abs)
        results['mse_error'].append(mse)
        
        print(f"Triton: {t_triton*1e6:.2f} µs, {mem_triton:.1f} MiB")
        print(f"PyTorch: {t_ref*1e6:.2f} µs, {mem_ref:.1f} MiB")
        print(f"Max abs error: {max_abs:.3e}, MSE: {mse:.3e}\n")
    
    return results


if __name__ == "__main__":
    import json
    results = run_attention_benchmark()
    print(json.dumps(results, indent=2, ensure_ascii=False))
