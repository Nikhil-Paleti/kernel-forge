# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/matrix-multiplication
"""

import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    c_ptrs = c + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        
        a_ptrs = a + (offs_m[:, None]*stride_am + offs_n[None, :]*stride_an)
        b_ptrs = b + (offs_n[:, None]*stride_bn + offs_k[None, :]*stride_bk)

        a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.)
        acc += tl.dot(a_block, b_block, input_precision="ieee")

    
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    # c = acc.to(tl.float16)
    tl.store(c_ptrs, acc, mask=c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_ck = c.stride()
    
    # A is MN
    # B is NK
    # C is MK

    BLOCK_M = 64 
    BLOCK_N = 32 # reduction dimension 
    BLOCK_K = 64 

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K), ) 

    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4, num_stages=1
    )