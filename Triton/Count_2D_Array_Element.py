# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

import torch
import triton
import triton.language as tl

"""
https://leetgpu.com/challenges/count-2d-array-element
"""

@triton.jit
def count_kernel(input_ptr, output_ptr, N, M, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N*M
    x = tl.load(input_ptr + offs, mask=mask)
    k_mask = x == K 
    tl.atomic_add(output_ptr, tl.sum(k_mask))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    
    N, M = input.shape
    total = N * M

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    count_kernel[grid](
        input,
        output,
        N, M, K,
        BLOCK_SIZE,
    )