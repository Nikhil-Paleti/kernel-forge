# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/reverse-array
"""

import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(
    input,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    first_ptrs = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
    mask = first_ptrs < (N // 2)
    first = tl.load(input + first_ptrs, mask=mask)
    
    last_ptrs = N - 1 - first_ptrs
    second = tl.load(input + last_ptrs, mask=mask)
    tl.store(input + first_ptrs, second, mask=mask)
    tl.store(input + last_ptrs, first, mask=mask)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)
    
    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    ) 