# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

import torch
import triton
import triton.language as tl

"""
https://leetgpu.com/challenges/matrix-copy
"""

@triton.jit
def copy_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N*N 
    a = tl.load(a_ptr + offs , mask = mask )
    tl.store(b_ptr + offs, a, mask = mask )

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    num_elements = N * N 
    BLOCK_SIZE = 1024 
    grid = (triton.cdiv(num_elements, BLOCK_SIZE), )

    copy_kernel[grid](
        a,
        b,
        N,
        BLOCK_SIZE
    )