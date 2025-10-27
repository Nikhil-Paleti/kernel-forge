# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

import torch
import triton
import triton.language as tl

"""
https://leetgpu.com/challenges/vector-addition
"""

@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements 

    x = tl.load(a + offs, mask=mask)
    y = tl.load(b + offs, mask=mask)
    tl.store(c + offs, x+y, mask=mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)