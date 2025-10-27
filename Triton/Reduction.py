# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/reduction
"""

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(input, output, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N 
    x = tl.load(input + offs, mask = mask, other=0.0)
    s = tl.sum(x, dtype=tl.float32)
    tl.atomic_add(output, s, sem = "relaxed")

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024 
    grid = (triton.cdiv(N , BLOCK_SIZE),)
    add_kernel[grid](
        input, output,
        N,
        BLOCK_SIZE
    )