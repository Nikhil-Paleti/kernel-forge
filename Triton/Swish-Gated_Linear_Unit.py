# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/swish-gated-linear-unit
"""

import torch
import triton
import triton.language as tl

@triton.jit
def swiglu(
    input, output, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N // 2 
    x1 = tl.load(input + offs, mask=mask)
    x2 = tl.load(input + offs + N//2, mask=mask)
    silu = x1 * tl.sigmoid(x1)
    y = silu * x2
    tl.store(output + offs, y, mask=mask) 

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](
        input, output, N, BLOCK_SIZE=BLOCK_SIZE
    )
