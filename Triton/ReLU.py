# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

import torch
import triton
import triton.language as tl

"""
https://leetgpu.com/challenges/relu
"""

@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(input + offs, mask=mask)
    tl.store(output + offs, tl.maximum(x,0), mask=mask)
    
# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)
