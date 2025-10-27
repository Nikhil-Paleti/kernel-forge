# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/count-array-element
"""

import torch
import triton
import triton.language as tl

@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N 
    x = tl.load(input_ptr + offs, mask=mask)
    valid = x == K 
    tl.atomic_add(output_ptr, tl.sum(valid))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)