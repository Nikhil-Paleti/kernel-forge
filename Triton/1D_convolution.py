# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/1d-convolution
"""

import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input, kernel, output,
    input_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    max_o = input_size - kernel_size + 1
    mask = offs < max_o 

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for k in range(kernel_size):
        x = tl.load(input + offs + k, mask = mask, other=0.0)
        w = tl.load(kernel + k)
        acc += x * w 

    tl.store(output + offs, acc, mask=mask)

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    
    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )