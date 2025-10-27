# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/matrix-transpose
"""

import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input, output,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    inp_offs = offs_x[:, None] * stride_ir + offs_y[None, :] * stride_ic
    inp_mask = (offs_x[:, None] < rows) & (offs_y[None, :] < cols)
    inp = tl.load(input + inp_offs, mask=inp_mask)

    output_offs = offs_y[:, None] * stride_or + offs_x[None, :] * stride_oc 
    output_mask = (offs_y[:, None] < cols) & (offs_x[None, :] < rows)
    tl.store(output+output_offs, inp.T, mask=output_mask) 

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    BLOCK_X = 8
    BLOCK_Y = 8

    grid = (triton.cdiv(rows, BLOCK_Y), triton.cdiv(cols, BLOCK_X), )
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_X, BLOCK_Y
    ) 