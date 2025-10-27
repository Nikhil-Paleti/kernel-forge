# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/2d-convolution
"""

import torch
import triton
import triton.language as tl


@triton.jit
def conv2d(
    input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
    input_sy, input_sx,
    kernel_sy, kernel_sx,
    output_sy, output_sx,
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr
):
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)

    out_H = input_rows - kernel_rows + 1
    out_W = input_cols - kernel_cols + 1

    oh = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    ow = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)

    mask_y = oh < out_H
    mask_x = ow < out_W
    mask_yx = mask_y[:, None] & mask_x[None, :]

    base_in = input + (oh[:, None] * input_sy + ow[None, :] * input_sx)
    acc = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)

    for kh in range(0, kernel_rows):
        for kw in range(0, kernel_cols):
            k = tl.load(kernel + kh * kernel_sy + kw * kernel_sx)
            inp = tl.load(base_in + kh * input_sy + kw * input_sx,
                          mask=mask_yx, other=0.0)
            acc += k * inp

    tl.store(output + (oh[:, None] * output_sy + ow[None, :] * output_sx),
             acc, mask=mask_yx)


def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
          
    input = input.reshape(input_rows, input_cols)
    kernel = kernel.reshape(kernel_rows, kernel_cols)
    output = output.reshape(input_rows - kernel_rows + 1, input_cols - kernel_cols + 1)

    input_sy, input_sx = input.stride()
    kernel_sy, kernel_sx = kernel.stride()
    output_sy, output_sx = output.stride()

    BLOCK_Y = 8
    BLOCK_X = 8
    M, N = output.shape  # rows, cols

    grid = (triton.cdiv(M, BLOCK_Y), triton.cdiv(N, BLOCK_X))

    conv2d[grid](
        input, kernel, output,
        input_rows, input_cols, kernel_rows, kernel_cols,
        input_sy, input_sx, kernel_sy, kernel_sx, output_sy, output_sx,
        BLOCK_Y, BLOCK_X
    )