# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

import torch
import triton
import triton.language as tl

"""
https://leetgpu.com/challenges/color-inversion
"""

@triton.jit
def invert_kernel(
    image,
    width, height,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE * 4) + tl.arange(0, BLOCK_SIZE * 4)
    mask = offs < (4 * width * height)
    x = tl.load(image + offs, mask=mask)

    ch = offs % 4
    is_alpha = ch == 3
    inv = 255 - x

    y = tl.where(is_alpha, x, inv)
    tl.store(image + offs, y, mask=mask)

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    
    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    ) 