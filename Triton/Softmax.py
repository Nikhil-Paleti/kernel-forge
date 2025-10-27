# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/softmax
"""

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input, output,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # input = input.to(tl.pointer_type(tl.float32))
    # output = output.to(tl.pointer_type(tl.float32))

    m = tl.full((), -float("inf"), tl.float32)
    s = tl.zeros((), tl.float32)

    off = 0
    while off < N:
        idx  = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(input + idx, mask=mask, other=-float("inf")).to(tl.float32)

        bmax = tl.max(x, axis=0)
        new_m = tl.maximum(m, bmax)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

        off += BLOCK_SIZE

    off = 0
    while off < N:
        idx  = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(input + idx, mask=mask, other=-float("inf")).to(tl.float32)
        y = tl.exp(x - m) / s
        tl.store(output + idx, y, mask=mask)
        off += BLOCK_SIZE

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    softmax_kernel[(1,)](
        input, output,
        N,
        BLOCK_SIZE
    ) 