# ------------------------------------------------------------------------------
# Backend : Triton
# Author  : Nikhil Paleti
# Note    : Original implementation by Nikhil Paleti
# ------------------------------------------------------------------------------

"""
https://leetgpu.com/challenges/fp16-dot-product
"""

import torch
import triton
import triton.language as tl

@triton.jit 
def dot(A, B, result, N, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N 
    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    output = tl.sum(a * b, dtype=tl.float32)

    tl.atomic_add(result, output, sem="relaxed")

# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    temp_result = torch.zeros((1,), dtype=torch.float32, device=result.device)
    BLOCK_SIZE = 1024 
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    dot[grid](
        A, B, temp_result, N, BLOCK_SIZE
    )
    result.copy_(temp_result.to(torch.float16))


