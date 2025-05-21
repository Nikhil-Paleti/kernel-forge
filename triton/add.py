import torch 
import triton 
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
print(f'using device: {DEVICE}')


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    PID = tl.program_id(axis=0)
    block_start = PID * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements 
    x = tl.load(x_ptr + offset, mask=mask, other=None)
    y = tl.load(y_ptr + offset, mask=mask, other=None)
    output = x + y 
    tl.store(output_ptr+offset, output, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE,\
        f'DEVICE: {DEVICE}, x.device: {x.device}, y.device: {y.device}, output.device: {output.device}'

    n_elements = x.numel()
    grid = lambda meta : (triton.cdiv(n_elements, meta['BLOCK_SIZE'] ), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
    
def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    z_tri = add(x, y) 
    z_ref = x + y
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print('PASSED')

if __name__ == '__main__':
    test_add_kernel(2048)
    test_add_kernel(9078)
    test_add_kernel(15000)
    
