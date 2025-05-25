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

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals = [2**i for i in range(12, 28, 1)],
        x_log = True,
        line_arg='provider',
        line_vals=['triton', 'pytorch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args = {}
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
     
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y , quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench( lambda: add(x,y) , quantiles=quantiles)
         
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3) 

    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    test_add_kernel(2048)
    test_add_kernel(9078)
    test_add_kernel(15000)

    import sys 

    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='./benchmark/', print_data=True)
