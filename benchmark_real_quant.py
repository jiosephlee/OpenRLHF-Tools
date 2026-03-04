import torch
import time
from openrlhf.utils.mxfp4_quantize import quantize_to_mxfp4
from openrlhf.utils.nvfp4_quantize import quantize_to_nvfp4

def benchmark_quant(fn, name, tensor, **kwargs):
    # Warmup
    for _ in range(5):
        fn(tensor, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(20):
        fn(tensor, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) * 1000 / 20
    print(f"{name}: {avg_ms:.2f} ms")

if __name__ == "__main__":
    device = "cuda"
    # Benchmark with a large tensor (100M elements, similar to the main benchmark)
    M, N = 10000, 10240
    tensor = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    
    print(f"Benchmarking real quantization (packing) for {tensor.numel()/1e6:.0f}M elements")
    
    # Check MXFP4
    benchmark_quant(quantize_to_mxfp4, "quantize_to_mxfp4", tensor, block_size=32)
    
    # Check NVFP4
    # Note: NVFP4 requires 2D tensor of [M, N] where N % 16 == 0
    benchmark_quant(quantize_to_nvfp4, "quantize_to_nvfp4", tensor, block_size=16)
