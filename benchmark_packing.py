import sys
import os
import torch
import time

# Setup ModelOpt import
MODELOPT_PATH = "/home/josephL/Model-Optimizer"
if MODELOPT_PATH not in sys.path:
    sys.path.insert(0, MODELOPT_PATH)
try:
    import modelopt
except ImportError:
    # Patch modelopt if needed (common for local clones)
    import types
    model_opt_pkg = types.ModuleType("modelopt")
    model_opt_pkg.__path__ = [os.path.join(MODELOPT_PATH, "modelopt")]
    model_opt_pkg.__version__ = "0.0.0-dev"
    sys.modules["modelopt"] = model_opt_pkg

from openrlhf.utils.mxfp4_quantize import quantize_to_mxfp4, _quantize_to_mxfp4_triton
from openrlhf.utils.nvfp4_quantize import quantize_to_nvfp4, _quantize_to_nvfp4_triton, _IS_BLACKWELL

def get_modelopt_mxfp4(tensor, block_size=32):
    from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor
    original_shape = tensor.shape
    qtensor, e8m0_scale = MXFP4QTensor.quantize(tensor, block_size=block_size)
    
    # Reshape scale to match our structured shape [..., last_dim // block_size]
    scale_shape = list(original_shape)
    scale_shape[-1] = scale_shape[-1] // block_size
    return qtensor._quantized_data, e8m0_scale.reshape(scale_shape)

def get_modelopt_nvfp4(tensor, block_size=16, global_scale=None):
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    # ModelOpt computes weights_scaling_factor_2 internally if not provided
    # but we want to be consistent with our global_scale
    if global_scale is not None:
        wsf2 = global_scale
    else:
        wsf2 = NVFP4QTensor.get_weights_scaling_factor_2(tensor)
    
    wsf, _ = NVFP4QTensor.get_weights_scaling_factor(tensor, block_size, weights_scaling_factor_2=wsf2)
    qtensor, wsf_out, wsf2_out = NVFP4QTensor.quantize(
        tensor, block_size,
        weights_scaling_factor=wsf,
        weights_scaling_factor_2=wsf2,
    )
    return qtensor._quantized_data, wsf_out, wsf2_out

def benchmark_fn(fn, name, tensor, **kwargs):
    # Warmup
    for _ in range(5):
        fn(tensor, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    iters = 50
    for _ in range(iters):
        fn(tensor, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) * 1000 / iters
    print(f"  {name:30}: {avg_ms:6.2f} ms")
    return avg_ms

def check_accuracy(ground_truth_packed, test_packed, ground_truth_scales, test_scales, label):
    packed_match = torch.equal(ground_truth_packed, test_packed)
    scales_match = torch.equal(ground_truth_scales, test_scales)
    
    print(f"\n[Correctness vs ModelOpt - {label}]")
    if packed_match:
        print(f"  ✅ {label} Packed uint8: EXACT MATCH")
    else:
        diff_count = (ground_truth_packed != test_packed).sum().item()
        print(f"  ❌ {label} Packed uint8: MISMATCH ({diff_count} elements differ)")
        
    if scales_match:
        print(f"  ✅ {label} Scales: EXACT MATCH")
    else:
        diff_count = (ground_truth_scales != test_scales).sum().item()
        print(f"  ❌ {label} Scales: MISMATCH ({diff_count} elements differ)")
    
    return packed_match and scales_match

if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(42)
    # 102.4M elements
    M, N = 10000, 10240
    tensor = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    
    print(f"\nBenchmarking Real Quantization (Packing) — {tensor.numel()/1e6:.1f}M elements")
    print("="*70)

    # --- MXFP4 ---
    print("\n[MXFP4 - block_size=32]")
    print("Ground Truth: ModelOpt")
    mo_packed_mx, mo_scales_mx = get_modelopt_mxfp4(tensor, block_size=32)
    
    pt_packed_mx, pt_scales_mx = quantize_to_mxfp4(tensor, block_size=32)
    tr_packed_mx, tr_scales_mx = _quantize_to_mxfp4_triton(tensor, block_size=32)
    
    check_accuracy(mo_packed_mx, pt_packed_mx, mo_scales_mx, pt_scales_mx, "PyTorch")
    check_accuracy(mo_packed_mx, tr_packed_mx, mo_scales_mx, tr_scales_mx, "Triton")
    
    print("\n[MXFP4 Timing]")
    benchmark_fn(get_modelopt_mxfp4, "ModelOpt (Reference)", tensor, block_size=32)
    pt_ms_mx = benchmark_fn(quantize_to_mxfp4, "PyTorch (Compiled)", tensor, block_size=32)
    tr_ms_mx = benchmark_fn(_quantize_to_mxfp4_triton, "Triton (Fused)", tensor, block_size=32)
    print(f"  Triton Speedup vs PyTorch: {pt_ms_mx / tr_ms_mx:.2f}x")

    # --- NVFP4 ---
    print("\n[NVFP4 - block_size=16]")
    if not _IS_BLACKWELL:
        print("  ⚠️ Skipping NVFP4 comparison: Native FP8 not supported on this GPU.")
    else:
        from openrlhf.utils.nvfp4_quantize import compute_nvfp4_global_scale
        gs = compute_nvfp4_global_scale(tensor)
        
        print("Ground Truth: ModelOpt")
        mo_packed_nv, mo_scales_nv, mo_gs_nv = get_modelopt_nvfp4(tensor, block_size=16, global_scale=gs)
        
        pt_packed_nv, pt_scales_nv, _ = quantize_to_nvfp4(tensor, block_size=16, global_scale=gs)
        tr_packed_nv, tr_scales_nv, _ = _quantize_to_nvfp4_triton(tensor, block_size=16, global_scale=gs)
        
        check_accuracy(mo_packed_nv, pt_packed_nv, mo_scales_nv, pt_scales_nv, "PyTorch")
        check_accuracy(mo_packed_nv, tr_packed_nv, mo_scales_nv, tr_scales_nv, "Triton")
        
        print("\n[NVFP4 Timing]")
        benchmark_fn(get_modelopt_nvfp4, "ModelOpt (Reference)", tensor, block_size=16, global_scale=gs)
        pt_ms_nv = benchmark_fn(quantize_to_nvfp4, "PyTorch (Compiled)", tensor, block_size=16, global_scale=gs)
        tr_ms_nv = benchmark_fn(_quantize_to_nvfp4_triton, "Triton (Fused)", tensor, block_size=16, global_scale=gs)
        print(f"  Triton Speedup vs PyTorch: {pt_ms_nv / tr_ms_nv:.2f}x")
