#!/usr/bin/env python3
"""Benchmark and correctness test for fake dequantization optimizations.

Tests the optimized fake_quantize_mxfp4 and fake_quantize_nvfp4 against:
  1. A baseline implementation (original chunked + torch.bucketize)
  2. NVIDIA ModelOpt's quantize→dequantize round-trip (if repo is available)

Simulates realistic MoE model dequantization workloads with configurable
model profiles (expert count, hidden size, layers).

Usage:
    conda run -n finetuning python tests/test_fake_quant_optimization.py
    conda run -n finetuning python tests/test_fake_quant_optimization.py --profile small
    conda run -n finetuning python tests/test_fake_quant_optimization.py --profile gpt-oss
"""

import sys
import os
import time
import torch
import argparse


# ---------------------------------------------------------------------------
# Model profiles: simulated MoE expert weight shapes per layer
# Each profile defines the shapes of expert weight tensors that get
# fake-quantized during QAT. In a real forward pass, the parametrization
# fires on every expert proj in every MoE layer.
#
# Shapes are [E, in_features, out_features] (Actor convention).
# gate_up_proj is [E, hidden, intermediate*2], down_proj is [E, intermediate, hidden].
# ---------------------------------------------------------------------------

MODEL_PROFILES = {
    # Small profile for quick iteration on a 24GB GPU
    "small": {
        "name": "Small MoE (fits 24GB)",
        "num_layers": 4,
        "expert_shapes": {
            "gate_up_proj": (8, 1024, 2048),   # [E=8, in=1024, out=2048]
            "down_proj":    (8, 1024, 1024),   # [E=8, in=1024, out=1024]
        },
    },
    # Medium: roughly Qwen3-MoE-A3B scale (128 experts, ~60 layers)
    # Scaled down to fit 24GB — fewer layers, fewer experts
    "medium": {
        "name": "Medium MoE (scaled for 24GB)",
        "num_layers": 8,
        "expert_shapes": {
            "gate_up_proj": (16, 2048, 4096),  # [E=16, in=2048, out=4096]
            "down_proj":    (16, 2048, 2048),  # [E=16, in=2048, out=2048]
        },
    },
    # GPT-OSS full scale (for reference — may OOM on 24GB GPU)
    "gpt-oss": {
        "name": "GPT-OSS MoE (full scale)",
        "num_layers": 36,
        "expert_shapes": {
            "gate_up_proj": (64, 4096, 14336),  # [E=64, in=4096, out=14336]
            "down_proj":    (64, 7168, 4096),   # [E=64, in=7168, out=4096]
        },
    },
    # Single tensor — for micro-benchmarking one weight
    "single": {
        "name": "Single expert tensor",
        "num_layers": 1,
        "expert_shapes": {
            "gate_up_proj": (8, 4096, 4096),
        },
    },
}

# ---------------------------------------------------------------------------
# ModelOpt cross-check helpers
# ---------------------------------------------------------------------------

MODELOPT_PATH = "/home/josephL/Model-Optimizer"


def _setup_modelopt_import():
    """Setup imports for ModelOpt: try pip-installed first, then fallback to local clone."""
    try:
        import modelopt
        # If it imports successfully, we don't need to patch sys.path
        return
    except ImportError:
        pass

    if MODELOPT_PATH not in sys.path:
        sys.path.insert(0, MODELOPT_PATH)
    # Patch modelopt.__init__ which calls importlib.metadata.version('nvidia-modelopt')
    import types
    modelopt_pkg = types.ModuleType("modelopt")
    modelopt_pkg.__path__ = [os.path.join(MODELOPT_PATH, "modelopt")]
    modelopt_pkg.__version__ = "0.0.0-dev"
    sys.modules["modelopt"] = modelopt_pkg


def modelopt_mxfp4_roundtrip(weight, block_size=32):
    """Quantize via ModelOpt MXFP4 → dequantize, returning the round-trip result."""
    _setup_modelopt_import()
    from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor

    qtensor, e8m0_scale = MXFP4QTensor.quantize(weight, block_size=block_size)
    deq = qtensor.dequantize(
        dtype=weight.dtype,
        scale=e8m0_scale,
        block_sizes={-1: block_size},
    )
    return deq


def modelopt_nvfp4_roundtrip(weight, block_size=16):
    """Quantize via ModelOpt NVFP4 → dequantize, returning the round-trip result."""
    _setup_modelopt_import()
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    original_shape = weight.shape
    w2d = weight.reshape(-1, weight.shape[-1]) if weight.ndim != 2 else weight

    wsf2 = NVFP4QTensor.get_weights_scaling_factor_2(w2d)
    wsf, _ = NVFP4QTensor.get_weights_scaling_factor(
        w2d, block_size, weights_scaling_factor_2=wsf2
    )
    qtensor, wsf_out, wsf2_out = NVFP4QTensor.quantize(
        w2d, block_size,
        weights_scaling_factor=wsf,
        weights_scaling_factor_2=wsf2,
    )
    deq = qtensor.dequantize(
        dtype=weight.dtype,
        scale=wsf_out,
        double_scale=wsf2_out,
        block_sizes={-1: block_size},
    )
    return deq.reshape(original_shape)


# ---------------------------------------------------------------------------
# OpenRLHF Weight Sync Baselines (Production Quantization)
# ---------------------------------------------------------------------------

def openrlhf_mxfp4_sync_roundtrip(weight, block_size=32):
    """Production quantization (weight sync) used in OpenRLHF."""
    from openrlhf.utils.mxfp4_quantize import quantize_to_mxfp4, E2M1_VALUES
    
    # Quantize
    packed, e8m0_scales = quantize_to_mxfp4(weight, block_size=block_size)
    
    # Dequantize (local emulation for benchmark)
    orig_shape = weight.shape
    device = weight.device
    
    # Unpack uint8 -> uint4
    left = packed & 0x0F
    right = (packed >> 4) & 0x0F
    unpacked = torch.stack([left, right], dim=-1).reshape(-1)
    
    # Extract sign and magnitude
    sign = 1 - 2 * ((unpacked & 0b1000) >> 3).float()
    magnitude = (unpacked & 0b0111).long()
    
    vals = E2M1_VALUES.to(device)
    deq = sign * vals[magnitude]
    
    # Scale
    deq = deq.reshape(-1, block_size)
    scale_factor = torch.exp2(e8m0_scales.float() - 127).reshape(-1, 1)
    deq = (deq * scale_factor).reshape(orig_shape)
    
    return deq.to(weight.dtype)


def openrlhf_nvfp4_sync_roundtrip(weight, block_size=16):
    """Production quantization (weight sync) used in OpenRLHF."""
    from openrlhf.utils.nvfp4_quantize import quantize_to_nvfp4, E2M1_VALUES
    
    # Quantize
    orig_shape = weight.shape
    w2d = weight.reshape(-1, weight.shape[-1]) if weight.ndim != 2 else weight
    packed, wsf, wsf2 = quantize_to_nvfp4(w2d, block_size=block_size)
    
    # Dequantize (local emulation for benchmark)
    device = weight.device
    
    # Unpack uint8 -> uint4
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).reshape(-1)
    
    # Sign bit is bit 3 (value 8)
    sign = 1 - 2 * ((unpacked & 0b1000) >> 3).float()
    # Magnitude is bits 0-2 (values 0-7)
    magnitude = (unpacked & 0b0111).long()
    
    vals = E2M1_VALUES.to(device)
    deq = sign * vals[magnitude]
    
    # Scale
    deq = deq.reshape(-1, block_size)
    # double_scale (wsf2) is per-tensor, scale (wsf) is per-block (FP8 E4M3)
    # ModelOpt convention: dequant = fp4 * (wsf_fp8 * wsf2)
    effective_scale = (wsf.float() * wsf2).reshape(-1, 1)
    deq = (deq * effective_scale).reshape(orig_shape)
    
    return deq.to(weight.dtype)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_model_fakequant(fn, weight_tensors, warmup=2, repeat=5, label=""):
    """Benchmark fake-quantizing all expert weight tensors (simulates one forward pass).

    Args:
        fn: The fake_quantize function to call on each tensor
        weight_tensors: List of (name, tensor) pairs representing all expert weights
        warmup: Number of warmup iterations
        repeat: Number of timed iterations
        label: Display label
    """
    total_elements = sum(t.numel() for _, t in weight_tensors)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        for _, w in weight_tensors:
            _ = fn(w)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _, w in weight_tensors:
            _ = fn(w)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated()
    mem_used = peak_mem - mem_before

    mean_t = sum(times) / len(times)
    std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5

    print(f"  {label:45s}  {mean_t*1000:9.1f} ± {std_t*1000:6.1f} ms   "
          f"peak={peak_mem/1e9:.2f} GB  delta={mem_used/1e9:.2f} GB  "
          f"({total_elements/1e6:.0f}M elements)")
    return mean_t, peak_mem


def check_correctness(fn_a, fn_b, weight_tensors, label_a, label_b, atol=0, rtol=0):
    """Check if two fake-quantize functions produce identical results on all tensors."""
    all_match = True
    max_diff_overall = 0.0
    for name, w in weight_tensors:
        result_a = fn_a(w)
        result_b = fn_b(w)
        if not torch.equal(result_a, result_b):
            max_diff = (result_a - result_b).abs().max().item()
            max_diff_overall = max(max_diff_overall, max_diff)
            if not torch.allclose(result_a, result_b, atol=atol, rtol=rtol):
                all_match = False

    if all_match and max_diff_overall == 0:
        print(f"  ✅ {label_a} vs {label_b}: EXACT MATCH (all {len(weight_tensors)} tensors)")
    elif all_match:
        print(f"  ✅ {label_a} vs {label_b}: CLOSE MATCH (max_diff={max_diff_overall:.2e})")
    else:
        print(f"  ❌ {label_a} vs {label_b}: MISMATCH (max_diff={max_diff_overall:.2e})")
    return all_match


def create_model_weights(profile, device, dtype):
    """Create synthetic expert weight tensors for the given model profile."""
    weights = []
    torch.manual_seed(42)
    for layer_idx in range(profile["num_layers"]):
        for proj_name, shape in profile["expert_shapes"].items():
            name = f"layers.{layer_idx}.mlp.experts.{proj_name}"
            w = torch.randn(shape, device=device, dtype=dtype)
            weights.append((name, w))
    return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark fake dequantization")
    parser.add_argument("--profile", type=str, default="small",
                        choices=list(MODEL_PROFILES.keys()),
                        help="Model profile to simulate")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-modelopt", action="store_true",
                        help="Skip ModelOpt cross-check")
    parser.add_argument("--mxfp4-only", action="store_true", help="Only test MXFP4")
    parser.add_argument("--nvfp4-only", action="store_true", help="Only test NVFP4")
    args = parser.parse_args()

    profile = MODEL_PROFILES[args.profile]
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n{'='*90}")
    print(f"Fake Dequantization Benchmark — Whole-Model Simulation")
    print(f"{'='*90}")
    print(f"Profile: {profile['name']}")
    print(f"Layers: {profile['num_layers']}, Projections per layer: {len(profile['expert_shapes'])}")
    for proj_name, shape in profile["expert_shapes"].items():
        numel = 1
        for s in shape:
            numel *= s
        print(f"  {proj_name}: {shape}  ({numel/1e6:.1f}M elements)")
    total_tensors = profile["num_layers"] * len(profile["expert_shapes"])
    total_elements = sum(
        profile["num_layers"] * torch.tensor(shape).prod().item()
        for shape in profile["expert_shapes"].values()
    )
    print(f"Total: {total_tensors} tensors, {total_elements/1e6:.0f}M elements "
          f"({total_elements * 2 / 1e9:.2f} GB in bf16)")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    # Create weight tensors
    weights = create_model_weights(profile, device, dtype)

    # Import optimized versions
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    run_mxfp4 = not args.nvfp4_only
    run_nvfp4 = not args.mxfp4_only

    # -----------------------------------------------------------------------
    # MXFP4 tests
    # -----------------------------------------------------------------------
    if run_mxfp4:
        print(f"{'─'*90}")
        print("MXFP4 Fake Quantization (block_size=32)")
        print(f"{'─'*90}")

        from openrlhf.utils.mxfp4_quantize import (
            fake_quantize_mxfp4,
            _fake_quantize_mxfp4_chunk,
            _HAS_TRITON as MXFP4_HAS_TRITON,
            E2M1_VALUES as MX_VALUES,
        )
        if MXFP4_HAS_TRITON:
            from openrlhf.utils.mxfp4_quantize import _fake_quantize_mxfp4_triton

        # Compiled PyTorch wrapper (uses @torch.compile'd _fake_quantize_mxfp4_chunk)
        def compiled_mxfp4(w, block_size=32):
            vals = MX_VALUES.to(w.device)
            w_blocks = w.float().reshape(-1, block_size)
            dq = _fake_quantize_mxfp4_chunk(w_blocks, block_size, vals)
            dq = dq.reshape(w.shape).to(w.dtype)
            return w + (dq - w).detach()

        # Triton wrapper
        def triton_mxfp4(w):
            from openrlhf.utils.mxfp4_quantize import _fake_quantize_mxfp4_triton
            dq = _fake_quantize_mxfp4_triton(w)
            return w + (dq - w).detach()

        # ModelOpt wrapper (handles 3D expert tensors)
        def modelopt_mxfp4(w):
            orig_shape = w.shape
            w2d = w.reshape(-1, w.shape[-1])
            return modelopt_mxfp4_roundtrip(w2d, block_size=32).reshape(orig_shape)

        print("\n[Timing — full model pass]")
        compiled_time, _ = benchmark_model_fakequant(
            compiled_mxfp4, weights,
            warmup=args.warmup, repeat=args.repeat,
            label="Compiled PyTorch (B+C+D: torch.compile)"
        )

        triton_time = None
        if MXFP4_HAS_TRITON:
            triton_time, _ = benchmark_model_fakequant(
                triton_mxfp4, weights,
                warmup=args.warmup, repeat=args.repeat,
                label="Triton kernel (fused)"
            )

        modelopt_time = None
        if not args.skip_modelopt:
            try:
                modelopt_time, _ = benchmark_model_fakequant(
                    modelopt_mxfp4, weights,
                    warmup=args.warmup, repeat=args.repeat,
                    label="ModelOpt (quantize→dequantize roundtrip)"
                )
            except Exception as e:
                print(f"  ⚠️ ModelOpt benchmark failed: {e}")

        sync_baseline_time, _ = benchmark_model_fakequant(
            openrlhf_mxfp4_sync_roundtrip, weights,
            warmup=args.warmup, repeat=args.repeat,
            label="Baseline (Production Weight Sync)"
        )

        ref_time = modelopt_time if modelopt_time is not None else compiled_time
        ref_name = "ModelOpt" if modelopt_time is not None else "Compiled"
        
        compiled_speedup = ref_time / compiled_time if compiled_time > 0 else float('inf')
        print(f"\n  Compiled speedup: {compiled_speedup:.1f}x  "
              f"({ref_time*1000:.0f}ms [{ref_name}] → {compiled_time*1000:.0f}ms)")
              
        if triton_time is not None:
            triton_speedup = ref_time / triton_time if triton_time > 0 else float('inf')
            print(f"  Triton speedup:   {triton_speedup:.1f}x  "
                  f"({ref_time*1000:.0f}ms [{ref_name}] → {triton_time*1000:.0f}ms)")

        print("\n[Correctness]")
        ref_fn = modelopt_mxfp4 if not args.skip_modelopt else compiled_mxfp4
        ref_name = "ModelOpt" if not args.skip_modelopt else "Compiled"
        
        if not args.skip_modelopt:
            try:
                check_correctness(
                    ref_fn, compiled_mxfp4,
                    weights, ref_name, "Compiled"
                )
            except Exception as e:
                print(f"  ⚠️ ModelOpt correctness check failed: {e}")
                
        if MXFP4_HAS_TRITON:
            check_correctness(
                ref_fn, triton_mxfp4,
                weights, ref_name, "Triton"
            )
            
        check_correctness(
            ref_fn, openrlhf_mxfp4_sync_roundtrip,
            weights, ref_name, "Sync Baseline"
        )

    # -----------------------------------------------------------------------
    # NVFP4 tests
    # -----------------------------------------------------------------------
    if run_nvfp4:
        # Ensure last dim divisible by 16 for NVFP4
        nvfp4_weights = []
        for name, w in weights:
            if w.shape[-1] % 16 != 0:
                pad = 16 - (w.shape[-1] % 16)
                w = torch.nn.functional.pad(w, (0, pad))
            nvfp4_weights.append((name, w))

        print(f"\n{'─'*90}")
        print("NVFP4 Fake Quantization (block_size=16)")
        print(f"{'─'*90}")

        from openrlhf.utils.nvfp4_quantize import (
            fake_quantize_nvfp4, compute_nvfp4_global_scale,
            _fake_quantize_nvfp4_chunk,
            _HAS_TRITON as NVFP4_HAS_TRITON,
            E2M1_VALUES as NV_VALUES,
        )
        if NVFP4_HAS_TRITON:
            from openrlhf.utils.nvfp4_quantize import _fake_quantize_nvfp4_triton

        # Compiled PyTorch wrapper (uses manual comparisons, no chunking)
        def compiled_nvfp4(w, block_size=16):
            gs = compute_nvfp4_global_scale(w)
            vals = NV_VALUES.to(w.device)
            w_blocks = w.float().reshape(-1, block_size)
            dq = _fake_quantize_nvfp4_chunk(w_blocks, block_size, gs, vals)
            dq = dq.reshape(w.shape).to(w.dtype)
            return w + (dq - w).detach()

        # Triton wrapper
        def triton_nvfp4(w):
            from openrlhf.utils.nvfp4_quantize import _fake_quantize_nvfp4_triton
            gs = compute_nvfp4_global_scale(w)
            dq = _fake_quantize_nvfp4_triton(w, 16, gs)
            return w + (dq - w).detach()

        # ModelOpt wrapper
        def modelopt_nvfp4(w):
            orig_shape = w.shape
            w2d = w.reshape(-1, w.shape[-1])
            return modelopt_nvfp4_roundtrip(w2d, block_size=16).reshape(orig_shape)

        print("\n[Timing — full model pass]")
        compiled_time_nv, _ = benchmark_model_fakequant(
            compiled_nvfp4, nvfp4_weights,
            warmup=args.warmup, repeat=args.repeat,
            label="Compiled PyTorch (B+C: manual cmp, no-chunk)"
        )

        triton_time_nv = None
        if NVFP4_HAS_TRITON:
            triton_time_nv, _ = benchmark_model_fakequant(
                triton_nvfp4, nvfp4_weights,
                warmup=args.warmup, repeat=args.repeat,
                label="Triton kernel (fused scale + quantize)"
            )

        modelopt_time_nv = None
        if not args.skip_modelopt:
            try:
                modelopt_time_nv, _ = benchmark_model_fakequant(
                    modelopt_nvfp4, nvfp4_weights,
                    warmup=args.warmup, repeat=args.repeat,
                    label="ModelOpt (quantize→dequantize roundtrip)"
                )
            except Exception as e:
                print(f"  ⚠️ ModelOpt benchmark failed: {e}")

        sync_baseline_time_nv, _ = benchmark_model_fakequant(
            openrlhf_nvfp4_sync_roundtrip, nvfp4_weights,
            warmup=args.warmup, repeat=args.repeat,
            label="Baseline (Production Weight Sync)"
        )

        ref_time_nv = modelopt_time_nv if modelopt_time_nv is not None else compiled_time_nv
        ref_name_nv = "ModelOpt" if modelopt_time_nv is not None else "Compiled"
        
        compiled_speedup_nv = ref_time_nv / compiled_time_nv if compiled_time_nv > 0 else float('inf')
        print(f"\n  Compiled speedup: {compiled_speedup_nv:.1f}x  "
              f"({ref_time_nv*1000:.0f}ms [{ref_name_nv}] → {compiled_time_nv*1000:.0f}ms)")
              
        if triton_time_nv is not None:
            triton_speedup_nv = ref_time_nv / triton_time_nv if triton_time_nv > 0 else float('inf')
            print(f"  Triton speedup:   {triton_speedup_nv:.1f}x  "
                  f"({ref_time_nv*1000:.0f}ms [{ref_name_nv}] → {triton_time_nv*1000:.0f}ms)")

        print("\n[Correctness]")
        ref_fn_nv = modelopt_nvfp4 if not args.skip_modelopt else compiled_nvfp4
        ref_name_nv = "ModelOpt" if not args.skip_modelopt else "Compiled"
        
        if not args.skip_modelopt:
            try:
                check_correctness(
                    ref_fn_nv, compiled_nvfp4,
                    nvfp4_weights, ref_name_nv, "Compiled"
                )
            except Exception as e:
                print(f"  ⚠️ ModelOpt correctness check failed: {e}")
                
        if NVFP4_HAS_TRITON:
            check_correctness(
                ref_fn_nv, triton_nvfp4,
                nvfp4_weights, ref_name_nv, "Triton"
            )

        check_correctness(
            ref_fn_nv, openrlhf_nvfp4_sync_roundtrip,
            nvfp4_weights, ref_name_nv, "Sync Baseline"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("Summary")
    print(f"{'='*90}")
    print(f"Profile: {profile['name']} ({total_tensors} tensors, "
          f"{total_elements/1e6:.0f}M elements)")
    if run_mxfp4:
        ref_time = modelopt_time if modelopt_time is not None else compiled_time
        ref_name = "ModelOpt" if modelopt_time is not None else "Compiled"
        line = f"  MXFP4: [Ref: {ref_name}] {ref_time*1000:.0f}ms"
        if modelopt_time is not None:
            line += f" | Compiled {compiled_time*1000:.0f}ms ({ref_time/compiled_time:.1f}x)"
        if triton_time is not None:
            line += f" | Triton {triton_time*1000:.0f}ms ({ref_time/triton_time:.1f}x)"
        line += f" | Sync {sync_baseline_time*1000:.0f}ms ({ref_time/sync_baseline_time:.1f}x)"
        print(line)
    if run_nvfp4:
        ref_time_nv = modelopt_time_nv if modelopt_time_nv is not None else compiled_time_nv
        ref_name_nv = "ModelOpt" if modelopt_time_nv is not None else "Compiled"
        line = f"  NVFP4: [Ref: {ref_name_nv}] {ref_time_nv*1000:.0f}ms"
        if modelopt_time_nv is not None:
            line += f" | Compiled {compiled_time_nv*1000:.0f}ms ({ref_time_nv/compiled_time_nv:.1f}x)"
        if triton_time_nv is not None:
            line += f" | Triton {triton_time_nv*1000:.0f}ms ({ref_time_nv/triton_time_nv:.1f}x)"
        line += f" | Sync {sync_baseline_time_nv*1000:.0f}ms ({ref_time_nv/sync_baseline_time_nv:.1f}x)"
        print(line)
    print()


if __name__ == "__main__":
    main()

