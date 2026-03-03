# NVFP4 / MXFP4 Fake Quantization — Findings

## Ground Truths

| Format | Ground Truth | Location |
|--------|-------------|----------|
| **MXFP4** | ModelOpt `MXFP4QTensor` | `~/Model-Optimizer/modelopt/torch/quantization/qtensor/mxfp4_tensor.py` |
| **NVFP4** | ModelOpt `NVFP4QTensor` | `~/Model-Optimizer/modelopt/torch/quantization/qtensor/nvfp4_tensor.py` |

**Why ModelOpt for NVFP4** (not vLLM's `ref_nvfp4_quant`): the production CUTLASS/FlashInfer inference kernels expect block scales in ModelOpt format (FP8 E4M3, range [0, 448]). Both checkpoint loaders (ModelOpt and Compressed-Tensors) normalize to `weight_global_scale = wsf2 = amax/(6*448)` before calling the kernel. `ref_nvfp4_quant` is a debug/emulation helper only — see evidence below.

## NVFP4: Two Conventions, One Math

Both produce the same result in infinite precision, but cast different values to FP8 E4M3:

| Convention | Per-block scale before FP8 cast | Magnitude | Dequant formula |
|---|---|---|---|
| **vLLM ref** | `gs * (vec_max / 6)` | tiny (~0.001) | `fp4 * (scale_fp8 / gs)` |
| **ModelOpt** | `vec_max / (6 * gs)` | large (0–448) | `fp4 * (scale_fp8 * gs)` |

Where `gs = amax / (6 * 448)` in both cases.

**ModelOpt is better** because FP8 E4M3 has range [0, 448] — casting values in [0, 448] uses the full dynamic range. The vLLM ref casts tiny values near 0, wasting FP8 precision.

The FP8 cast is lossy and nonlinear, so the two conventions produce slightly different dequantized values even though the math is equivalent.

## ModelOpt NVFP4 Roundtrip (the reference)

```python
# Global scale
wsf2 = amax / (6.0 * 448.0)           # small scalar, ~0.0004

# Per-block scale
pbs = per_block_amax / (6.0 * wsf2)   # = 448 * per_block_amax / amax, range [0, 448]
pbs[pbs == 0] = 1.0                   # zero-guard (all-zero blocks)
pbs_fp8 = pbs.to(float8_e4m3fn)       # FP8 cast at good magnitude

# Quantize
block_scale = pbs_fp8.to(float32) * wsf2   # effective dequant scale
normalized = x / block_scale               # puts values in [-6, 6]
fp4_val = snap_to_e2m1(normalized)

# Dequantize
result = fp4_val * block_scale
```

**Key differences from old vLLM-ref code:**
- Scale formula inverted: `vec_max / (6 * gs)` instead of `gs * (vec_max / 6)`
- Zero-guard: `pbs[pbs == 0] = 1.0` instead of `where(scale == 0, 0, ...)`
- Normalize: `x / (pbs_fp8 * gs)` instead of `x * (gs / scale_fp8)`
- Dequant: `fp4 * (pbs_fp8 * gs)` instead of `fp4 * (scale_fp8 / gs)`

## ModelOpt NVFP4 `_cast_fp4` Rounding

ModelOpt uses `searchsorted` + rounding adjustment at odd-indexed bounds:
```python
ord = searchsorted(e2m1_bounds, abs_weight)
# Round up at odd bounds [0.75, 1.75, 2.5] (indices 1, 3, 5)
equals_odd = any(abs_weight == odd_bounds, dim=-1)
return (sign_bit << 3) + ord + equals_odd
```
This implements IEEE round-to-nearest-even at tie points. Equivalent comparison-sum: use `>=` at odd bounds (0.75, 1.75, 3.5):
```python
ord_ = (abs_w > 0.25) + (abs_w >= 0.75) + (abs_w > 1.25) +
       (abs_w >= 1.75) + (abs_w > 2.5)  + (abs_w >= 3.5)  + (abs_w > 5.0)
```

## How We Know vLLM Uses ModelOpt Convention

### Key files in the vLLM codebase

| File | Role |
|---|---|
| `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py` | Backend selection, `apply_nvfp4_linear`, `convert_to_nvfp4_linear_kernel_format` |
| `vllm/model_executor/layers/quantization/modelopt.py` | ModelOpt checkpoint loader (`process_weights_after_loading`) |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` | Compressed-Tensors format loader |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` | CT MoE loader |
| `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py` | **Debug/emulation only** — not the production path |

### Both checkpoint formats normalize to ModelOpt convention

**ModelOpt checkpoints** (`modelopt.py:1192–1205`):
```python
weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
# weight_scale_2 IS wsf2 = amax/(6*448) from the checkpoint — used directly, no inversion
layer.weight_global_scale = Parameter(weight_global_scale, ...)
layer.alpha = Parameter(layer.input_global_scale * layer.weight_global_scale, ...)
layer.input_global_scale_inv = Parameter(1.0 / layer.input_global_scale, ...)
```

**Compressed-Tensors checkpoints** (`compressed_tensors_w4a4_nvfp4.py:92–108`):
```python
# CT stores global scales as DIVISORS (1/wsf2), comment says so explicitly:
# "Process global scales (CT stores as divisors, i.e. 1/scale)"
weight_global_scale = layer.weight_global_scale.max().to(torch.float32)  # = 1/wsf2
layer.weight_global_scale = Parameter(1.0 / weight_global_scale, ...)    # inverted → wsf2
layer.alpha = Parameter(layer.input_global_scale * layer.weight_global_scale, ...)
```

Both converge to `layer.weight_global_scale = wsf2 = amax/(6*448)` and the same `alpha`.

### Production inference path (`nvfp4_utils.py:170–263`)

`apply_nvfp4_linear` is called for every linear layer at inference time:

```python
# Quantize activations to FP4 (uses input_global_scale_INV = 1/gs_input)
x_fp4, x_blockscale = scaled_fp4_quant(x, input_global_scale_inv, ...)

# Call CUTLASS / FlashInfer / FBGEMM kernel
cutlass_scaled_fp4_mm(x_fp4, weight, x_blockscale, weight_scale, alpha, output_dtype)
#                                                                  ^^^^^
#                     alpha = input_gs * weight_gs — combined output rescaling scalar
```

The kernel receives:
- `weight_scale`: FP8 E4M3 block scales in [0, 448] (ModelOpt format, loaded from checkpoint)
- `alpha = input_gs * weight_gs`: the two global scales multiplied together

The CUTLASS/FlashInfer FP4 GEMM kernel internally computes:
```
out ≈ (x_fp4 * x_blockscale) @ (weight * weight_scale).T * alpha
```
Which dequantizes weights as `fp4 * (block_scale_fp8 * weight_gs)` — exactly ModelOpt convention.

### Why `ref_nvfp4_quant` is NOT the ground truth

`nvfp4_emulation_utils.py` is only used via `NvFp4LinearBackend.EMULATION`, activated by the env var `VLLM_USE_NVFP4_CT_EMULATIONS`. This is a debug/fallback mode, never used in production on B200s (which use CUTLASS or FlashInfer). The function `ref_nvfp4_quant` is called with `input_global_scale_inv` (= `1/gs`), so its internal scale formula `gs * vec_max/6` evaluates to `(1/gs) * vec_max/6` = large values — which coincidentally matches ModelOpt format. But the function itself is ambiguous and not the canonical reference.

## MXFP4: Already Correct

Our MXFP4 code matches ModelOpt exactly (E8M0 exponent-based scales, block_size=32). No formula changes needed.

## Files

| File | Role |
|---|---|
| `openrlhf/utils/mxfp4_quantize.py` | MXFP4 fake quant (QAT) + weight sync quantization |
| `openrlhf/utils/nvfp4_quantize.py` | NVFP4 fake quant (QAT) + weight sync quantization |
| `tests/test_fake_quant_optimization.py` | Benchmark + correctness vs ModelOpt |

## Weight Sync Note

`quantize_to_nvfp4` (weight sync, NOT fake quant) should also be updated to ModelOpt convention, but was deferred. When done: same formula change as fake quant, PLUS the `quantize_to_nvfp4` function docstring says "vLLM ref" but should say "ModelOpt".

## Performance Context

- MXFP4: Triton kernel (fused, one pass) + `@torch.compile` PyTorch fallback
- NVFP4: Two-pass (PyTorch FP8 cast → Triton snap+dequant); can't fuse because FP8 cast isn't supported in Triton on Ampere. `combined_scale = pbs_fp8 * gs` precomputed in pass 1 and passed to kernel to avoid recomputing.
- `torch.compile` can't be used for NVFP4 on Ampere (current dev GPU); will work on B200s.
