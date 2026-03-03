# Getting QAT to Work for FP4 Models

Quantization-Aware Training (QAT) inserts a fake-quantize operation into the forward pass so the model learns to be robust to the quantization error it will see at inference. This doc covers the specific challenges and current state for MXFP4 and NVFP4.

---

## The Core Idea

A fake-quantizer simulates the quantize→dequantize roundtrip:

```
weight_bf16  →  [quantize to FP4]  →  [dequantize back to BF16]  →  weight_fq_bf16
```

`weight_fq_bf16` has the same dtype/shape as `weight_bf16` but its values are restricted to whatever the FP4 format can represent. The model trains with these degraded weights, and gradients flow through unchanged (straight-through estimator). The result is a model that is robust to FP4 quantization at inference.

**What matters**: the fake-quantizer must simulate the *exact same roundtrip* that will happen at inference. If the fake-quantizer uses a different formula than the inference kernel, the model trains against the wrong noise and QAT won't help.

---

## MXFP4

### Inference path
- **Format**: FP4 E2M1 weights, E8M0 per-block scales, block_size=32, no global scale
- **Kernel**: W4A16 — **activations stay in BF16**
  - SM90/Ampere: Marlin GEMM (`apply_fp4_marlin_linear` in `marlin_utils_fp4.py`)
  - SM100/Blackwell: FlashInfer MXFP4+MXFP8 CUTLASS (activations quantized to MXFP8, not MXFP4)
- **Key file**: `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py`

### QAT situation: clean
Because inference is W4A16, **weight-only fake-quantization is an exact simulation** — there is no activation quantization gap to worry about. The fake-quantizer just needs to match the weight roundtrip precisely.

### Ground truth: ModelOpt
`~/Model-Optimizer/modelopt/torch/quantization/qtensor/mxfp4_tensor.py`

Roundtrip:
```python
# Per-block E8M0 scale
amax = block.abs().max()
e8m0_exp = ceil(max(log2(amax / 6.0), -127))
scale = 2 ** e8m0_exp          # exact power of two, no precision loss

# Quantize
normalized = x / scale         # puts values in [-6, 6]
fp4_val = snap_to_e2m1(normalized)

# Dequantize
result = fp4_val * scale
```

### Implementation: `openrlhf/utils/mxfp4_quantize.py`
- **Triton kernel** (`_fake_quantize_mxfp4_triton`): fused single-pass, preferred path on CUDA
- **PyTorch fallback** (`_fake_quantize_mxfp4_chunk`): `@torch.compile(mode="reduce-overhead")`, used when Triton unavailable

### Correctness fixes needed to match ModelOpt exactly

**1. IEEE 754 division in Triton (`tl.div_rn`)**

Triton's `/` operator defaults to `div.approx.f32` (reciprocal multiplication, ~2 ULP error). For values exactly on E2M1 bucket boundaries, this was enough to push them into the wrong bucket, causing a persistent `max_diff=0.244` mismatch against the PyTorch baseline.

Fix: use `tl.div_rn(w, scale)` which emits the IEEE-compliant `div.rn.f32` instruction.

**2. IEEE 754 tie-breaking in bucketization**

`torch.bucketize` always rounds down at exact boundaries. ModelOpt uses round-to-nearest-even (RNE): at the midpoint between two representable values, it rounds to whichever has the even mantissa bit. For E2M1, the three tie-point boundaries are `0.75`, `1.75`, and `3.5` (between odd and even mantissa values).

Fix: use `>=` instead of `>` at these three bounds:
```python
ord_ = (abs_w > 0.25)  + (abs_w >= 0.75) + (abs_w > 1.25) +
       (abs_w >= 1.75) + (abs_w > 2.5)   + (abs_w >= 3.5)  + (abs_w > 5.0)
```
And in PyTorch: `torch.bucketize` + `torch.any(abs_w == [0.75, 1.75, 3.5])` rounding adjustment.

After both fixes: **exact match with ModelOpt** on all test shapes.

---

## NVFP4

### Inference path
- **Format**: FP4 E2M1 weights, FP8 E4M3 per-block scales, block_size=16, FP32 global scale
- **Kernel**: W4A4 — **activations are also quantized to FP4** (default on B200)
  - SM100/Blackwell (B200): `FLASHINFER_CUTLASS` — native Blackwell FP4 tensor cores (W4A4)
  - Fallback: `MARLIN` — W4A16 (activations in BF16), but not Blackwell-optimized
- **Key file**: `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`

Backend auto-selection (from `select_nvfp4_linear_backend`):
```python
if current_platform.has_device_capability(100) and has_flashinfer():
    backend = FLASHINFER_CUTLASS   # B200 default — W4A4
elif cutlass_fp4_supported():
    backend = VLLM_CUTLASS         # W4A4
elif is_fp4_marlin_supported():
    backend = MARLIN               # W4A16 fallback (SM75+)
```

### QAT situation: harder

#### Option A: Marlin backend (W4A16), forced via `VLLM_NVFP4_GEMM_BACKEND=marlin`
- Activations stay in BF16 → **weight-only fake-quant is an exact simulation**
- But Marlin is not optimized for Blackwell — won't use SM100 FP4 tensor cores → slower inference

#### Option B: FlashInfer CUTLASS (W4A4), the B200 default
- Activations are quantized to FP4 on-the-fly at inference → weight-only fake-quant has an **activation gap**
- To close the gap fully, activations would also need to be fake-quantized during training
- In practice, weight quantization error usually dominates, so weight-only QAT is still a useful approximation

**Current status**: weight-only fake-quant only. Activation fake-quantization is not yet implemented and is a future TODO if the W4A4 gap proves significant.

### Ground truth: ModelOpt (not vLLM's `ref_nvfp4_quant`)
`~/Model-Optimizer/modelopt/torch/quantization/qtensor/nvfp4_tensor.py`

The key difference from vLLM's emulation helper: ModelOpt stores per-block scales as large values (range [0, 448]), which makes full use of the FP8 E4M3 dynamic range. vLLM's `ref_nvfp4_quant` (used only in the debug/emulation backend, not production) stores scales as tiny values near 0, wasting FP8 precision.

Both are mathematically equivalent in infinite precision, but differ after the lossy FP8 cast.

ModelOpt roundtrip:
```python
# Global scale (per-tensor)
wsf2 = amax / (6.0 * 448.0)             # small scalar, e.g. ~0.0004

# Per-block scale (in FP8-friendly range [0, 448])
pbs = per_block_amax / (6.0 * wsf2)     # = 448 * per_block_amax / amax
pbs = max(pbs, 1.0) if pbs == 0 else pbs  # zero-guard
pbs_fp8 = fp8_e4m3(pbs)                 # FP8 cast at good magnitude

# Quantize
block_scale = pbs_fp8.to(float32) * wsf2  # effective scale
normalized = x / block_scale              # puts values in [-6, 6]
fp4_val = snap_to_e2m1(normalized)

# Dequantize
result = fp4_val * block_scale
```

### Implementation: `openrlhf/utils/nvfp4_quantize.py`
- **Two-pass design**: FP8 E4M3 cast must happen in PyTorch (not Triton — Ampere doesn't support FP8 in Triton), then snap+dequant runs in a fused Triton kernel
  - Pass 1 (PyTorch): `pbs → fp8_cast → combined_scale = pbs_fp8 * wsf2`
  - Pass 2 (Triton): `normalize(w, combined_scale) → snap_to_e2m1 → dequant`
- **PyTorch fallback** (`_fake_quantize_nvfp4_chunk`): no `@torch.compile` (can't compile FP8 cast on Ampere)

### Correctness: same two fixes as MXFP4
- `tl.div_rn` in Triton kernel
- `>=` at odd-indexed E2M1 bounds (0.75, 1.75, 3.5) for RNE tie-breaking

---

## Summary Table

| | **MXFP4** | **NVFP4** |
|---|---|---|
| Inference kernel | W4A16 (Marlin/SM90, MXFP8-acts/SM100) | W4A4 (FlashInfer CUTLASS on B200) |
| Weight-only QAT accuracy | Exact (no activation gap for W4A16) | Approximate (activation gap on W4A4) |
| Block size | 32 | 16 |
| Scale format | E8M0 uint8 (exact power-of-two, no precision loss) | FP8 E4M3 (lossy) + FP32 global scale |
| Ground truth | ModelOpt `MXFP4QTensor` | ModelOpt `NVFP4QTensor` |
| Triton kernel | Fused single-pass | Two-pass (PyTorch FP8 cast + Triton snap/dequant) |
| Exact match with ModelOpt | Yes (after div_rn + RNE fixes) | Yes (after div_rn + RNE fixes) |

## Open TODOs

- **NVFP4 activation fake-quantization**: if the W4A4 activation gap proves significant empirically, fake-quantize activations during training too. Same ModelOpt roundtrip formula, but applied per-token with a calibrated static `input_global_scale`.
- **NVFP4 weight sync** (`quantize_to_nvfp4`): still uses old vLLM-ref convention. Should be updated to ModelOpt convention (same formula change as fake-quant).
- **SM100 MXFP4 QAT**: SM100 uses MXFP4 weights + MXFP8 activations — weight-only fake-quant may have a similar (smaller) activation gap there too.
