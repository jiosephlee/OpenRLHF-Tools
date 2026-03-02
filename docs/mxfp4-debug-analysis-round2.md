# MXFP4 Weight Sync Debug Analysis — Round 2

> **Date**: 2026-03-01
> **Status**: 🟢 Root cause confirmed, fix applied

## Summary

Enhanced `debug_weight_snapshot` with raw uint8 view for float8 tensors and NaN/Inf counting.
Results confirm the corruption source: **uninitialized padding in layerwise reload materialization**.

## New Diagnostic Output

### BEFORE weight sync

| Tensor | dtype | NaN | Inf | Raw uint8 max | Notes |
|--------|-------|-----|-----|---------------|-------|
| `w13_weight_scale` | float8_e4m3fn | False | False | **126** | All bytes ≤ 126 → no float8 NaN patterns |
| `w13_bias` | float32 | False | False | — | Clean |
| `w2_weight_scale` | float8_e4m3fn | True (256) | False | **129** | 256 bytes = 0x7F (E8M0 exponent=0 → scale=1.0, misread as float8 NaN) |
| `w2_bias` | float32 | False | False | — | Clean |

### AFTER weight sync

| Tensor | dtype | NaN | Inf | Raw uint8 max | Notes |
|--------|-------|-----|-----|---------------|-------|
| `w13_weight_scale` | float8_e4m3fn | True (1844) | False | **255** | Garbage bytes from padding now scattered throughout |
| `w13_bias` | float32 | **24** | **201** | — | ⚠ Real corruption: padding garbage shuffled into valid region |
| `w2_weight_scale` | float8_e4m3fn | True (967) | False | **255** | Same: padding garbage bytes include 0x7F and 0xFF |
| `w2_bias` | float32 | **2** | **39** | — | ⚠ Real corruption |

**Key observation**: First 8 values (hash) are **identical** before/after — the garbage is in the padding, then permuted into the interior by `process_weights_after_loading`.

## Root Cause

### The Bug

1. [`materialize_meta_tensor()`](file:///Users/jlee0/Desktop/research/vllm/vllm/model_executor/model_loader/reload/meta.py#L46) uses `torch.empty_strided()` — **no zero initialization**
2. FusedMoE params have **padding** beyond loaded weights:
   - `w13_bias` vLLM shape `[32, 6144]`, only `[:, :5760]` loaded → 12,288 uninitialized padding elements
   - `w2_bias` vLLM shape `[32, 3072]`, only `[:, :2880]` loaded → 6,144 uninitialized padding elements
3. `Mxfp4MoEMethod.process_weights_after_loading` runs `swap_every_two_rows` + `get_w2_permute_indices_with_cache` on the **full** tensor → **scatters NaN/Inf garbage from padding into valid positions**

### Why `create_weights` initializes with zeros but reload doesn't

In [`Mxfp4MoEMethod.create_weights()`](file:///Users/jlee0/Desktop/research/vllm/vllm/model_executor/layers/quantization/mxfp4.py#L341-L410), all parameters are created via `torch.zeros(...)` — so initial model load is clean. But during layerwise reload, `materialize_layer()` recreates tensors via `torch.empty_strided()`, losing the zero-fill guarantee.

### Scale NaN: Confirmed Artifact

- BEFORE: `w13_weight_scale` raw uint8 max=126 → **no** float8_e4m3fn NaN bit patterns (NaN = 0x7F=127 or 0xFF=255)
- BEFORE: `w2_weight_scale` raw uint8 max=129 → 256 values at byte 0x7F are **valid E8M0 exponent=0** (scale=2⁰=1.0), misinterpreted as NaN by `float8_e4m3fn` dtype when cast to `.float()` for snapshotting
- AFTER: both have max=255 → uninitialized padding bytes (including 0x7F and 0xFF) scattered throughout by permutation

## Fix Applied

Zero-fill after `materialize_layer()` in [`_layerwise_process()`](file:///Users/jlee0/Desktop/research/vllm/vllm/model_executor/model_loader/reload/layerwise.py#L224-L235):

```python
# Zero-fill materialized tensors to prevent uninitialized padding from
# contaminating process_weights_after_loading.
for tensor in get_layer_tensors(layer).values():
    if tensor.device.type != "meta":
        tensor.data.zero_()
```

Safe because all valid data is overwritten by the cached weight loading step immediately after. Only the padding (which should be zero) is affected.

## Action Items from Round 1 — Status

- [x] **Item 1**: Raw uint8 view for float8 scale tensors → confirmed NaN is E8M0 artifact
- [x] **Item 2**: NaN/Inf counting → quantified corruption extent
- [x] **Item 3**: Root cause identified and fixed
- [ ] **Item 4**: Re-run training to verify fix eliminates NaN/Inf and produces coherent generation
