# MXFP4 Weight Sync Debug Analysis — Round 1

> **Date**: 2026-03-01
> **Status**: 🔴 Root causes identified → action items below

## Summary

The debug logs confirm:
1. **`process_weights_after_loading` IS running** for FusedMoE with `Mxfp4MoEMethod` ✅ — H2 partially eliminated
2. **All weight names map correctly** (no `no_weights` anomalies) ✅ — H3 eliminated
3. **BUT: critical NaN/Inf corruption** in scale and bias tensors 🔴

---

## BEFORE vs AFTER Comparison (Layer 0)

| Tensor | BEFORE | AFTER | Verdict |
|--------|--------|-------|---------|
| `embedding.weight` | mean=0.000591, hash=[-0.299, -0.695, 1.914, ...] | **Same** hash, same stats | ✅ Unchanged (expected — not an expert weight) |
| `layers.0.attn.qkv_proj.weight` | mean=0.000010, hash=[-0.000113, 0.00510, ...] | **Same** hash, same stats | ✅ Unchanged |
| `layers.0.mlp.experts.w13_weight` | `uint8` mean=96.4 hash=[130, 248, 206, 168, ...] | `uint8` mean=**109.2** hash=[**2**, **240**, 206, **160**, ...] | ✅ Changed — new quantized weights written |
| `layers.0.mlp.experts.w13_weight_scale` | `float8_e4m3fn` mean=262.0, **nan=False** | `float8_e4m3fn` **mean=nan, nan=True** | 🔴 **CORRUPTED** |
| `layers.0.mlp.experts.w13_bias` | `float32` mean=-0.635, **nan=False, inf=False** | `float32` **mean=nan, nan=True, inf=True** | 🔴 **CORRUPTED** |
| `layers.0.mlp.experts.w2_weight` | `uint8` mean=103.4 hash=[123, 220, 34, 90, ...24] | `uint8` mean=**116.1** hash=[123, 220, 34, 90, ...**16**] | ✅ Changed — new quantized weights written |
| `layers.0.mlp.experts.w2_weight_scale` | `float8_e4m3fn` **mean=nan, nan=True** | `float8_e4m3fn` **mean=nan, nan=True** | ⚠️ Already NaN BEFORE sync |
| `layers.0.input_layernorm.weight` | hash=[1.391, 1.258, 1.078, ...] | **Same** hash | ✅ Unchanged |
| `lm_head.weight` | hash=[-0.00601, 0.000736, ...] | **Same** hash | ✅ Unchanged |

---

## Key Findings

### Finding 1: `w13_weight_scale` goes from clean to NaN AFTER sync

**BEFORE**: `mean=262.0, nan=False` — scale tensor was healthy
**AFTER**: `mean=nan, nan=True` — scale tensor is now full of NaN

The hash stays the same: `[288.0, 288.0, 320.0, 288.0, 288.0, 320.0, 320.0, 320.0]` both BEFORE and AFTER. This suggests the `float()`→`float8_e4m3fn` conversion during snapshot is producing NaN from certain `float8_e4m3fn` bit patterns, OR the `process_weights_after_loading` is corrupting the scales.

> [!IMPORTANT]
> The hash values are **identical BEFORE and AFTER** (288, 288, 320, ...). This means the underlying bytes may not have changed — the NaN could be coming from `process_weights_after_loading` converting the scale representation (e.g., dtype change, interleave/swizzle).

### Finding 2: `w13_bias` goes from clean to NaN+Inf AFTER sync

**BEFORE**: `float32 mean=-0.635, nan=False, inf=False`
**AFTER**: `float32 mean=nan, nan=True, inf=True`

The hash is **identical**: `[-0.773, -0.895, -0.910, ...]`. Same first 8 elements, but now has NaN and Inf somewhere in the tensor. This is **H4 (bias dtype mismatch)** — the `process_weights_after_loading` may be doing an operation that produces Inf/NaN in the bias.

### Finding 3: `w2_weight_scale` is NaN BEFORE sync (pre-existing)

This tensor had `mean=nan, nan=True` even BEFORE any weight sync. This means the **initial model load** by vLLM already has NaN in the w2 scales. This could be:
- Normal for `float8_e4m3fn` (certain bit patterns are NaN in float8 but valid as scale indices E8M0)
- Or a pre-existing corruption from the initial checkpoint load

> [!WARNING]
> If `float8_e4m3fn` NaN values are expected for E8M0 scales (since E8M0 uses exponent-only encoding stored as uint8, but vLLM may store them as float8_e4m3fn dtype), then the "NaN" in the snapshot may be a **red herring** — the `debug_weight_snapshot` converts to `.float()` which interprets E8M0 bit patterns through float8_e4m3fn semantics, producing NaN for values that are actually valid E8M0 exponents.

### Finding 4: `finalize_layerwise_reload` stats look healthy

```
Stats: {'attention': 24, 'no_weights': 102, 'delayed': 24, 'already_done': 123}
```

- `attention: 24` — 24 attention layers processed in finalize (expected for 24-layer model)
- `delayed: 24` — 24 FusedMoE layers deferred, then processed in finalize ✅
- `already_done: 123` — most layers processed inline as weights arrived ✅
- `no_weights: 102` — layers with no trainable weights (e.g., activation functions, norms handled separately)

### Finding 5: MXFP4 quantization shapes look correct

```
[MXFP4 Quantize] gate_up_proj: bf16 [32, 2880, 5760] → uint8 packed [32, 5760, 1440], scales [32, 5760, 90]
```

This matches the expected pipeline from the design doc.

---

## Root Cause Hypothesis

### Most Likely: E8M0 Scale Dtype Misinterpretation (Red Herring NaN)

E8M0 scales use **exponent-only encoding** (8 bits = biased exponent, no mantissa, no sign). These are stored as `torch.uint8` in the quantization code but may be stored as `torch.float8_e4m3fn` in vLLM's parameter storage.

When `debug_weight_snapshot` calls `.cpu().float()`, it reinterprets the bit pattern through `float8_e4m3fn` semantics, where certain valid E8M0 exponent values (e.g., exponent 255 = NaN in float8_e4m3fn) produce NaN.

**Evidence**: The hash values for `w13_weight_scale` are **identical** BEFORE and AFTER (`[288.0, 288.0, 320.0, ...]`). The first 8 values convert fine, but `mean()` shows NaN because some values elsewhere in the tensor have E8M0 bit patterns that map to float8_e4m3fn NaN.

**If this is the case**: The NaN in scales is harmless — the MXFP4 kernel reads them as raw uint8 exponents, not as float8 values.

### Possibly Real: Bias Corruption (needs investigation)

The `w13_bias` going from clean float32 to containing Inf is more concerning. `process_weights_after_loading` in `Mxfp4MoEMethod` may be performing operations on the bias that produce Inf (e.g., dtype conversion, scaling).

**Evidence**: Hash is identical for first 8 elements, meaning the corruption is NOT in the first 8 values but somewhere else in the `[32, 6144]` tensor.

---

## Action Items

### 1. Verify scale NaN is a dtype-interpretation artifact
Add to `debug_weight_snapshot`: if dtype is `float8_e4m3fn`, also print the raw uint8 view:
```python
if data.dtype == torch.float8_e4m3fn:
    raw = data.view(torch.uint8).flatten()
    print(f"    (raw uint8 view: mean={raw.float().mean():.1f} max={raw.max()} nan_count=N/A)")
```

### 2. Investigate bias Inf source
Add a targeted print in `process_weights_after_loading` or after `finalize_layerwise_reload` to check `w13_bias` specifically:
```python
bias = model.layers[0].mlp.experts.w13_bias
print(f"w13_bias after finalize: nan={bias.isnan().any()} inf={bias.isinf().any()} "
      f"nan_count={bias.isnan().sum()} inf_count={bias.isinf().sum()}")
```

### 3. Test generation quality
If the NaN in scales is truly a dtype artifact, the model might actually produce correct output. Run a quick generation test after sync to check.
