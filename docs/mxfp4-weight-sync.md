# MXFP4 On-the-Fly Weight Sync: Design & Lessons Learned

## Problem Statement

When the Actor trains in bf16 but vLLM serves with MXFP4-quantized weights (e.g., GPT-OSS MoE), we need to quantize weights on the fly during the RLHF weight sync. This is non-trivial because:

1. vLLM's `process_weights_after_loading()` replaces `Parameter` objects (destroying `weight_loader` attributes) and transforms parameters into kernel-specific layouts (FlashInfer swizzle/interleave).
2. After this transformation, you can't call `load_weights()` again — the shapes and `weight_loaders` are gone.
3. The quantization must produce weights in exactly the format vLLM's model-specific `_load_weights_mxfp4()` expects, including correct shapes for packed weights AND scales.

## Solution Architecture

### End-to-End Flow

```
Actor (bf16)
  │
  ├─→ initialize_weight_reload() on all vLLM workers     [NEW]
  │     └─→ vLLM layerwise reload API: saves kernel tensors,
  │         restores params to model format (meta device),
  │         wraps weight_loaders for deferred per-layer processing
  │
  ├─→ For each parameter:
  │     update_weight_cuda_ipc(..., mxfp4_quantize_on_the_fly=True)
  │       └─→ WorkerWrap._maybe_quantize_for_vllm()
  │             bf16 [E, in, out] → transpose → quantize_to_mxfp4()
  │             → yields (packed_uint8, scales_uint8)
  │           └─→ model.load_weights([(name, tensor)])
  │                 └─→ wrapped weight_loaders auto-trigger
  │                     process_weights_after_loading() per-layer
  │                     when all weights for that layer arrive
  │
  └─→ post_weight_sync() on all vLLM workers              [MODIFIED]
        └─→ finalize_layerwise_reload(model, model_config)
              unwraps loaders, processes remaining layers,
              restores kernel tensors
```

### Files Modified

| File | Changes |
|------|---------|
| `openrlhf/trainer/ray/vllm_worker_wrap.py` | Added `initialize_weight_reload()`, modified `post_weight_sync()` to use layerwise reload API |
| `openrlhf/trainer/ray/vllm_engine.py` | Added `initialize_weight_reload()` RPC method |
| `openrlhf/trainer/ray/ppo_actor.py` | Call `initialize_weight_reload()` before weight sync loop |
| `openrlhf/utils/mxfp4_quantize.py` | Fixed scale tensor shapes in `quantize_to_mxfp4()` |
| `vllm/.../quantization/mxfp4.py` | Reverted `weight_loader` re-set hack (no longer needed) |

### Key Components

**`_maybe_quantize_for_vllm(name, weight)`** — Intercepts expert weights during broadcast:
- Identifies MoE experts by name (`gate_up_proj`, `down_proj`)
- Transposes `[E, in, out]` → `[E, out, in]` (checkpoint convention)
- Quantizes each expert independently via `quantize_to_mxfp4()`
- Yields two tensors per expert weight: packed uint8 + E8M0 scales
- Uses HF naming convention; vLLM's `hf_to_vllm_mapper` handles remapping

**`quantize_to_mxfp4(tensor, block_size=32)`** — Core quantization:
- Per-32-element block E8M0 scaling (power-of-2 scales)
- Rounds to nearest FP4 E2M1 value: `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`
- Packs two 4-bit values per uint8 byte
- Returns `(packed_uint8[..., last_dim//2], scales_uint8[..., last_dim//block_size])`

**vLLM Layerwise Reload API** (PR #32133):
- `initialize_layerwise_reload(model)` — saves kernel tensors, restores params to meta, wraps weight_loaders
- `finalize_layerwise_reload(model, model_config)` — unwraps, processes remaining layers, restores kernel tensors
- Automatically runs `process_weights_after_loading()` per-layer as weights arrive

---

## Bugs Encountered & Fixes

### Bug 1: `weight_loader` Destroyed After `process_weights_after_loading()`

**Symptom:** `default_weight_loader` used instead of FusedMoE's `weight_loader`, causing shape mismatches on the second weight sync.

**Root Cause:** FlashInfer's `process_weights_after_loading()` creates fresh `Parameter` objects (for swizzled weights), losing the `weight_loader` attribute that `FusedMoE.create_weights()` set via `set_weight_attrs()`. On subsequent `load_weights()` calls, params fall back to `default_weight_loader`.

**Initial Fix (reverted):** Save `weight_loader` before processing and re-apply it after:
```python
# In mxfp4.py process_weights_after_loading():
_saved_weight_loader = getattr(layer.w13_weight, "weight_loader", None)
# ... process ...
if _saved_weight_loader is not None:
    for attr in ("w13_weight", "w13_weight_scale", ...):
        param = getattr(layer, attr, None)
        if param is not None and not hasattr(param, "weight_loader"):
            param.weight_loader = _saved_weight_loader
```

**Proper Fix:** Use vLLM's layerwise reload API, which handles saving/restoring weight_loaders and kernel tensors correctly. The hack was reverted.

---

### Bug 2: Scale Tensor Shape Mismatch (AssertionError in meta dispatch)

**Symptom:**
```
File "layer.py", line 1071, in weight_loader
    param.data[:, :dim1, :dim2].copy_(loaded_weight)
File "meta.py", line 126, in __torch_dispatch__
    assert args[0].numel() == args[1].numel()
AssertionError
```

**Root Cause:** `quantize_to_mxfp4()` flattened the input tensor to `(-1, block_size)` before computing scales but never reshaped the scales back. This produced scales with shape `[total_blocks, 1]` instead of the 2D shape `[rows, cols // block_size]` that vLLM's `_load_weights_mxfp4()` expects.

**Example (down_proj, 32 experts, hidden=2880):**
- Input per expert: `[2880, 2880]`
- Scales returned: `[259200, 1]` (flat: 2880×2880/32) ← **WRONG**
- Scales expected: `[2880, 90]` (2D: rows × cols/block_size) ← **CORRECT**

**Why gate_up_proj didn't crash but down_proj did:**

The `_load_weights_mxfp4` handler narrows differently for w13 vs w2 scales:
- w13_weight_scale: `weight[:, 0:5760, ...]` on `[32, 518400, 1]` → `[32, 5760, 1]`
  - FusedMoE weight_loader: `param.data[:, :5760, :1]` on meta `[32, 6144, 96]` → `[32, 5760, 1]` (numel matches by accident since 5760 < 6144)
- w2_weight_scale: `weight[..., 0:90]` on `[32, 259200, 1]` → `[32, 259200, 1]` (narrowing on last dim, size 1 < 90, so no-op)
  - FusedMoE weight_loader: `param.data[:, :259200, :1]` on meta `[32, 3072, 96]` → clamped to `[32, 3072, 1]` (numel 98,304 ≠ 8,294,400) → **CRASH**

**Fix:** Reshape scales in `quantize_to_mxfp4()` to preserve the 2D structure:
```python
# Before:
e8m0_scales = (e8m0_exp + 127).to(torch.uint8)

# After:
scale_shape = list(original_shape)
scale_shape[-1] = scale_shape[-1] // block_size
e8m0_scales = (e8m0_exp + 127).to(torch.uint8).reshape(scale_shape)
```

---

### Bug 3: `device must be a cuda device` in FlashInfer

**Symptom:**
```
File "mxfp4.py", line 561, in process_weights_after_loading
    nvfp4_block_scale_interleave(
File "flashinfer/utils.py", line 260, in get_compute_capability
    raise ValueError("device must be a cuda device")
```

**Root Cause:** `materialize_meta_tensor()` in vLLM's layerwise reload creates tensors via `torch.empty_strided()` **without specifying a device**. It relies on being called within a `torch.device` context manager. Without this context, tensors are created on CPU, and FlashInfer's `nvfp4_block_scale_interleave` (which needs CUDA tensors) fails.

**How vLLM does it internally** (in `gpu_worker.py:1076` and `gpu_model_runner.py:4417`):
```python
with torch.device(self.device):
    initialize_layerwise_reload(model)
    # ... load weights ...
    finalize_layerwise_reload(model, self.model_config)
```

**Fix:** Wrap both calls in `with torch.device(self.device):` in `vllm_worker_wrap.py`:
```python
def initialize_weight_reload(self):
    with torch.device(self.device):
        initialize_layerwise_reload(self.model_runner.model)

def post_weight_sync(self):
    with torch.device(self.device):
        finalize_layerwise_reload(self.model_runner.model, self.model_config)
    torch.cuda.synchronize()
```

---

### Bug 4: OOM in `fake_quantize_mxfp4` during QAT backward pass

**Symptom:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 27.69 GiB.
  File "mxfp4_quantize.py", line 96, in fake_quantize_mxfp4
    ord_ = torch.sum((w_normalized.abs().unsqueeze(-1) - bounds) > 0, dim=-1)
```

**Root Cause:** The original NVIDIA Model-Optimizer code uses a broadcast pattern to find
which FP4 E2M1 bucket each value falls into:
```python
# bounds has 7 elements → broadcasts [N, 32] to [N, 32, 7] float32 intermediate
ord_ = torch.sum((w_normalized.abs().unsqueeze(-1) - bounds) > 0, dim=-1)
```

This works fine for ModelOpt's use case (single-layer quantization, e.g. `[4096, 4096]` = 16M
elements → 0.5 GiB intermediate). But GPT-OSS stacked expert weights are `[32, 2880, 5760]`
= 530M elements. The broadcast creates a `[16.6M, 32, 7]` intermediate (14 GiB) plus a
same-sized boolean mask, totaling ~27 GiB — matching the error message exactly.

The same pattern exists in both `quantize_to_mxfp4()` (line 51) and `fake_quantize_mxfp4()`
(line 96). The weight sync path calls `quantize_to_mxfp4()` per-expert so it doesn't OOM,
but `fake_quantize_mxfp4()` is called via QAT parametrization on the full stacked tensor.

**Fix:** Replace with `torch.bucketize()`, which does the identical bucket lookup in O(N)
memory instead of O(N×7):
```python
# Before (O(N×7) memory — broadcasts to [N, 32, 7]):
ord_ = torch.sum((w_normalized.abs().unsqueeze(-1) - bounds) > 0, dim=-1)

# After (O(N) memory — element-wise binary search):
ord_ = torch.bucketize(w_normalized.abs(), bounds)
```

**Why this is equivalent:** Both compute `bisect_left(x, bounds)` — the number of bounds
strictly less than x. `sum((x - b) > 0)` counts bounds where `x > b` (strict inequality);
`torch.bucketize(x, bounds, right=False)` returns the leftmost insertion index, which is
the same value. Verified for all FP4 E2M1 representable values including exact boundary hits.

---

## Previous Approach (Pre-Layerwise Reload)

Before adopting the layerwise reload API, the flow was:

```
update_weight_cuda_ipc() × N params
  → _maybe_quantize_for_vllm() → model.load_weights([(name, tensor)])
post_weight_sync()
  → process_weights_after_loading(model, model_config, device)
```

This required hacking `process_weights_after_loading()` in vLLM's `mxfp4.py` to re-apply `weight_loader` attributes after it replaced Parameter objects. The hack was fragile:
- It saved one `weight_loader` from `w13_weight` and applied it to ALL params (w13, w2, scales, biases) — semantically incorrect but happened not to crash on the first sync.
- On subsequent syncs, the kernel-format shapes (post-swizzle) would be incompatible with the weight_loader's narrowing logic.
- The fundamental issue — `process_weights_after_loading()` destroys the ability to call `load_weights()` again — remained unsolvable without the layerwise reload API.

## Optional: MXFP4 Quantization-Aware Training (QAT)

Enabled via `--qat_mxfp4` (requires `--mxfp4_dequantize`). Closes the train/inference distribution gap by fake-quantizing expert weights during the Actor's forward pass.

- **Plain nn.Linear experts:** `register_parametrization(module, "weight", _Mxfp4FakeQuant())`
- **Stacked-parameter experts (GPT-OSS):** `GptOssExperts` holds `gate_up_proj`, `down_proj` as bare `nn.Parameter` (not `nn.Linear` submodules). `register_mxfp4_qat_parametrization()` detects these via `named_parameters(recurse=False)` and registers `_Mxfp4FakeQuant(transpose=True)` — `transpose=True` transposes to `[E, out, in]` before quantizing (blocks run along in_features, matching vLLM's layout) and transposes back after.
- **LoRA experts:** Monkey-patches `forward()` to fake-quantize the *merged* weight (`base + lora_B @ lora_A * scaling`)
- **STE backward:** `weight + (dequantized - weight).detach()`
- **Weight sync unaffected:** `named_parameters()` yields `.parametrizations.weight.original` (true bf16)
- **Memory optimization:** Uses `torch.bucketize()` instead of ModelOpt's broadcast pattern for FP4 bucket lookup (see Bug 4 above)

---

## Active Debug: Post-Sync Gibberish

> **Status**: 🔴 Model produces gibberish after 1st weight sync
> **Last updated**: 2026-02-28

### HF Dequantization Layout (Confirmed)

From `transformers/integrations/mxfp4.py`, `_convert_moe_packed_tensors`:

```python
# blocks: [E, out_dim, G, B]  where out_dim=2*intermediate_size (interleaved gate/up)
out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
return out.transpose(1, 2).contiguous()  # → [E, hidden_size, 2*intermediate_size] = [E, in, out]
```

**Result**: Actor's `gate_up_proj` = `[E, in, out]` in **bf16**. Interleaving preserved. No float8_e5m2 conversion.

### Investigation Checklist

**✅ Verified Correct**
- [x] Name mapping — HF→mapper→`_load_weights_mxfp4` dispatches correctly
- [x] Shape pipeline — `[32,2880,5760]`→transpose→quantize→`[32,5760,1440]`+scales `[32,5760,90]`
- [x] Gate/up interleaving — preserved through dequant (`.transpose(1,2)` swaps in/out, not interleaving dim)
- [x] Transpose direction — `[E,in,out]`→transpose→`[E,out,in]` matches checkpoint format
- [x] HF dequant layout — `[E, hidden_size, 2*intermediate_size]` in bf16 confirmed from source

**🔍 Under Investigation**
- [ ] `process_weights_after_loading` copy-back — SM100 TRTLLM creates NEW Parameters with dtype changes. Waiting on debug logs.
- [ ] Layerwise reload numel tracking — if biases not counted by `MetaCopyCounter`, processing defers. Waiting on logs.
- [ ] Non-expert weight name mapping — all weights must map HF→vLLM or get zeros. Waiting on logs.
- [ ] Bias sync names — does HF use `gate_up_proj_bias` (flat) matching `hf_to_vllm_mapper` entry?

### Hypothesis Log

| # | Hypothesis | Status | Evidence |
|---|-----------|--------|----------|
| H1 | Gate/up interleaving broken | ❌ Eliminated | HF source confirms `.transpose(1,2)` preserves interleaving |
| H2 | `process_weights_after_loading` not running or copy-back failing | 🔍 Active | SM100 TRTLLM does `swap_every_two_rows` + shuffling + dtype changes. If not run → raw MXFP4 → gibberish |
| H3 | Weight name mismatches → zeros | 🔍 Active | `initialize_layerwise_reload` moves params to meta. Unmapped weights = zeros |
| H4 | Bias dtype mismatch in copy-back | 🔍 Active | Processing converts bias bf16→float32. Copy-back to bf16 kernel tensor may fail |

### Debug Outputs Needed

1. **`[DEBUG finalize_layerwise_reload] Stats:`** — `already_done` vs `delayed` vs `no_weights` vs `attention`
2. **`[DEBUG _layerwise_process]`** — is `process_weights_after_loading` running for FusedMoE? Is `quant_method=Mxfp4MoEMethod`?
3. **`debug_weight_snapshot` BEFORE/AFTER** — are weights changing? NaN? Zero?
4. **Any errors/warnings** — dtype/shape mismatches in `copy_()`

### Interpretation Guide

| Observation | Likely Cause |
|-------------|-------------|
| Many `delayed` FusedMoE layers | `get_numel_loaded` not counting biases → delayed path (should still work) |
| `no_weights` > 0 for non-trivial layers | Name mapping failure — weights sent but not received |
| `NO quant_method` for FusedMoE | `process_weights_after_loading` not running → no swizzle → gibberish |
| Weights unchanged BEFORE/AFTER | Sync didn't propagate |
| Weights are NaN/zero AFTER | Corruption in `copy_()` or meta device leak |

---

## Debugging Tips

- **`[MXFP4 Quantize]` log:** Confirms quantization is happening. Check shapes:
  ```
  [MXFP4 Quantize] model.layers.0.mlp.experts.gate_up_proj: bf16 [32, 2880, 5760] →
    uint8 packed [32, 5760, 1440], scales [32, 5760, 90]
  ```
- **`[WorkerWrap] initialize_weight_reload`:** Should appear before any weight updates.
- **`[WorkerWrap] post_weight_sync: finalize_layerwise_reload complete`:** Should appear after all weights sync.
- **Scale shape sanity check:** For input `[E, rows, cols]`, scales should be `[E, rows, cols // 32]`, NOT `[E, rows*cols//32, 1]`.
- **Device context:** `device must be a cuda device` → missing `with torch.device(self.device):` around layerwise reload calls.

---

## Code References

| Component | File | Key Functions |
|-----------|------|---------------|
| Weight sync orchestration | `openrlhf/trainer/ray/ppo_actor.py` | `broadcast_to_vllm()`, `_build_vllm_sync_params()` |
| MXFP4 on-the-fly quantize | `openrlhf/trainer/ray/vllm_worker_wrap.py` | `_maybe_quantize_for_vllm()`, `update_weight()` |
| MXFP4 quantize utilities | `openrlhf/utils/mxfp4_quantize.py` | `quantize_to_mxfp4()`, `fake_quantize_mxfp4()` |
| vLLM GPT-OSS model | `vllm/.../models/gpt_oss.py` | `_load_weights_mxfp4()`, `hf_to_vllm_mapper` |
| vLLM MXFP4 quant method | `vllm/.../quantization/mxfp4.py` | `Mxfp4MoEMethod.process_weights_after_loading()` |
| Layerwise reload API | `vllm/.../reload/layerwise.py` | `initialize/finalize_layerwise_reload()`, `_layerwise_process()` |
| HF dequantization | `transformers/integrations/mxfp4.py` | `_convert_moe_packed_tensors()`, `Mxfp4Dequantize` |
