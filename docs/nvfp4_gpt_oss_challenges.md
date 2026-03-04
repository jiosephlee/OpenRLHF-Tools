# NVFP4 GPT-OSS: Conversion, Integration, and Debugging

This document covers the full lifecycle of NVFP4 for GPT-OSS: the offline checkpoint conversion process, the conventions that must be consistent across conversion/sync/QAT, and every bug encountered when integrating with vLLM.

---

## Conversion Overview

Script: `scripts/convert_to_nvfp4.py`

1. Load BF16 model
2. For each MoE expert projection (`gate_up_proj`, `down_proj`): transpose, quantize to packed NVFP4
3. Save HF checkpoint with quantized weights + scale tensors
4. Upload to HF Hub, delete local copy

Only expert projections are quantized. Attention, router, embedding, and lm_head stay BF16.

Two backends:
- `--backend modelopt` (default): uses `NVFP4QTensor.quantize` from `nvidia-modelopt`
- `--backend builtin`: uses `openrlhf/utils/nvfp4_quantize.py` (no extra deps, identical math)

---

## Conventions

### Transposition

BF16 HF checkpoints store expert weights as `[E, in_features, out_features]` (confirmed: `gate_up_proj` shape `[32, 2880, 5760]`, `d1 < d2`). vLLM expects `[E, out_features, in_features]` so quantization blocks run along the input dimension. The conversion applies `.transpose(-1, -2).contiguous()` before quantizing. Verify: if `d1 < d2` in the original, the transpose is correct.

### Scale Convention

NVFP4 uses a two-level scale system. The dequantization formula is:
```
output = fp4_val * block_scale_fp8 * global_scale_fp32
```

Both ModelOpt and vLLM's actual CUTLASS/FlashInfer inference kernel use the same (ModelOpt) convention:
- `global_scale = amax / (E4M3_MAX * E2M1_MAX) = amax / 2688`  — small number, multiply
- `block_scale = vec_max_in_block / (E2M1_MAX * global_scale) = 448 * vec_max / amax`  — up to 448, fills E4M3 range

The block scales always reduce to `448 * vec_max / amax` regardless of which convention is used for `global_scale`. Both ModelOpt and the builtin implementation produce identical `global_scale` and block scale values (confirmed by diagnostic: ratio = 1.0000, same sample bytes).

> **Note:** vLLM's `ref_nvfp4_quant` reference utility uses the inverse convention (large global_scale, divide). This is a test helper only — not what the inference kernel uses. Don't treat it as ground truth.

### Packed Uint8 Format

Two FP4 values per byte: even-index element in low nibble, odd-index in high nibble:
```python
packed = (fp4_val[..., 1::2] << 4) | fp4_val[..., 0::2]
```

Shapes after quantizing `[out, in]` per expert:
- `packed`: `[E, out, in//2]`
- `block_scales`: `[E, out, in//block_size]` (one E4M3 per 16 inputs)
- `global_scales`: `[E]` (one FP32 per expert)

ModelOpt and builtin produce near-identical packed bytes. The only difference: for zero-valued weights, builtin encodes `0b1000` (negative zero) and ModelOpt encodes `0b0000` (positive zero). Both decode to `0.0` in the kernel — cosmetic only.

### QAT and Live Weight Sync Consistency

All three paths use the same math and must stay consistent with vLLM's fixed dequant kernel:

| Path | Code | Convention |
|---|---|---|
| Offline checkpoint | `convert_to_nvfp4.py` | ModelOpt ✓ |
| Live weight sync | `vllm_worker_wrap.py:_maybe_quantize_nvfp4_for_vllm` | ModelOpt ✓ |
| QAT fake-quant | `nvfp4_quantize.py:fake_quantize_nvfp4` | ModelOpt ✓ |
| vLLM inference kernel | CUTLASS/FlashInfer | ModelOpt ✓ |

---

## Key Naming Convention (Critical)

The checkpoint key names must exactly match what `hf_to_vllm_mapper` in `gpt_oss.py` expects. Unrecognized names are **silently dropped** — no error, no warning, weights just never load.

The mapper (`GptOssForCausalLM.hf_to_vllm_mapper`) translates HF names → vLLM names before `_load_weights_nvfp4` sees them:

| HF checkpoint key suffix | → vLLM name | Handled in `_load_weights_nvfp4` |
|---|---|---|
| `gate_up_proj_blocks` | → `w13_weight` | packed weight load |
| `gate_up_proj_scales` | → `w13_weight_scale` | block scale load |
| `gate_up_proj_scales_2` | → `w13_weight_scales_2` | global scale load (normalizes `scales_2→scale_2`) |
| `down_proj_blocks` | → `w2_weight` | packed weight load |
| `down_proj_scales` | → `w2_weight_scale` | block scale load |
| `down_proj_scales_2` | → `w2_weight_scales_2` | global scale load |

**Both suffixes must be plural (`_scales`, `_scales_2`).** Using singular `_scale` or `_scale_2` has no mapper entry.

This was the root cause of "model loads cleanly but produces garbage": packed weights loaded fine (the `_blocks` entry existed), but block scales and global scales were silently skipped — leaving them at uninitialized values.

---

## Bug Log

### 1. `quant_method` Resolution and Loading

**Problem:** vLLM fell back to the MXFP4 loader because `config.json` had `"quant_method": "modelopt"` + `"quant_algo": "NVFP4"` — neither was recognized as NVFP4.

**Solution:** Updated `GptOssModel.load_weights` in `gpt_oss.py` to check `self.quant_config.get_name()`, then `self.config.quantization_config["quant_algo"]`, then finally whether `'nvfp4'` appears in `self.config._name_or_path`.

---

### 2. Scale Key Names Silently Dropped (Garbage Outputs)

**Problem:** Conversion script saved `gate_up_proj_scale` and `gate_up_proj_scale_2` (singular). The `hf_to_vllm_mapper` has entries for `gate_up_proj_scales` (plural) → `w13_weight_scale` but **no entry** for singular or for `_scale_2` / `_scales_2`. Result: block and global scales were never loaded into the model. The model ran with uninitialized scales → garbage outputs with no error.

**Solution:**
- Conversion script: save as `_scales` and `_scales_2` (plural)
- `hf_to_vllm_mapper`: add entries for both plural and singular `_scale_2` variants:
  ```python
  ".gate_up_proj_scales_2": ".w13_weight_scales_2",
  ".down_proj_scales_2":    ".w2_weight_scales_2",
  ".gate_up_proj_scale_2":  ".w13_weight_scale_2",
  ".down_proj_scale_2":     ".w2_weight_scale_2",
  ```

---

### 3. `_scales_2` KeyError in `params_dict`

**Problem:** `KeyError: 'layers.0.mlp.experts.w2_weight_scales_2'`. Even after the mapper produced `w13_weight_scales_2`, the vLLM `params_dict` only has the singular key `w13_weight_scale_2`.

**Solution:** In `_load_weights_nvfp4`, normalize before lookup:
```python
param_name = name.replace("scales_2", "scale_2")
param = params_dict[param_name]
```
This handles both plural and singular input names from any checkpoint.

---

### 4. Global Scale Shape Mismatch

**Problem:** `RuntimeError: size of tensor a (2) must match tensor b (32) at non-singleton dimension 1`. Global scales are saved as `[E]` (one per expert), but vLLM's `FusedMoE` allocates:
- `w13_weight_scale_2`: `[E, 2]` (gate and up share one scale, stored twice)
- `w2_weight_scale_2`: `[E, 1]`

**Solution:** In `_load_weights_nvfp4`, reshape before copy:
```python
# w13
weight_to_copy = weight.unsqueeze(1).expand(-1, 2)   # [E] → [E, 2]
# w2
weight_to_copy = weight.unsqueeze(1)                  # [E] → [E, 1]
```
Applies to both offline checkpoint loading and live weight sync.

---

### 5. `exclude_modules` vs `modules_to_not_convert`

**Problem:** `convert_to_nvfp4.py` saved unquantized-layer exclusions in `config.json` under `"modules_to_not_convert"`. vLLM's ModelOpt quant config reads `"exclude_modules"` (or `"ignore"`). Due to the mismatch, vLLM instantiated attention and router layers as NVFP4, then failed with `AssertionError` when loading the BF16 weights.

**Solution:** Runtime override in `GptOssModel.__init__`:
```python
if (
    self.quant_config is not None
    and getattr(self.quant_config, "exclude_modules", None) is not None
):
    self.quant_config.exclude_modules.extend(["*.attn.*", "*.router"])
```
Forces exclusion at runtime regardless of what `config.json` says.

---

## Diagnostics

### Built-in conversion diagnostic

The `[DIAG]` block in `_convert_nvfp4_modelopt` (runs on the first expert layer):

```bash
python scripts/convert_to_nvfp4.py \
    --model_path unsloth/gpt-oss-20b-BF16 \
    --hub_repo_id <any-id> \
    --backend modelopt
# Kill after [DIAG] block prints
```

Checks:
1. Original weight shape → verifies transpose direction
2. `global_scale` ratio ModelOpt/builtin → should be ~1.0
3. Block scale values → should match between backends
4. Packed uint8 comparison → detects nibble-swap or other packing differences

### Inspect a saved checkpoint's key names

```bash
python scripts/inspect_nvfp4_checkpoint.py \
    --repo_id jiosephlee/gpt-oss-20B-NVFP4-packed-clean \
    --filter experts
```

---

## False Positive: "N NVFP4 params were NOT loaded"

The warning fired by `_load_weights_nvfp4`:
```
[nvfp4] 102 NVFP4 params were NOT loaded (likely silent drop due to missing checkpoint key or mapper mismatch)
```
is **almost always a false positive**. Here's why.

### Mechanism

`_load_weights_nvfp4` ends with a diagnostic that compares `params_dict` (all 24 layers × 6 param types = 144 params) against `loaded_params` (what was added during this specific call's loading loop).

`GptOssForCausalLM.load_weights` uses `AutoWeightsLoader`, which groups incoming weights by their top-level prefix using `itertools.groupby`. **`itertools.groupby` only groups consecutive elements.** If the checkpoint shard files yield keys in an order where the "model" prefix is interrupted:

```
model.layers.0.*    (shard 1)   ← "model" group #1
...
model.layers.16.*   (shard 1)
lm_head.weight      (shard 1)   ← "lm_head" breaks the consecutive run
model.layers.17.*   (shard 2)   ← "model" group #2 — new group!
...
model.layers.23.*   (shard 2)
```

`GptOssModel.load_weights` → `_load_weights_nvfp4` is called **once per group**, each time with a fresh `loaded_params = set()`. The diagnostic at the end of each call sees only the partial set of layers loaded so far.

### The Complementary Counts

With 24 layers × 6 param types = 144 total:
- Call 1 (layers 0–16, 17 layers): `loaded_params` has 102 entries → warning "**102** NOT loaded" (layers 17–23 not yet seen)
- Call 2 (layers 17–23, 7 layers): `loaded_params` has 42 entries → warning "**42** NOT loaded" (layers 0–16 done in call 1)

Note 42 + 102 = 144 — they're perfectly complementary. Both warnings fire on the same run; which one you see depends on which log line you happen to notice.

### Why It's Actually Fine

`_log_nvfp4_post_load_check` is called from `GptOssForCausalLM.load_weights` **after** `AutoWeightsLoader` finishes processing all shard groups. At that point all 24 layers have their weights correctly loaded. The POST-LOAD CHECK is the authoritative diagnostic:
```
[nvfp4] POST-LOAD CHECK: all 24 MoE layers have valid non-zero w13+w2 scale_2 AND non-zero packed weights ✓
```
If this passes, ignore the "N params were NOT loaded" warnings entirely.

### When the Warning IS Real

The warning would indicate a genuine problem only if:
1. The POST-LOAD CHECK also fails (zero or NaN scales after loading), **or**
2. The listed unloaded params don't follow the complementary pattern (e.g., the same 6 params repeatedly across all 24 layers → suggests a mapper key mismatch)

---

## Failure Mode Reference

| Symptom | Root cause | Fix |
|---|---|---|
| Model loads, outputs garbage | Scale keys silently dropped (wrong suffix) | Use `_scales` / `_scales_2` (plural) in conversion |
| `KeyError: w2_weight_scales_2` | vLLM `params_dict` uses singular key | `_load_weights_nvfp4` normalizes `scales_2→scale_2` |
| `RuntimeError: size mismatch dim 1` | Global scale `[E]` vs vLLM's `[E,2]`/`[E,1]` | Reshape in `_load_weights_nvfp4` with `unsqueeze`/`expand` |
| `AssertionError` on attention weight shapes | vLLM instantiated attn as NVFP4 | `gpt_oss.py` runtime override extends `exclude_modules` |
| Falls through to MXFP4 loader | `quant_method: "modelopt"` not recognized as NVFP4 | Check `_name_or_path` as fallback in `load_weights` |

---

## Weight Loading Architecture (Why `_load_weights_nvfp4` Exists)

### The General vLLM Weight Loading Flow

When vLLM loads a model, each layer parameter has a `weight_loader` attribute attached by the quantization method's `create_weights()`. For `FusedMoE`, the single `FusedMoE.weight_loader()` method handles all expert parameters. It is called once per weight tensor per expert with three key arguments:

- `weight_name`: the parameter name (e.g. `"w13_weight"`)
- `shard_id`: `"w1"`, `"w2"`, or `"w3"` — which projection this tensor is for
- `expert_id`: the global expert index this tensor belongs to

Internally it maps `expert_id` to a local id (for Expert Parallelism), does TP sharding by narrowing the loaded tensor, and dispatches to quantization-method-specific logic (e.g. `ModelOptNvFp4FusedMoE`'s handler).

### Why `_load_weights_nvfp4` Bypasses This (and What We Changed)

The gpt-oss NVFP4 checkpoint stores **all experts fused into one tensor** — e.g. `w13_weight` has shape `(E, 2*intermediate, hidden//2)` rather than one tensor per expert. The generic `weight_loader` API expects to be called once per expert with a 2D per-expert slice, so it cannot directly consume the all-experts-combined format without special handling.

Before our change, `_load_weights_nvfp4` responded to this by **skipping `weight_loader` entirely**: it computed TP/EP slices manually and called `param.copy_()` directly. This bypassed:
- The ModelOpt-specific loading logic (`_load_combined_w13_weight_scale`, `_load_model_weight_or_group_weight_scale`)
- `process_weights_after_loading()` format assumptions (correct packed layout per kernel backend)

After our change, we use `weight_loader` for expert weights by:
1. **Pre-slicing by EP rank** when EP is enabled, so the tensor has shape `(local_E, ...)` before being passed in
2. **Splitting the fused `w13_weight`** into `w1` and `w3` halves along dim 1, then calling `weight_loader` once for each with `shard_id="w1"` and `shard_id="w3"` — the weight_loader's `_load_w13` handles TP narrowing and writes each half into the correct slot of the fused param. **The param itself stays fused** (`[w1_tp_slice | w3_tp_slice]`); the split only controls what `weight_loader` receives as input.
3. **Passing `expert_id=ep_rank_start`** for all full-load calls. Since `full_load=True` (3D tensor), `expert_data = param.data` regardless of expert_id. Using `ep_rank_start` (which is always a valid expert on the current EP rank) prevents the `expert_id == -1` guard from short-circuiting the load on non-zero EP ranks.

### What Still Uses Direct Copy and Why

**Global scales (`w13_weight_scale_2`, `w2_weight_scale_2`):**
The weight_loader does have a path for these — ModelOpt's `_load_per_tensor_weight_scale` is triggered when `"weight_scale_2"` is in the weight name. However, it does `param_data[expert_id][idx] = loaded_weight`, which expects `loaded_weight` to be a single scalar for one expert. The gpt-oss checkpoint stores shape `(E,)` for all experts combined, so calling weight_loader would assign an entire `(E,)` tensor into one expert's scalar slot — wrong. Direct copy (after EP-slicing and shape expansion) is the correct approach here.

**Biases (`w13_bias`, `w2_bias`):**
The ModelOpt weight_loader handler checks `"weight" in weight_name` to decide whether to dispatch to `_load_model_weight_or_group_weight_scale`. Since `"weight"` is not a substring of `"w13_bias"`, this check is False — the handler returns `True` (success) without loading anything. A silent no-op. Manual TP/EP slicing + `param.copy_()` is required.

Note: the `mxfp4` quant_config path in `weight_loader` (line 1063 of `layer.py`) *does* handle bias via a special `"bias" in weight_name` check. That path is for the MXFP4 checkpoint format, not ModelOpt NVFP4.

### The Duplicate `_load_weights_other` Bug

There were two definitions of `_load_weights_other` in `GptOssModel`:
- Lines ~972–994: incomplete stub — set up `params_dict` and TP params, then the function body ended with no loading loop
- Lines ~1168+: the real, complete implementation

Python silently uses the second definition (it overwrites the first). The stub had no runtime effect but was confusing and was removed.

## How vLLM does FP4

It's right there in nvfp4_utils.py that we already read — in the production apply_nvfp4_linear path (lines 213–216):

  # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
  x_fp4, x_blockscale = scaled_fp4_quant(
      x, input_global_scale_inv, is_sf_swizzled_layout=True, backend=backend.value
  )

  This runs on every forward pass. x is the bf16 activation coming in; x_fp4 is what actually gets fed into the GEMM. Then the call is:

  cutlass_scaled_fp4_mm(x_fp4, weight, x_blockscale, weight_scale, alpha, output_dtype)

  Both inputs to the matmul are FP4 — this is W4A4, not W4A16. The filename compressed_tensors_w4a4_nvfp4.py also spells it out explicitly.

compressed_tensors_w4a16_nvfp4.py — W4A16, Marlin GEMM, bf16 activations (same as MXFP4)
compressed_tensors_w4a4_nvfp4.py — W4A4, CUTLASS/FlashInfer, FP4 activations

---

## Inference Debugging: Hang / No Output (B200 + hidden_size=2880)

**Symptom:** The NVFP4 checkpoint (`jiosephlee/gpt-oss-20B-NVFP4-packed-clean`) loads cleanly — both POST-LOAD CHECK and KERNEL CONFIG CHECK report "all layers OK" — but inference either hangs indefinitely or produces garbage output. No trace is written.

### Quant Method Clarification (Critical)

For ModelOpt NVFP4 checkpoints, vLLM uses **`ModelOptNvFp4FusedMoE`** (in `modelopt.py`), **NOT** `CompressedTensorsW4A4Nvfp4MoEMethod`. These two register completely different param names:

| | `ModelOptNvFp4FusedMoE` (what we use) | `CompressedTensorsW4A4Nvfp4MoEMethod` (wrong) |
|---|---|---|
| Packed weights | `w13_weight` | `w13_weight_packed` |
| Global scale | `w13_weight_scale_2` | `w13_weight_global_scale` |
| Activation scale | `w13_input_scale` | `w13_weight_global_scale` |

`ModelOptNvFp4FusedMoE.create_weights` (line 1274 of `modelopt.py`) registers:
- `w13_weight`: shape `[E, 2*intermediate, hidden//2]`
- `w13_weight_scale_2`: shape `[E, 2]` (w13_num_shards=2)
- `w13_input_scale`: scalar, pre-filled with 1.0 by `_load_weights_nvfp4`

### What Is Confirmed Correct (Weight Loading)

- All param names match: `_load_weights_nvfp4` finds all params via `params_dict`
- Block scales loaded via `_load_combined_w13_weight_scale` (handles combined w1+w3 case at line 857 of `layer.py`)
- Global scales reshaped `[E] → [E, 2]` via `unsqueeze(1).expand(-1, 2)` before `copy_()`
- Packed weights split into w1/w3 halves; each half loaded via `weight_loader` with `shard_id="w1"` / `shard_id="w3"`
- POST-LOAD CHECK genuinely passes: `w13_weight_scale_2` is a Parameter registered by `create_weights`, present before `process_weights_after_loading`
- KERNEL CONFIG CHECK genuinely passes: `g1_alphas = a13_scale * w13_scale_2` is non-zero and non-NaN

### Root Cause: Wrong MoE Backend for 2880 Hidden Size

Backend selection order in `select_nvfp4_moe_backend` (`oracle/nvfp4.py` line 106):

1. **FLASHINFER_TRTLLM** → **REJECTED** (`hidden_size=2880`, `2880 % 512 = 320 ≠ 0` at `is_supported_config_trtllm` line 98 of `flashinfer_fp4_moe.py`)
2. **FLASHINFER_CUTEDSL** → selected if FlashInfer is installed (likely on this cluster)
3. **FLASHINFER_CUTLASS** → selected if CUTEDSL is rejected
4. **VLLM_CUTLASS** → `cutlass_scaled_mm_supports_fp4(10.0)` → True on B200 (family 100)
5. **MARLIN** → fallback

The selected backend is logged by `_log_nvfp4_kernel_config_check`:
```
[nvfp4] KERNEL CONFIG CHECK: all N NVFP4 MoE layers have valid kernel quant config ✓ (MoE backend=<name>, ...)
```

FLASHINFER_CUTEDSL or FLASHINFER_CUTLASS is likely selected and hangs or errors on B200 for `hidden_size=2880` (not aligned to FlashInfer tile size requirements).

### Fix: Disable FlashInfer MoE FP4

```bash
export VLLM_USE_FLASHINFER_MOE_FP4=0
```

Setting this before launching vLLM forces the backend to fall through to **VLLM_CUTLASS** (the well-tested B200 path). `CutlassExpertsFp4._supports_current_device` (line 672 of `cutlass_moe.py`) accepts capability families 100/110/120, and B200 (10.0) is family 100.

Add to `scripts/train_grpo_tdc_gpt_oss.sh` inside the `nvfp4)` case block, before the training command:
```bash
export VLLM_USE_FLASHINFER_MOE_FP4=0
```

### Fallback: Force Marlin — DOES NOT WORK for group_size=16

`VLLM_TEST_FORCE_FP8_MARLIN=1` selects the `moe_wna16_marlin_gemm` kernel (W4A16, BF16 activations), which is conceptually correct for a W4A16 checkpoint. However it crashes with:

```
RuntimeError: Invalid thread config: thread_m_blocks=4, thread_k=-1, thread_n=-1, num_threads=-1
    for MKN=[65536, 2880, 2880] and num_bits=4, group_size=16
```

`thread_k=-1` means the kernel's tile-config selector found no valid config for `group_size=16`. Marlin's `moe_wna16_marlin_gemm` only supports group sizes ≥ 32 (typical: 128, 64, 32). NVFP4's block size is 16 — not in the supported set. **Do not use this flag for NVFP4 checkpoints.**

### process_weights_after_loading Format Notes

`ModelOptNvFp4FusedMoE.process_weights_after_loading` (line 1413 of `modelopt.py`):
- Takes `w13_weight_scale_2[:, 0]` as the per-expert global scale
- Routes to `convert_to_nvfp4_moe_kernel_format` → `prepare_nvfp4_moe_layer_for_fi_or_cutlass`
- For FLASHINFER_CUTLASS only: `reorder_w1w3_to_w3w1` is applied (w1/w3 axis swapped)
- For all non-TRTLLM backends: `swizzle_blockscale` applied to block scales
- For VLLM_CUTLASS/CUTEDSL: no weight reordering, only scale swizzle

Checkpoint shape `[E=32, out=5760, in//2=1440]` matches `w13_weight` param `[E, 2*intermediate, hidden//2] = [32, 5760, 1440]` ✓

### Full Backend Status for GPT-OSS 20B (hidden_size=2880, B200)

All four NVFP4 MoE backends have been tested and all fail:

| Backend | Env var | Result | Root cause |
|---|---|---|---|
| FlashInfer TRTLLM | (auto) | rejected at init | `hidden_size=2880`, `2880 % 512 ≠ 0` |
| FlashInfer CUTEDSL | (auto) | hangs at inference | tile alignment requirement on B200 |
| FlashInfer CUTLASS | (auto) | hangs at inference | tile alignment requirement on B200 |
| VLLM_CUTLASS | `VLLM_USE_FLASHINFER_MOE_FP4=0` | garbage output | W4A4 kernel; `a1_gscale=1.0` (uncalibrated) saturates activations |
| Marlin W4A16 | `VLLM_TEST_FORCE_FP8_MARLIN=1` | crashes | `moe_wna16_marlin_gemm` doesn't support `group_size=16` |

### Paths Forward

**Option A — Calibrate activation scales for VLLM_CUTLASS (proper W4A4 fix)**

Run the BF16 model on a calibration dataset and compute per-expert activation statistics. Store calibrated `w13_input_scale` / `w2_input_scale` in the checkpoint. With accurate `a1_gscale`, VLLM_CUTLASS should produce correct output. This is the "right" fix but requires calibration infrastructure.

**Option B — Serve BF16 in vLLM (bypasses all FP4 kernel issues)**

Load `unsloth/gpt-oss-20b-BF16` as the vLLM model (not the NVFP4 checkpoint). Actor still initializes from the NVFP4 checkpoint + `--nvfp4_dequantize_base_model` for correct weights. Weight sync becomes BF16→BF16 (no quantization for vLLM). Requires either a separate `--vllm_pretrain` arg or passing the BF16 path as `PRETRAIN_PATH` and having the actor load from `NVFP4_BASE` separately. Memory cost: vLLM uses ~40 GB for BF16 vs ~10 GB for FP4.

**Option C — Use MXFP4 for vLLM instead**

Convert the workflow to MXFP4 format (load BF16 base in vLLM + on-the-fly MXFP4 quantization during weight sync via `--vllm_sync_fp4 mxfp4`). MXFP4 uses a different kernel path that may support hidden_size=2880 on B200.

### Updated Failure Mode Reference

| Symptom | Root cause | Fix |
|---|---|---|
| Model loads, inference hangs | FlashInfer CUTEDSL/CUTLASS selected for 2880 hidden | `VLLM_USE_FLASHINFER_MOE_FP4=0` (forces VLLM_CUTLASS) |
| VLLM_CUTLASS produces garbage output | W4A4 kernel; `a1_gscale=1.0` saturates uncalibrated activations | Calibrate `w13_input_scale` from BF16 activation statistics |
| `VLLM_TEST_FORCE_FP8_MARLIN=1` crashes | Marlin `moe_wna16_marlin_gemm` doesn't support `group_size=16` | Don't use Marlin for NVFP4; NVFP4 block size (16) is too small |
| No working NVFP4 kernel for this shape | All four backends fail for hidden_size=2880 + group_size=16 | Serve BF16 in vLLM (Option B) or calibrate for VLLM_CUTLASS (Option A) |