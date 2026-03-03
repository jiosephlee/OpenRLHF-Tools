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

## Failure Mode Reference

| Symptom | Root cause | Fix |
|---|---|---|
| Model loads, outputs garbage | Scale keys silently dropped (wrong suffix) | Use `_scales` / `_scales_2` (plural) in conversion |
| `KeyError: w2_weight_scales_2` | vLLM `params_dict` uses singular key | `_load_weights_nvfp4` normalizes `scales_2→scale_2` |
| `RuntimeError: size mismatch dim 1` | Global scale `[E]` vs vLLM's `[E,2]`/`[E,1]` | Reshape in `_load_weights_nvfp4` with `unsqueeze`/`expand` |
| `AssertionError` on attention weight shapes | vLLM instantiated attn as NVFP4 | `gpt_oss.py` runtime override extends `exclude_modules` |
| Falls through to MXFP4 loader | `quant_method: "modelopt"` not recognized as NVFP4 | Check `_name_or_path` as fallback in `load_weights` |
