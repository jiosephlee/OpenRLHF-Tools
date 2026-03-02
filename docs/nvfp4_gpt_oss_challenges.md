# NVFP4 GPT-OSS Integration Challenges

This document outlines the various challenges, bugs, and edge cases encountered when integrating and debugging NVFP4 quantization for the `GptOssModel` within vLLM and OpenRLHF.

## 1. `quant_method` Resolution and Loading

**Problem:** vLLM failed to properly route the model weights to the NVFP4 loader `_load_weights_nvfp4`. It was falling back to the default MXFP4 loader because the model config had `"quant_method": "modelopt"` and `"quant_algo": "NVFP4"`.

**Solution:** Updated the `load_weights` method in `vllm/model_executor/models/gpt_oss.py` to first check `self.quant_config.get_name()`, then fall back to `self.config.quantization_config`, and finally check if `'nvfp4'` is present in the model's physical name or path.

## 2. Scale Parameter Naming Mismatches (`KeyError`)

**Problem:** A `KeyError` occurred during weight loading: `KeyError: 'layers.0.mlp.experts.w2_weight_scales_2'`. The checkpoint contained pluralized scale parameter names (e.g., `_scales_2`), but vLLM expects the singular form (e.g., `_scale_2`).

**Solution:** Intercepted the parameter names during `_load_weights_nvfp4` in `gpt_oss.py`. If a name contains `.w13_weight_scale_2`, `.w13_weight_scales_2`, `.w2_weight_scale_2`, or `.w2_weight_scales_2`, we use string replacement (`name.replace("scales_2", "scale_2")`) to dynamically map it to the correct vLLM parameter dictionary key before copying.

## 3. Scale Tensor Shape Mismatches for Fused Kernels

**Problem:** When loading `w13` (fused gate and up projections), vLLM's `FusedMoE` memory layout allocates the scales as shape `[num_experts, 2]`. However, the mathematical conversion script simply assigned one master scale per expert naturally resulting in shape `[num_experts]`. When PyTorch tries to copy the memory during loading, dimensions collide violently resulting in `RuntimeError: The size of tensor a (2) must match the size of tensor b (32) at non-singleton dimension 1`. For `w2` (the down projection), the expected shape was `[num_experts, 1]`.

**Solution:** Added dynamic reshaping and broadcasting during copy logic in `gpt_oss.py`:
- For `w13_weight_scale_2` matching `[32] -> [32, 2]`:
  `weight_to_copy = weight.unsqueeze(1).expand(-1, 2)`
- For `w2_weight_scale_2` matching `[32] -> [32, 1]`:
  `weight_to_copy = weight.unsqueeze(1)`

*Note: This solution safely fixes both offline checkpoint loading and online PyTorch actor to vLLM engine weight syncs during OpenRLHF PPO training.*

## 4. `exclude_modules` vs `modules_to_not_convert`

**Problem:** The conversion script `convert_to_nvfp4.py` correctly skipped quantizing attention layers and router weights. However, it saved this list in `config.json` under `"modules_to_not_convert"`. Unfortunately, vLLM's `ModelOpt` expects this list under either `"ignore"` or `"exclude_modules"`. Due to this naming disconnect, vLLM instantiated the QKV and Output projections as NVFP4 layers instead of standard BF16 Linears. When the engine proceeded to load the actual BF16 matrices, brutal shape mismatches threw an `AssertionError`.

**Solution:** Rather than requiring the user to wait hours while deleting and rebuilding a 20B checkpoint from scratch, an override hack was injected directly into `gpt_oss.py` initialization (`GptOssModel.__init__`):
```python
if (
    self.quant_config is not None
    and getattr(self.quant_config, "exclude_modules", None) is not None
):
    self.quant_config.exclude_modules.extend(["*.attn.*", "*.router"])
```
This forces vLLM to skip quantization for attention layers overriding any config.json formatting mistakes dynamically at runtime.
