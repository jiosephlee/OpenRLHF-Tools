# Self-contained NVFP4 quantization utility.
# Ported from vLLM's reference implementation (Apache 2.0):
#   vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
#
# Packs a bfloat16/float16 tensor into uint8 NVFP4 format with E4M3 block scales
# and FP32 per-tensor global scale.
# Used during RLHF weight sync to quantize the Actor's bf16 weights on the fly
# before writing them into vLLM's NVFP4 parameter storage.

import torch
import torch.nn as nn

# FP4 E2M1 representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
E2M1_MAX = 6.0
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# E4M3 max: 448.0
E4M3_MAX = 448.0


def compute_nvfp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute per-tensor FP32 global scale for NVFP4 quantization.

    global_scale = max(abs(tensor)) / (E4M3_MAX * E2M1_MAX)

    This ensures that after scaling by global_scale, block-level scales
    fit within the E4M3 range.
    """
    amax = tensor.float().abs().max()
    global_scale = amax / (E4M3_MAX * E2M1_MAX)
    # Clamp to avoid zero global_scale
    global_scale = torch.clamp(global_scale, min=1e-12)
    return global_scale.to(torch.float32)


def quantize_to_nvfp4(
    tensor: torch.Tensor,
    block_size: int = 16,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a bfloat16/float16 tensor to packed NVFP4 uint8 + E4M3 scales + global scale.

    Follows vLLM's ref_nvfp4_quant() reference implementation.

    Args:
        tensor: 2D input tensor [M, N] (N must be divisible by block_size).
        block_size: Number of elements per scaling block (default 16, per NVFP4 spec).
        global_scale: Per-tensor FP32 scale. If None, computed from weight statistics.

    Returns:
        packed_uint8: Packed FP4 values (two per byte), shape [M, N // 2]
        e4m3_scales: E4M3 block scale factors, shape [M, N // block_size]
        global_scale: FP32 per-tensor global scale (scalar tensor)
    """
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
    m, n = tensor.shape
    assert n % block_size == 0, f"Last dim {n} not divisible by block_size {block_size}"

    if global_scale is None:
        global_scale = compute_nvfp4_global_scale(tensor)

    # Reshape into blocks
    x = tensor.float().reshape(m, n // block_size, block_size)

    # Compute per-block E4M3 scales (following ref_nvfp4_quant)
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max / E2M1_MAX)
    scale = torch.clamp(scale, max=E4M3_MAX, min=-E4M3_MAX)
    # Cast to E4M3 and back to simulate E4M3 precision loss
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)

    # Compute inverse scale for normalization
    output_scale = torch.where(
        scale == 0,
        torch.zeros_like(scale),
        1.0 / (scale / global_scale),
    )

    # Normalize values into FP4 range and clamp
    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -E2M1_MAX, E2M1_MAX).reshape(m, n)

    # Cast to nearest FP4 E2M1 representable value
    sign = torch.sign(clipped_x)
    sign_bit = (2 - sign) // 2
    bounds = E2M1_BOUNDS.to(clipped_x.device)
    ord_ = torch.bucketize(clipped_x.abs(), bounds)
    fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)

    # Pack two 4-bit values into one uint8
    # Even indices → low nibble, odd indices → high nibble
    left = fp4_val[..., 0::2]
    right = fp4_val[..., 1::2]
    packed = (right << 4) + left

    # E4M3 scales: squeeze keepdim and cast to float8_e4m3fn
    e4m3_scales = scale.squeeze(-1).to(torch.float8_e4m3fn)

    return packed, e4m3_scales, global_scale


def _fake_quantize_nvfp4_chunk(
    w_flat: torch.Tensor, block_size: int, global_scale: torch.Tensor,
    bounds: torch.Tensor, values: torch.Tensor,
) -> torch.Tensor:
    """Quantize a flat (N, block_size) float32 tensor to fake-NVFP4. Returns float32 result."""
    vec_max = torch.max(torch.abs(w_flat), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max / E2M1_MAX)
    scale = torch.clamp(scale, max=E4M3_MAX, min=-E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)

    output_scale = torch.where(
        scale == 0,
        torch.zeros_like(scale),
        1.0 / (scale / global_scale),
    )

    scaled_x = w_flat * output_scale
    clipped_x = torch.clamp(scaled_x, -E2M1_MAX, E2M1_MAX)

    # Snap to nearest FP4 value
    sign = torch.sign(clipped_x)
    ord_ = torch.bucketize(clipped_x.abs(), bounds)
    dequantized = sign * values[ord_]

    # Dequantize back: value * (scale / global_scale)
    dequantized = dequantized * (scale / global_scale)
    return dequantized


def fake_quantize_nvfp4(
    weight: torch.Tensor, block_size: int = 16,
    global_scale: torch.Tensor | None = None,
    max_chunk_rows: int = 4096,
) -> torch.Tensor:
    """Differentiable NVFP4 fake-quantizer for QAT.

    Forward: returns NVFP4-dequantized approximation (same shape/dtype as input).
    Backward: straight-through estimator — gradient flows unchanged.

    Processes in row-chunks to cap peak intermediate memory.
    """
    original_shape = weight.shape
    original_dtype = weight.dtype

    if global_scale is None:
        global_scale = compute_nvfp4_global_scale(weight)

    total_rows = weight.numel() // block_size
    bounds = E2M1_BOUNDS.to(weight.device)
    values = E2M1_VALUES.to(weight.device)

    if total_rows <= max_chunk_rows:
        w_blocks = weight.float().reshape(-1, block_size)
        dequantized = _fake_quantize_nvfp4_chunk(w_blocks, block_size, global_scale, bounds, values)
        dequantized = dequantized.reshape(original_shape).to(original_dtype)
    else:
        w_blocks = weight.float().reshape(-1, block_size)
        out = torch.empty_like(w_blocks)
        for start in range(0, total_rows, max_chunk_rows):
            end = min(start + max_chunk_rows, total_rows)
            out[start:end] = _fake_quantize_nvfp4_chunk(
                w_blocks[start:end], block_size, global_scale, bounds, values,
            )
        dequantized = out.reshape(original_shape).to(original_dtype)

    # STE: forward = dequantized value, backward = identity through weight
    return weight + (dequantized - weight).detach()


class _Nvfp4FakeQuant(nn.Module):
    """Parametrization that applies NVFP4 fake-quantization to a weight.

    Args:
        block_size: Number of elements per scaling block.
        transpose: If True, transpose last two dims before quantizing and back
            after. Use for stacked expert params [E, in, out] where quantization
            blocks should run along in_features (matching vLLM's [E, out, in] layout).
    """

    def __init__(self, block_size: int = 16, transpose: bool = False):
        super().__init__()
        self.block_size = block_size
        self.transpose = transpose

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            weight = weight.transpose(-1, -2).contiguous()
            weight = fake_quantize_nvfp4(weight, block_size=self.block_size)
            return weight.transpose(-1, -2).contiguous()
        return fake_quantize_nvfp4(weight, block_size=self.block_size)


def _patch_lora_layer_nvfp4_qat(lora_module, block_size: int = 16) -> None:
    """Patch a PEFT LoraLayer's forward to fake-quantize the merged expert weight with NVFP4.

    Same pattern as MXFP4 version but uses NVFP4 fake-quantization (block_size=16,
    E4M3 scales, per-tensor global scale).
    """
    import types
    import torch.nn.functional as F

    def qat_forward(self, x, *args, **kwargs):
        adapter = getattr(self, "active_adapter", None)
        if adapter is None:
            adapters = getattr(self, "active_adapters", None) or list(self.lora_A.keys())
            adapter = adapters[0]

        base_w = self.base_layer.weight
        lora_delta = (self.lora_B[adapter].weight @ self.lora_A[adapter].weight) * self.scaling[adapter]
        merged_w = base_w + lora_delta

        fq_w = fake_quantize_nvfp4(merged_w, block_size=block_size)

        bias = self.base_layer.bias if hasattr(self.base_layer, "bias") else None
        return F.linear(x.to(fq_w.dtype), fq_w, bias)

    lora_module.forward = types.MethodType(qat_forward, lora_module)


# Name fragments identifying NVFP4-quantized MoE expert projections.
_NVFP4_EXPERT_NAME_FRAGMENTS = ("gate_up_proj", "down_proj", "w13_weight", "w2_weight")


def register_nvfp4_qat_parametrization(model: nn.Module, block_size: int = 16) -> int:
    """Register NVFP4 fake-quantization on MoE expert weight layers.

    Same pattern as register_mxfp4_qat_parametrization but uses NVFP4 params
    (block_size=16, E4M3 scales, per-tensor global scale).

    Returns: count of layers registered.
    """
    import torch.nn.utils.parametrize as parametrize
    import logging

    logger = logging.getLogger(__name__)

    try:
        from peft.tuners.lora import LoraLayer
        _has_peft = True
    except ImportError:
        _has_peft = False

    count = 0
    for name, module in model.named_modules():
        if "experts" not in name:
            continue

        if any(frag in name for frag in _NVFP4_EXPERT_NAME_FRAGMENTS):
            if isinstance(module, nn.Linear):
                parametrize.register_parametrization(module, "weight", _Nvfp4FakeQuant(block_size))
                count += 1
            elif _has_peft and isinstance(module, LoraLayer):
                _patch_lora_layer_nvfp4_qat(module, block_size)
                count += 1
            continue

        for param_name, param in list(module.named_parameters(recurse=False)):
            if not any(frag in param_name for frag in _NVFP4_EXPERT_NAME_FRAGMENTS):
                continue
            if "bias" in param_name or "_scale" in param_name:
                continue
            parametrize.register_parametrization(module, param_name, _Nvfp4FakeQuant(block_size, transpose=True))
            count += 1

    logger.info(f"[QAT NVFP4] Registered fake-quantization on {count} expert weight layers.")
    return count
