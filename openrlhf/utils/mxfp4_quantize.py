# Self-contained MXFP4 quantization utility.
# Adapted from NVIDIA Model-Optimizer (Apache 2.0):
#   modelopt/torch/quantization/qtensor/mxfp4_tensor.py
#
# Packs a bfloat16/float16 tensor into uint8 MXFP4 format with E8M0 block scales.
# Used during RLHF weight sync to quantize the Actor's bf16 weights on the fly
# before writing them into vLLM's MXFP4 parameter storage.

import torch

# FP4 E2M1 representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
E2M1_MAX = 6.0
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])


def quantize_to_mxfp4(
    tensor: torch.Tensor,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bfloat16/float16 tensor to packed MXFP4 uint8 + E8M0 scales.

    Args:
        tensor: Input tensor (any shape, last dim must be divisible by block_size).
        block_size: Number of elements per scaling block (default 32, per OCP spec).

    Returns:
        packed_uint8: Packed FP4 values (two per byte), shape [..., last_dim // 2]
        e8m0_scales: E8M0 block scale factors (uint8), shape [..., last_dim // block_size]
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    # Reshape into blocks of `block_size`
    tensor = tensor.reshape(-1, block_size)

    # Compute per-block E8M0 scale factors
    # Casting to float for numerical stability
    amax = tensor.float().abs().max(dim=-1, keepdim=True).values
    descale = amax / E2M1_MAX
    min_exp = torch.tensor(-127.0, device=descale.device)
    e8m0_exp = torch.ceil(torch.maximum(torch.log2(descale), min_exp))

    # Normalize values into FP4 range
    normalized = (tensor / torch.exp2(e8m0_exp)).reshape(original_shape)

    # Cast each value to the nearest FP4 E2M1 representable value
    sign = torch.sign(normalized)
    sign_bit = (2 - sign) // 2
    bounds = E2M1_BOUNDS.to(normalized.device)
    ord_ = torch.sum(
        (normalized.abs().unsqueeze(-1) - bounds) > 0, dim=-1
    )
    fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)

    # Pack two 4-bit values into one uint8
    # Even indices → low nibble, odd indices → high nibble
    left = fp4_val[..., 0::2]
    right = fp4_val[..., 1::2]
    packed = (right << 4) + left

    # Convert exponent to biased E8M0 format
    e8m0_scales = (e8m0_exp + 127).to(torch.uint8)

    return packed, e8m0_scales
