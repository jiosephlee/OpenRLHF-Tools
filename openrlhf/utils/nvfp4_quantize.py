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

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

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

    # Compute per-block E4M3 scales (ModelOpt convention)
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = vec_max / (E2M1_MAX * global_scale)
    scale = torch.clamp(scale, max=E4M3_MAX, min=-E4M3_MAX)
    # Cast to E4M3 and back to simulate E4M3 precision loss
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)

    # Normalize values into FP4 range and clamp
    scaled_x = torch.where(
        scale == 0,
        torch.zeros_like(x, dtype=torch.float32),
        x.to(torch.float32) / (scale * global_scale),
    )
    clipped_x = torch.clamp(scaled_x, -E2M1_MAX, E2M1_MAX).reshape(m, n)

    # Cast to nearest FP4 E2M1 representable value
    sign = torch.sign(clipped_x)
    sign_bit = (2 - sign) // 2
    bounds = E2M1_BOUNDS.to(clipped_x.device)
    abs_x = clipped_x.abs()
    ord_ = torch.bucketize(abs_x, bounds)
    
    # IEEE round-to-nearest-even tie-breaking for odd E2M1 bounds
    odd_bounds = bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(abs_x.unsqueeze(-1) == odd_bounds, dim=-1)
    ord_ = ord_ + equals_odd_bounds.to(ord_.dtype)
    
    fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)

    # Pack two 4-bit values into one uint8
    # Even indices → low nibble, odd indices → high nibble
    left = fp4_val[..., 0::2]
    right = fp4_val[..., 1::2]
    packed = (right << 4) + left

    # E4M3 scales: squeeze keepdim and cast to float8_e4m3fn
    e4m3_scales = scale.squeeze(-1).to(torch.float8_e4m3fn)

    return packed, e4m3_scales, global_scale


# ---------------------------------------------------------------------------
# Triton kernel for NVFP4 fake quantization
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _triton_fake_quant_nvfp4_kernel(
        w_ptr, combined_scale_ptr, out_ptr, N,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
    ):
        """NVFP4 fake-quantize kernel with pre-computed combined scales.

        Takes pre-computed per-block combined_scale = E4M3_scale * global_scale
        and performs normalize → snap to E2M1 → dequantize in one fused kernel.
        """
        pid = tl.program_id(0)
        
        for i in range(BLOCKS_PER_PROGRAM):
            block_idx = pid * BLOCKS_PER_PROGRAM + i
            base_offs = block_idx * BLOCK_SIZE
            offs = base_offs + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            # If block is completely out of bounds, skip
            if base_offs < N:
                # Load block data and pre-computed combined scale
                w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                combined_scale = tl.load(combined_scale_ptr + block_idx).to(tl.float32)

                # Normalize and clamp to FP4 range
                # Use tl.div_rn for IEEE 754 round-to-nearest division
                scaled_x = tl.where(combined_scale == 0.0, 0.0, tl.div_rn(w, combined_scale))
                clipped_x = tl.minimum(tl.maximum(scaled_x, -6.0), 6.0)
                abs_x = tl.abs(clipped_x)

                # Comparison-sum bucketize with IEEE round-to-nearest-even tie-breaking
                # For odd E2M1 bounds (0.75, 1.75, 3.5), we use >= to round up to the even mantissa.
                ord_ = ((abs_x > 0.25).to(tl.int32) + (abs_x >= 0.75).to(tl.int32) +
                        (abs_x > 1.25).to(tl.int32) + (abs_x >= 1.75).to(tl.int32) +
                        (abs_x > 2.5).to(tl.int32) + (abs_x >= 3.5).to(tl.int32) +
                        (abs_x > 5.0).to(tl.int32))

                # Inline E2M1 decode: no table lookup needed
                # Subnormals (ord_ < 2): val = ord_ * 0.5 → {0.0, 0.5}
                # Normals (ord_ >= 2): val = (1 + mantissa_bit * 0.5) * 2^(exp_bits - 1)
                #   where mantissa_bit = ord_ & 1, exp_bits = ord_ >> 1
                m_bit = (ord_ & 1).to(tl.float32)
                e_bits = (ord_ >> 1).to(tl.float32)
                normal_val = (1.0 + m_bit * 0.5) * tl.math.exp2(e_bits - 1.0)
                subnormal_val = ord_.to(tl.float32) * 0.5
                q_val = tl.where(ord_ < 2, subnormal_val, normal_val)

                # Dequantize: sign * fp4_val * combined_scale
                sign = tl.where(clipped_x >= 0, 1.0, -1.0)
                result = sign * q_val * combined_scale

                tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _triton_fused_nvfp4_kernel(
        w_ptr, global_scale_ptr, out_ptr, N,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
    ):
        """Fused NVFP4 fake-quantize kernel."""
        pid = tl.program_id(0)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        
        for i in range(BLOCKS_PER_PROGRAM):
            block_idx = pid * BLOCKS_PER_PROGRAM + i
            base_offs = block_idx * BLOCK_SIZE
            offs = base_offs + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            if base_offs < N:
                w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                abs_w = tl.abs(w)
                vec_max = tl.max(abs_w)
                
                scale = vec_max / (6.0 * global_scale)
                scale = tl.minimum(tl.maximum(scale, -448.0), 448.0)
                # cast to E4M3 and back
                scale = scale.to(tl.float8e4nv).to(tl.float32)
                
                combined_scale = scale * global_scale

                scaled_x = tl.where(combined_scale == 0.0, 0.0, tl.div_rn(w, combined_scale))
                clipped_x = tl.minimum(tl.maximum(scaled_x, -6.0), 6.0)
                abs_x = tl.abs(clipped_x)

                ord_ = ((abs_x > 0.25).to(tl.int32) + (abs_x >= 0.75).to(tl.int32) +
                        (abs_x > 1.25).to(tl.int32) + (abs_x >= 1.75).to(tl.int32) +
                        (abs_x > 2.5).to(tl.int32) + (abs_x >= 3.5).to(tl.int32) +
                        (abs_x > 5.0).to(tl.int32))

                m_bit = (ord_ & 1).to(tl.float32)
                e_bits = (ord_ >> 1).to(tl.float32)
                normal_val = (1.0 + m_bit * 0.5) * tl.math.exp2(e_bits - 1.0)
                subnormal_val = ord_.to(tl.float32) * 0.5
                q_val = tl.where(ord_ < 2, subnormal_val, normal_val)

                sign = tl.where(clipped_x >= 0, 1.0, -1.0)
                result = sign * q_val * combined_scale

                tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)

def _fake_quantize_nvfp4_triton(
    weight: torch.Tensor, block_size: int, global_scale: torch.Tensor,
) -> torch.Tensor:
    """Two-pass NVFP4 fake quantization: PyTorch scale + Triton snap+dequant.

    Pass 1 (PyTorch): Compute per-block E4M3 scale and combined_scale = scale * global_scale.
    Pass 2 (Triton): Fused normalize + snap-to-E2M1 + dequantize.
    """
    original_shape = weight.shape
    N = weight.numel()
    num_blocks = triton.cdiv(N, block_size)
    assert N % block_size == 0

    # Pass 1: compute per-block E4M3 scales in PyTorch (exact fp8 cast, ModelOpt convention)
    w_blocks = weight.float().reshape(num_blocks, block_size)
    vec_max = w_blocks.abs().max(dim=-1, keepdim=True).values
    scale = vec_max / (E2M1_MAX * global_scale)
    scale = torch.clamp(scale, max=E4M3_MAX, min=-E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32).squeeze(-1)  # [num_blocks]
    
    # Precompute combined scale to match PyTorch evaluation exactly
    combined_scale = scale * global_scale  # [num_blocks], float32

    # Pass 2: Triton kernel — normalize, snap, dequant
    w_flat = weight.reshape(-1)
    out = torch.empty(N, dtype=torch.bfloat16, device=weight.device)
    
    # Process 64 blocks (1024 elements) per program to improve occupancy
    BLOCKS_PER_PROGRAM = 64
    num_programs = triton.cdiv(num_blocks, BLOCKS_PER_PROGRAM)
    grid = (num_programs,)

    _triton_fake_quant_nvfp4_kernel[grid](
        w_flat, combined_scale, out, N,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
    )
    return out.reshape(original_shape)

def _fake_quantize_nvfp4_triton_fused(
    weight: torch.Tensor, block_size: int, global_scale: torch.Tensor,
) -> torch.Tensor:
    """Triton V2 fused."""
    original_shape = weight.shape
    N = weight.numel()
    num_blocks = triton.cdiv(N, block_size)
    assert N % block_size == 0

    w_flat = weight.reshape(-1)
    out = torch.empty(N, dtype=torch.bfloat16, device=weight.device)
    
    BLOCKS_PER_PROGRAM = 64
    num_programs = triton.cdiv(num_blocks, BLOCKS_PER_PROGRAM)
    grid = (num_programs,)

    gs_tensor = global_scale.view(1)

    _triton_fused_nvfp4_kernel[grid](
        w_flat, gs_tensor, out, N,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
    )
    return out.reshape(original_shape)


def _fake_quantize_nvfp4_chunk(
    w_flat: torch.Tensor, block_size: int, global_scale: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """PyTorch fallback: quantize a flat (N, block_size) float32 tensor to fake-NVFP4."""
    vec_max = torch.max(torch.abs(w_flat), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = vec_max / (E2M1_MAX * global_scale)
    scale = torch.clamp(scale, max=E4M3_MAX, min=-E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)

    scaled_x = torch.where(
        scale == 0,
        torch.zeros_like(w_flat),
        w_flat / (scale * global_scale),
    )
    clipped_x = torch.clamp(scaled_x, -E2M1_MAX, E2M1_MAX)

    # Snap to nearest FP4 value — manual bucketize matching torch.bucketize(right=False)
    sign = torch.sign(clipped_x)
    abs_x = clipped_x.abs()
    ord_ = (
        (abs_x > 0.25).int() + (abs_x >= 0.75).int() + (abs_x > 1.25).int() +
        (abs_x >= 1.75).int() + (abs_x > 2.5).int() + (abs_x >= 3.5).int() +
        (abs_x > 5.0).int()
    )
    dequantized = sign * values[ord_]

    # Dequantize back: value * (scale * global_scale)
    dequantized = dequantized * (scale * global_scale)
    return dequantized

@torch.compile(mode="reduce-overhead")
def _fake_quantize_nvfp4_chunk_v2(
    w_flat: torch.Tensor, block_size: int, global_scale: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """PyTorch compiled fallback v2."""
    vec_max = torch.max(torch.abs(w_flat), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = vec_max / (E2M1_MAX * global_scale)
    scale = torch.clamp(scale, max=E4M3_MAX, min=-E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)

    scaled_x = torch.where(
        scale == 0,
        torch.zeros_like(w_flat),
        w_flat / (scale * global_scale),
    )
    clipped_x = torch.clamp(scaled_x, -E2M1_MAX, E2M1_MAX)

    sign = torch.sign(clipped_x)
    abs_x = clipped_x.abs()
    ord_ = (
        (abs_x > 0.25).int() + (abs_x >= 0.75).int() + (abs_x > 1.25).int() +
        (abs_x >= 1.75).int() + (abs_x > 2.5).int() + (abs_x >= 3.5).int() +
        (abs_x > 5.0).int()
    )
    dequantized = sign * values[ord_]

    dequantized = dequantized * (scale * global_scale)
    return dequantized


def fake_quantize_nvfp4(
    weight: torch.Tensor, block_size: int = 16,
    global_scale: torch.Tensor | None = None,
    max_chunk_rows: int = 65536,
) -> torch.Tensor:
    """Differentiable NVFP4 fake-quantizer for QAT.

    Forward: returns NVFP4-dequantized approximation (same shape/dtype as input).
    Backward: straight-through estimator — gradient flows unchanged.

    Uses a fused Triton kernel when available, falls back to PyTorch with
    large-chunk processing to bound peak memory.
    """
    original_shape = weight.shape
    original_dtype = weight.dtype

    if global_scale is None:
        global_scale = compute_nvfp4_global_scale(weight)

    # Triton path: fused kernel, zero intermediate memory
    if _HAS_TRITON and weight.is_cuda and weight.dtype == torch.bfloat16:
        dequantized = _fake_quantize_nvfp4_triton(weight, block_size, global_scale)
        return weight + (dequantized - weight).detach()

    # PyTorch fallback with large-chunk processing
    values = E2M1_VALUES.to(weight.device)
    total_rows = weight.numel() // block_size

    if total_rows <= max_chunk_rows:
        w_blocks = weight.float().reshape(-1, block_size)
        dequantized = _fake_quantize_nvfp4_chunk(w_blocks, block_size, global_scale, values)
        dequantized = dequantized.reshape(original_shape).to(original_dtype)
    else:
        w_blocks = weight.float().reshape(-1, block_size)
        out = torch.empty_like(w_blocks)
        for start in range(0, total_rows, max_chunk_rows):
            end = min(start + max_chunk_rows, total_rows)
            out[start:end] = _fake_quantize_nvfp4_chunk(
                w_blocks[start:end], block_size, global_scale, values,
            )
        dequantized = out.reshape(original_shape).to(original_dtype)

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
