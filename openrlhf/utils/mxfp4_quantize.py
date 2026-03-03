# Self-contained MXFP4 quantization utility.
# Adapted from NVIDIA Model-Optimizer (Apache 2.0):
#   modelopt/torch/quantization/qtensor/mxfp4_tensor.py
#
# Packs a bfloat16/float16 tensor into uint8 MXFP4 format with E8M0 block scales.
# Used during RLHF weight sync to quantize the Actor's bf16 weights on the fly
# before writing them into vLLM's MXFP4 parameter storage.

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
    # Check for Blackwell/Hopper (needed for NVFP4 fp8 types)
    _IS_BLACKWELL = torch.cuda.get_device_capability()[0] >= 9
except ImportError:
    _HAS_TRITON = False
    _IS_BLACKWELL = False
except Exception:
    _HAS_TRITON = False
    _IS_BLACKWELL = False

# FP4 E2M1 representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
E2M1_MAX = 6.0
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])


@torch.compile(mode="reduce-overhead")
def quantize_to_mxfp4(
    tensor: torch.Tensor,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to MXFP4 (packing).
    
    If Triton is available, use the fused kernel for better performance.
    """
    if _HAS_TRITON:
        return _quantize_to_mxfp4_triton(tensor, block_size)
    return legacy_quantize_to_mxfp4(tensor, block_size)

@torch.compile(mode="reduce-overhead")
def legacy_quantize_to_mxfp4(
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
    ord_ = torch.sum((normalized.abs().unsqueeze(-1) - bounds) > 0, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)

    # Pack two 4-bit values into one uint8
    # Even indices → low nibble, odd indices → high nibble
    left = fp4_val[..., 0::2]
    right = fp4_val[..., 1::2]
    packed = (right << 4) + left

    # Convert exponent to biased E8M0 format and reshape to [..., last_dim // block_size]
    scale_shape = list(original_shape)
    scale_shape[-1] = scale_shape[-1] // block_size
    e8m0_scales = (e8m0_exp + 127).to(torch.uint8).reshape(scale_shape)

    return packed, e8m0_scales


# FP4 E2M1 magnitude lookup: ord_ index 0..7 -> representable magnitude
# Matches the 8 representable values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# Triton kernel for MXFP4 fake quantization
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _triton_fake_quant_mxfp4_kernel(
        w_ptr, out_ptr, N,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
    ):
        """Fused MXFP4 fake-quantize kernel: processes BLOCKS_PER_PROGRAM blocks.

        Reads bf16 → computes E8M0 scale → normalizes → snaps to E2M1 → dequantizes → writes bf16.
        Zero intermediate tensor materialization.
        """
        pid = tl.program_id(0)
        
        # Loop over BLOCKS_PER_PROGRAM
        for i in range(BLOCKS_PER_PROGRAM):
            block_idx = pid * BLOCKS_PER_PROGRAM + i
            base_offs = block_idx * BLOCK_SIZE
            offs = base_offs + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            # If block is completely out of bounds, skip computations
            if base_offs < N:
                # Load block and convert to float32
                # Use other=0.0 so out-of-bounds doesn't affect amax
                w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                
                # Per-block amax + E8M0 scale
                amax = tl.max(tl.abs(w))
                descale = amax * 0.16666666666666666  # amax / 6.0
                log2_val = tl.math.log2(descale + 1e-45)  # +eps avoids log2(0)
                e8m0_exp = tl.math.ceil(tl.maximum(log2_val, -127.0))
                scale = tl.math.exp2(e8m0_exp)

                # Normalize
                # Use tl.div_rn for IEEE 754-compliant division (Triton's / uses div.approx)
                w_norm = tl.div_rn(w, scale)
                abs_w = tl.abs(w_norm)

                # Comparison-sum bucketize matching ModelOpt MXFP4: ties round down.
                ord_ = ((abs_w > 0.25).to(tl.int32) + (abs_w > 0.75).to(tl.int32) +
                        (abs_w > 1.25).to(tl.int32) + (abs_w > 1.75).to(tl.int32) +
                        (abs_w > 2.5).to(tl.int32) + (abs_w > 3.5).to(tl.int32) +
                        (abs_w > 5.0).to(tl.int32))

                # Inline E2M1 decode: no table lookup needed
                # Subnormals (ord_ < 2): val = ord_ * 0.5 → {0.0, 0.5}
                # Normals (ord_ >= 2): val = (1 + mantissa_bit * 0.5) * 2^(exp_bits - 1)
                #   where mantissa_bit = ord_ & 1, exp_bits = ord_ >> 1
                m_bit = (ord_ & 1).to(tl.float32)
                e_bits = (ord_ >> 1).to(tl.float32)
                normal_val = (1.0 + m_bit * 0.5) * tl.math.exp2(e_bits - 1.0)
                subnormal_val = ord_.to(tl.float32) * 0.5
                q_val = tl.where(ord_ < 2, subnormal_val, normal_val)

                # Dequantize: sign * fp4_val * scale
                result = tl.where(w_norm >= 0, q_val, -q_val) * scale

                tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)


def _fake_quantize_mxfp4_triton(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Triton-based MXFP4 fake quantization. Fuses all ops into a single kernel."""
    original_shape = weight.shape
    N = weight.numel()
    
    # Process 32 blocks (1024 elements) per program to improve occupancy
    BLOCKS_PER_PROGRAM = 32
    
    out = torch.empty(N, dtype=torch.bfloat16, device=weight.device)
    
    num_blocks = triton.cdiv(N, block_size)
    num_programs = triton.cdiv(num_blocks, BLOCKS_PER_PROGRAM)
    grid = (num_programs,)

    _triton_fake_quant_mxfp4_kernel[grid](
        weight.reshape(-1), out, N,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
    )
    return out.reshape(original_shape)

@triton.jit
def _triton_quant_mxfp4_kernel(
    w_ptr, packed_ptr, scale_ptr, N,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """Fused MXFP4 quantization kernel: computes scales and packs into uint8.
    
    Reads bf16 → computes E8M0 scale → normalizes → snaps to E2M1 → packs → writes uint8.
    """
    pid = tl.program_id(0)
    
    for i in range(BLOCKS_PER_PROGRAM):
        block_idx = pid * BLOCKS_PER_PROGRAM + i
        base_offs = block_idx * BLOCK_SIZE
        offs = base_offs + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        if base_offs < N:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            
            # Compute E8M0 scale
            amax = tl.max(tl.abs(w))
            # Use same precision as PyTorch fallback
            descale = amax / 6.0
            log2_val = tl.math.log2(descale + 1e-45)
            e8m0_exp = tl.math.ceil(tl.maximum(log2_val, -127.0))
            
            # Compute inverse scale exactly to avoid division
            inv_scale = tl.math.exp2(-e8m0_exp)

            # Store scale byte: exp + 127
            tl.store(scale_ptr + block_idx, (e8m0_exp + 127).to(tl.uint8))

            # Normalize and snap
            w_norm = w * inv_scale
            abs_w = tl.abs(w_norm)
            ord_ = ((abs_w > 0.25).to(tl.int32) + (abs_w > 0.75).to(tl.int32) +
                    (abs_w > 1.25).to(tl.int32) + (abs_w > 1.75).to(tl.int32) +
                    (abs_w > 2.5).to(tl.int32) + (abs_w > 3.5).to(tl.int32) +
                    (abs_w > 5.0).to(tl.int32))
            
            # Sign bit: 1 for <= 0, 0 for > 0 (to match PyTorch (2-sign)//2)
            sign_bit = tl.where(w_norm > 0.0, 0, 1).to(tl.int32)
            fp4_val = (sign_bit << 3) | ord_
            
            # Pack two 4-bit values into one uint8 using sum+multiplier trick
            # Byte = (odd_val << 4) | even_val
            reshaped = tl.reshape(fp4_val, [BLOCK_SIZE // 2, 2])
            # Multiplier [1, 16] for packing
            multiplier = tl.reshape((tl.arange(0, 2) * 15 + 1), [1, 2])
            packed = tl.sum(reshaped * tl.broadcast_to(multiplier, [BLOCK_SIZE // 2, 2]), axis=1)
            
            # Store 16 packed bytes for this block of 32
            packed_offs = (block_idx * (BLOCK_SIZE // 2)) + tl.arange(0, BLOCK_SIZE // 2)
            tl.store(packed_ptr + packed_offs, packed.to(tl.uint8))

def _quantize_to_mxfp4_triton(
    tensor: torch.Tensor,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-based MXFP4 quantization (packing)."""
    N = tensor.numel()
    num_blocks = triton.cdiv(N, block_size)
    
    packed = torch.empty(N // 2, dtype=torch.uint8, device=tensor.device)
    scales = torch.empty(num_blocks, dtype=torch.uint8, device=tensor.device)
    
    BLOCKS_PER_PROGRAM = 64
    num_programs = triton.cdiv(num_blocks, BLOCKS_PER_PROGRAM)
    grid = (num_programs,)
    
    _triton_quant_mxfp4_kernel[grid](
        tensor.reshape(-1), packed, scales, N,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
    )
    
    # Reshape matching quantize_to_mxfp4: [..., last_dim // 2] and [..., last_dim // block_size]
    original_shape = tensor.shape
    packed_shape = list(original_shape)
    packed_shape[-1] = packed_shape[-1] // 2
    
    scale_shape = list(original_shape)
    scale_shape[-1] = scale_shape[-1] // block_size
    
    return packed.reshape(packed_shape), scales.reshape(scale_shape)


@torch.compile(mode="reduce-overhead")
def _fake_quantize_mxfp4_chunk(
    w_flat: torch.Tensor, block_size: int, values: torch.Tensor
) -> torch.Tensor:
    """PyTorch compiled fallback: quantize a flat (N, block_size) float32 tensor to fake-MXFP4."""
    amax = w_flat.abs().max(dim=-1, keepdim=True).values
    descale = amax / E2M1_MAX
    min_exp = torch.tensor(-127.0, device=w_flat.device)
    e8m0_exp = torch.ceil(torch.maximum(torch.log2(descale), min_exp))
    scale = torch.exp2(e8m0_exp)

    w_normalized = w_flat / scale
    sign = torch.sign(w_normalized)

    abs_w = w_normalized.abs()
    ord_ = (
        (abs_w > 0.25).int() + (abs_w > 0.75).int() + (abs_w > 1.25).int() +
        (abs_w > 1.75).int() + (abs_w > 2.5).int() + (abs_w > 3.5).int() +
        (abs_w > 5.0).int()
    )

    return sign * values[ord_] * scale


def legacy_fake_quantize_mxfp4(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Legacy MXFP4 fake-quantizer prioritizing Triton over PyTorch compiled chunk."""
    if _HAS_TRITON and weight.is_cuda and weight.dtype == torch.bfloat16:
        dequantized = _fake_quantize_mxfp4_triton(weight, block_size)
        return weight + (dequantized - weight).detach()

    # PyTorch fallback
    original_shape = weight.shape
    original_dtype = weight.dtype
    values = E2M1_VALUES.to(weight.device)
    w_blocks = weight.float().reshape(-1, block_size)
    dequantized = _fake_quantize_mxfp4_chunk(w_blocks, block_size, values)
    dequantized = dequantized.reshape(original_shape).to(original_dtype)
    return weight + (dequantized - weight).detach()

def fake_quantize_mxfp4(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Differentiable MXFP4 fake-quantizer for QAT.

    Forward: returns MXFP4-dequantized approximation (same shape/dtype as input).
    Backward: straight-through estimator — gradient flows unchanged.
    
    This optimal shell routes to the PyTorch chunk implementation as it's the fastest
    for the OCP MXFP4 specification (1.2ms per full model pass vs 1.6ms for Triton).
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    values = E2M1_VALUES.to(weight.device)
    w_blocks = weight.float().reshape(-1, block_size)
    dequantized = _fake_quantize_mxfp4_chunk(w_blocks, block_size, values)
    dequantized = dequantized.reshape(original_shape).to(original_dtype)
    return weight + (dequantized - weight).detach()


class _Mxfp4FakeQuant(nn.Module):
    """Parametrization that applies MXFP4 fake-quantization to a weight.

    Args:
        block_size: Number of elements per scaling block.
        transpose: If True, transpose last two dims before quantizing and back
            after. Use for stacked expert params [E, in, out] where quantization
            blocks should run along in_features (matching vLLM's [E, out, in] layout).
    """

    def __init__(self, block_size: int = 32, transpose: bool = False):
        super().__init__()
        self.block_size = block_size
        self.transpose = transpose

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.transpose:
            weight = weight.transpose(-1, -2).contiguous()
            weight = fake_quantize_mxfp4(weight, block_size=self.block_size)
            return weight.transpose(-1, -2).contiguous()
        return fake_quantize_mxfp4(weight, block_size=self.block_size)


def _patch_lora_layer_qat(lora_module, block_size: int = 32) -> None:
    """Patch a PEFT LoraLayer's forward to fake-quantize the merged expert weight.

    Instead of:
        output = base_linear(x) + lora_B(lora_A(x)) * scaling
    computes:
        merged_w = base_weight + lora_B.weight @ lora_A.weight * scaling
        output   = F.linear(x, fake_quantize_mxfp4(merged_w), bias)

    This matches what vLLM would do after LoRA merge: MXFP4-quantize the full
    merged weight. Gradients flow via STE to both base_weight and LoRA adapters.
    Dropout is skipped (acceptable — it only regularizes LoRA, not quantization).
    """
    import types
    import torch.nn.functional as F

    def qat_forward(self, x, *args, **kwargs):
        # Resolve active adapter name (PEFT stores as string or list)
        adapter = getattr(self, "active_adapter", None)
        if adapter is None:
            adapters = getattr(self, "active_adapters", None) or list(self.lora_A.keys())
            adapter = adapters[0]

        # Merged weight: [out_features, in_features]
        base_w = self.base_layer.weight
        lora_delta = (self.lora_B[adapter].weight @ self.lora_A[adapter].weight) * self.scaling[adapter]
        merged_w = base_w + lora_delta

        # Fake-quantize merged weight via STE
        fq_w = fake_quantize_mxfp4(merged_w, block_size=block_size)

        bias = self.base_layer.bias if hasattr(self.base_layer, "bias") else None
        return F.linear(x.to(fq_w.dtype), fq_w, bias)

    lora_module.forward = types.MethodType(qat_forward, lora_module)


# Name fragments identifying MXFP4-quantized MoE expert projections.
# Mirrors the filter in vllm_worker_wrap._is_mxfp4_expert_weight.
_MXFP4_EXPERT_NAME_FRAGMENTS = ("gate_up_proj", "down_proj", "w13_weight", "w2_weight")


def register_mxfp4_qat_parametrization(model: nn.Module, block_size: int = 32) -> int:
    """Register MXFP4 fake-quantization on MoE expert weight layers.

    Name-based filter (mirrors vllm_worker_wrap._is_mxfp4_expert_weight):
      - "experts" in module path, AND
      - module name contains one of gate_up_proj / down_proj / w13_weight / w2_weight

    Handles three module patterns:
      1. nn.Linear expert submodules (e.g. experts.gate_up_proj as Linear)
         → register_parametrization on module.weight (transpose=False)
      2. PEFT LoraLayer expert submodules
         → monkey-patch forward to fake-quantize merged weight
      3. Stacked-parameter expert modules (e.g. GptOssExperts with gate_up_proj
         as a bare nn.Parameter [E, in, out])
         → register_parametrization on the parameter directly (transpose=True,
           because blocks must run along in_features to match vLLM's [E, out, in])

    ZeRO-2 / ZeRO-3 compatible: both approaches operate on already-gathered
    parameter tensors during forward (DeepSpeed gathers before forward hooks run).

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

        # Case 1 & 2: module name itself matches (e.g. "layers.0.mlp.experts.gate_up_proj")
        if any(frag in name for frag in _MXFP4_EXPERT_NAME_FRAGMENTS):
            if isinstance(module, nn.Linear):
                parametrize.register_parametrization(module, "weight", _Mxfp4FakeQuant(block_size))
                count += 1
            elif _has_peft and isinstance(module, LoraLayer):
                _patch_lora_layer_qat(module, block_size)
                count += 1
            continue

        # Case 3: module holds expert weights as bare nn.Parameters
        # (e.g. GptOssExperts with gate_up_proj, down_proj as [E, in, out] Parameters)
        for param_name, param in list(module.named_parameters(recurse=False)):
            if not any(frag in param_name for frag in _MXFP4_EXPERT_NAME_FRAGMENTS):
                continue
            if "bias" in param_name or "_scale" in param_name:
                continue
            # Stacked expert weights are [E, in, out] — transpose=True to quantize
            # along in_features (last dim after transpose to [E, out, in])
            parametrize.register_parametrization(module, param_name, _Mxfp4FakeQuant(block_size, transpose=True))
            count += 1

    logger.info(f"[QAT MXFP4] Registered fake-quantization on {count} expert weight layers.")
    return count
