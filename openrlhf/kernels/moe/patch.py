# MoE kernel patching - applies grouped GEMM kernels to HF Transformers MoE models
# Adapted from Unsloth Zoo (AGPL-3.0)
# Source: https://github.com/unslothai/unsloth-zoo

import logging

from .moe_utils import forward_moe_backend

logger = logging.getLogger(__name__)


def _patch_experts_class(cls, label):
    """Replace the forward method on an MoE experts class with grouped GEMM dispatch."""
    original_forward = cls.forward
    cls._original_forward = original_forward
    cls.forward = forward_moe_backend
    logger.info(f"[unsloth_moe] Patched {label}.forward → forward_moe_backend")
    return True


def patch_moe_kernels():
    """
    Monkey-patch HF Transformers MoE expert modules to use grouped GEMM kernels.

    Supports:
    - Qwen3MoE (Qwen3MoeExperts / Qwen3MoeSparseMoeBlock)
    - Qwen3.5MoE (Qwen3_5MoeExperts / Qwen3_5MoeSparseMoeBlock)
    - Qwen2MoE (Qwen2MoeExperts / Qwen2MoeSparseMoeBlock)
    - GPT-OSS (GptOssExperts via trust_remote_code / instance-level)

    The patching replaces the expert-level forward (the inner loop over experts)
    with `forward_moe_backend` which dispatches to grouped_mm / Triton / loop fallback.

    For models loaded via trust_remote_code (GPT-OSS), patching happens at the
    instance level in `patch_moe_model_instance` instead.
    """
    patched = []

    # --- Qwen3 MoE ---
    try:
        from transformers.models.qwen3_moe import modeling_qwen3_moe

        # New transformers (v5+): Qwen3MoeExperts has stacked weights
        if hasattr(modeling_qwen3_moe, "Qwen3MoeExperts"):
            _patch_experts_class(modeling_qwen3_moe.Qwen3MoeExperts, "Qwen3MoeExperts")
            patched.append("Qwen3MoeExperts")
        # Old transformers: expert loop is inside SparseMoeBlock
        elif hasattr(modeling_qwen3_moe, "Qwen3MoeSparseMoeBlock"):
            _patch_experts_class(modeling_qwen3_moe.Qwen3MoeSparseMoeBlock, "Qwen3MoeSparseMoeBlock")
            patched.append("Qwen3MoeSparseMoeBlock")
    except (ImportError, ModuleNotFoundError):
        pass

    # --- Qwen3.5 MoE ---
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

        if hasattr(modeling_qwen3_5_moe, "Qwen3_5MoeExperts"):
            _patch_experts_class(modeling_qwen3_5_moe.Qwen3_5MoeExperts, "Qwen3_5MoeExperts")
            patched.append("Qwen3_5MoeExperts")
        elif hasattr(modeling_qwen3_5_moe, "Qwen3_5MoeSparseMoeBlock"):
            _patch_experts_class(modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock, "Qwen3_5MoeSparseMoeBlock")
            patched.append("Qwen3_5MoeSparseMoeBlock")
    except (ImportError, ModuleNotFoundError):
        pass

    # --- Qwen2 MoE ---
    try:
        from transformers.models.qwen2_moe import modeling_qwen2_moe

        if hasattr(modeling_qwen2_moe, "Qwen2MoeExperts"):
            _patch_experts_class(modeling_qwen2_moe.Qwen2MoeExperts, "Qwen2MoeExperts")
            patched.append("Qwen2MoeExperts")
        elif hasattr(modeling_qwen2_moe, "Qwen2MoeSparseMoeBlock"):
            _patch_experts_class(modeling_qwen2_moe.Qwen2MoeSparseMoeBlock, "Qwen2MoeSparseMoeBlock")
            patched.append("Qwen2MoeSparseMoeBlock")
    except (ImportError, ModuleNotFoundError):
        pass

    if patched:
        logger.info(f"[unsloth_moe] Patched MoE classes: {patched}")
    else:
        logger.warning("[unsloth_moe] No MoE classes found to patch at the class level. "
                       "For trust_remote_code models (GPT-OSS), instance-level patching will be applied.")

    return patched


def patch_moe_model_instance(model):
    """
    Patch MoE expert modules at the instance level (for trust_remote_code models
    like GPT-OSS where the class isn't available in transformers at import time).

    Should be called after model loading. Walks all modules and patches any that
    have stacked expert weights (gate_up_proj / down_proj or w1/w3/w2) and num_experts.
    """
    patched_count = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        # Skip if already patched
        if hasattr(module, "_original_forward"):
            continue

        # Detect expert modules by structure: must have num_experts + stacked weights
        has_num_experts = hasattr(module, "num_experts")
        has_stacked_weights = (
            (hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"))
            or (hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3"))
        )

        if has_num_experts and has_stacked_weights:
            # Patch at instance level via bound method
            import types
            module._original_forward = module.forward
            module.forward = types.MethodType(forward_moe_backend, module)
            patched_count += 1
            logger.debug(f"[unsloth_moe] Instance-patched {name} ({cls_name})")

    if patched_count > 0:
        logger.info(f"[unsloth_moe] Instance-patched {patched_count} expert modules in model")
    else:
        logger.warning("[unsloth_moe] No expert modules found to patch in model instance")

    return patched_count
