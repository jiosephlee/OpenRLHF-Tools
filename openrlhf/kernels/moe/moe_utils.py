# MoE kernel dispatch utilities - adapted from Unsloth Zoo (AGPL-3.0)
# Source: https://github.com/unslothai/unsloth-zoo
# Stripped to essentials for full-parameter training (no LoRA support).

import os
import logging
from functools import lru_cache

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Backend availability checks
# ============================================================================

_GROUPED_GEMM_AVAILABLE = None
_TORCH_GROUPED_MM_AVAILABLE = hasattr(torch, "_grouped_mm")
_TORCH_GROUPED_MM_SUPPORTED = None


def _check_torch_grouped_mm_supported():
    global _TORCH_GROUPED_MM_SUPPORTED
    if _TORCH_GROUPED_MM_SUPPORTED is not None:
        return _TORCH_GROUPED_MM_SUPPORTED

    if not _TORCH_GROUPED_MM_AVAILABLE or not torch.cuda.is_available():
        _TORCH_GROUPED_MM_SUPPORTED = False
        return False

    try:
        device = torch.cuda.current_device()
        x = torch.ones((1, 8), device=device, dtype=torch.float16)
        w = torch.ones((1, 8, 8), device=device, dtype=torch.float16)
        offs = torch.tensor([1], device=device, dtype=torch.int32)
        torch._grouped_mm(x, w, offs=offs)
        del x, w, offs
        _TORCH_GROUPED_MM_SUPPORTED = True
    except Exception:
        _TORCH_GROUPED_MM_SUPPORTED = False

    return _TORCH_GROUPED_MM_SUPPORTED


def _check_grouped_gemm_available():
    if os.environ.get("UNSLOTH_DISABLE_MOE_TRITON", "0") == "1":
        return False
    global _GROUPED_GEMM_AVAILABLE
    if _GROUPED_GEMM_AVAILABLE is not None:
        return _GROUPED_GEMM_AVAILABLE
    try:
        from openrlhf.kernels.moe.grouped_gemm.interface import grouped_gemm, supports_tma
        _GROUPED_GEMM_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        _GROUPED_GEMM_AVAILABLE = False
    return _GROUPED_GEMM_AVAILABLE


@lru_cache(maxsize=1)
def select_moe_backend():
    """
    Select MoE backend: "grouped_mm" (H100+), "unsloth_triton" (A100+), or "native_torch".
    Override with UNSLOTH_MOE_BACKEND env variable.
    """
    requested = os.environ.get("UNSLOTH_MOE_BACKEND")
    if requested:
        if requested == "grouped_mm" and _check_torch_grouped_mm_supported():
            return "grouped_mm"
        if requested == "unsloth_triton" and _check_grouped_gemm_available():
            return "unsloth_triton"
        if requested == "native_torch":
            return "native_torch"
        logger.warning(f"Requested MoE backend '{requested}' not available, falling back.")

    if _check_torch_grouped_mm_supported():
        logger.info("Using MoE backend: grouped_mm (torch._grouped_mm)")
        return "grouped_mm"
    if _check_grouped_gemm_available():
        logger.info("Using MoE backend: unsloth_triton")
        return "unsloth_triton"
    logger.info("Using MoE backend: native_torch (loop fallback)")
    return "native_torch"


# ============================================================================
# Routing helpers
# ============================================================================

@torch.no_grad()
def _get_routing_indices(selected_experts, num_experts):
    flat_experts = selected_experts.view(-1)
    token_counts_by_expert = torch.bincount(flat_experts, minlength=num_experts).to(torch.int32)
    gather_indices = flat_experts.argsort(stable=True)
    return token_counts_by_expert, gather_indices


def _silu_and_mul(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def _grouped_mm_with_backward_fix(inputs, weight, offsets):
    return torch._grouped_mm(inputs, weight, offs=offsets)


# ============================================================================
# Backend implementations
# ============================================================================

def forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights):
    """
    Native torch._grouped_mm MoE forward. Full-parameter training only (no LoRA).
    """
    is_2d = hidden_states.dim() == 2
    if is_2d:
        seq_len, hidden_dim = hidden_states.shape
        batch_size = 1
    else:
        batch_size, seq_len, hidden_dim = hidden_states.shape

    hidden_states = hidden_states.view(-1, hidden_dim)

    flat_top_k = top_k_index.view(-1)
    num_tokens_per_expert = torch.bincount(flat_top_k, minlength=self.num_experts).int()
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k_index.shape[-1]
    permuted_input = hidden_states[token_indices]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # Gate + Up projection
    if hasattr(self, "gate_up_proj"):
        gate_up_base = self.gate_up_proj
        if gate_up_base.shape[-1] == hidden_dim:
            w1 = gate_up_base
        else:
            w1 = gate_up_base.transpose(-2, -1).contiguous()
        mm1_out = _grouped_mm_with_backward_fix(permuted_input, w1, offsets)

        if "GptOssExperts" in self.__class__.__name__:
            gate = mm1_out[..., ::2]
            up = mm1_out[..., 1::2]
        else:
            gate, up = mm1_out.chunk(2, dim=-1)
    elif hasattr(self, "w1") and hasattr(self, "w3"):
        w1 = self.w1.transpose(-2, -1)
        w3 = self.w3.transpose(-2, -1)
        gate = _grouped_mm_with_backward_fix(permuted_input, w1, offsets)
        up = _grouped_mm_with_backward_fix(permuted_input, w3, offsets)
    else:
        raise AttributeError("MoE layer must have 'gate_up_proj' or 'w1'/'w3'.")

    # Activation
    if "GptOssExperts" in self.__class__.__name__:
        limit = getattr(self, "limit", 7.0)
        alpha = getattr(self, "alpha", 1.702)
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        inter = (up + 1.0) * glu
    else:
        inter = F.silu(gate) * up

    # Down projection
    if hasattr(self, "down_proj"):
        down_base = self.down_proj
        if down_base.shape[2] == hidden_dim:
            w2 = down_base
        else:
            w2 = down_base.transpose(-2, -1).contiguous()
        mm2_out = _grouped_mm_with_backward_fix(inter, w2, offsets)
    elif hasattr(self, "w2"):
        w2 = self.w2.transpose(-2, -1)
        mm2_out = _grouped_mm_with_backward_fix(inter, w2, offsets)
    else:
        raise AttributeError("MoE layer must have 'down_proj' or 'w2'.")

    # Apply routing weights and scatter-add
    flat_weights = top_k_weights.view(-1)
    permuted_weights = flat_weights[sorted_indices]
    mm2_out = mm2_out * permuted_weights.unsqueeze(-1)

    final_hidden_states = torch.zeros(
        (batch_size * seq_len, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    final_hidden_states.index_add_(0, token_indices, mm2_out.to(hidden_states.dtype))

    if is_2d:
        return final_hidden_states
    return final_hidden_states.view(batch_size, seq_len, hidden_dim)


def forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights):
    """
    Triton grouped GEMM MoE forward. Full-parameter training only (no LoRA).
    """
    from openrlhf.kernels.moe.grouped_gemm.interface import grouped_gemm
    from openrlhf.kernels.moe.autotune_cache import get_or_autotune_moe_kernels

    if not hasattr(self, "_unsloth_moe_configs"):
        self._unsloth_moe_configs = None

    is_3d = hidden_states.dim() == 3
    if is_3d:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = batch_size * seq_len
        if top_k_index.dim() == 3:
            top_k_index = top_k_index.view(-1, top_k_index.shape[-1])
        if top_k_weights.dim() == 3:
            top_k_weights = top_k_weights.view(-1, top_k_weights.shape[-1])
    else:
        num_tokens, hidden_dim = hidden_states.shape

    top_k = top_k_index.shape[1]

    if self._unsloth_moe_configs is None:
        intermediate_dim = self.gate_up_proj.shape[1] // 2

        gemm1_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim * 2,
            top_k=top_k,
            dtype=hidden_states.dtype,
        )
        gemm2_configs = get_or_autotune_moe_kernels(
            num_experts=self.num_experts,
            hidden_dim=intermediate_dim,
            intermediate_dim=hidden_dim,
            top_k=top_k,
            dtype=hidden_states.dtype,
        )
        self._unsloth_moe_configs = (intermediate_dim, gemm1_configs, gemm2_configs)
        torch.cuda.empty_cache()

    intermediate_dim, gemm1_configs, gemm2_configs = self._unsloth_moe_configs
    fwd_config_1, bwd_dX_config_1, bwd_dW_config_1 = gemm1_configs
    fwd_config_2, bwd_dX_config_2, bwd_dW_config_2 = gemm2_configs

    token_counts_by_expert, gather_indices = _get_routing_indices(top_k_index, self.num_experts)

    if self.gate_up_proj.shape[-1] == hidden_dim:
        w1 = self.gate_up_proj
    else:
        w1 = self.gate_up_proj.transpose(-2, -1).contiguous()

    first_gemm_output = grouped_gemm(
        X=hidden_states, W=w1, m_sizes=token_counts_by_expert,
        topk=top_k, gather_indices=gather_indices,
        permute_x=True, permute_y=False, autotune=False,
        kernel_config_fwd=fwd_config_1,
        kernel_config_bwd_dX=bwd_dX_config_1,
        kernel_config_bwd_dW=bwd_dW_config_1,
        is_first_gemm=True,
    )

    intermediate = _silu_and_mul(first_gemm_output)

    if self.down_proj.shape[-1] == intermediate.shape[-1]:
        w2 = self.down_proj
    else:
        w2 = self.down_proj.transpose(-2, -1).contiguous()

    second_gemm_output = grouped_gemm(
        X=intermediate, W=w2, m_sizes=token_counts_by_expert,
        topk=top_k, gather_indices=gather_indices,
        permute_x=False, permute_y=True, autotune=False,
        kernel_config_fwd=fwd_config_2,
        kernel_config_bwd_dX=bwd_dX_config_2,
        kernel_config_bwd_dW=bwd_dW_config_2,
        is_first_gemm=False,
    )

    top_k_weights_casted = top_k_weights.to(hidden_states.dtype)
    final_hidden_states = (
        second_gemm_output.view(num_tokens, top_k, hidden_dim)
        * top_k_weights_casted[..., None]
    )
    final_hidden_states = final_hidden_states.sum(dim=1)

    if is_3d:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

    return final_hidden_states


@torch.compiler.disable
def forward_native_moe_loop(self, hidden_states, top_k_index, top_k_weights):
    """Loop-based MoE forward (fallback). Disabled for torch.compile."""
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t.item()
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        if hasattr(self, "gate_up_proj"):
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        else:
            gate = F.linear(current_state, self.w1[expert_idx])
            up = F.linear(current_state, self.w3[expert_idx])

        current_hidden_states = self.act_fn(gate) * up

        if hasattr(self, "down_proj"):
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
        else:
            current_hidden_states = F.linear(current_hidden_states, self.w2[expert_idx])

        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def forward_moe_backend(self, hidden_states, top_k_index, top_k_weights):
    """Dispatch MoE forward to the selected backend."""
    backend = select_moe_backend()
    if backend == "grouped_mm":
        return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)
    if backend == "unsloth_triton":
        return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)
    return forward_native_moe_loop(self, hidden_states, top_k_index, top_k_weights)
