import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean

logger = logging.getLogger(__name__)


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        loss = (
            masked_mean(-per_token_logps, loss_mask, dim=None)
            if self.token_level_loss
            else masked_mean(-per_token_logps, loss_mask, dim=-1).mean()
        )

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        vllm_is_correction_type: str = "tis",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type

        # GSPO requires sequence-level loss
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if self.vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self.vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
            )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.policy_loss_type == "ppo":
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = log_probs - old_log_probs
            ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            log_ratio = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
                vllm_is = torch.exp(log_ratio).detach()
                mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
                vllm_is = vllm_is * mask
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                # seq-mask-tis: use sequence-level geometric mean only for filtering,
                # correction coefficients still use TIS (token-level clamp)
                seq_log_ratio = masked_mean(log_ratio, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
                vllm_is = torch.exp(log_ratio).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:
                # TIS: token-level clamp with low and high thresholds
                vllm_is = torch.exp(log_ratio).clamp(min=low_threshold, max=high_threshold).detach()
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        #### token_level_loss reduction ####
        if not self.token_level_loss:
            # grpo/ppo/gspo: per-sequence mean, then batch mean
            loss = masked_mean(loss, action_mask, dim=-1).mean()
        else:
            # dapo/bnpo: flat token mean within this rank
            loss = masked_mean(loss, action_mask, dim=None)
        #### end token_level_loss reduction ####
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl


#### Liger fused GRPO loss wrapper (triton backend) ####
class LigerPolicyLoss(nn.Module):
    """
    Fused lm_head + GRPO loss via liger-kernel (>= 0.7.0).

    Uses the Triton backend: fused Triton kernels for log-softmax + loss + backward.
    Avoids storing the log-softmax tensor by recomputing from saved LSE.

    Supports the same vLLM off-policy correction methods as PolicyLoss
    (TIS, ICEPOP, seq-mask-tis).

    Returns the same (loss, clip_ratio, ppo_kl, vllm_kl) tuple as PolicyLoss.
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        beta: float = 0.0,
        temperature: float = 1.0,
        loss_type: str = "grpo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        vllm_is_correction_type: str = "tis",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.temperature = temperature
        self.loss_type = loss_type
        self.use_ref_model = beta > 0
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type

        if enable_vllm_is_correction and vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
            )

        from liger_kernel.transformers.grpo_loss import triton_grpo_loss

        self._triton_grpo_loss = triton_grpo_loss
        logger.info(
            f"[Liger GRPO] Initialized Triton backend "
            f"(beta={beta}, eps=[{clip_eps_low}, {clip_eps_high}], loss_type={loss_type})"
        )

        self._eps_low = clip_eps_low
        self._eps_high = clip_eps_high

    def _compute_vllm_is_ratio(
        self,
        old_log_probs: torch.Tensor,
        rollout_log_probs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute vLLM importance sampling ratio (TIS/ICEPOP/seq-mask-tis)."""
        if not self.enable_vllm_is_correction or rollout_log_probs is None:
            return None, None

        low_threshold, high_threshold = self.vllm_is_truncated_threshold
        log_ratio = old_log_probs - rollout_log_probs

        if self.vllm_is_correction_type == "icepop":
            vllm_is = torch.exp(log_ratio).detach()
            mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
            vllm_is_ratio = vllm_is * mask
        elif self.vllm_is_correction_type == "seq-mask-tis":
            seq_log_ratio = masked_mean(log_ratio, action_mask, dim=-1)
            seq_is = torch.exp(seq_log_ratio)
            seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
            vllm_is = torch.exp(log_ratio).detach()
            vllm_is_ratio = seq_mask.unsqueeze(-1) * vllm_is
        else:
            vllm_is_ratio = torch.exp(log_ratio).clamp(min=low_threshold, max=high_threshold).detach()

        vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)
        return vllm_is_ratio, vllm_kl

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Linear,
        completion_ids: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
        ref_log_probs: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Backbone output, shape [B, L+1, D] (triton needs next-token positions).
            lm_head: The model's lm_head linear layer.
            completion_ids: Token IDs for the completion/action region, shape [B, L].
            action_mask: Mask for valid action tokens, shape [B, L].
            advantages: Per-token or per-sequence advantages.
            old_log_probs: Log probs from the rollout policy, shape [B, L].
            ref_log_probs: Log probs from the reference model, shape [B, L] (None if beta=0).
            rollout_log_probs: Log probs from vLLM rollout (for IS correction), shape [B, L].

        Returns:
            (loss, clip_ratio, ppo_kl, vllm_kl) — same format as PolicyLoss.forward().
        """
        ref_lp = ref_log_probs if self.use_ref_model else None

        vllm_is_ratio, vllm_kl = self._compute_vllm_is_ratio(
            old_log_probs, rollout_log_probs, action_mask
        )

        logits = F.linear(
            hidden_states, lm_head.weight, getattr(lm_head, "bias", None)
        ).contiguous()

        loss, metrics = self._triton_grpo_loss(
            logits=logits,
            old_logp=old_log_probs,
            ref_logp=ref_lp,
            completion_ids=completion_ids.contiguous(),
            advantages=advantages.contiguous(),
            completion_mask=action_mask.contiguous(),
            temperature=self.temperature,
            beta=self.beta,
            eps_low=self._eps_low,
            eps_high=self._eps_high,
            loss_type=self.loss_type,
            reduce=True,
            vllm_is_ratio=vllm_is_ratio,
        )

        clip_ratio = metrics[-1]

        #### Compute ppo_kl from hidden states (approx KL between new and old policy) ####
        # The Liger kernel computes new log probs internally but doesn't expose them.
        # Recompute per-sequence in no_grad to avoid materializing full (B,L,V) logits.
        with torch.no_grad():
            if self.backend == "triton":
                hs_for_lp = hidden_states[:, :-1, :]  # (B, L, D)
            else:
                hs_for_lp = hidden_states  # already (B, L, D)
            per_seq_lp = []
            for b in range(hs_for_lp.shape[0]):
                logits_b = F.linear(hs_for_lp[b], lm_head.weight, getattr(lm_head, "bias", None))
                lp_b = logits_b.log_softmax(-1).gather(-1, completion_ids[b].unsqueeze(-1)).squeeze(-1)
                per_seq_lp.append(lp_b)
            new_log_probs = torch.stack(per_seq_lp)
            log_ratio = new_log_probs - old_log_probs
            ppo_kl = masked_mean(-log_ratio, action_mask, dim=None)
        #### end compute ppo_kl ####

        return loss, clip_ratio, ppo_kl, vllm_kl
#### end Liger fused GRPO loss wrapper ####


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = (
            masked_mean(loss, action_mask, dim=None)
            if self.token_level_loss
            else masked_mean(loss, action_mask, dim=-1).mean()
        )
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask].squeeze(1)
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
