import math
import os
import socket
from abc import ABC
from typing import Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Actor, LigerPolicyLoss, PolicyLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger

from ..ppo_utils import NaiveReplayBuffer

logger = init_logger(__name__)

from .launcher import BaseModelActor
from .utils import get_physical_gpu_id


def _build_vllm_sync_params(model, zero_stage: int):
    """Build (hf_name, tensor) list for vLLM weight sync.

    For PEFT LoRA models:
      - Merges lora_B @ lora_A * scaling into base weight, broadcasts under
        canonical HF name (strips "base_model.model." prefix and ".base_layer.").
      - Skips lora_A / lora_B tensors (absorbed into merged weight above).
    For plain models: returns list(model.named_parameters()) unchanged.

    ZeRO-3 + LoRA: raises NotImplementedError. All three tensors (base,
    lora_A, lora_B) must be gathered simultaneously before merging — not
    yet supported. Use ZeRO-2.
    """
    try:
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer

        is_peft = isinstance(model, PeftModel)
    except ImportError:
        is_peft = False

    if not is_peft:
        import re
        result = []
        for name, param in model.named_parameters():
            # nn.utils.parametrize stores the original parameter under
            # "module.parametrizations.PNAME.original" rather than "module.PNAME".
            # Strip that so vLLM receives the canonical HF weight name.
            name = re.sub(r'\.parametrizations\.([^.]+)\.original$', r'.\1', name)
            result.append((name, param))
        return result

    if zero_stage == 3:
        raise NotImplementedError(
            "LoRA weight sync to vLLM with ZeRO-3 is not yet supported. Use ZeRO-2 when training with lora_rank > 0."
        )

    # Build peft_module_path -> LoraLayer
    lora_module_map = {name: mod for name, mod in model.named_modules() if isinstance(mod, LoraLayer)}

    skip_names = set()
    result = []

    # Pass 1: merged LoRA weights
    for peft_name, lora_mod in lora_module_map.items():
        adapter = getattr(lora_mod, "active_adapters", None) or getattr(lora_mod, "active_adapter", None)
        if isinstance(adapter, list):
            adapter = adapter[0]
        if adapter is None:
            adapter = list(lora_mod.lora_A.keys())[0]

        prefix = peft_name + "."
        skip_names.add(f"{prefix}base_layer.weight")
        skip_names.add(f"{prefix}lora_A.{adapter}.weight")
        skip_names.add(f"{prefix}lora_B.{adapter}.weight")

        base_w = lora_mod.base_layer.weight.data
        lora_A = lora_mod.lora_A[adapter].weight.data
        lora_B = lora_mod.lora_B[adapter].weight.data
        merged = (base_w + (lora_B @ lora_A) * lora_mod.scaling[adapter]).to(base_w.dtype)

        # Canonical HF name: strip PEFT prefix and .base_layer path segment
        hf_name = peft_name
        if hf_name.startswith("base_model.model."):
            hf_name = hf_name[len("base_model.model.") :]
        result.append((hf_name + ".weight", merged))

        # Bias (if present in base layer, rare)
        if lora_mod.base_layer.bias is not None:
            skip_names.add(f"{prefix}base_layer.bias")
            result.append((hf_name + ".bias", lora_mod.base_layer.bias))

    # Pass 2: all remaining non-LoRA params
    for name, param in model.named_parameters():
        if name in skip_names:
            continue
        if ".lora_A." in name or ".lora_B." in name or ".lora_embedding" in name:
            continue
        hf_name = name
        if hf_name.startswith("base_model.model."):
            hf_name = hf_name[len("base_model.model.") :]
        hf_name = hf_name.replace(".base_layer.", ".")  # safety strip
        result.append((hf_name, param))

    return result


class ActorPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        actor: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.ema_beta = ema_beta

        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.max_epochs

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        #### Policy loss: route to PolicyLoss or LigerPolicyLoss ####
        self.use_liger_grpo_loss = getattr(self.args, "use_liger_grpo_loss", False)
        if self.use_liger_grpo_loss:
            assert self.args.zero_stage != 3, (
                "--use_liger_grpo_loss requires direct access to lm_head.weight, "
                "which is incompatible with ZeRO-3 parameter sharding. Use ZeRO-2."
            )
            assert self.args.entropy_loss_coef is None, (
                "--use_liger_grpo_loss cannot compute entropy (requires full logits). "
                "Remove --entropy_loss_coef when using Liger fused GRPO loss."
            )
            self.actor_loss_fn = LigerPolicyLoss(
                clip_eps_low=self.args.eps_clip_low_high[0],
                clip_eps_high=self.args.eps_clip_low_high[1],
                beta=0.0,  # KL handled externally in shared post-loss code
                temperature=getattr(self.args, "temperature", 1.0),
                loss_type=getattr(self.args, "liger_loss_type", "grpo"),
                backend=getattr(self.args, "liger_grpo_backend", "chunked"),
                chunk_size=getattr(self.args, "liger_chunk_size", 1),
                enable_vllm_is_correction=self.args.enable_vllm_is_correction,
                vllm_is_truncated_threshold=(
                    self.args.vllm_is_truncated_threshold if self.args.enable_vllm_is_correction else None
                ),
                vllm_is_correction_type=self.args.vllm_is_correction_type,
            )
        else:
            self.actor_loss_fn = PolicyLoss(
                clip_eps_low=self.args.eps_clip_low_high[0],
                clip_eps_high=self.args.eps_clip_low_high[1],
                dual_clip=self.args.dual_clip,
                token_level_loss=getattr(self.args, "token_level_loss", "local_rank"),
                policy_loss_type=self.args.policy_loss_type,
                enable_vllm_is_correction=self.args.enable_vllm_is_correction,
                vllm_is_truncated_threshold=(
                    self.args.vllm_is_truncated_threshold if self.args.enable_vllm_is_correction else None
                ),
                vllm_is_correction_type=self.args.vllm_is_correction_type,
            )
        #### end policy loss init ####

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            getattr(self.args, "use_dynamic_batch", False),
            getattr(self.args, "use_adaptive_batch", False),
            loss_type=getattr(self.args, "loss_type", "ppo"),
        )

        # Init torch group for weights sync
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = False
        if backend == "nccl" and self.args.colocate_all_models and not self.args.async_train:
            self.use_cuda_ipc = True

        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and each of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            group_name = "openrlhf"
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            if use_ray:
                import ray.util.collective as collective

                collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
                self._model_update_group = group_name
            else:
                self._model_update_group = stateless_init_process_group(
                    master_address, master_port, 0, world_size, torch.cuda.current_device()
                )

            ray.get(refs)

        torch_dist_barrier_and_cuda_sync()

    def ppo_train(self, kl_ctl: float):
        # replay buffer may be empty at first, we should rebuild at each training
        if getattr(self.args, "use_adaptive_batch", False):
            self.replay_buffer.setup_adaptive_batch(self.strategy)
        elif getattr(self.args, "use_dynamic_batch", False):
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        # clear cache at the start
        import torch

        torch.cuda.empty_cache()

        not_shuffle = (
            self.strategy.ring_attn_group is not None
            or self.args.ds_tensor_parallel_size > 1
            or self.args.use_dynamic_batch
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=not not_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, kl_ctl, step)
                status["kl"] *= status["response_length"]
                if "logprobs_diff" in status:
                    status["logprobs_diff"] *= status["response_length"]

                # Normalize sparse keys across ranks so all_reduce sees
                # identical key sets. parse_method__* keys are data-dependent
                # (e.g. parse_method__regex only appears when regex fallback
                # fires), so different ranks may have different subsets.
                if torch.distributed.is_initialized():
                    sparse_keys = [k for k in status if k.startswith("parse_method__")]
                    all_sparse = [None] * self.strategy.world_size
                    torch.distributed.all_gather_object(all_sparse, sparse_keys)
                    union_keys = sorted(set(k for rank_keys in all_sparse for k in rank_keys))
                    for k in union_keys:
                        if k not in status:
                            status[k] = 0.0

                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]
                if "logprobs_diff" in status:
                    status["logprobs_diff"] /= status["response_length"]

                short_status = {
                    "act_loss": status["policy_loss"],
                    "reward": status["reward"],
                    "return": status["return"],
                    "gen_len": status["response_length"],
                    "tot_len": status["total_length"],
                    "kl": status["kl"],
                    "act_lr": status["actor_lr"],
                }

                if "entropy_loss" in status:
                    short_status["ent_loss"] = status["entropy_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0].copy()
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] = status_mean.get(k, 0.0) + v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)

        # Inject micro batch partition stats (computed once per setup, not per step).
        if self.replay_buffer.micro_batch_stats:
            status_mean.update(self.replay_buffer.micro_batch_stats)

        return status_mean

    def _assert_finite_actor_state(self, step: int, stage: str, check_grad: bool) -> None:
        model = self.actor.model.module if hasattr(self.actor.model, "module") else self.actor.model
        for name, param in model.named_parameters():
            if not torch.isfinite(param.data).all():
                nonfinite = int((~torch.isfinite(param.data)).sum().item())
                raise RuntimeError(
                    f"Non-finite actor parameter detected at step={step}, stage={stage}, "
                    f"name={name}, shape={tuple(param.shape)}, nonfinite_count={nonfinite}, dtype={param.dtype}"
                )
            if check_grad and param.grad is not None and not torch.isfinite(param.grad).all():
                nonfinite = int((~torch.isfinite(param.grad)).sum().item())
                raise RuntimeError(
                    f"Non-finite actor gradient detected at step={step}, stage={stage}, "
                    f"name={name}, shape={tuple(param.grad.shape)}, nonfinite_count={nonfinite}, dtype={param.grad.dtype}"
                )

    def _log_vram_audit(self, tag: str):
        """Log a detailed VRAM breakdown. Enable with OPENRLHF_VRAM_AUDIT=1."""
        if not getattr(self, "_vram_audit", False):
            return
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / 1024**3
        reserved = torch.cuda.memory_reserved(dev) / 1024**3
        max_alloc = torch.cuda.max_memory_allocated(dev) / 1024**3
        total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
        # Parameter memory breakdown
        model = self.actor.model.module if hasattr(self.actor.model, "module") else self.actor.model
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        grad_mem = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None) / 1024**3
        trainable_mem = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1024**3
        frozen_mem = param_mem - trainable_mem
        logger.info(
            f"[VRAM Audit: {tag}] "
            f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB max_alloc={max_alloc:.2f}GB total={total:.2f}GB | "
            f"params={param_mem:.2f}GB (trainable={trainable_mem:.2f}GB frozen={frozen_mem:.2f}GB) "
            f"grads={grad_mem:.2f}GB | "
            f"other={alloc - param_mem - grad_mem:.2f}GB (activations+buffers+DS)"
        )

    def training_step(self, experience: Experience, kl_ctl: float, step: int) -> Dict[str, float]:
        self.actor.train()
        # Enable VRAM audit on the first step only
        if step == 0 and not hasattr(self, "_vram_audit"):
            self._vram_audit = os.environ.get("OPENRLHF_VRAM_AUDIT", "0") == "1"
            if self._vram_audit:
                torch.cuda.reset_peak_memory_stats()
                self._log_vram_audit("pre_forward")
        nan_guard = os.environ.get("OPENRLHF_DEBUG_NAN_GUARD", "0") == "1"
        if nan_guard:
            self._assert_finite_actor_state(step, stage="pre_forward", check_grad=False)

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        packed_seq_lens = None
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        #### Forward pass + policy loss (divergent) ####
        # Both paths produce: actor_loss, clip_ratio, ppo_kl, vllm_kl, aux_loss
        # Liger path also sets action_log_probs=None, model_output=None
        action_log_probs = None
        model_output = None
        aux_loss = None

        if self.use_liger_grpo_loss:
            # Liger: backbone-only forward → fused lm_head + loss
            hidden_states, aux_loss = self.actor.forward_hidden_states(
                sequences,
                attention_mask=attention_mask,
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
            )
            L = action_mask.shape[1]
            lm_head = self.actor.get_lm_head()
            backend = getattr(self.args, "liger_grpo_backend", "chunked")
            # hidden_states is (B, S, D) — full sequence, no pre-slicing.
            # Triton needs L+1 positions: h[t] → logits[t] → predicts token[t+1].
            #   For L completion tokens we need L+1 hidden states (the one before
            #   the first completion token through the last completion token).
            # Chunked needs L positions: same L hidden states but shifted by one
            #   (h[t-1] for each completion token t), so we slice -(L+1):-1.
            if backend == "triton":
                hs_slice = hidden_states[:, -(L + 1) :, :]   # (B, L+1, D)
            else:
                hs_slice = hidden_states[:, -(L + 1) : -1, :]  # (B, L, D)

            actor_loss, clip_ratio, ppo_kl, vllm_kl = self.actor_loss_fn(
                hidden_states=hs_slice,
                lm_head=lm_head,
                completion_ids=sequences[:, -L:],
                action_mask=action_mask,
                advantages=advantages,
                old_log_probs=old_action_log_probs,
                rollout_log_probs=experience.rollout_log_probs,
            )
        else:
            # Standard: full forward → PolicyLoss on log_probs
            action_log_probs, model_output = self.actor(
                sequences,
                action_mask,
                attention_mask=attention_mask,
                return_output=True,
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
                return_entropy=self.args.entropy_loss_coef is not None,
            )
            actor_loss, clip_ratio, ppo_kl, vllm_kl = self.actor_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                action_mask=experience.action_mask,
                rollout_log_probs=experience.rollout_log_probs,
            )
            aux_loss = getattr(model_output, "aux_loss", None)
        #### end forward pass + policy loss ####

        # Non-finite loss check
        if not torch.isfinite(actor_loss):
            action_tokens = int(experience.action_mask.sum().item())
            diag = (
                f"step={step}, action_tokens={action_tokens}, "
                f"advantages_finite={bool(torch.isfinite(advantages).all())}, "
                f"old_log_probs_finite={bool(torch.isfinite(old_action_log_probs).all())}"
            )
            if action_log_probs is not None:
                diag += (
                    f", new_log_probs_finite={bool(torch.isfinite(action_log_probs).all())}, "
                    f"old_log_probs_excerpt={str(old_action_log_probs)[:100]} ... {str(old_action_log_probs)[-100:]}, "
                    f"new_log_probs_excerpt={str(action_log_probs)[:100]} ... {str(action_log_probs)[-100:]}"
                )
            raise RuntimeError(f"Non-finite actor_loss detected. {diag}")

        #### Shared post-loss: metrics, KL, aux_loss, entropy, distill ####
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()

        loss = actor_loss

        # KL loss (requires action_log_probs — not available in Liger path)
        if self.args.use_kl_loss and action_log_probs is not None:
            if self.args.init_kl_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.kl_estimator,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                kl = torch.zeros_like(action_log_probs)
                logprobs_diff = torch.zeros_like(action_log_probs)
            kl_loss = masked_mean(kl, experience.action_mask)
            logprobs_diff = masked_mean(logprobs_diff, experience.action_mask)
            experience.info["kl"] = kl_loss.detach()
            experience.info["logprobs_diff"] = logprobs_diff.detach()
            loss = loss + kl_loss * kl_ctl

        # Aux loss (MoE load balancing — available from both paths)
        if aux_loss is not None and self.aux_loss:
            loss = loss + aux_loss * self.args.aux_loss_coef

        # Entropy loss (requires full logits — not available in Liger path)
        if model_output is not None and self.args.entropy_loss_coef is not None:
            entropy_loss = masked_mean(
                model_output.entropy[:, -experience.action_mask.shape[1] :], experience.action_mask
            )
            if self.args.entropy_loss_coef != 0:
                loss -= entropy_loss * self.args.entropy_loss_coef

        # Distillation loss (requires action_log_probs — not available in Liger path)
        distill_coef = getattr(self.args, "distill_coef", 0.0)
        if distill_coef > 0 and action_log_probs is not None:
            distill_mask = experience.info.get("distill_mask")
            if distill_mask is not None and distill_mask.any():
                distill_action_mask = distill_mask.unsqueeze(-1) * experience.action_mask
                distill_loss = -masked_mean(action_log_probs, distill_action_mask)
                loss = loss + distill_coef * distill_loss
        #### end shared post-loss ####

        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        if step == 0:
            self._log_vram_audit("post_forward")
        self.strategy.backward(loss, self.actor, self.actor_optim)
        if step == 0:
            self._log_vram_audit("post_backward")
        if nan_guard:
            self._assert_finite_actor_state(step, stage="post_backward", check_grad=True)
        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        else:
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if nan_guard:
            self._assert_finite_actor_state(step, stage="post_optimizer", check_grad=False)

        if self.ema_model:
            if self.args.use_dynamic_batch:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
            else:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        # status
        status = {"policy_loss": actor_loss.detach().item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.args.entropy_loss_coef is not None:
            status["entropy_loss"] = entropy_loss.detach().item()

        # merge logs from info field
        # Skip tool_count__* keys: they are sparse across ranks (different
        # ranks may have different sets) and would cause NCCL deadlock in
        # the downstream strategy.all_reduce() call which iterates over
        # every key.  Tool counts are only consumed during eval, not training.
        # NOTE: parse_method__* keys are also sparse but are normalized
        # across ranks via all_gather_object in ppo_train() before all_reduce.
        # Keys that are internal signals, not metrics to log.
        _SKIP_INFO_KEYS = {"distill_mask"}
        for k in sorted(experience.info.keys()):
            v = experience.info[k]
            if k in _SKIP_INFO_KEYS or k.startswith("tool_count__"):
                continue
            if isinstance(v, list):
                status[k] = torch.tensor(v, dtype=torch.float).mean().item()
            elif isinstance(v, torch.Tensor):
                status[k] = v.float().mean().item()

        # Sanity check: reward/score/clip_ratio should never exceed reasonable bounds.
        _BOUNDED_KEYS = {"reward": 10, "score": 10, "return": 100, "response_clip_ratio": 1.01}
        for k, bound in _BOUNDED_KEYS.items():
            if k in status and abs(status[k]) > bound:
                logger.warning(
                    f"[METRIC SANITY] {k}={status[k]:.4f} exceeds bound {bound}. "
                    f"info type={type(experience.info.get(k))}, "
                    f"info value={experience.info.get(k)}"
                )
        return status

    def broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        
        import time
        build_start_time = time.time()
        broadcast_params = _build_vllm_sync_params(model, self.strategy.args.zero_stage)
        build_time = time.time() - build_start_time
        if getattr(self.strategy.args, "lora_rank", 0) > 0:
            logger.info(f"[Timing] LoRA weight merge and sync param collection took {build_time:.2f}s")
            
        count, num_params = 0, len(broadcast_params)

        def _broadcast_param(param, count, num_params):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = (
                    param.ds_shape
                    if (self.strategy.args.zero_stage == 3 and hasattr(param, "ds_shape"))
                    else param.shape
                )
                _fp4_cfg = getattr(self.strategy.args, "fp4_config", None)
                fp4_format = _fp4_cfg.sync_format if _fp4_cfg is not None else None
                refs = [
                    engine.update_weight.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        empty_cache=count == num_params,
                        fp4_quantize_format=fp4_format,
                    )
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = (
                    param.ds_shape
                    if (self.strategy.args.zero_stage == 3 and hasattr(param, "ds_shape"))
                    else param.shape
                )
                _fp4_cfg = getattr(self.strategy.args, "fp4_config", None)
                fp4_format = _fp4_cfg.sync_format if _fp4_cfg is not None else None
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                        fp4_quantize_format=fp4_format,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        # Initialize layerwise reload on vLLM workers before syncing weights.
        # This prepares the model for deferred per-layer processing.
        _fp4_cfg_init = getattr(self.strategy.args, "fp4_config", None)
        if _fp4_cfg_init is not None and _fp4_cfg_init.sync_format and torch.distributed.get_rank() == 0:
            init_refs = [engine.initialize_weight_reload.remote() for engine in self.vllm_engines]
            ray.get(init_refs)

        for name, param in broadcast_params:
            count += 1  # empty_cache at last param
            # Merged LoRA tensors are plain torch.Tensor (no ds_shape) — skip ZeRO gather
            is_ds_param = hasattr(param, "ds_shape")

            # broadcast
            if not self.use_cuda_ipc:
                # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
                if self.strategy.args.ds_tensor_parallel_size > 1 and is_ds_param:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _broadcast_param(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters(
                        [param], enabled=self.strategy.args.zero_stage == 3 and is_ds_param
                    ):
                        _broadcast_param(param, count, num_params)
            # CUDA IPC
            else:
                if self.strategy.args.ds_tensor_parallel_size > 1 and is_ds_param:
                    with deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True):
                        _handle_cuda_ipc(param, count, num_params)
                else:
                    with deepspeed.zero.GatheredParameters(
                        [param], enabled=self.strategy.args.zero_stage == 3 and is_ds_param
                    ):
                        _handle_cuda_ipc(param, count, num_params)

        if cache_reset_refs:
            ray.get(cache_reset_refs)

        # After all weights are synced, trigger post-load processing
        # (MXFP4 swizzling, kernel prep, etc.) via vLLM's canonical API.
        _fp4_cfg_post = getattr(self.strategy.args, "fp4_config", None)
        if _fp4_cfg_post is not None and _fp4_cfg_post.sync_format and torch.distributed.get_rank() == 0:
            post_sync_refs = [engine.post_weight_sync.remote() for engine in self.vllm_engines]
            ray.get(post_sync_refs)

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        #### Raise torch.compile recompile limit for flex_attention compat ####
        # Must be set HERE (inside the Ray actor process), not in train_ppo_ray.py,
        # because torch._dynamo.config doesn't cross process boundaries.
        # flex_attention with variable sequence lengths (adaptive batching) triggers
        # recompilation for each unique BlockMask shape. When the limit is hit,
        # dynamo falls back to eager, which produces different tensor metadata and
        # breaks gradient checkpointing (CheckpointError).
        _recompile_limit = int(os.environ.get("TORCH_DYNAMO_RECOMPILE_LIMIT", "0"))
        _cache_size_limit = int(os.environ.get("TORCH_DYNAMO_CACHE_SIZE_LIMIT", "0"))
        if _recompile_limit > 0 or _cache_size_limit > 0:
            import torch._dynamo.config as dynamo_config
            if _recompile_limit > 0:
                dynamo_config.recompile_limit = _recompile_limit
                strategy.print(f"[dynamo] recompile_limit set to {_recompile_limit}")
            if _cache_size_limit > 0:
                dynamo_config.cache_size_limit = _cache_size_limit
                strategy.print(f"[dynamo] cache_size_limit set to {_cache_size_limit}")
        #### end raise torch.compile recompile limit ####

        # Skip for vLLM >= 0.16 where NCCL_CUMEM_ENABLE=0 causes ncclCommInitRank to fail
        # with "unhandled cuda error" under NCCL 2.27+.
        if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
            import vllm
            from packaging import version as pkg_version

            if pkg_version.parse(vllm.__version__) < pkg_version.parse("0.16"):
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            param_dtype=strategy.args.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
            mxfp4_dequantize=getattr(strategy.args, "mxfp4_dequantize", False),
            fp4_config=getattr(strategy.args, "fp4_config", None),
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                attn_implementation=strategy.args.attn_implementation,
                param_dtype=strategy.args.param_dtype,  # default: bf16
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # The scheduler is stepped once per optimizer step inside ppo_train().
        # With dynamic batch (or without), each ppo_train() call runs
        # (rollout_batch_size * n_samples_per_prompt / train_batch_size) optimizer steps,
        # but only 1 global_step is logged. So total scheduler steps = max_steps * steps_per_ppo_train.
        steps_per_ppo_train = max(1, args.rollout_batch_size * args.n_samples_per_prompt // args.train_batch_size)
        total_scheduler_steps = max_steps * steps_per_ppo_train

        warmup_multiplier = getattr(args, "warm_steps_multiplier_for_correction", steps_per_ppo_train)
        raw_warmup = getattr(args, "warmup_steps", None)
        if raw_warmup:
            # warmup_steps is in global-step (outer) units; multiply by correction factor for scheduler steps
            num_warmup_steps = int(raw_warmup * warmup_multiplier)
        else:
            num_warmup_steps = math.ceil(total_scheduler_steps * args.lr_warmup_ratio)

        strategy.print(
            f"[Scheduler] lr_scheduler={args.lr_scheduler}, "
            f"outer_max_steps={max_steps}, steps_per_ppo_train={steps_per_ppo_train}, "
            f"total_scheduler_steps={total_scheduler_steps}, "
            f"warm_steps_multiplier={warmup_multiplier}, "
            f"num_warmup_steps={num_warmup_steps} ({raw_warmup or num_warmup_steps // steps_per_ppo_train} global steps)"
        )
        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_scheduler_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states = states

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
        )

    def fit(self, kl_ctl: float = 0):
        """Train actor model with the replay buffer."""
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self.actor(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")

    def broadcast_to_vllm(self):
        self.trainer.broadcast_to_vllm()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        torch_dist_barrier_and_cuda_sync()
