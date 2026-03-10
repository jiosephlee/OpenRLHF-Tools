import logging
import os
from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

try:
    from transformers import Mxfp4Config
except ImportError:
    Mxfp4Config = None
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from openrlhf.utils.fp4_config import FP4Config
from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits

logger = logging.getLogger(__name__)


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        param_dtype (str, optional): Model data type ("bf16", "fp16"). Defaults to "bf16".
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation="flash_attention_2",
        param_dtype="bf16",
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        mxfp4_dequantize=False,
        fp4_config: Optional["FP4Config"] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            # Support multiple attention mechanism implementations
            attn_impl = attn_implementation

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            # Determine torch dtype based on param_dtype parameter, default: bf16
            from openrlhf.utils.utils import convert_to_torch_dtype

            torch_dtype = convert_to_torch_dtype(param_dtype)

            if mxfp4_dequantize:
                assert Mxfp4Config is not None, (
                    "Mxfp4Config requires transformers >= 4.52. Please upgrade: pip install -U transformers"
                )
                quant_config = Mxfp4Config(dequantize=True)
                # Allow flash_attention_2 instead of forcing eager to avoid OOM
                # attn_impl = "eager"
                logger.info(f"Using Mxfp4Config(dequantize=True) with {attn_impl} for GPT-OSS model")
            elif load_in_4bit:
                assert param_dtype == "bf16", "we only support bnb_4bit_compute_dtype = bf16"
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                quant_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=quant_config,
                torch_dtype=torch_dtype,  # default: bf16
                device_map=device_map,
            )

            # Verify dequantization worked — log param dtypes and requires_grad
            if mxfp4_dequantize:
                dtypes = {}
                non_trainable = []
                for name, p in self.model.named_parameters():
                    dt = str(p.dtype)
                    dtypes[dt] = dtypes.get(dt, 0) + 1
                    if not p.requires_grad:
                        non_trainable.append(name)
                logger.info(f"[mxfp4_dequantize] Parameter dtype distribution: {dtypes}")
                if non_trainable:
                    logger.warning(
                        f"[mxfp4_dequantize] {len(non_trainable)} params have requires_grad=False! "
                        f"First 5: {non_trainable[:5]}"
                    )
                else:
                    logger.info("[mxfp4_dequantize] All parameters are trainable (requires_grad=True)")
                # Check if model still thinks it's quantized
                if hasattr(self.model, "is_quantized"):
                    logger.info(f"[mxfp4_dequantize] model.is_quantized = {self.model.is_quantized}")
                if hasattr(self.model.config, "quantization_config"):
                    logger.info(
                        f"[mxfp4_dequantize] model.config.quantization_config = {self.model.config.quantization_config}"
                    )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # QAT: fake-quantize MoE expert weights during training forward passes (STE)
            if fp4_config is not None and fp4_config.qat_enabled:
                if fp4_config.sync_format == "mxfp4":
                    assert fp4_config.dequantize_base, (
                        "MXFP4 QAT requires --mxfp4_dequantize. The model must be loaded "
                        "with Mxfp4Config(dequantize=True) so expert weights are in bf16."
                    )
                    from openrlhf.utils.mxfp4_quantize import register_mxfp4_qat_parametrization

                    n = register_mxfp4_qat_parametrization(self.model)
                    logger.info(f"[QAT fp4_fake_quantize/mxfp4] Applied to {n} expert weight layers in actor.")
                elif fp4_config.sync_format == "nvfp4":
                    from openrlhf.utils.nvfp4_quantize import register_nvfp4_qat_parametrization

                    n = register_nvfp4_qat_parametrization(self.model)
                    logger.info(f"[QAT fp4_fake_quantize/nvfp4] Applied to {n} expert weight layers in actor.")
                else:
                    raise ValueError(
                        f"--qat fp4_fake_quantize requires --vllm_sync_fp4 (mxfp4 or nvfp4), got '{fp4_config.sync_format}'"
                    )

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

                # set_z3_leaf_modules is required for MoE models
                for m in self.model.modules():
                    # https://github.com/microsoft/DeepSpeed/pull/4966
                    if "SparseMoeBlock" in m.__class__.__name__:
                        deepspeed.utils.set_z3_leaf_modules(self.model, [m.__class__])
                        print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                        break

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)
        logits = output["logits"]
        debug_logits = os.environ.get("OPENRLHF_DEBUG_LOGITS", "0") == "1"
        # Use scalar reductions only — boolean indexing / str(logits) would copy the full tensor (~18 GiB)
        logits_all_finite = bool(torch.isfinite(logits).all().item())
        if debug_logits or not logits_all_finite:
            nonfinite_count = 0 if logits_all_finite else int((~torch.isfinite(logits)).sum().item())
            logger.warning(
                f"[DEBUG logits] shape={tuple(logits.shape)}, dtype={logits.dtype}, finite={logits_all_finite}, "
                f"nonfinite_count={nonfinite_count}, min={logits.min().item():.4f}, max={logits.max().item():.4f}"
            )
        if not logits_all_finite:
            bad_idx = (~torch.isfinite(logits)).nonzero(as_tuple=False)[0]
            b_idx, s_idx, v_idx = bad_idx.tolist()
            token_id = int(sequences[b_idx, s_idx].item())
            label_id = int(rolled_sequences[b_idx, s_idx].item())
            if attention_mask is not None:
                attn_val = int(attention_mask[b_idx, s_idx].item())
            else:
                attn_val = -1
            if action_mask is not None:
                action_slice = action_mask[b_idx, -min(16, action_mask.shape[1]) :].int().tolist()
            else:
                action_slice = []
            logger.warning(
                f"[DEBUG logits nonfinite] b={b_idx}, s={s_idx}, v={v_idx}, input_token={token_id}, label_token={label_id}, "
                f"attention_mask_val={attn_val}, action_mask_tail={action_slice}"
            )

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)
        log_probs_finite = torch.isfinite(log_probs)
        if debug_logits or (not bool(log_probs_finite.all().item())):
            finite_log_probs = log_probs[log_probs_finite]
            finite_lp_min = finite_log_probs.min().item() if finite_log_probs.numel() > 0 else float("nan")
            finite_lp_max = finite_log_probs.max().item() if finite_log_probs.numel() > 0 else float("nan")
            logger.warning(
                f"[DEBUG log_probs] shape={tuple(log_probs.shape)}, finite={bool(log_probs_finite.all().item())}, "
                f"nonfinite_count={(~log_probs_finite).sum().item()}, finite_min={finite_lp_min:.4f}, finite_max={finite_lp_max:.4f}"
            )

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :]
        action_log_probs = torch.where(action_mask.bool(), action_log_probs, torch.zeros_like(action_log_probs))

        return (action_log_probs, output) if return_output else action_log_probs

    def forward_hidden_states(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> tuple:
        """Forward pass returning last hidden state (before lm_head) for fused loss kernels.

        Returns:
            (hidden_states, aux_loss): hidden_states shape [B, T-1, D], aux_loss scalar or None.
        """
        batch, seqlen = sequences.size()
        forward_attention_mask = attention_mask
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            forward_attention_mask = None
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        # Get the transformer backbone (before lm_head).
        # NOTE: cannot use hasattr(model, "base_model") — PreTrainedModel defines
        # a base_model property (returns self.model), so it's True for ALL HF models.
        # Must check for PeftModel explicitly (same approach as TRL's is_peft_model).
        causal_lm = self.model
        try:
            from peft import PeftModel

            if isinstance(causal_lm, PeftModel):
                causal_lm = causal_lm.base_model.model  # PeftModel → LoraModel → CausalLM
        except ImportError:
            pass
        backbone = causal_lm.model  # e.g., LlamaModel, MistralModel, Qwen3Model

        output = backbone(sequences, attention_mask=forward_attention_mask, position_ids=position_ids)
        last_hidden_state = output.last_hidden_state

        if self.packing_samples:
            last_hidden_state = gather_and_pad_tensor(
                last_hidden_state, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
            )

        # Slice off last token (next-token prediction: predict token t+1 from hidden state t)
        last_hidden_state = last_hidden_state[:, :-1, :]

        # aux_loss (MoE load-balancing loss) — available when output_router_logits=True
        aux_loss = getattr(output, "aux_loss", None)
        if aux_loss is None:
            # Backbone forward (used by Liger path) doesn't compute aux_loss,
            # but it does return router_logits. Recompute aux_loss manually —
            # same logic the CausalLM wrapper uses internally.
            router_logits = getattr(output, "router_logits", None)
            if router_logits is not None:
                logger.info(
                    "[forward_hidden_states] aux_loss not in backbone output; "
                    "recomputing from router_logits (Liger path MoE fix)"
                )
                from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

                #### Use causal_lm.config (not self.model.config which may be DS dict) ####
                num_experts = causal_lm.config.num_local_experts
                top_k = causal_lm.config.num_experts_per_tok
                #### end config ####
                aux_loss = load_balancing_loss_func(router_logits, num_experts, top_k, attention_mask)

        return last_hidden_state, aux_loss

    def get_lm_head(self) -> nn.Linear:
        """Return the lm_head module, handling PEFT wrapping."""
        model = self.model
        try:
            from peft import PeftModel

            if isinstance(model, PeftModel):
                model = model.base_model.model  # PeftModel → LoraModel → CausalLM
        except ImportError:
            pass
        return model.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
