import logging
from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

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

            if load_in_4bit:
                assert param_dtype == "bf16", "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=nf4_config,
                torch_dtype=torch_dtype,  # default: bf16
                device_map=device_map,
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

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

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

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :]
        #### NaN-safe masking — torch.where prevents masked NaN from propagating gradients ####
        action_log_probs = torch.where(action_mask.bool(), action_log_probs, torch.zeros_like(action_log_probs))
        #### end NaN-safe masking ####

        return (action_log_probs, output) if return_output else action_log_probs

    #### forward_hidden_states — returns last hidden state (before lm_head) for fused loss kernels ####
    def forward_hidden_states(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> tuple:
        """Forward pass returning last hidden state (before lm_head) for fused loss kernels.

        Returns:
            (hidden_states, aux_loss): hidden_states shape [B, T, D] (full sequence),
            aux_loss scalar or None.  The caller is responsible for slicing the
            appropriate positions for next-token prediction.
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
        # Must check for PeftModel explicitly.
        causal_lm = self.model
        if hasattr(causal_lm, "module"):
            causal_lm = causal_lm.module
        try:
            from peft import PeftModel

            if isinstance(causal_lm, PeftModel):
                causal_lm = causal_lm.base_model.model
        except ImportError:
            pass
        backbone = causal_lm.model  # e.g., LlamaModel, MistralModel, Qwen3Model

        output = backbone(sequences, attention_mask=forward_attention_mask, position_ids=position_ids)
        last_hidden_state = output.last_hidden_state

        if self.packing_samples:
            last_hidden_state = gather_and_pad_tensor(
                last_hidden_state, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
            )

        # aux_loss (MoE load-balancing loss)
        aux_loss = getattr(output, "aux_loss", None)
        if aux_loss is None:
            router_logits = getattr(output, "router_logits", None)
            if router_logits is not None:
                if not getattr(self, "_logged_aux_loss_recompute", False):
                    logger.info(
                        "[forward_hidden_states] aux_loss not in backbone output; "
                        "recomputing from router_logits (Liger path MoE fix)"
                    )
                    self._logged_aux_loss_recompute = True
                from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

                cfg = backbone.config
                num_experts = cfg.num_local_experts
                top_k = cfg.num_experts_per_tok
                aux_loss = load_balancing_loss_func(router_logits, num_experts, top_k, attention_mask)

        return last_hidden_state, aux_loss

    def get_lm_head(self) -> nn.Linear:
        """Return the lm_head module, handling PEFT wrapping."""
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        try:
            from peft import PeftModel

            if isinstance(model, PeftModel):
                model = model.base_model.model
        except ImportError:
            pass
        return model.lm_head
    #### end forward_hidden_states ####

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
