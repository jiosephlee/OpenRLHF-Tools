import argparse
import json
import os
import torch

# Set Dynamo limits from environment variables if provided
if "TORCH_DYNAMO_RECOMPILE_LIMIT" in os.environ:
    torch._dynamo.config.recompile_limit = int(os.environ["TORCH_DYNAMO_RECOMPILE_LIMIT"])
if "TORCH_DYNAMO_CACHE_SIZE_LIMIT" in os.environ:
    torch._dynamo.config.cache_size_limit = int(os.environ["TORCH_DYNAMO_CACHE_SIZE_LIMIT"])

from datetime import datetime

import ray
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
    RewardModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.ppo_critic import CriticModelActor
from openrlhf.utils import get_strategy
from openrlhf.utils.fp4_config import FP4Config


def _strip_quantization_config(pretrain_path: str) -> None:
    """Remove quantization_config from config.json in-place.

    Dequantized checkpoints have BF16 weights but stale quantization_config
    metadata, which causes vLLM to misroute to _load_weights_mxfp4.
    """
    config_path = os.path.join(pretrain_path, "config.json")
    if not os.path.isfile(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    if "quantization_config" not in config:
        return

    del config["quantization_config"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[mxfp4_dequantize] Stripped quantization_config from {config_path}")


def train(args):
    # initialize ray if not initialized
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "INFO"}})

    # configure strategy
    strategy = get_strategy(args)
    strategy.print(args)

    # init vllm / actor /critic /ref /reward model
    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes
                and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # When using HF dequantization (--mxfp4_dequantize), the checkpoint's
    # config.json still has quantization_config which causes vLLM to route
    # to _load_weights_mxfp4 instead of the normal BF16 loader. Strip it.
    if getattr(args, "mxfp4_dequantize", False) and not getattr(args, "vllm_sync_fp4", None):
        _strip_quantization_config(args.pretrain)

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        if args.colocate_all_models and not args.async_train:
            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
            )

        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.full_determinism,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            pg if args.colocate_all_models and not args.async_train else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
            args.vllm_sleep_level,
            "processed_logprobs" if args.enable_vllm_is_correction else None,
            agent_func_path=args.agent_func_path,
            remote_rm_url=args.remote_rm_url,
            agent_max_steps=args.agent_max_steps,
            vllm_stop_strings=args.vllm_stop_strings,
            chat_protocol=args.chat_protocol,
            tool_version=args.tool_version,
            length_penalty_max_length=args.length_penalty_max_length,
            enable_tool_calling_rewards=args.enable_tool_calling_rewards,
            reduce_cuda_graph=args.optimal_flags_b200_gpt_oss,
            vllm_cudagraph_max_capture_size=args.vllm_cudagraph_max_capture_size,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.vllm_max_num_seqs,
            erl_hard_threshold=args.erl_hard_threshold,
            erl_k=args.erl_k,
            erl_memory=args.erl_memory,
            erl_max_memory=args.erl_max_memory,
            erl_max_reflection_tokens=args.erl_max_reflection_tokens,
            language_model_only=args.language_model_only,
        )

    actor_model = RayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        PolicyModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
    )

    if args.init_kl_coef > 0:
        ref_model = RayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            ReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    else:
        ref_model = None

    if not args.colocate_all_models:
        pg = None

    # if colocated, create placement group for critic and reward model explicitly.
    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    if args.critic_pretrain:
        critic_model = RayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    else:
        critic_model = None

    # multiple reward models
    if not args.remote_rm_url:
        reward_model = RayActorGroup(
            args.reward_num_nodes,
            args.reward_num_gpus_per_node,
            RewardModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ring_attn_size * args.ds_tensor_parallel_size,
        )
    else:
        reward_model = None

    # Select trainer by mode
    if args.async_train:
        from openrlhf.trainer.ppo_trainer_async import PPOTrainerAsync as PPOTrainer
    else:
        from openrlhf.trainer.ppo_trainer import PPOTrainer

    # init PPO trainer (Single controller)
    ppo_trainer = PPOTrainer.remote(
        args.pretrain,
        strategy,
        actor_model,
        critic_model,
        reward_model,
        ref_model,
        vllm_engines,
        # generate kwargs
        do_sample=True,
        prompt_max_len=args.prompt_max_len,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # training update steps
    max_steps = ray.get(ppo_trainer.get_max_steps.remote())

    # init actor/reference/reward model
    refs = []
    actor_pretrain = args.nvfp4_dequantize_base_model if args.nvfp4_dequantize_base_model else args.pretrain

    refs.extend(actor_model.async_init_model_from_pretrained(strategy, actor_pretrain, max_steps, vllm_engines))
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, actor_pretrain))
    if reward_model is not None and args.reward_pretrain:
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, args.reward_pretrain))
    ray.get(refs)

    if critic_model is not None and args.critic_pretrain:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        # TODO: use first reward model as critic model
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    # train actor and critic model
    ray.get(ppo_trainer.fit.remote())

    # save model
    ray.get(actor_model.async_save_model())

    if args.critic_pretrain and args.save_value_network and critic_model is not None:
        ray.get(critic_model.async_save_model())

    # Save training config alongside model
    config_path = os.path.join(args.save_path, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Push to HuggingFace Hub and optionally clean up
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=args.push_to_hub_private, exist_ok=True)
        api.upload_folder(
            folder_path=args.save_path,
            repo_id=args.push_to_hub,
            commit_message="Upload model from OpenRLHF training",
        )
        if args.delete_local_after_push:
            import shutil

            shutil.rmtree(args.save_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )

    # vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--optimal_flags_b200_gpt_oss",
        action="store_true",
        default=False,
        help="Apply optimal B200/GPT-OSS vLLM flags: reduced CUDAGraph capture sizes and async scheduling to lower GPU memory during vLLM init",
    )
    parser.add_argument(
        "--vllm_cudagraph_max_capture_size",
        type=int,
        default=None,
        help="Maximum sequence length to capture in vLLM CUDAGraphs. When set, overrides the default capture size range. "
        "Corresponds to vLLM's compilation_config.cudagraph_capture_sizes upper bound.",
    )
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_sleep_level",
        type=int,
        default=1,
        choices=[1, 2],
        help="vLLM sleep level: 1=offload weights to CPU + discard KV cache, "
        "2=discard both weights and KV cache (recommended for RLHF weight sync)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        type=str,
        default="auto",
        help="KV cache data type for vLLM (auto, fp8, etc.)",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per iteration in vLLM",
    )
    parser.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences in vLLM",
    )
    # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
    parser.add_argument("--enable_vllm_is_correction", action="store_true", default=False)
    parser.add_argument(
        "--vllm_is_truncated_threshold",
        type=float,
        nargs=2,
        default=[0.5, 5.0],
        help="Low and high thresholds for vllm importance sampling truncation",
    )
    parser.add_argument(
        "--vllm_is_correction_type",
        type=str,
        default="tis",
        choices=["tis", "icepop", "seq-mask-tis"],
        help="vLLM IS correction type: tis (token-level clamp), icepop (token-level filter), seq-mask-tis (sequence-level geom mean)",
    )

    # Async training using ray
    parser.add_argument("--async_train", action="store_true", default=False, help="Enable async training")
    parser.add_argument("--async_queue_size", type=int, default=1, help="Queue size for async sampler<->trainer")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--skip_eval_step_zero", action="store_true", default=False)
    parser.add_argument(
        "--skip_training",
        action="store_true",
        default=False,
        help="Run only the initial step-0 evaluation (if eval_dataset is set) and exit. "
        "Useful for benchmarking evaluation speed and efficiency reports without running training.",
    )
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--use_ds_universal_ckpt", action="store_true", help="Use deepspeed universal checkpoint", default=False
    )

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--deepcompile", action="store_true", default=False)
    parser.add_argument(
        "--param_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model data type",
    )
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--ema_beta", type=float, default=0.992, help="EMA beta coefficient")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument(
        "--adam_8bit",
        action="store_true",
        default=False,
        help="Use bitsandbytes 8-bit Adam (keeps optimizer on GPU with ~2x less memory)",
    )
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument(
        "--use_unsloth_moe_kernels",
        action="store_true",
        default=False,
        help="Replace MoE expert loops with grouped GEMM kernels (Triton on A100+, torch._grouped_mm on H100+)",
    )
    parser.add_argument(
        "--use_liger_grpo_loss",
        action="store_true",
        default=False,
        help="Use Liger fused lm_head+GRPO loss to reduce peak memory (requires liger-kernel-nightly>=0.7.0)",
    )
    #### Liger GRPO loss args ####
    parser.add_argument(
        "--liger_grpo_backend",
        type=str,
        default="triton",
        choices=["triton", "chunked"],
        help="Liger GRPO loss backend: 'triton' (default) uses fused Triton kernels "
        "(still materializes logits but saves log-softmax memory by recomputing in backward); "
        "'chunked' fuses lm_head+loss and processes chunk_size sequences at a time "
        "(never materializes full logits tensor).",
    )
    parser.add_argument(
        "--liger_chunk_size",
        type=int,
        default=1,
        help="Chunk size for Liger fused GRPO loss. chunk_size=1 means max chunking (one sequence per chunk, "
        "minimum memory). Higher values process more sequences together (faster but more memory).",
    )
    #### end Liger GRPO loss args ####

    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for deepspeed when using --colocate_all_models",
    )
    parser.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed tensor parallel size")

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # dynamic batch size
    parser.add_argument("--use_dynamic_batch", action="store_true", default=False)
    parser.add_argument("--use_adaptive_batch", action="store_true", default=False, help="Use padded adaptive batching instead of packed dynamic batching")
    parser.add_argument("--rollout_max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train_max_tokens_per_gpu", type=int, default=16192)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--mxfp4_dequantize",
        action="store_true",
        default=False,
        help="Use Mxfp4Config(dequantize=True) for MXFP4-packed GPT-OSS checkpoints. Forces eager attention.",
    )
    parser.add_argument(
        "--nvfp4_dequantize_base_model",
        type=str,
        default=None,
        help="Path to the BF16 base model for NVFP4 loading. OpenRLHF will load this unquantized BF16 model while vLLM handles the packed NVFP4 model natively.",
    )
    parser.add_argument(
        "--vllm_sync_fp4",
        type=str,
        default=None,
        choices=["mxfp4", "nvfp4"],
        help=(
            "Quantize bf16 actor weights to FP4 on the fly during vLLM weight sync. "
            "'mxfp4': OCP MXFP4 (block=32, E8M0 scales). "
            "'nvfp4': NVIDIA NVFP4 (block=16, E4M3 scales, per-tensor global scale)."
        ),
    )
    parser.add_argument(
        "--qat",
        type=str,
        default=None,
        choices=["fp4_fake_quantize"],
        help=(
            "Enable Quantization-Aware Training (QAT) during actor forward passes. "
            "'fp4_fake_quantize': fake-quantize expert weights (bf16 -> nearest FP4 -> bf16) via STE; "
            "the concrete FP4 format (mxfp4/nvfp4) is derived from --vllm_sync_fp4. "
            "Default: None (no QAT)."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024, help="Batch size for make experience")
    parser.add_argument(
        "--vllm_generate_batch_size", type=int, default=None, help="Batch size for vLLM generating samples"
    )
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None, help="PPO-clip low and high")
    parser.add_argument("--dual_clip", type=float, default=None, help="Dual-clip PPO")
    parser.add_argument("--value_clip", type=float, default=0.5, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HF Hub repo ID to push model after training (e.g. 'username/my-model')",
    )
    parser.add_argument(
        "--push_to_hub_private", action="store_true", default=False, help="Make the HF Hub repo private"
    )
    parser.add_argument(
        "--delete_local_after_push",
        action="store_true",
        default=False,
        help="Delete local save_path after successful push to Hub",
    )
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--kl_horizon", type=int, default=10000)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    #### Unified loss_type: controls ratio, reduction, and Liger variant ####
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ppo",
        choices=["ppo", "dapo", "bnpo", "dr_grpo", "gspo", "cispo", "sapo"],
        help=(
            "Unified loss type controlling ratio computation, reduction strategy, and Liger variant. "
            "'ppo' (default): token-level PPO ratio, per-sequence mean with cross-rank seq-count sync. "
            "'dapo': token-level PPO ratio, flat token mean with cross-rank all-reduce. "
            "'bnpo': token-level PPO ratio, flat token mean within rank. "
            "'dr_grpo': token-level PPO ratio, per-sequence mean with cross-rank seq-count sync. "
            "'gspo': sequence-level IS ratio, per-sequence mean with cross-rank seq-count sync. "
            "'cispo'/'sapo': Liger-only variants (require --use_liger_grpo_loss)."
        ),
    )
    #### end unified loss_type ####
    parser.add_argument(
        "--kl_estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--entropy_loss_coef",
        type=float,
        default=None,
        help="Entropy loss coef, set to 0 means only enable entropy logs",
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce/GRPO, etc
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo",
    )
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")
    parser.add_argument(
        "--no_advantage_std_norm",
        action="store_true",
        default=False,
        help="disable dividing by std for advantages while keeping mean normalization",
    )
    parser.add_argument(
        "--overlong_buffer_len", type=float, default=None, help="reward with optional overlong penalty"
    )
    parser.add_argument("--overlong_penalty_factor", type=float, default=1, help="overlong penalty factor")
    parser.add_argument(
        "--stop_properly_penalty_coef",
        type=float,
        default=None,
        help="Penalty coefficient [0,1] for truncated samples (finish_reason='length'). "
        "Truncated sample rewards are scaled by this coefficient to encourage proper stopping.",
    )

    # Context Parallel
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)
    parser.add_argument("--agent_func_path", type=str, default=None, help="Agent script path")
    parser.add_argument("--agent_max_steps", type=int, default=5, help="Maximum number of agent turns per episode")
    parser.add_argument(
        "--length_penalty_max_length",
        type=int,
        default=0,
        help="Upper limit of length penalty; 0 disables it",
    )
    parser.add_argument(
        "--enable_tool_calling_rewards",
        action="store_true",
        default=False,
        help="Enable format/parse shaping rewards for tool-calling turns. "
        "When off (default), all tool-calling rewards (format_reward, parse_failed penalty) are zeroed.",
    )
    parser.add_argument(
        "--vllm_stop_strings",
        type=str,
        nargs="+",
        default=None,
        help="Stop strings for vLLM generation (e.g., '</tool_call>')",
    )
    parser.add_argument("--is_vlm", action="store_true", default=False, help="Enable Vision-Language Model support")
    parser.add_argument(
        "--chat_protocol",
        type=str,
        default="glm_flash",
        choices=["glm_flash", "intern_s1", "gpt_oss", "qwen3", "qwen3_5"],
        help="Chat protocol for tool-calling format.",
    )
    parser.add_argument(
        "--language_model_only",
        action="store_true",
        default=False,
        help="Skip loading vision encoder in VLMs (e.g. Qwen3.5) to save memory for text-only tasks.",
    )

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--eval_temperature", type=float, default=0.6, help="Temperature for evaluation")
    parser.add_argument(
        "--eval_n_samples_per_prompt", type=int, default=4, help="Number of samples per prompt for evaluation"
    )

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument(
        "--tdc_tools",
        type=str,
        default=None,
        help="Path to JSON mapping {task_name: [tool_schemas]} for per-task tool injection into apply_chat_template",
    )
    parser.add_argument(
        "--tool_version",
        type=str,
        choices=["v1", "v2", "v3", "v4"],
        default=None,
        help="Tool version for training (v1: RDKit+AccFG, v2: +salts, v3: +pKa/logD/ePSA, v4: +Haydn)",
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # Dynamic filtering
    parser.add_argument("--dynamic_filtering", action="store_true", default=False, help="Enable dynamic filtering")
    parser.add_argument(
        "--dynamic_filtering_reward_range", nargs=2, default=(0, 1), type=float, help="Dynamic filtering rewards range"
    )
    # Smart replay (selective prompt repetition after primary pass)
    parser.add_argument(
        "--smart_replay",
        action="store_true",
        default=False,
        help="After each episode, replay filtered prompts the model can still learn from",
    )
    #### Prompt-level oversampling with early termination ####
    parser.add_argument(
        "--oversample_ratio",
        type=float,
        default=1.0,
        help="Dispatch ceil(batch_size * ratio) prompts; early-terminate once batch_size accepted. "
        "Cancelled prompts recycled via LeftOverPrompts phase. Default 1.0 (no oversampling).",
    )
    #### end oversampling ####
    parser.add_argument(
        "--constant_lr_with_warm_up",
        action="store_true",
        default=False,
        help="Force a constant LR with linear warmup (see --warmup_steps)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=20,
        help="Number of global warmup steps for constant_lr_with_warm_up (default: 20)",
    )
    parser.add_argument(
        "--warm_steps_multiplier_for_correction",
        type=float,
        default=None,
        help="Multiplier applied to warmup steps for scheduler correction. "
        "Default: rollout_batch_size * n_samples_per_prompt / train_batch_size",
    )
    parser.add_argument("--max_replay_rounds", type=int, default=2, help="Max replay rounds per episode (default: 2)")
    parser.add_argument(
        "--curriculum_balanced",
        action="store_true",
        default=False,
        help="Evenly interleave samples from each dataset across training",
    )
    # ERL (Experiential Reinforcement Learning)
    parser.add_argument(
        "--erl_hard_threshold",
        type=float,
        default=None,
        help="Avg reward threshold for hard prompt gating. None=disabled. 0.2 recommended for TDC.",
    )
    parser.add_argument(
        "--erl_k", type=int, default=4, help="Number of diverse reflection+retry pairs per hard prompt"
    )
    parser.add_argument(
        "--erl_memory",
        action="store_true",
        default=False,
        help="Enable cross-episode reflection memory (off by default)",
    )
    parser.add_argument("--erl_max_memory", type=int, default=5, help="Max reflections per task in memory")
    parser.add_argument(
        "--erl_max_reflection_tokens", type=int, default=512, help="Max tokens for reflection generation"
    )
    # Distillation loss (generic — any executor can tag experiences for SFT)
    parser.add_argument(
        "--distill_coef",
        type=float,
        default=0.0,
        help="Distillation loss coefficient for experiences tagged with distill_mask (0=disabled)",
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    # Build and validate FP4 configuration
    args.fp4_config = FP4Config.from_args(args)
    args.fp4_config.validate()

    # Validate arguments
    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)

    if args.agent_func_path:
        args.remote_rm_url = "agent"

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.ring_attn_size > 1:
        if not args.packing_samples:
            print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
            args.packing_samples = True

    #### Derive internal flags from --loss_type ####
    # token_level_loss: controls reduction in PolicyLoss
    if args.loss_type in ("ppo", "gspo", "dr_grpo", "sapo"):
        args.token_level_loss = None  # per-sequence mean
    elif args.loss_type in ("dapo", "cispo"):
        args.token_level_loss = "global"  # flat token mean with cross-rank all-reduce
    elif args.loss_type == "bnpo":
        args.token_level_loss = "local_rank"  # flat token mean within rank
    else:
        args.token_level_loss = None

    # policy_loss_type: controls ratio computation in PolicyLoss
    args.policy_loss_type = "gspo" if args.loss_type == "gspo" else "ppo"

    # liger_loss_type: maps to Liger's internal loss_type enum
    LIGER_LOSS_TYPE_MAP = {"ppo": "grpo", "gspo": "grpo"}
    args.liger_loss_type = LIGER_LOSS_TYPE_MAP.get(args.loss_type, args.loss_type)

    # Validate Liger-only variants
    if args.loss_type in ("cispo", "sapo") and not getattr(args, "use_liger_grpo_loss", False):
        raise ValueError(f"--loss_type {args.loss_type} requires --use_liger_grpo_loss")
    #### end derive from loss_type ####

    if args.use_adaptive_batch:
        args.use_dynamic_batch = True
        if args.rollout_max_tokens_per_gpu is None:
            print("[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu.")
            args.rollout_max_tokens_per_gpu = args.train_max_tokens_per_gpu

    if args.use_dynamic_batch:
        if not args.packing_samples and not args.use_adaptive_batch:
            print("[Warning] Please --packing_samples to accelerate when --use_dynamic_batch is enabled.")
            args.packing_samples = True
        if args.rollout_max_tokens_per_gpu is None:
            print("[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu.")
            args.rollout_max_tokens_per_gpu = args.train_max_tokens_per_gpu

    if args.packing_samples:
        if "flash_attention" not in args.attn_implementation:
            print(
                "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.attn_implementation = "flash_attention_2"
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.colocate_all_models and args.async_train:
        print("[Warning] Using --colocate_all_models in async RLHF only colocates DeepSpeed models.")

    if args.async_train:
        assert not args.vllm_enable_sleep, "Async RLHF is not supported with --vllm_enable_sleep."

    if args.eval_dataset:
        assert args.remote_rm_url, "`--eval_dataset` is only supported with `--remote_rm_url`."

    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.kl_estimator not in ["k1"]:
            print(f"Recommend setting {args.kl_estimator} to 'k1' when not using KL as a loss.")

    # Set vLLM generate_batch_size to rollout_batch_size if not specified
    if not args.vllm_generate_batch_size:
        args.vllm_generate_batch_size = args.rollout_batch_size

    #### Oversample ratio validation ####
    assert args.oversample_ratio >= 1.0, f"--oversample_ratio must be >= 1.0, got {args.oversample_ratio}"
    if args.oversample_ratio > 1.0:
        assert args.dynamic_filtering, "--oversample_ratio > 1.0 requires --dynamic_filtering"
    #### end oversample ratio validation ####

    if args.dynamic_filtering:
        assert args.dynamic_filtering_reward_range[0] < args.dynamic_filtering_reward_range[1], (
            "reward_clip_range[0] must be less than reward_clip_range[1]"
        )
        assert args.remote_rm_url or args.agent_func_path, (
            "remote_rm_url or agent_func_path must be specified when using dynamic filtering"
        )
        assert args.n_samples_per_prompt > 1, (
            "n_samples_per_prompt must be greater than 1 when using dynamic filtering"
        )

    if args.smart_replay:
        assert args.dynamic_filtering, "--smart_replay requires --dynamic_filtering"
        assert args.constant_lr_with_warm_up, (
            "--smart_replay requires --constant_lr_with_warm_up because smart replay has variable step counts "
            "and needs a constant LR schedule (with warmup) to remain stable"
        )

    if args.constant_lr_with_warm_up:
        print(
            f"[SmartReplay/ConstantLR] Overriding LR scheduler to constant_with_warmup (warmup={args.warmup_steps} steps)"
        )
        args.lr_scheduler = "constant_with_warmup"
        args.lr_warmup_ratio = 0.0

    # Default warm_steps_multiplier_for_correction = rollout_batch_size * n_samples_per_prompt / train_batch_size
    if args.warm_steps_multiplier_for_correction is None:
        args.warm_steps_multiplier_for_correction = (
            args.rollout_batch_size * args.n_samples_per_prompt / args.train_batch_size
        )

    assert (
        args.n_samples_per_prompt * args.rollout_batch_size // args.micro_rollout_batch_size
        >= args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    ), "The number of sample batches must be greater than or equal to the effective number of actor processes."

    if args.use_liger_grpo_loss:
        if args.entropy_loss_coef is not None:
            raise ValueError(
                "--use_liger_grpo_loss and --entropy_loss_coef are incompatible. "
                "Liger fused loss cannot compute entropy (requires full logits)."
            )
        if args.zero_stage == 3:
            raise ValueError(
                "--use_liger_grpo_loss requires direct access to lm_head.weight, "
                "which is incompatible with ZeRO-3. Use --zero_stage 2."
            )

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)
