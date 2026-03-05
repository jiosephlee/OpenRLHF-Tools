# OpenRLHF-Tools: Extended OpenRLHF Fork

## Overview

This fork extends OpenRLHF with multi-turn tool-calling support for GRPO training, along with several infrastructure improvements: transformers v5 compatibility, multi-stage GPU dispatch, DeepSpeed OOM fixes, eval improvements, and TDC (Therapeutics Data Commons) dataset integration.

**Key Features:**
- Multi-turn agent-based rollouts with tool execution (including parallel tool calls)
- Token-level masking (only LLM actions contribute to loss, observations excluded)
- Clean abstraction layer (ToolCallingTurn + ChatProtocol)
- Multiple chat protocol support: GLM Flash XML, Intern-S1, Qwen3, GPT-OSS
- Transformers v4/v5 backward compatibility
- 3-stage deferred GPU dispatch for better load balancing
- AutoTP OOM fix (free pre-sharded weights before DeepSpeed init)
- NaN-safe masked operations (`torch.where` instead of `tensor * mask` in action log probs)
- Eval at step 0, macro-F1 for TDC, `eval/global_step` W&B axis
- Checkpoint uploading to HF Hub
- Rollout trace logging to `runs/<run_name>/traces/`
- On-the-fly MXFP4 quantization for bf16→MXFP4 weight sync to vLLM
- Unified bash scripting system and 1a1v lightweight distributed training

## Major Changes from Upstream OpenRLHF

### 1. Transformers v5 Compatibility
**Files:** `openrlhf/cli/batch_inference.py`, `openrlhf/cli/interactive_chat.py`

Detects transformers major version at import time and branches on `batch_decode` (v4) vs `decode` (v5). `requirements.txt` allows either version.

### 2. Deferred Dispatch (75/25)
**File:** `openrlhf/trainer/ppo_utils/experience_maker.py`

Enabled via `--deferred_dispatch`. Upstream dispatches all prompts at once; with this flag we split into 2 stages (75/25):
- Stage 1 (75%): dispatched immediately via heap-balanced `_dispatch_prompts_to_vllm`
- Stage 2 (25%): held as reserve, dispatched when any engine's pending count drops to ≤4

Improves GPU utilization when generation times vary (common with multi-turn tool calling). Uses heap-based balancer with per-engine pending counts.

### 3. DeepSpeed AutoTP OOM Fix
**File:** `openrlhf/utils/deepspeed/deepspeed.py`

After `tp_model_init()` shards the model, the old optimizer still holds references to full-size pre-sharded parameters (~80 GiB). Fix: explicitly delete old optimizer, break scheduler reference, recreate optimizer over sharded params, force `gc.collect()` + `torch.cuda.empty_cache()` before `deepspeed.initialize()`.

### 4. NaN-Safe Masked Operations
**File:** `openrlhf/models/actor.py`

Changed `(tensor * mask).sum()` to `torch.where(mask.bool(), tensor, torch.zeros_like(tensor)).sum()` in `action_log_probs`. Prevents NaN propagation through masked positions during action log prob calculation. (Note: `masked_mean` was reverted to `tensor * mask`).

### 5. Eval Improvements
**Files:** `openrlhf/trainer/ppo_trainer.py`, `openrlhf/trainer/ppo_trainer_async.py`, `openrlhf/utils/logging_utils.py`

- Moved `evaluate()` from `PPOTrainer` to `BasePPOTrainer` (shared by sync/async)
- Added eval at step 0 (`--skip_eval_step_zero` to disable)
- Changed W&B metric axis from `eval/epoch` to `eval/global_step`
- Added macro-F1 computation for TDC binary classification tasks

### 6. TDC Dataset Integration
**Files:** `openrlhf/datasets/tdc_loader.py`, `openrlhf/utils/tdc_reward_model.py`, `openrlhf/datasets/prompts_dataset.py`

- `TDCDatasetLoader`: converts TDC CSVs to OpenAI message format with fuzzy prompt matching, Tox21 multi-subtask support
- Per-task tool schema injection via `--tdc_tools` pointing to `tools_per_task.json`
- Binary answer extraction (A/B) for eval with macro-F1

### 7. Checkpoint Uploading
**Files:** `openrlhf/cli/train_ppo_ray.py`, `openrlhf/trainer/ppo_trainer.py`

New CLI args: `--push_to_hub`, `--push_to_hub_private`, `--delete_local_after_push`, `--save_steps`

### 8. Rollout Trace Logging
**File:** `openrlhf/trainer/ppo_utils/experience_maker.py`

Saves one decoded rollout trace per step to `runs/<run_name>/traces/`. Annotates each record with prompt/action/observation sections decoded from token IDs using action ranges.

### 9. Memory Optimization & ZeRO-2 Fixes
- **Liger Kernels**: Experimental support to reduce vRAM OOM issues. We also implemented Liger GRPO loss (currently experimental/WIP) along with Liger PEFT detection fixes.
- **DeepSpeed ZeRO-2 & Optimizers**: Testing `adam_offload` vs `8bit_adam`:
  - `adam_offload` avoids CUDA OOM but faces unpredictable SIGKILLs (usually mitigated if sufficient system RAM/GPUs are available).
  - `8bit_adam` is technically faster but requires smaller batch sizes and has potential compatibility issues on B200s.

### 10. Unified Bash Scripts & 1a1v Setup
- Refactored shell scripts into a simplified unified bash staging system. Recently heavily revamped to sync `unsloth`, `intern-s1`, and `gpt-oss` scripts. Added TIS and GSPO feature flags.
- Fixed multiple Ray cluster setup bugs in both interactive and bash scripts.
- Added `1a1v` distributed training setup (1 actor + 1 vLLM) for a 2-GPU footprint, improving rapid iteration and debugging compared to full 1a3v multi-node sweeps.

### 11. Parallel Tool Calls & GPT-OSS
- Added support for executing parallel tool calls natively.
- Full support and bug fixes for OpenAI/GPT-OSS schema parsing and generation formatting.

### 12. Colocate Mode CUDA Cache Clearing
**Files:** `openrlhf/trainer/ray/vllm_engine.py`

In colocate mode, vLLM's `sleep()` releases weights but doesn't return the memory to the CUDA driver. The Actor process then OOMs during backward passes because `torch.cuda.memory.caching_allocator` still holds the pages. Fix: call `torch.cuda.empty_cache()` in both `sleep()` and `gc_collect()` so freed GPU memory is actually returned to the driver and available to the Actor.

### 14. On-the-Fly FP4 Quantization for Weight Sync
**Files:** `openrlhf/trainer/ray/vllm_worker_wrap.py`, `openrlhf/utils/mxfp4_quantize.py`

When the Actor trains in bf16 but vLLM serves with FP4-quantized weights (e.g. GPT-OSS MoE), the weight sync must quantize on the fly. `quantize_to_mxfp4()` reimplements NVIDIA ModelOpt's E8M0-scaled FP4 E2M1 packing (transpose, per-expert block scaling, uint8 nibble packing). After writing packed weights + scales into vLLM's parameter storage, `reprocess_mxfp4_weights()` calls `process_weights_after_loading()` on dirty layers to re-run FlashInfer's swizzle/interleave pass. Recently, we fixed MXFP4 sync bugs and added support for NVFP4 formats (WIP on the vLLM end).

### 15. FP4 Quantization-Aware Training (QAT)
**Files:** `openrlhf/utils/mxfp4_quantize.py`, `openrlhf/models/actor.py`, `openrlhf/trainer/ray/ppo_actor.py`, `openrlhf/cli/train_ppo_ray.py`

Closes the train/inference distribution gap when vLLM serves with FP4-quantized MoE expert weights but the actor trains in bf16. Uses a batched Triton kernel prefetch: before each forward pass, all expert weights are concatenated and fake-quantized in a single kernel launch on the default stream, eliminating the previous multi-stream approach and its race conditions.

- Only MoE expert projections are targeted (same name filter as `vllm_worker_wrap`: `"experts"` in path AND one of `gate_up_proj`/`down_proj`/`w13_weight`/`w2_weight`).
- Weight sync unaffected: `named_parameters()` yields true bf16.
- Enabled via `--qat fp4_fake_quantize` (along with `--mxfp4_dequantize_base_model` / `--nvfp4_dequantize_base_model`).

### 16. Ceiling Fix for Dynamic Batch Splitting
**File:** `openrlhf/trainer/ppo_utils/experience_maker.py`

`minimum_batch_num` was rounded down with floor division (`//`), which could produce 0 microbatches when `minimum_batch_num < effective_actor_num`, causing packed sequences to accidentally exceed `rollout_max_tokens_per_gpu`. Fix: use `math.ceil()` to round up, ensuring at least one microbatch per actor and respecting the token budget.

### 17. Asymmetric PPO Clipping (Clip-Higher)

Uses asymmetric `--eps_clip_low_high 0.3 0.372`, for instance, to give exploration tokens more room to increase probability per update (inspired by DAPO's Clip-Higher). GPT-OSS runs show ~24% clip ratio vs ~0.6% for smaller baselines, so wider bounds help avoid suppressing the gradient signal. Lower clip (ε=0.3) limits how aggressively bad actions are suppressed; upper clip (ε=0.372) limits reinforcement of good actions, with the asymmetry favoring exploration. Usual values are 0.2 and 0.272.

### 18. ERL: Experiential Reinforcement Learning (EXPERIMENTAL — not yet tested)
**Files:** `openrlhf/utils/erl_executor.py`, `openrlhf/utils/erl_tdc_agent.py`, `openrlhf/utils/chat_protocol.py`, `openrlhf/trainer/ppo_utils/experience_maker.py`, `openrlhf/trainer/ray/ppo_actor.py`

Based on [Experiential Reinforcement Learning](https://arxiv.org/abs/2602.13949) (Shi et al., Feb 2026). For hard prompts where all samples fail (avg reward < threshold), generates structured reflections from failed attempts and retries with reflection-augmented prompts. Key components:

- **`ERLExecutor`** (`openrlhf/utils/erl_executor.py`): Wraps any `AgentExecutorBase` with ERL reflection+retry logic. Loaded via `--agent_func_path` pointing to an agent .py that exports it as `AgentExecutor` (see `erl_tdc_agent.py`). Implements `execute_batch()` which `LLMRayActor.generate_responses()` calls when available, returning variable-size result lists. Config read from env vars (`OPENRLHF_ERL_HARD_THRESHOLD`, `OPENRLHF_ERL_K`, etc.).
- **`erl_tdc_agent.py`**: Agent file for TDC tasks — exports `ERLExecutor(ToolCallingTurn executor)` as `AgentExecutor`. Point `--agent_func_path` here (or set `AGENT_FUNC_PATH` env var before running the training script).
- **Prompt-level gating**: Only hard prompts (avg r1 < threshold) trigger reflection+retry; easy prompts use standard path
- **k diverse reflections**: k reflection+retry pairs per hard prompt, each conditioned on a different failed attempt
- **Variable group sizes**: Hard prompt groups expand from n to n+k; advantage computation uses dynamic `torch.split` instead of fixed reshape
- **Reflection injection**: `ChatProtocol.inject_reflection()` inserts reflection into system message (implemented for InternS1Protocol and Qwen3Protocol)
- **Distillation loss**: Optional SFT loss (`--erl_distill_coef`) on successful retry action tokens (r2==1 only)
- **Per-task memory**: Optional (`OPENRLHF_ERL_MEMORY=1`) cross-episode reflection storage keyed by TDC task name
- **Training script**: `scripts/train_grpo_tdc_erl.sh` sets ERL env vars + `AGENT_FUNC_PATH` and delegates to intern_s1 script

## Architecture

### Tool-Calling Components

#### ToolCallingTurn (`openrlhf/utils/tool_calling_turn.py`)
Single-class agent implementing `AgentInstanceBase`. Handles conversation history, tool execution, reward computation, and format rendering.

```python
class ToolCallingTurn(AgentInstanceBase):
    async def reset(self, states) -> dict    # Returns {"observation": formatted_prompt}
    async def step(self, state_dict) -> dict  # Returns {"environment_feedback", "rewards", "done", ...}
```

#### ChatProtocol (`openrlhf/utils/chat_protocol.py`)
Abstract interface for model-specific tool-call formats.

**Implementations:**
- `GLMFlashProtocol`: XML format (`<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`)
- `InternS1Protocol`: JSON format with `<|action_start|><|plugin|>` delimiters, includes SMILES-safe JSON escape repair
- `Qwen3Protocol`: JSON format with `<tool_call>` / `</tool_call>` delimiters, `<|im_start|>tool` observation wrappers (subclass of `InternS1Protocol`)
- `GPTOSSProtocol`: Harmony-format parser for `gpt_oss` using generated token IDs

Selected via `OPENRLHF_CHAT_PROTOCOL` env var (propagated by `vllm_engine.py`).

### Token-Level Masking

Only LLM-generated action tokens contribute to policy loss:
```
Trajectory: [prompt | action_1 | observation_1 | action_2 | observation_2 | final_answer]
action_ranges = [(N, M), (K, L), (P, Q)]  # Only LLM-generated spans
loss_mask = action_mask * attention_mask
actor_loss = -(log_probs * advantages * loss_mask).sum() / loss_mask.sum()
```

### Multi-Turn Flow
```
GRPO Training Loop
  -> Experience Maker (deferred dispatch 75/25)
    -> vLLM Engine (LLMRayActor)
      -> MultiTurnAgentExecutor (agent.py, tracks action_ranges)
        -> ToolCallingTurn (tool_calling_turn.py)
          -> ChatProtocol (parse/render)
```

## CLI Arguments

**Tool-Calling:**
- `--agent_func_path`: Path to agent implementation
- `--agent_max_steps`: Max turns per episode (default: 5)
- `--vllm_stop_strings`: Stop generation tokens (e.g., `"</tool_call>"`)
- `--chat_protocol`: Protocol name (`glm_flash`, `intern_s1`, `gpt_oss`, `qwen3`)
- `--tool_version`: Tool schema version descriptor

**TDC:**
- `--tdc_tools`: Path to per-task tool schema JSON

**Eval:**
- `--skip_eval_step_zero`: Skip evaluation at step 0
- `--skip_training`: Run only the step-0 eval and exit (skips training loop, for benchmarking eval speed and efficiency reports)

**Dispatch:**
- `--deferred_dispatch`: Dispatch 75% upfront, hold 25% as reserve until any engine drops to ≤4 pending

**Checkpointing:**
- `--push_to_hub <repo_id>`: Upload checkpoints to HF Hub
- `--push_to_hub_private`: Make repo private
- `--delete_local_after_push`: Delete local checkpoint after upload
- `--save_steps <int>`: Save checkpoint every N steps

**ERL (Experimental):**
- `--agent_func_path <path>`: Point to `openrlhf/utils/erl_tdc_agent.py` to enable ERL
- `--erl_hard_threshold <float>`: Avg reward threshold for hard prompt gating (None=disabled, 0.2 recommended for TDC)
- `--erl_k <int>`: Number of reflection+retry pairs per hard prompt (default: 4)
- `--erl_memory`: Enable cross-episode reflection memory (off by default)
- `--erl_max_memory <int>`: Max reflections per task in memory (default: 5)
- `--erl_max_reflection_tokens <int>`: Max tokens for reflection generation (default: 512)
- `--erl_distill_coef <float>`: Distillation loss coefficient for successful retries (0=disabled)

**Environment Variables:**
Set automatically by vllm_engine.py:
- `OPENRLHF_MODEL_PATH`: Model path for tokenizer
- `OPENRLHF_MAX_STEPS`: Max agent steps
- `OPENRLHF_CHAT_PROTOCOL`: Chat protocol name
- `OPENRLHF_TOOL_VERSION`: Tool version descriptor

Debug flags:
- `OPENRLHF_DEBUG_NAN_GUARD=1`: Enable NaN assertions in actor forward/backward
- `OPENRLHF_DEBUG_LOGITS=1`: Enable verbose logit/log_prob diagnostics

## Key Files Changed from Upstream

| File | Changes |
|---|---|
| `openrlhf/utils/chat_protocol.py` | New: ChatProtocol ABC, GLMFlashProtocol, InternS1Protocol, Qwen3Protocol, GPTOSSProtocol, `inject_reflection()` for ERL |
| `openrlhf/utils/tool_calling_turn.py` | New: ToolCallingTurn agent class; injectable `reward_fn` param; `_default_reward_fn` fallback |
| `openrlhf/utils/erl_executor.py` | New: ERLExecutor wrapping AgentExecutorBase with `execute_batch()`, reflection+retry, memory, protocol handling |
| `openrlhf/utils/erl_tdc_agent.py` | New: Agent file exporting ERLExecutor(ToolCallingTurn) as AgentExecutor for --agent_func_path |
| `openrlhf/utils/fp4_config.py` | New: FP4Config dataclass consolidating vllm_sync_fp4/qat/dequantize_base flags |
| `openrlhf/utils/tdc_reward_model.py` | New: binary answer extractor for TDC eval |
| `openrlhf/datasets/tdc_loader.py` | New: TDCDatasetLoader |
| `openrlhf/datasets/prompts_dataset.py` | Per-task tool schema injection via tools_map |
| `openrlhf/trainer/ppo_utils/experience_maker.py` | Deferred dispatch (75/25 via `--deferred_dispatch`), trace logging, filtered count logging, ERL variable group sizes |
| `openrlhf/trainer/ppo_trainer.py` | evaluate() in BasePPOTrainer, step-0 eval, macro-F1, hub push |
| `openrlhf/trainer/ppo_trainer_async.py` | Eval wired into async trainer, missing logging/cleanup fixes |
| `openrlhf/trainer/ray/vllm_engine.py` | Passes chat_protocol env var to Ray actors, reduced CUDA graphs, MXFP4 weight sync, `execute_batch()` dispatch in `generate_responses()` |
| `openrlhf/trainer/ray/vllm_worker_wrap.py` | On-the-fly bf16→MXFP4 quantization for vLLM weight sync |
| `openrlhf/utils/mxfp4_quantize.py` | MXFP4 quantization utility + QAT: `fake_quantize_mxfp4`, `_Mxfp4FakeQuant`, `register_mxfp4_qat_parametrization` |
| `openrlhf/trainer/ray/ppo_actor.py` | NaN guard assertions; `fp4_config` forwarded to Actor(); weight sync reads `fp4_config.sync_format`; ERL distillation loss |
| `openrlhf/models/actor.py` | torch.where NaN fix, logit diagnostics; `fp4_config: FP4Config` param replaces `qat`/`qat_fp4_format` |
| `openrlhf/models/utils.py` | torch.where in masked_mean |
| `openrlhf/utils/deepspeed/deepspeed.py` | Recreate optimizer after AutoTP to free pre-sharded weights |
| `openrlhf/utils/distributed_util.py` | NCCL diagnostic logging |
| `openrlhf/utils/logging_utils.py` | eval/global_step W&B axis |
| `openrlhf/cli/batch_inference.py` | Transformers v4/v5 compat |
| `openrlhf/cli/interactive_chat.py` | Transformers v4/v5 compat |
| `openrlhf/cli/train_ppo_ray.py` | New CLI args for tools, eval, checkpointing, `--deferred_dispatch`, ERL args; validation builds `args.fp4_config` |
| `openrlhf/utils/agent.py` | Pass hf_tokenizer + `**agent_kwargs` through to agent instance |

## Storage Guidelines

**IMPORTANT:** The Slurm personal home directory has very limited storage. Always save large files (model checkpoints, datasets, logs) to the shared project directory instead:

- **Large file storage:** `/vast/projects/myatskar/design-documents/hf_home/`
- **Never** save large checkpoints or model weights under `$PROJECT_ROOT/saves/` or commit them to the repo.
- Scripts use `LOCAL_SAVE_DIR=/vast/projects/myatskar/design-documents/hf_home` as the default save path.

---

**Last Updated:** 2026-03-03
**Base Version:** OpenRLHF (latest main branch)
