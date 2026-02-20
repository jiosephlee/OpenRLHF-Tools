# OpenRLHF-Tools: Extended OpenRLHF Fork

## Overview

This fork extends OpenRLHF with multi-turn tool-calling support for GRPO training, along with several infrastructure improvements: transformers v5 compatibility, multi-stage GPU dispatch, DeepSpeed OOM fixes, eval improvements, and TDC (Therapeutics Data Commons) dataset integration.

**Key Features:**
- Multi-turn agent-based rollouts with tool execution
- Token-level masking (only LLM actions contribute to loss, observations excluded)
- Clean abstraction layer (ToolCallingTurn + ChatProtocol)
- Multiple chat protocol support: GLM Flash XML, Intern-S1, Qwen3
- Transformers v4/v5 backward compatibility
- 3-stage deferred GPU dispatch for better load balancing
- AutoTP OOM fix (free pre-sharded weights before DeepSpeed init)
- NaN-safe masked operations (`torch.where` instead of `tensor * mask`)
- Eval at step 0, macro-F1 for TDC, `eval/global_step` W&B axis
- Checkpoint uploading to HF Hub
- Rollout trace logging to `runs/<run_name>/<date>/traces/`

## Major Changes from Upstream OpenRLHF

### 1. Transformers v5 Compatibility
**Files:** `openrlhf/cli/batch_inference.py`, `openrlhf/cli/interactive_chat.py`

Detects transformers major version at import time and branches on `batch_decode` (v4) vs `decode` (v5). `requirements.txt` allows either version.

### 2. Multi-Stage GPU Dispatch (3-Stage Deferred Dispatch)
**File:** `openrlhf/trainer/ppo_utils/experience_maker.py`

Upstream dispatches all prompts to vLLM engines at once. We split into 3 stages (50/25/25):
- Stage 1 (50%): dispatched immediately
- Stage 2 (25%): dispatched when any engine's pending count drops to <=1
- Stage 3 (25%): same trigger

This dramatically improves GPU utilization when generation times vary (common with multi-turn tool calling). Uses heap-based balancer with per-engine pending counts.

### 3. DeepSpeed AutoTP OOM Fix
**File:** `openrlhf/utils/deepspeed/deepspeed.py`

After `tp_model_init()` shards the model, the old optimizer still holds references to full-size pre-sharded parameters (~80 GiB). Fix: explicitly delete old optimizer, break scheduler reference, recreate optimizer over sharded params, force `gc.collect()` + `torch.cuda.empty_cache()` before `deepspeed.initialize()`.

### 4. NaN-Safe Masked Operations
**Files:** `openrlhf/models/actor.py`, `openrlhf/models/utils.py`

Changed `(tensor * mask).sum()` to `torch.where(mask.bool(), tensor, torch.zeros_like(tensor)).sum()` in `masked_mean()` and `action_log_probs`. Prevents NaN propagation through masked positions.

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

New CLI args: `--push_to_hub`, `--push_to_hub_private`, `--delete_local_after_push`, `--save_steps_ratio`

### 8. Rollout Trace Logging
**File:** `openrlhf/trainer/ppo_utils/experience_maker.py`

Saves one decoded rollout trace per step to `runs/<run_name>/<date>/traces/`. Annotates each record with prompt/action/observation sections decoded from token IDs using action ranges.

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
- `GPTOSSProtocol`: Harmony-format parser for `gpt_oss` using generated token IDs
- Qwen3 support via `--chat_protocol qwen3`

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
  -> Experience Maker (3-stage dispatch)
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

**TDC:**
- `--tdc_tools`: Path to per-task tool schema JSON

**Eval:**
- `--skip_eval_step_zero`: Skip evaluation at step 0

**Checkpointing:**
- `--push_to_hub <repo_id>`: Upload checkpoints to HF Hub
- `--push_to_hub_private`: Make repo private
- `--delete_local_after_push`: Delete local checkpoint after upload
- `--save_steps_ratio <float>`: Compute save_steps as fraction of total steps

## Environment Variables

Set automatically by vllm_engine.py:
- `OPENRLHF_MODEL_PATH`: Model path for tokenizer
- `OPENRLHF_MAX_STEPS`: Max agent steps
- `OPENRLHF_CHAT_PROTOCOL`: Chat protocol name

Debug flags:
- `OPENRLHF_DEBUG_NAN_GUARD=1`: Enable NaN assertions in actor forward/backward
- `OPENRLHF_DEBUG_LOGITS=1`: Enable verbose logit/log_prob diagnostics

## Key Files Changed from Upstream

| File | Changes |
|---|---|
| `openrlhf/utils/chat_protocol.py` | New: ChatProtocol ABC, GLMFlashProtocol, InternS1Protocol, GPTOSSProtocol |
| `openrlhf/utils/tool_calling_turn.py` | New: ToolCallingTurn agent class |
| `openrlhf/utils/tdc_reward_model.py` | New: binary answer extractor for TDC eval |
| `openrlhf/datasets/tdc_loader.py` | New: TDCDatasetLoader |
| `openrlhf/datasets/prompts_dataset.py` | Per-task tool schema injection via tools_map |
| `openrlhf/trainer/ppo_utils/experience_maker.py` | 3-stage dispatch, trace logging, filtered count logging |
| `openrlhf/trainer/ppo_trainer.py` | evaluate() in BasePPOTrainer, step-0 eval, macro-F1, hub push |
| `openrlhf/trainer/ppo_trainer_async.py` | Eval wired into async trainer |
| `openrlhf/trainer/ray/vllm_engine.py` | Passes chat_protocol env var to Ray actors |
| `openrlhf/trainer/ray/ppo_actor.py` | NaN guard assertions |
| `openrlhf/models/actor.py` | torch.where NaN fix, logit diagnostics |
| `openrlhf/models/utils.py` | torch.where in masked_mean |
| `openrlhf/utils/deepspeed/deepspeed.py` | Recreate optimizer after AutoTP to free pre-sharded weights |
| `openrlhf/utils/distributed_util.py` | NCCL diagnostic logging |
| `openrlhf/utils/logging_utils.py` | eval/global_step W&B axis |
| `openrlhf/cli/batch_inference.py` | Transformers v4/v5 compat |
| `openrlhf/cli/interactive_chat.py` | Transformers v4/v5 compat |
| `openrlhf/cli/train_ppo_ray.py` | New CLI args for tools, eval, checkpointing |
| `openrlhf/utils/agent.py` | Pass hf_tokenizer through to agent instance |

---

**Last Updated:** 2026-02-16
**Base Version:** OpenRLHF (latest main branch)
