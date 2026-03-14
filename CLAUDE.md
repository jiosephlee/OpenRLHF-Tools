# OpenRLHF-Tools: Extended OpenRLHF Fork

## Section 1: Original OpenRLHF Architecture & Abstractions

OpenRLHF is built around a flexible, Ray-based distributed architecture designed to decouple stateful components for massive scaling. 

### Ray-Based Distributed Infrastructure
OpenRLHF provisions different models (e.g., Actor, Critic, Reference Model, Reward Model) inside independent Ray actor groups:
- **Actor process**: Runs training (forward/backward passes) on the policy model using PyTorch + DeepSpeed.
- **Reference process**: Hosts the frozen reference model for KL penalty calculations during RL.
- **vLLM Engine**: Inference and rollout generation happens entirely in separate `LLMRayActor` instances wrapping vLLM, decoupled from the memory-intensive Actor training loop.
- **Colocate Mode**: A cost-saving technique where vLLM and the PyTorch Actor map to the same physical GPUs. They take turns using the GPUs by aggressively unloading and reloading weights into VRAM.

### Core Training Loop Abstractions
The main orchestrator of the training loop lies in `PPOTrainer` (or `GRPOTrainer`):
- **`ExperienceMaker`**: Before adjusting gradients, the trainer delegates to the Experience Maker. It takes the prompt dataset, ships prompts to the associated vLLM engine, extracts generated tokens, and runs them through the Reward and Reference models to prepare full trajectory tensors (states, actions, logprobs, rewards, advantages).
- **Mini-Batches & Updates**: The Trainer slices the massive experience buffer into micro-batches for DeepSpeed ZeRO gradient accumulation.

---

## Section 2: ML Flow of the GRPO Recipe

Group Relative Policy Optimization (GRPO) modifies standard PPO by dropping the Critic model (Value network). Instead, it normalizes rewards within groups to compute advantages.

### 1. Rollout Generation
The `ExperienceMaker` extracts a batch of prompts. For each prompt, it asks vLLM to sample $G$ independent completions.

### 2. Group Advantage Estimation
Instead of querying a trained Critic model to determine baseline expected return, GRPO looks at all $G$ completions for a single prompt.
Calculates the mean $\mu$ and standard deviation $\sigma$ of the rewards for these $G$ responses.
The advantage for standard completion $i$ is calculated as: $A_i = \frac{R_i - \mu}{\sigma}$.

### 3. PPO-Style Optimization
Using the pre-computed advantages, the Actor model runs mini-batch gradient descent for multiple optimization epochs. To prevent catastrophically large policy shifts:
- The likelihood ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ is clipped using standard PPO bounds: $clip(r_t(\theta), 1-\epsilon, 1+\epsilon)$.

### 4. KL Divergence & Masking
- The Reference Model's log probabilities are computed for the generated actions.
- An exact KL divergence penalty ensures the updated policy stays anchored close to the base model.
- **Token-Level Masking**: Only the LLM's generated response tokens contribute to the backward pass loss. Prompt tokens and intermediate observation tokens (in multi-turn tool calling) are completely masked out using `action_ranges`, ensuring the model isn't penalized for environmental output.

---

## Section 3: Our Custom Changes

This fork extends OpenRLHF extensively with multi-turn tool-calling, quantization algorithms, and infrastructural stability patches.

### A. Multi-turn Tool Calling & Parallel Tool Execution natively
- **Parallel Tool Calls**: Added support for executing parallel tool calls natively within trajectories.
- **`ChatProtocol` Extensions**: Modular format-handling interface to support varying LLM prompt schemes.
  - implementations: `GLMFlashProtocol`, `InternS1Protocol` (with SMILES JSON-escape repairs), `Qwen3Protocol`, `GPTOSSProtocol` (Harmony-format syntax).
- **`ToolCallingTurn` & Agent Flow**: Single-class agent abstracting the back-and-forth multi-turn flow of prompting -> parsing -> executing tools -> generating observations -> appending history.
- **TDC Agent Integration**: `TDCDatasetLoader` loads Therapeutics Data Commons tasks and automatically formats tool schemas via the `--tdc_tools` mapping file.

### B. Tracking, Monitoring and Logging
- **Rollout Traces**: Auto-saves one readable, decoded rollout trace per step into `runs/<run_name>/traces/` for rapid prompting iteration and debugging.
- **Improved Evals**: Moved evaluate logic to `BasePPOTrainer`, disabled by default until later steps but `--skip_eval_step_zero` allows testing zero-shot behavior. Supports macro-F1 computations for TDC binary classification and logs cleanly to `eval/global_step` in wandb.
- **SmartReplay Telemetry**: Added new logging metrics and buffer logging after each experience-making step to closely monitor recycling phases and W&B QoL improvements tracking oversample/missed prompt counts.
- **Performance Timing & Bottleneck Auditing**: Implemented a precise, end-to-end timing tracking mechanism for the entire training loop. Distinct time intervals between stages (e.g., `gap_train_to_rollout`, `rollout_time`, `advantage_time`, `train_time`) are captured and stored in `runs/<run_name>/vllm_stats/run_timing.jsonl` to reliably audit run bottlenecks.

### C. Learning Recipes & Core Modifications
- **ERL (Experiential Reinforcement Learning) & Smart Replay** *(Experimental)*:
  Based on [Experiential Reinforcement Learning](https://arxiv.org/abs/2602.13949) (Shi et al., Feb 2026). For hard prompts where all samples fail (avg reward < threshold), generates structured reflections from failed attempts and retries with reflection-augmented prompts. Key components:
  - **`ERLExecutor`**: Wraps any `AgentExecutorBase` with ERL reflection+retry logic.
  - **Prompt-level gating**: Only hard prompts trigger reflection+retry; easy prompts use standard path.
  - **k diverse reflections**: $k$ reflection+retry pairs per hard prompt, each conditioned on a different failed attempt.
  - **Variable group sizes**: Hard prompt groups expand from $n$ to $n+k$; advantage computation uses dynamic `torch.split` instead of a fixed reshape.
  - **Reflection injection**: Inserts reflection into the system message (implemented for InternS1Protocol and Qwen3Protocol).
  - **Distillation loss**: Optional SFT loss (`--distill_coef`) on experiences tagged with `extra_logs["distill"] = 1`.
  - **Per-task memory**: Optional (`OPENRLHF_ERL_MEMORY=1`) cross-episode reflection storage keyed by TDC task name.

### D. Performance Tuning & Stability
- **Prompt-Level Oversampling with Early Termination**: Added `--oversample_ratio` flag to dispatch additional prompts per step. Drains unused or slow generation references mid-flight aggressively utilizing a `ray.cancel(ref)` early termination to abort stragglers and clean up vLLM scheduler state without waiting. 
- **Ceiling fix for dynamic batch splitting**: Ensures `math.ceil()` is used during microbatch generation, preventing packing algorithms from emitting 0 batches and accidentally exceeding `rollout_max_tokens_per_gpu`.
- **Liger Kernels Fused GRPO Loss (`loss.py`)**: Abstracted Liger's fused kernels into `LigerPolicyLoss` keeping API signatures compatible with `PolicyLoss`. Adding the chunked backend (`--liger_grpo_backend chunked`) fusing `lm_head` dramatically solves Actor OOM issues.
- **Token-Level Loss Normalization**: Replaced uniform 1/N batch division with token-proportional adaptive batch scaling `--token_level_loss` (`none`, `local_rank`, `global`). Guarantees equal accumulated gradients from all valid tokens across all dynamic micro-batches.
- **DeepSpeed AutoTP OOM Fix**: Enforces a `gc.collect()` and `cuda.empty_cache()` immediately after model sharding alongside explicit wiping of the PyTorch optimizer references. Saves ~80 GiB memory spikes.
- **Colocate Mode CUDA cache**: Forces `torch.cuda.empty_cache()` inside the `vllm_engine`'s `sleep()` sequence to forcibly relinquish reserved memory pages back to the driver before the Actor begins its backward pass.
- **NaN-Safe Masked Operations**: Prevents masked NaN errors propagating backwards by rewriting `(tensor * mask).sum()` explicitly into `torch.where(mask.bool(), tensor, 0).sum()`.

### E. Quantization
- **On-the-Fly FP4 Quantization for Weight Sync *(Experimental)***: Automatically dynamically compresses full bf16 actor weights down to FP4/MXFP4 packing using ModelOpt logic prior to broadcasting synced parameters to the vLLM Actor.
- **FP4 Quantization-Aware Training (QAT) *(Experimental)***: Bridges the domain shift when training against FP4-served targets. Batched Triton prefetches concatenate and fake-quantize all MoE expert weights dynamically within a single kernel launch on the default stream before each standard PyTorch forward pass.

### F. Infrastructure, Refactors, and bugs
- **Fixing bug for PyTorch Nightly**: Added checks for nightly PyTorch `isinstance` on the DeepSpeed optimizers and deferred LR schedule definitions appropriately to avoid crashes.
- **Unified Bash Staging**: Complete refactor of run scripts (e.g. `1a1v` light, `intern_s1`, `gpt_oss`) simplifying configurations between local nodes, distributed SLURM setups, interactive debugging modes, and Liger backend permutations.
- **Transformers v4 / v5**: Dual compatibility at runtime via import branching `batch_decode` (v4) vs `decode` (v5).
- **Seamless HF Hub Uploads**: Added native CLI options for automatically streaming checkpoints (`--push_to_hub`, `--push_to_hub_private`, dropping local temp files).

---
**Last Updated:** 2026-03-10
**Base Version:** OpenRLHF (latest main branch)

---

## Storage Guidelines

**IMPORTANT:** The Slurm personal home directory has very limited storage. Always save large files (model checkpoints, datasets, logs) to the shared project directory instead:

- **Large file storage:** `/vast/projects/myatskar/design-documents/hf_home/`
- **Never** save large checkpoints or model weights under `$PROJECT_ROOT/saves/` or commit them to the repo.
- Scripts use `LOCAL_SAVE_DIR=/vast/projects/myatskar/design-documents/hf_home` as the default save path.