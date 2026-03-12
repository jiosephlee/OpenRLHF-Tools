# PolicyLoss: Changes from Upstream OpenRLHF

## Overview

The core GRPO loss logic — token-level PPO ratio, clipping, per-sequence-mean reduction — is **unchanged from upstream**. Our changes add alternative reduction strategies, an alternative ratio computation (GSPO), and cross-rank sequence-count synchronization for correctness under unequal sample counts.

In the default configuration (`--loss_type ppo`), with equal sample counts across ranks (the common case — no ERL variable group sizes, no partial batches), the loss is **mathematically equivalent** to the original upstream formulation.

## What Changed

### 1. `token_level_loss` reduction parameter

Controls how per-token losses are reduced to a scalar. Derived from `--loss_type`:

| Value | Reduction | Used by |
|-------|-----------|---------|
| `None` | Per-sequence mean, then batch mean | `ppo`, `gspo`, `dr_grpo`, `sapo` |
| `"global"` | Flat token mean with cross-rank all-reduce normalizer | `dapo`, `cispo` |
| `"local_rank"` | Flat token mean within rank (no cross-rank sync) | `bnpo` |

`None` is the original upstream behavior.

### 2. `policy_loss_type` ratio parameter

Controls how the importance sampling ratio is computed:

| Value | Ratio | Used by |
|-------|-------|---------|
| `"ppo"` (default) | Token-level: `exp(log_pi - log_pi_old)` | All except `gspo` |
| `"gspo"` | Sequence-level: `exp(mean(log_pi - log_pi_old))` | `gspo` only |

`"ppo"` is the original upstream behavior. GSPO implements [arXiv:2507.18071](https://arxiv.org/pdf/2507.18071).

### 3. Cross-rank sequence-count synchronization (replay buffer)

When using per-sequence reduction (`token_level_loss=None`), the replay buffer now all-reduces the local sequence count `N_r` across ranks to get `N_global`, then scales each microbatch's loss by `|microbatch| * world_size / N_global`. This ensures each sequence gets exactly `1/N_global` weight in the final gradient, even if ranks have unequal sample counts.

**When ranks have equal counts** (the typical case), this reduces to the same `1/N_local` scaling as upstream — the all-reduce is a no-op in effect.

**When ranks have unequal counts** (e.g., ERL variable group sizes, partial batches from oversampling), this prevents the gradient from being biased toward ranks with fewer samples.

Both `--use_dynamic_batch` and `--use_adaptive_batch` support all loss types. For token-level reduction losses (`dapo`, `bnpo`, `cispo`), both modes use token-proportional scaling — each microbatch's loss scale is proportional to its action token count, ensuring every token contributes equally to the gradient.

### 4. Unified `--loss_type` CLI flag

The three previously independent flags (`--policy_loss_type`, `--token_level_loss`, `--liger_loss_type`) are replaced by a single `--loss_type` flag. Internal parameters are derived automatically:

| `--loss_type` | `policy_loss_type` | `token_level_loss` | Liger `loss_type` |
|---------------|--------------------|--------------------|-------------------|
| `ppo` | `ppo` | `None` | `grpo` |
| `dapo` | `ppo` | `global` | `dapo` |
| `bnpo` | `ppo` | `local_rank` | `bnpo` |
| `dr_grpo` | `ppo` | `None` | `dr_grpo` |
| `gspo` | `gspo` | `None` | `grpo` |
| `cispo` | — (Liger-only) | — | `cispo` |
| `sapo` | — (Liger-only) | — | `sapo` |

### 5. LigerPolicyLoss wrapper

Wraps `liger-kernel`'s fused lm_head+GRPO loss for reduced peak memory. Supports both `triton` and `chunked` backends, and the same vLLM off-policy IS correction methods (TIS, ICEPOP, seq-mask-tis) as `PolicyLoss`. This is entirely new code (no upstream equivalent).

## Files Involved

- `openrlhf/models/loss.py` — `PolicyLoss` (modified), `LigerPolicyLoss` (new)
- `openrlhf/cli/train_ppo_ray.py` — `--loss_type` flag and derivation logic
- `openrlhf/trainer/ppo_utils/replay_buffer.py` — cross-rank seq-count sync in `setup_dynamic_batch` / `setup_adaptive_batch`
- `openrlhf/trainer/ray/ppo_actor.py` — wires derived params to loss constructors
