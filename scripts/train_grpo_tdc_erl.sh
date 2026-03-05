#!/bin/bash
#
# Intern-S1 GRPO training with ERL (Experiential Reinforcement Learning).
#
# This wraps the standard intern_s1 script with ERL-specific flags.
# For hard prompts (avg reward < threshold), generates diverse reflections
# from failed attempts, retries with injected reflections, and includes
# both original and retry experiences in the advantage group.
#
# Usage:
#   bash scripts/train_grpo_tdc_erl.sh
#
#   # With custom ERL settings:
#   ERL_K=4 ERL_THRESHOLD=0.2 ERL_DISTILL_COEF=0.1 bash scripts/train_grpo_tdc_erl.sh
#
#   # Distributed with ERL:
#   MODE=distributed ACTOR_GPUS=2 VLLM_NUM_ENGINES=6 bash scripts/train_grpo_tdc_erl.sh
#
# ERL-specific env vars:
#   ERL_THRESHOLD=0.2        # Hard prompt gating threshold (default: 0.2)
#   ERL_K=4                  # Number of reflection+retry pairs per hard prompt (default: 4)
#   ERL_MEMORY=0             # Enable cross-episode reflection memory (default: 0)
#   ERL_MAX_MEMORY=5         # Max reflections per task in memory (default: 5)
#   ERL_MAX_REFL_TOKENS=512  # Max tokens for reflection generation (default: 512)
#   ERL_DISTILL_COEF=0.1     # Distillation loss coef for successful retries (default: 0.1)
#

### ERL FEATURE FLAGS ###
ERL_THRESHOLD="${ERL_THRESHOLD:-0.2}"
ERL_K="${ERL_K:-4}"
ERL_MEMORY="${ERL_MEMORY:-0}"
ERL_MAX_MEMORY="${ERL_MAX_MEMORY:-5}"
ERL_MAX_REFL_TOKENS="${ERL_MAX_REFL_TOKENS:-512}"
ERL_DISTILL_COEF="${ERL_DISTILL_COEF:-0.1}"

### BUILD ERL CLI FLAGS ###
ERL_FLAGS="--erl_hard_threshold $ERL_THRESHOLD --erl_k $ERL_K --erl_max_reflection_tokens $ERL_MAX_REFL_TOKENS"

if [ "$ERL_MEMORY" = "1" ]; then
    ERL_FLAGS+=" --erl_memory --erl_max_memory $ERL_MAX_MEMORY"
fi

if [ "$(echo "$ERL_DISTILL_COEF > 0" | bc -l)" = "1" ]; then
    ERL_FLAGS+=" --distill_coef $ERL_DISTILL_COEF"
fi

### OVERRIDE DEFAULTS FOR ERL ###
# ERL generates extra samples for hard prompts (n + k), so we use fewer
# base samples per prompt to keep compute budget manageable.
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"

# Point agent_func_path to the ERL agent (wraps ToolCallingTurn with ERL).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export AGENT_FUNC_PATH="${AGENT_FUNC_PATH:-$PROJECT_ROOT/openrlhf/utils/erl_tdc_agent.py}"

# Append ERL flags to EXTRA_ARGS and delegate to the standard intern_s1 script.
export EXTRA_ARGS="${EXTRA_ARGS:-} $ERL_FLAGS"

# Add -erl suffix to run name for identification
export WANDB_GROUP="${WANDB_GROUP:-TDC-InternS1-ERL}"

echo "========================================"
echo "ERL Configuration"
echo "========================================"
echo "ERL Threshold: $ERL_THRESHOLD"
echo "ERL K (retries): $ERL_K"
echo "ERL Memory: $ERL_MEMORY"
echo "ERL Distill Coef: $ERL_DISTILL_COEF"
echo "ERL Max Refl Tokens: $ERL_MAX_REFL_TOKENS"
echo "N Samples Per Prompt: $N_SAMPLES_PER_PROMPT"
echo "Agent Func Path: $AGENT_FUNC_PATH"
echo "========================================"

# Delegate to the standard intern_s1 script
exec bash "$SCRIPT_DIR/train_grpo_tdc_intern_s1.sh" "$@"
