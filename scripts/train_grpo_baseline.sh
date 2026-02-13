#!/bin/bash
#
# BASELINE GRPO training script — verifies the core training loop works.
# Uses standard HuggingFace datasets, NO tool-calling, NO custom agents.
# This script is compatible with both the modified and unmodified OpenRLHF codebase.
#
# Model:   DeepSeek-R1-Distill-Qwen-1.5B  (~3GB, fits easily on 2 GPUs)
# Dataset: OpenRLHF/dapo-math-17k         (math problems with \boxed{answer} labels)
# Reward:  Math answer verification        (examples/python/math_reward_func.py)
# Method:  GRPO (group_norm advantage)     (no critic, no separate reward model)
#
# Usage:
#   1. Get an interactive node:
#      srun --partition=<partition> --gpus=2 --mem-per-gpu=64G --cpus-per-gpu=4 --time=2:00:00 --pty bash
#   2. Activate env:
#      module load MAMBA && module load cuda/13.1.0
#      micromamba activate <your_openrlhf_env>
#   3. Run:
#      bash scripts/train_grpo_baseline.sh [num_gpus]
#
# The script runs ~200 samples with small batch sizes for a quick smoke test.
# Increase --max_samples for a longer run.
#

set -euo pipefail

### ARGS ###
NUM_GPUS=${1:-${SLURM_GPUS_ON_NODE:-2}}

### PROJECT ROOT ###
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/openrlhf" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -d "$PROJECT_ROOT/openrlhf" ]; then
    echo "Error: Cannot find project root (no 'openrlhf' directory found)" >&2
    exit 1
fi

### CONFIG ###
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="OpenRLHF/dapo-math-17k"
REWARD_FUNC="$PROJECT_ROOT/examples/python/math_reward_func.py"
LEARNING_RATE="1e-6"

### RUN CONFIG ###
RUN_ID="baseline-grpo-math_$(date +%Y-%m-%d_%H-%M-%S)"
SAVE_PATH="$PROJECT_ROOT/saves/baseline/$RUN_ID"
HUB_REPO_ID="jiosephlee/grpo-baseline-math"
mkdir -p "$PROJECT_ROOT/saves/baseline"

### W&B ###
WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_baseline}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Warning: WANDB_API_KEY is not set. W&B logging will be disabled." >&2
fi

### GPU LAYOUT (colocated — shared GPUs) ###
TRAIN_BATCH_SIZE=$((NUM_GPUS * 8))
VLLM_NUM_ENGINES=$NUM_GPUS

### ENVIRONMENT VARIABLES ###
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"

export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1

### RAY ###
export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
ulimit -n 65535 2>/dev/null || true

ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/ray/session_* 2>/dev/null || true

echo "Starting Ray head node at $RAY_NODE_IP_ADDRESS"
ray start --head \
    --node-ip-address "$RAY_NODE_IP_ADDRESS" \
    --num-gpus "$NUM_GPUS" \
    --temp-dir "$RAY_TMPDIR" &

echo "Waiting for Ray..."
for i in {1..60}; do
    curl -fsS http://127.0.0.1:8265/api/version >/dev/null 2>&1 && break
    sleep 1
done

export RAY_ADDRESS="auto"

### PRINT CONFIG ###
echo "========================================"
echo "Baseline GRPO Training (Math)"
echo "========================================"
echo "Model:    $MODEL"
echo "Dataset:  $DATASET"
echo "Reward:   $REWARD_FUNC"
echo "LR:       $LEARNING_RATE"
echo "Run ID:   $RUN_ID"
echo "----------------------------------------"
echo "NUM_GPUS:         $NUM_GPUS (colocated)"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "VLLM_ENGINES:     $VLLM_NUM_ENGINES"
echo "Save:             $SAVE_PATH"
echo "Hub:              $HUB_REPO_ID"
echo "W&B:              project=$WANDB_PROJECT run=$RUN_ID"
echo "========================================"

### TRAINING ###
CUDA_VISIBLE_DEVICES=0,1 python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$MODEL" \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node $NUM_GPUS \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $NUM_GPUS \
    --vllm_num_engines $VLLM_NUM_ENGINES \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.94 \
    --advantage_estimator group_norm \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.27 \
    --dynamic_filtering \
    --dynamic_filtering_reward_range 0 1 \
    --remote_rm_url "$REWARD_FUNC" \
    --save_steps -1 \
    --logging_steps 1 \
    --n_samples_per_prompt 8 \
    --micro_train_batch_size 4 \
    --micro_rollout_batch_size 8 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $TRAIN_BATCH_SIZE \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 16384 \
    --max_samples 640 \
    --zero_stage 1 \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$DATASET" \
    --input_key prompt \
    --label_key label \
    --apply_chat_template \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --enable_prefix_caching \
    --eval_dataset OpenRLHF/aime-2024 \
    --eval_steps 4 \
    --eval_temperature 0.7 \
    --eval_n_samples_per_prompt 4 \
    --save_path "$SAVE_PATH" \
    --save_hf_ckpt \
    --push_to_hub "$HUB_REPO_ID" \
    --delete_local_after_push \
    --use_wandb "${WANDB_API_KEY:+1}" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$RUN_ID"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
ray stop --force || true
