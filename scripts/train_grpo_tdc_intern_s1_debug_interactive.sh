#!/bin/bash
#
# INTERACTIVE DEBUG version of the Intern-S1 GRPO training script.
#
# Usage:
#   1. Get an interactive node:  srun --partition=dgx-b200 --gpus=4 --mem-per-gpu=128G --cpus-per-gpu=8 --time=1:00:00 --pty bash
#   2. Activate env:             module load MAMBA && module load cuda/13.1.0 && micromamba activate /vast/projects/myatskar/design-documents/conda_env/open_rlhf_intern
#   3. Run:                      bash scripts/train_grpo_tdc_intern_s1_debug_interactive.sh <task_name> [model_path] [learning_rate] [num_gpus]
#
# Example:
#   bash scripts/train_grpo_tdc_intern_s1_debug_interactive.sh AMES jiosephlee/sft_intern_distillation_Intern-S1-mini-lm_complet_only_chat_think_lr5e-05 1e-6 4
#

set -euo pipefail

### ARGS ###
TASK_NAME=${1:-"AMES"}
PRETRAIN_PATH=${2:-"jiosephlee/sft_intern_distillation_Intern-S1-mini-lm_complet_only_chat_think_lr5e-05"}
LEARNING_RATE=${3:-"1e-6"}
NUM_GPUS=${4:-4}

# ### NCCL / IB / NETWORK CONFIG ###
# export OMP_NUM_THREADS=$(( NUM_GPUS * 2 ))
# export NCCL_NVLS_ENABLE=1
# export NCCL_IB_ADAPTIVE_ROUTING=1
# export NCCL_IB_SL=1
# export NCCL_IB_QPS_PER_CONNECTION=2
# export NCCL_IB_SPLIT_DATA_ON_QPS=0
# # GPU-affine IB NICs on DGX B200 (curated list — must all be present)
# REQUIRED_IB_HCAS=(mlx5_15 mlx5_10 mlx5_14 mlx5_13 mlx5_8 mlx5_7 mlx5_9 mlx5_4)
# AVAILABLE_IB_HCAS=$(ls /sys/class/infiniband/ 2>/dev/null)
# MISSING=()
# for hca in "${REQUIRED_IB_HCAS[@]}"; do
#     if ! echo "$AVAILABLE_IB_HCAS" | grep -qw "$hca"; then
#         MISSING+=("$hca")
#     fi
# done
# if [ ${#MISSING[@]} -gt 0 ]; then
#     echo "Error: Missing required IB HCAs: ${MISSING[*]}" >&2
#     echo "Available: $AVAILABLE_IB_HCAS" >&2
#     exit 1
# fi
# export NCCL_IB_HCA=$(IFS=,; echo "${REQUIRED_IB_HCAS[*]}")
# echo "NCCL_IB_HCA: $NCCL_IB_HCA"
# export NCCL_SOCKET_IFNAME=bond0
# export UCX_TLS=rc

### W&B ###
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set." >&2
    exit 1
fi
# export WANDB_API_KEY

### PROJECT ROOT ###
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/openrlhf" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -d "$PROJECT_ROOT/openrlhf" ]; then
    echo "Error: Cannot find project root (no 'openrlhf' directory found)" >&2
    exit 1
fi

### DATA ###
DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format"
TRAIN_DATA="$DATA_DIR/${TASK_NAME}_train.jsonl"
VAL_DATA="$DATA_DIR/${TASK_NAME}_val.jsonl"
mkdir -p "$PROJECT_ROOT/logs"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found: $TRAIN_DATA"
    echo "Available tasks:"
    ls "$DATA_DIR" | grep "_train.jsonl" | sed 's/_train.jsonl//' | sort
    exit 1
fi

### RUN CONFIG ###
RUN_ID="S-grpo-debug-${TASK_NAME}_$(date +%Y-%m-%d_%H-%M-%S)_lr${LEARNING_RATE}"
SAVE_PATH="$PROJECT_ROOT/saves/tdc/${TASK_NAME}/$RUN_ID"

### GPU LAYOUT ###
ACTOR_GPUS=$((NUM_GPUS / 4))
[ $ACTOR_GPUS -lt 1 ] && ACTOR_GPUS=1
VLLM_GPUS=$((NUM_GPUS - ACTOR_GPUS))
[ $VLLM_GPUS -ge 1 ]
VLLM_NUM_ENGINES=$VLLM_GPUS
VLLM_TENSOR_PARALLEL_SIZE=1
TRAIN_BATCH_SIZE=$((ACTOR_GPUS * 16))

### TOOL-CALLING CONFIG ###
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_agent.py"
AGENT_MAX_STEPS=40
PROMPT_CONSTRUCTION_MODE="auto"
CHAT_PROTOCOL="intern_s1"

### GRPO CONFIG ###
N_SAMPLES_PER_PROMPT=16
ADVANTAGE_ESTIMATOR="dr_grpo"
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0 1"

WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"
TEMPERATURE=0.7
TOP_P=0.95

### ENVIRONMENT VARIABLES ###
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"

export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1

export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
export OPENRLHF_PROMPT_CONSTRUCTION_MODE="$PROMPT_CONSTRUCTION_MODE"
export OPENRLHF_CHAT_PROTOCOL="$CHAT_PROTOCOL"
export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"

# Enable masked_mean debug dumps
export OPENRLHF_MASKED_MEAN_DEBUG_DIR="/tmp/debug_masked_mean_${USER}"
mkdir -p "$OPENRLHF_MASKED_MEAN_DEBUG_DIR"

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
echo "TDC GRPO Training — Intern-S1-mini (INTERACTIVE DEBUG)"
echo "========================================"
echo "Task: $TASK_NAME"
echo "Model: $PRETRAIN_PATH"
echo "Chat Protocol: $CHAT_PROTOCOL"
echo "Learning Rate: $LEARNING_RATE"
echo "Run ID: $RUN_ID"
echo "----------------------------------------"
echo "Training Data: $TRAIN_DATA"
echo "Save Path: $SAVE_PATH"
echo "----------------------------------------"
echo "NUM_GPUS: $NUM_GPUS  ACTOR: $ACTOR_GPUS  VLLM: $VLLM_GPUS"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "VLLM_NUM_ENGINES: $VLLM_NUM_ENGINES"
echo "----------------------------------------"
echo "masked_mean debug dir: $OPENRLHF_MASKED_MEAN_DEBUG_DIR"
echo "========================================"

### TRAINING ###
python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$PRETRAIN_PATH" \
    --ref_num_nodes 0 \
    --ref_num_gpus_per_node 0 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $ACTOR_GPUS \
    --vllm_num_engines $VLLM_NUM_ENGINES \
    --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL_SIZE \
    --vllm_gpu_memory_utilization 0.7 \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps -1 \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size 8 \
    --micro_rollout_batch_size 16 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $TRAIN_BATCH_SIZE \
    --max_epochs 1 \
    --prompt_max_len 8192 \
    --generate_max_len 1536 \
    --max_samples 1000000 \
    --zero_stage 0 \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --input_key messages \
    --label_key answer \
    --apply_chat_template \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --async_train \
    --async_queue_size 1 \
    --enforce_eager \
    $([ "$DYNAMIC_FILTERING" = true ] && echo "--dynamic_filtering --dynamic_filtering_reward_range $DYNAMIC_FILTERING_REWARD_RANGE" || echo "") \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    --agent_func_path "$AGENT_FUNC_PATH" \
    --agent_max_steps $AGENT_MAX_STEPS \
    --vllm_stop_strings "<|action_end|>" "<|im_end|>" \
    --prompt_construction_mode "$PROMPT_CONSTRUCTION_MODE" \
    --chat_protocol "$CHAT_PROTOCOL" \
    --use_wandb 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group "TDC-InternS1-debug-$TASK_NAME" \
    --wandb_run_name "$RUN_ID" \
    --rollout_trace_dir "$SAVE_PATH/rollout_traces"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
ray stop --force || true
