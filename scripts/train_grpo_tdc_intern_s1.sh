#!/bin/bash
#
# TDC GRPO Training Script for Intern-S1-mini — HYBRID (colocated) mode
#
# Actor and vLLM share the same GPUs via sleep mode.
# Uses the Intern-S1 JSON tool-calling format:
#   <|action_start|><|plugin|>{"name": "...", "parameters": {...}}<|action_end|>
#
# Usage:
#   export WANDB_API_KEY=...   # required for wandb tracking
#   # SLURM: sbatch scripts/train_grpo_tdc_intern_s1.sh <task_name> [model_path] [learning_rate]
#   # Direct: bash scripts/train_grpo_tdc_intern_s1.sh <task_name> [model_path] [learning_rate] [num_gpus]
#
# Examples:
#   sbatch scripts/train_grpo_tdc_intern_s1.sh AMES
#   sbatch scripts/train_grpo_tdc_intern_s1.sh Skin_Reaction jiosephlee/sft_intern_distillation_Intern-S1-mini-lm_complet_only_chat_think_lr5e-05
#   bash scripts/train_grpo_tdc_intern_s1.sh hERG jiosephlee/sft_intern_distillation_Intern-S1-mini-lm_complet_only_chat_think_lr5e-05 1e-6 4
#

### SLURM DIRECTIVES ###
#SBATCH --job-name=S-grpo
#SBATCH --partition=dgx-b200
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4                    # Shared GPUs (override with --gpus=N)
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=0:20:00

set -euo pipefail

############################
#   CONFIGURATION          #
############################

# Detect if running under SLURM
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
    IS_SLURM=true
    NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
else
    echo "Running in standalone mode"
    IS_SLURM=false
    NUM_GPUS=${4:-4}  # Use 4th arg or default to 4
fi

# Parse arguments
TASK_NAME=${1:-"AMES"}
PRETRAIN_PATH=${2:-"jiosephlee/sft_intern_distillation_Intern-S1-mini-lm_complet_only_chat_think_lr5e-05"}
LEARNING_RATE=${3:-"1e-6"}

# Resolve PROJECT_ROOT by walking up from a known starting directory until
# we find the 'openrlhf' package dir. This handles both:
#   - SLURM: BASH_SOURCE points to spool copy, so start from SLURM_SUBMIT_DIR
#   - Standalone: BASH_SOURCE is the real script path
if [ "$IS_SLURM" = true ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/openrlhf" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -d "$PROJECT_ROOT/openrlhf" ]; then
    echo "Error: Cannot find project root (no 'openrlhf' directory found)" >&2
    exit 1
fi

DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format"
TRAIN_DATA="$DATA_DIR/${TASK_NAME}_train.jsonl"
VAL_DATA="$DATA_DIR/${TASK_NAME}_val.jsonl"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Verify data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found: $TRAIN_DATA"
    echo "Available tasks:"
    ls "$DATA_DIR" | grep "_train.jsonl" | sed 's/_train.jsonl//' | sort
    exit 1
fi

# Run configuration
RUN_ID="S-grpo-${TASK_NAME}_$(date +%Y-%m-%d_%H-%M-%S)_lr${LEARNING_RATE}"
SAVE_PATH="$PROJECT_ROOT/saves/tdc/${TASK_NAME}/$RUN_ID"

# Training hyperparameters
TRAIN_BATCH_SIZE=$((NUM_GPUS * 16))
VLLM_NUM_ENGINES=$((NUM_GPUS / 2))
[ $VLLM_NUM_ENGINES -lt 1 ] && VLLM_NUM_ENGINES=1

# Tool-calling configuration — Intern-S1 format
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"
AGENT_MAX_STEPS=40
PROMPT_CONSTRUCTION_MODE="auto"   # "manual" (fast) or "auto" (robust)
CHAT_PROTOCOL="intern_s1"           # Intern-S1 JSON format with <|action_start|><|plugin|> markers

# GRPO configuration
N_SAMPLES_PER_PROMPT=8
ADVANTAGE_ESTIMATOR="dr_grpo"
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0.2 0.8"

# W&B (required for tracking)
echo $WANDB_API_KEY
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set. Set it for wandb tracking (e.g. export WANDB_API_KEY=...)." >&2
    exit 1
fi
WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"

# Intern-S1 sampling parameters (match inference-time settings from recipe)
TEMPERATURE=0.8
TOP_P=0.8

############################
#   ENVIRONMENT SETUP      #
############################

# Ray configuration
if [ "$IS_SLURM" = true ]; then
    export RAY_TMPDIR="/tmp/ray_${USER}/${SLURM_JOB_ID}"
else
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
fi
mkdir -p "$RAY_TMPDIR"
PERSIST_RAY_DIR="$PROJECT_ROOT/logs/ray/latest"

# vLLM configuration
export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1

# OpenRLHF environment variables (for agent)
export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
export OPENRLHF_PROMPT_CONSTRUCTION_MODE="$PROMPT_CONSTRUCTION_MODE"
export OPENRLHF_CHAT_PROTOCOL="$CHAT_PROTOCOL"
export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"

# NCCL/distributed training settings
export OMP_NUM_THREADS=16
export NCCL_NVLS_ENABLE=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_SL=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
# export NCCL_DEBUG=INFO  # Uncomment for debugging

# Uncomment if you need to specify IB adapters (adjust for your hardware)
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
# export NCCL_SOCKET_IFNAME=bond0
# export UCX_TLS=rc

############################
#   RAY LOG MANAGEMENT     #
############################

copy_ray_logs() {
    set +e
    local base="$RAY_TMPDIR/ray"
    local latest="$base/session_latest"
    local real="$latest"
    [ -L "$latest" ] && real="$(readlink -f "$latest")"

    echo "Copying Ray logs from: $real"
    sync
    sleep 2

    mkdir -p "$PERSIST_RAY_DIR"
    rm -rf "$PERSIST_RAY_DIR/session_latest" 2>/dev/null
    cp -a "$real" "$PERSIST_RAY_DIR/session_latest" 2>/dev/null || true

    echo "Top non-empty logs (source):"
    find "$real/logs" -type f -size +0c 2>/dev/null | head -n 20 || true

    echo "Top non-empty logs (dest):"
    find "$PERSIST_RAY_DIR/session_latest/logs" -type f -size +0c 2>/dev/null | head -n 20 || true
}
trap copy_ray_logs EXIT

############################
#   RAY INITIALIZATION     #
############################

# Get node IP
export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Increase file descriptor limit
ulimit -n 65535 2>/dev/null || true

# Clean up any previous Ray state
ray stop --force 2>/dev/null || true

# Start Ray head node
echo "Starting Ray head node at $RAY_NODE_IP_ADDRESS"
ray start --head \
    --node-ip-address "$RAY_NODE_IP_ADDRESS" \
    --num-gpus "$NUM_GPUS" \
    --temp-dir "$RAY_TMPDIR" &

# Wait for Ray to be ready
echo "Waiting for Ray to be ready..."
for i in {1..60}; do
    curl -fsS http://127.0.0.1:8265/api/version >/dev/null 2>&1 && break
    sleep 1
done

# Tell ray.init() to connect to the cluster we just started,
# instead of spawning a second local instance.
export RAY_ADDRESS="auto"

############################
#   PRINT CONFIGURATION    #
############################

echo "========================================"
echo "TDC GRPO Training — Intern-S1-mini"
echo "========================================"
echo "Task: $TASK_NAME"
echo "Model: $PRETRAIN_PATH"
echo "Chat Protocol: $CHAT_PROTOCOL"
echo "Learning Rate: $LEARNING_RATE"
echo "Run ID: $RUN_ID"
echo "----------------------------------------"
if [ "$IS_SLURM" = true ]; then
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Node ID: ${SLURM_NODEID:-0}"
    echo "Node Name: ${SLURM_NODELIST:-$(hostname)}"
    echo "----------------------------------------"
fi
echo "Training Data: $TRAIN_DATA"
echo "Save Path: $SAVE_PATH"
echo "----------------------------------------"
echo "NUM_GPUS: $NUM_GPUS"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "VLLM_NUM_ENGINES: $VLLM_NUM_ENGINES"
echo "RAY_NODE_IP_ADDRESS: $RAY_NODE_IP_ADDRESS"
echo "----------------------------------------"
echo "Agent Max Steps: $AGENT_MAX_STEPS"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "Prompt Mode: $PROMPT_CONSTRUCTION_MODE"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "----------------------------------------"
echo "W&B: project=$WANDB_PROJECT group=TDC-InternS1-$TASK_NAME run=$RUN_ID"
echo "========================================"

############################
#   GENERATE TOOLS JSON    #
############################
TDC_TOOLS_JSON="$PROJECT_ROOT/data/tdc/metadata/tools_per_task.json"
python "$PROJECT_ROOT/scripts/generate_tools_json.py" "$TDC_TOOLS_JSON"

############################
#   TRAINING COMMAND       #
############################

python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$PRETRAIN_PATH" \
    --ref_num_nodes 0 \
    --ref_num_gpus_per_node 0 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $NUM_GPUS \
    --vllm_num_engines $VLLM_NUM_ENGINES \
    --vllm_tensor_parallel_size $((NUM_GPUS > 1 ? 2 : 1)) \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.7 \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps -1 \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size 4 \
    --micro_rollout_batch_size 16 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $TRAIN_BATCH_SIZE \
    --max_epochs 1 \
    --prompt_max_len 4096 \
    --generate_max_len 8192 \
    --max_samples 1000000 \
    --zero_stage 0 \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --input_key messages \
    --label_key answer \
    --apply_chat_template \
    --tdc_tools "$TDC_TOOLS_JSON" \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
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
    --wandb_group "TDC-InternS1-$TASK_NAME" \
    --wandb_run_name "$RUN_ID"

############################
#   CLEANUP                #
############################

echo "Training complete!"
echo "Stopping Ray..."
ray stop --force || true

echo ""
echo "========================================"
echo "Training Summary"
echo "========================================"
echo "Saved model to: $SAVE_PATH"
echo "W&B: project=$WANDB_PROJECT group=TDC-InternS1-$TASK_NAME run=$RUN_ID"
echo "Ray logs at: $PERSIST_RAY_DIR/session_latest"
if [ "$IS_SLURM" = true ]; then
    echo "SLURM output: S-grpo_${SLURM_JOB_ID}.out"
fi
echo "========================================"
