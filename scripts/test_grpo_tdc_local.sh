#!/bin/bash
#
# TDC GRPO Local Testing Script (Single GPU)
#
# Usage:
#   bash scripts/test_grpo_tdc_local.sh <task_name> <model_path> [learning_rate]
#
# Examples:
#   bash scripts/test_grpo_tdc_local.sh AMES internlm/internlm2_5-7b-chat
#   bash scripts/test_grpo_tdc_local.sh Tox21 /path/to/model 1e-6

set -euo pipefail

############################
#   CONFIGURATION          #
############################

# Parse arguments
TASK_NAME=${1:-"Tox21"}
PRETRAIN_PATH=${2:-"internlm/internlm2_5-7b-chat"}
LEARNING_RATE=${3:-"1e-6"}
NUM_GPUS=1

# TDC dataset paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format"
TRAIN_DATA="$DATA_DIR/${TASK_NAME}_train.jsonl"

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
RUN_ID="test-${TASK_NAME}_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="$PROJECT_ROOT/saves/test/$RUN_ID"
CKPT_PATH="$PROJECT_ROOT/checkpoints/test/$RUN_ID"

# Reduced hyperparameters for local testing
TRAIN_BATCH_SIZE=8
MICRO_TRAIN_BATCH_SIZE=4
MICRO_ROLLOUT_BATCH_SIZE=8
MAX_SAMPLES=50  # Small for quick testing

# Tool-calling configuration
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_agent.py"
AGENT_MAX_STEPS=20  # Reduced for testing
PROMPT_CONSTRUCTION_MODE="manual"

# GRPO configuration
N_SAMPLES_PER_PROMPT=4  # Reduced for testing

############################
#   ENVIRONMENT SETUP      #
############################

# Ray configuration
export RAY_TMPDIR="/tmp/ray_${USER}_test"
mkdir -p "$RAY_TMPDIR"
PERSIST_RAY_DIR="$PROJECT_ROOT/logs/ray/test_latest"

# vLLM configuration
export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1

# OpenRLHF environment variables
export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
export OPENRLHF_PROMPT_CONSTRUCTION_MODE="$PROMPT_CONSTRUCTION_MODE"
export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"

############################
#   RAY SETUP              #
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
}
trap copy_ray_logs EXIT

# Get node IP
export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}' || echo "127.0.0.1")

# Clean up previous Ray state
ray stop --force 2>/dev/null || true

# Start Ray
echo "Starting Ray head node at $RAY_NODE_IP_ADDRESS"
ray start --head \
    --node-ip-address "$RAY_NODE_IP_ADDRESS" \
    --num-gpus $NUM_GPUS \
    --temp-dir "$RAY_TMPDIR" &

# Wait for Ray
echo "Waiting for Ray to be ready..."
for i in {1..30}; do
    curl -fsS http://127.0.0.1:8265/api/version >/dev/null 2>&1 && break
    sleep 1
done

############################
#   PRINT CONFIGURATION    #
############################

echo "========================================"
echo "TDC GRPO Local Test Configuration"
echo "========================================"
echo "Task: $TASK_NAME"
echo "Model: $PRETRAIN_PATH"
echo "Learning Rate: $LEARNING_RATE"
echo "Run ID: $RUN_ID"
echo "----------------------------------------"
echo "Training Data: $TRAIN_DATA"
echo "Save Path: $SAVE_PATH"
echo "----------------------------------------"
echo "NUM_GPUS: $NUM_GPUS"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "MAX_SAMPLES: $MAX_SAMPLES"
echo "Agent Max Steps: $AGENT_MAX_STEPS"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "========================================"

############################
#   TRAINING COMMAND       #
############################

python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 0 \
    --ref_num_gpus_per_node 0 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $NUM_GPUS \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.8 \
    --advantage_estimator dr_grpo \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --pretrain "$PRETRAIN_PATH" \
    --save_path "$SAVE_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps 10 \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
    --micro_rollout_batch_size $MICRO_ROLLOUT_BATCH_SIZE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $TRAIN_BATCH_SIZE \
    --max_epochs 1 \
    --prompt_max_len 4096 \
    --generate_max_len 2048 \
    --max_samples $MAX_SAMPLES \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --input_key messages \
    --label_key answer \
    --apply_chat_template \
    --gradient_checkpointing \
    --enforce_eager \
    --dynamic_filtering \
    --dynamic_filtering_reward_range 0.2 0.8 \
    --top_p 0.95 \
    --temperature 1.0 \
    --agent_func_path "$AGENT_FUNC_PATH" \
    --agent_max_steps $AGENT_MAX_STEPS \
    --vllm_stop_strings "</tool_call>" \
    --prompt_construction_mode "$PROMPT_CONSTRUCTION_MODE"

############################
#   CLEANUP                #
############################

echo "Training complete!"
echo "Stopping Ray..."
ray stop --force || true

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Saved model to: $SAVE_PATH"
echo "Checkpoints at: $CKPT_PATH"
echo "Ray logs at: $PERSIST_RAY_DIR/session_latest"
echo "========================================"
