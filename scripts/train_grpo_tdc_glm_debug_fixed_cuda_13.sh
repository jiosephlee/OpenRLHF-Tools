#!/bin/bash
#
# INTERACTIVE DEBUG version of the GLM-4.7-Flash GRPO training script (FIXED, CUDA 13).
#
# Hybrid (colocated) mode — Actor and vLLM share the same GPUs via sleep mode.
# Uses the GLM Flash XML tool-calling format:
#   <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
#
# Usage:
#   1. Get an interactive node:  srun --partition=dgx-b200 --gpus=2 --mem-per-gpu=128G --cpus-per-gpu=4 --time=1:00:00 --pty bash
#   2. Activate env:             module load MAMBA && module load cuda/13.1.0 && micromamba activate /vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4
#   3. Run:                      bash scripts/train_grpo_tdc_glm_debug_fixed_cuda_13.sh <task_name> [model_path] [learning_rate]
#
# Example:
#   bash scripts/train_grpo_tdc_glm_debug_fixed_cuda_13.sh AMES zai-org/GLM-4.7-Flash 1e-6
#

set -euo pipefail

### ARGS ###
TASK_NAME=${1:-"BBB_Martins"}
PRETRAIN_PATH=${2:-"zai-org/GLM-4.7-Flash"}
LEARNING_RATE=${3:-"2e-6"}
NUM_GPUS=$SLURM_GPUS_ON_NODE
DEBUG_TRACES=${4:-"0"}

# ### NCCL / IB / NETWORK CONFIG ###
# export OMP_NUM_THREADS=16
# export NCCL_NVLS_ENABLE=1
# export NCCL_IB_ADAPTIVE_ROUTING=1
# export NCCL_IB_SL=1
# export NCCL_IB_QPS_PER_CONNECTION=2
# export NCCL_IB_SPLIT_DATA_ON_QPS=0
# export NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
# export NCCL_SOCKET_IFNAME=bond0
# export UCX_TLS=rc

### W&B ###
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set." >&2
    exit 1
fi

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
RUN_ID="GLM-grpo-fixed-debug-${TASK_NAME}_$(date +%Y-%m-%d_%H-%M-%S)_lr${LEARNING_RATE}"
SAVE_PATH="$PROJECT_ROOT/saves/tdc/${TASK_NAME}/$RUN_ID"
HUB_REPO_ID="jiosephlee/grpo-tdc-glm-flash-${TASK_NAME}"

### GPU LAYOUT (colocated — shared GPUs) ###
TRAIN_BATCH_SIZE=$((NUM_GPUS * 4))
VLLM_NUM_ENGINES=$((NUM_GPUS / 2))

### TOOL-CALLING CONFIG ###
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"
AGENT_MAX_STEPS=40
PROMPT_CONSTRUCTION_MODE="auto"
CHAT_PROTOCOL="glm_flash"

### GRPO CONFIG ###
N_SAMPLES_PER_PROMPT=8
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
export DEBUG_TRACES="$DEBUG_TRACES"
export OPENRLHF_DEBUG_LOGITS=0
export OPENRLHF_DEBUG_NAN_GUARD=0

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
echo "TDC GRPO Training — GLM-4.7-Flash (FIXED INTERACTIVE DEBUG)"
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
echo "NUM_GPUS: $NUM_GPUS (colocated — shared between actor and vLLM)"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "VLLM_NUM_ENGINES: $VLLM_NUM_ENGINES"
echo "----------------------------------------"
echo "Agent Max Steps: $AGENT_MAX_STEPS"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "Prompt Mode: $PROMPT_CONSTRUCTION_MODE"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "----------------------------------------"
echo "masked_mean debug dir: $OPENRLHF_MASKED_MEAN_DEBUG_DIR"
echo "W&B: project=$WANDB_PROJECT group=TDC-GLMFlash-fixed-$TASK_NAME run=$RUN_ID"
echo "========================================"

### GENERATE PER-TASK TOOLS JSON ###
TDC_TOOLS_JSON="$PROJECT_ROOT/data/tdc/metadata/tools_per_task.json"
python "$PROJECT_ROOT/scripts/generate_tools_json.py" "$TDC_TOOLS_JSON"

### TRAINING ###
python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$PRETRAIN_PATH" \
    --ref_num_nodes 0 \
    --ref_num_gpus_per_node 0 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $NUM_GPUS \
    --vllm_num_engines $VLLM_NUM_ENGINES \
    --vllm_tensor_parallel_size 2 \
    --colocate_all_models \
    --vllm_gpu_memory_utilization 0.835 \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps -1 \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size 2 \
    --micro_rollout_batch_size 8 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $TRAIN_BATCH_SIZE \
    --max_epochs 1 \
    --prompt_max_len 8192 \
    --generate_max_len 2048 \
    --max_samples 1000000 \
    --zero_stage 2 \
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
    --vllm_stop_strings "</tool_call>" \
    --prompt_construction_mode "$PROMPT_CONSTRUCTION_MODE" \
    --chat_protocol "$CHAT_PROTOCOL" \
    --use_wandb 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group "TDC-GLMFlash-fixed-$TASK_NAME" \
    --wandb_run_name "$RUN_ID" \
    --save_path "$SAVE_PATH" \
    --push_to_hub "$HUB_REPO_ID" \
    --delete_local_after_push \
    --rollout_trace_dir "$SAVE_PATH/rollout_traces"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
ray stop --force || true
