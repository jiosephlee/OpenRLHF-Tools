#!/bin/bash
#
# TDC GRPO Training Script — DISTRIBUTED (non-hybrid) mode
#
# Actor and vLLM run on SEPARATE GPUs (no colocation, no sleep modes).
# Modeled after examples/scripts/train_ppo_ray_slurm.sh.
#
# Usage:
#   export WANDB_API_KEY=...   # required for wandb tracking
#   # SLURM: sbatch scripts/train_grpo_tdc_distributed.sh <task_name> <model_path> [learning_rate]
#   # Direct: bash scripts/train_grpo_tdc_distributed.sh <task_name> <model_path> [learning_rate] [num_gpus]
#
# Examples:
#   sbatch scripts/train_grpo_tdc_distributed.sh AMES zai-org/GLM-4.7-Flash
#   bash scripts/train_grpo_tdc_distributed.sh hERG /path/to/glm-flash 1e-6 4
#

### SLURM DIRECTIVES ###
#SBATCH --job-name=D-grpo
#SBATCH --partition=dgx-b200
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8                    # 2 actor + 2 vLLM
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=0:30:00
#SBATCH --exclude=dgx011

set -euo pipefail
export RAY_TMPDIR=/tmp/jojolee/ray

############################
#   CONFIGURATION          #
############################

# Detect SLURM vs standalone
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
    IS_SLURM=true
    NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
else
    echo "Running in standalone mode"
    IS_SLURM=false
    NUM_GPUS=${4:-4}
fi

# GPU split: actor (training) vs vLLM (inference)
ACTOR_GPUS=2
VLLM_GPUS=2
TOTAL_REQUIRED=$((ACTOR_GPUS + VLLM_GPUS))
if [ "$NUM_GPUS" -lt "$TOTAL_REQUIRED" ]; then
    echo "Error: need at least $TOTAL_REQUIRED GPUs (${ACTOR_GPUS} actor + ${VLLM_GPUS} vLLM), got $NUM_GPUS" >&2
    exit 1
fi

# Parse arguments
TASK_NAME=${1:-"AMES"}
PRETRAIN_PATH=${2:-"zai-org/GLM-4.7-Flash"}
LEARNING_RATE=${3:-"1e-6"}

# Resolve PROJECT_ROOT
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

# Data paths
DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format"
TRAIN_DATA="$DATA_DIR/${TASK_NAME}_train.jsonl"

mkdir -p "$PROJECT_ROOT/logs"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found: $TRAIN_DATA"
    echo "Available tasks:"
    ls "$DATA_DIR" | grep "_train.jsonl" | sed 's/_train.jsonl//' | sort
    exit 1
fi

# Run ID / paths
RUN_ID="D-grpo-${TASK_NAME}_$(date +%Y-%m-%d_%H-%M-%S)_lr${LEARNING_RATE}"
SAVE_PATH="$PROJECT_ROOT/saves/tdc/${TASK_NAME}/$RUN_ID"
CKPT_PATH="$PROJECT_ROOT/checkpoints/tdc/${TASK_NAME}/$RUN_ID"

# ── Distributed-mode engine layout ──────────────────────────────
# 1 vLLM engine with TP = VLLM_GPUS (each engine gets its own GPUs)
VLLM_NUM_ENGINES=1
VLLM_TENSOR_PARALLEL_SIZE=$VLLM_GPUS

# ── Training hyperparameters ────────────────────────────────────
TRAIN_BATCH_SIZE=$((ACTOR_GPUS * 16))

# ── Tool-calling configuration ──────────────────────────────────
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"
AGENT_MAX_STEPS=50

# ── GRPO configuration ──────────────────────────────────────────
N_SAMPLES_PER_PROMPT=8
ADVANTAGE_ESTIMATOR="dr_grpo"
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0.2 0.8"

# ── W&B (required for tracking) ──────────────────────────────────
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set. Set it for wandb tracking (e.g. export WANDB_API_KEY=...)." >&2
    exit 1
fi
WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"

############################
#   ENVIRONMENT SETUP      #
############################

# Ray temp dir
if [ "$IS_SLURM" = true ]; then
    export RAY_TMPDIR="/tmp/ray_${USER}/${SLURM_JOB_ID}"
else
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
fi
mkdir -p "$RAY_TMPDIR"
PERSIST_RAY_DIR="$PROJECT_ROOT/logs/ray/latest"

# vLLM
export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1

# Agent env vars (also set by vllm_engine.py, but export here for visibility)
export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"

# NCCL — keep debug on until the init issue is resolved
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
# Disable P2P: crashes during ncclCommInitRank for the weight sync group.
export NCCL_P2P_DISABLE=1
# Disable IB: DGX has mixed IB+RoCE NICs that NCCL can't merge.
# Intra-node weight sync only needs SHM, not IB.
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=16

# IB tuning (uncomment / adjust for your cluster fabric)
# export NCCL_IB_ADAPTIVE_ROUTING=1
# export NCCL_IB_SL=1
# export NCCL_IB_QPS_PER_CONNECTION=2
# export NCCL_IB_SPLIT_DATA_ON_QPS=0
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
# export NCCL_SOCKET_IFNAME=bond0

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
    sync; sleep 2

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

export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
ulimit -n 65535 2>/dev/null || true

ray stop --force 2>/dev/null || true

echo "Starting Ray head node at $RAY_NODE_IP_ADDRESS"
ray start --head \
    --node-ip-address "$RAY_NODE_IP_ADDRESS" \
    --num-gpus "$NUM_GPUS" \
    --temp-dir "$RAY_TMPDIR" &

echo "Waiting for Ray to be ready..."
for i in {1..60}; do
    curl -fsS http://127.0.0.1:8265/api/version >/dev/null 2>&1 && break
    sleep 1
done

export RAY_ADDRESS="auto"

############################
#   PRINT CONFIGURATION    #
############################

echo "========================================"
echo "TDC GRPO Training — DISTRIBUTED mode"
echo "========================================"
echo "Task: $TASK_NAME"
echo "Model: $PRETRAIN_PATH"
echo "Learning Rate: $LEARNING_RATE"
echo "Run ID: $RUN_ID"
echo "----------------------------------------"
if [ "$IS_SLURM" = true ]; then
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Node: ${SLURM_NODELIST:-$(hostname)}"
    echo "----------------------------------------"
fi
echo "Training Data: $TRAIN_DATA"
echo "Save Path: $SAVE_PATH"
echo "Checkpoint Path: $CKPT_PATH"
echo "----------------------------------------"
echo "Actor GPUs:  $ACTOR_GPUS  (DeepSpeed ZeRO-2)"
echo "vLLM GPUs:   $VLLM_GPUS  ($VLLM_NUM_ENGINES engine, TP=$VLLM_TENSOR_PARALLEL_SIZE)"
echo "Total GPUs:  $NUM_GPUS"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "----------------------------------------"
echo "Agent Max Steps: $AGENT_MAX_STEPS"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "----------------------------------------"
echo "W&B: project=$WANDB_PROJECT group=TDC-$TASK_NAME run=$RUN_ID"
echo "========================================"

############################
#   GENERATE TOOLS JSON    #
############################
TOOL_VERSION="${TOOL_VERSION:-v3}"
TDC_TOOLS_JSON="$PROJECT_ROOT/data/tdc/metadata/tools_per_task_${TOOL_VERSION}.json"
python "$PROJECT_ROOT/scripts/generate_tools_json.py" --version "$TOOL_VERSION"

############################
#   TRAINING COMMAND       #
############################

python -m openrlhf.cli.train_ppo_ray \
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
    --pretrain "$PRETRAIN_PATH" \
    --save_path "$SAVE_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps 20 \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size 8 \
    --micro_rollout_batch_size 16 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $TRAIN_BATCH_SIZE \
    --max_epochs 1 \
    --prompt_max_len 4096 \
    --generate_max_len 8192 \
    --max_samples 1000000 \
    --zero_stage 2 \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --input_key messages \
    --label_key answer \
    --apply_chat_template \
    --tdc_tools "$TDC_TOOLS_JSON" \
    --tool_version "$TOOL_VERSION" \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --enforce_eager \
    $([ "$DYNAMIC_FILTERING" = true ] && echo "--dynamic_filtering --dynamic_filtering_reward_range $DYNAMIC_FILTERING_REWARD_RANGE" || echo "") \
    --top_p 0.95 \
    --temperature 1.0 \
    --agent_func_path "$AGENT_FUNC_PATH" \
    --agent_max_steps $AGENT_MAX_STEPS \
    --vllm_stop_strings "</tool_call>" \
    --use_wandb 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group "TDC-$TASK_NAME" \
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
echo "Checkpoints at: $CKPT_PATH"
echo "W&B: project=$WANDB_PROJECT group=TDC-$TASK_NAME run=$RUN_ID"
echo "Ray logs at: $PERSIST_RAY_DIR/session_latest"
if [ "$IS_SLURM" = true ]; then
    echo "SLURM output: D-grpo_${SLURM_JOB_ID}.out"
fi
echo "========================================"
