#!/bin/bash
#
# INTERACTIVE DEBUG version of the GLM-4.7-Flash GRPO training script — DISTRIBUTED (non-colocated).
#
# Actor and vLLM run on separate GPU sets (no colocation).
# Uses the GLM Flash XML tool-calling format:
#   <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
#
# Usage:
#   1. Get an interactive node:  srun --partition=dgx-b200 --gpus=8 --mem-per-gpu=128G --cpus-per-gpu=8 --time=1:00:00 --pty bash
#   2. Activate env:             module load MAMBA && module load cuda/13.1.0 && micromamba activate /vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4
#   3. Run:                      bash scripts/train_grpo_tdc_glm_debug_distributed_adam_offload.sh [model_path] [learning_rate] [vllm_gpu_mem_util]
#
# Example:
#   bash scripts/train_grpo_tdc_glm_debug_distributed_adam_offload.sh zai-org/GLM-4.7-Flash 1e-6 0.12
#

set -euo pipefail
export RAY_TMPDIR=/tmp/jojolee/ray

### ARGS ###
PRETRAIN_PATH=${1:-"zai-org/GLM-4.7-Flash"}
LEARNING_RATE=${2:-"1e-6"}
NUM_GPUS=$SLURM_GPUS_ON_NODE
DEBUG_TRACES=${3:-"0"}
VLLM_GPU_MEMORY_UTILIZATION=${4:-"0.95"}

### MULTI-TASK ###
TASK_NAMES=(Bioavailability_Ma HIA_Hou PAMPA_NCATS Pgp_Broccatelli BBB_Martins CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels CYP3A4_Substrate_CarbonMangels SARSCoV2_3CLPro_Diamond SARSCoV2_Vitro_Touret Carcinogens_Lagunin hERG ClinTox DILI Skin_Reaction AMES)
TASK_LABEL="Base"

### NCCL / IB / NETWORK CONFIG ###
unset NCCL_NVLS_ENABLE
unset NCCL_IB_ADAPTIVE_ROUTING
unset NCCL_IB_SL
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_SPLIT_DATA_ON_QPS
unset UCX_TLS
# Keep
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_14,mlx5_15

# Add for diagnosis/stability
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
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
mkdir -p "$PROJECT_ROOT/logs"

TRAIN_PARTS=()
for t in "${TASK_NAMES[@]}"; do
    f="$DATA_DIR/${t}_train.jsonl"
    if [ ! -f "$f" ]; then
        echo "Error: Training data not found: $f"
        echo "Available tasks:"
        ls "$DATA_DIR" 2>/dev/null | grep "_train.jsonl" | sed 's/_train.jsonl//' | sort
        exit 1
    fi
    TRAIN_PARTS+=("$f")
done
IFS=,; TRAIN_DATA="${TRAIN_PARTS[*]}"; unset IFS

### RUN CONFIG ###
RUN_ID="GLM-grpo-fixed-debug-distributed-${TASK_LABEL}_$(date +%Y-%m-%d_%H-%M-%S)_lr${LEARNING_RATE}"
DATE_STAMP=$(date +%Y%m%d)
RUNS_DIR="$PROJECT_ROOT/runs/${RUN_ID}/${DATE_STAMP}"
mkdir -p "$RUNS_DIR"
SAVE_PATH="$PROJECT_ROOT/saves/tdc/${TASK_LABEL}/$RUN_ID"
HUB_REPO_ID="jiosephlee/grpo-tdc-glm-flash-${TASK_LABEL}"

### GPU LAYOUT (distributed — separate actor and vLLM GPUs) ###
ACTOR_GPUS=4
VLLM_GPUS=4
VLLM_NUM_ENGINES=4
VLLM_TENSOR_PARALLEL_SIZE=1
TRAIN_BATCH_SIZE=32
MIN_GPUS=$((ACTOR_GPUS + VLLM_NUM_ENGINES * VLLM_TENSOR_PARALLEL_SIZE))
if [ "$NUM_GPUS" -lt "$MIN_GPUS" ]; then
    echo "Error: Need at least $MIN_GPUS GPUs for non-colocated run (actor=$ACTOR_GPUS, vLLM=$((VLLM_NUM_ENGINES * VLLM_TENSOR_PARALLEL_SIZE))), got $NUM_GPUS." >&2
    exit 1
fi

### TOOL-CALLING CONFIG ###
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"
AGENT_MAX_STEPS=30
PROMPT_CONSTRUCTION_MODE="auto"
CHAT_PROTOCOL="glm_flash"

### GRPO CONFIG ###
N_SAMPLES_PER_PROMPT=8
ADVANTAGE_ESTIMATOR="group_norm"
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
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID:-$$}_$(date +%s)"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton"
rm -rf "/tmp/torchinductor_${USER}" 2>/dev/null || true
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

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
echo "TDC GRPO Training — GLM-4.7-Flash (DISTRIBUTED INTERACTIVE DEBUG)"
echo "========================================"
echo "Tasks: ${TASK_NAMES[*]}"
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
echo "vLLM GPU memory utilization: $VLLM_GPU_MEMORY_UTILIZATION"
echo "----------------------------------------"
echo "Agent Max Steps: $AGENT_MAX_STEPS"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "Prompt Mode: $PROMPT_CONSTRUCTION_MODE"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "----------------------------------------"
echo "Runs Dir: $RUNS_DIR"
echo "W&B: project=$WANDB_PROJECT group=TDC-GLMFlash-fixed-$TASK_LABEL run=$RUN_ID"
echo "========================================"

### GENERATE PER-TASK TOOLS JSON ###
TDC_TOOLS_JSON="$PROJECT_ROOT/data/tdc/metadata/tools_per_task.json"
python "$PROJECT_ROOT/scripts/generate_tools_json.py" "$TDC_TOOLS_JSON"

### BUILD TDC EVAL DATASET ###
EVAL_DATA="$DATA_DIR/eval_tdc.jsonl"
python -c "
import json, sys
tasks = sys.argv[1:]
with open('$EVAL_DATA', 'w') as out:
    for task in tasks:
        with open(f'$DATA_DIR/{task}_val.jsonl') as f:
            for line in f:
                rec = json.loads(line)
                rec['datasource'] = task
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
print(f'Built TDC eval dataset: {sum(1 for _ in open(\"$EVAL_DATA\"))} samples from {len(tasks)} tasks')
" "${TASK_NAMES[@]}"

### TRAINING ###
RUN_LOG="$RUNS_DIR/run.log"
echo "Logging to: $RUN_LOG"
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
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps -1 \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size 1 \
    --micro_rollout_batch_size 2 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $TRAIN_BATCH_SIZE \
    --max_epochs 1 \
    --prompt_max_len 6144 \
    --generate_max_len 2048 \
    --max_samples 1000000 \
    --enable_prefix_caching \
    --zero_stage 2 \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps 25 \
    --eval_temperature $TEMPERATURE \
    --eval_n_samples_per_prompt 1 \
    --input_key messages \
    --label_key answer \
    --apply_chat_template \
    --tdc_tools "$TDC_TOOLS_JSON" \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend gloo \
    --async_train \
    --async_queue_size 1 \
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
    --wandb_group "TDC-GLMFlash-fixed-$TASK_LABEL" \
    --wandb_run_name "$RUN_ID" \
    --save_path "$SAVE_PATH" \
    --push_to_hub "$HUB_REPO_ID" \
    --delete_local_after_push \
    --use_dynamic_batch \
    --adam_offload \
    2>&1 | tee "$RUN_LOG"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
ray stop --force || true
