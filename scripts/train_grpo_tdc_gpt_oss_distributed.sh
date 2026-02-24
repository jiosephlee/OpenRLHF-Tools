#!/bin/bash
#
# GPT-OSS GRPO training — DISTRIBUTED (configurable actor + vLLM GPUs).
#
# Distributed (non-colocated) mode — Actor and vLLM run on separate GPU sets.
# When ACTOR_GPUS > 1, DeepSpeed AutoTP is enabled automatically (TP=ACTOR_GPUS).
#
# Uses GPT-OSS Harmony tool-calling format:
#   <|start|>assistant to=functions.<name><|channel|>commentary json<|message|>...
#
# Usage:
#   bash scripts/train_grpo_tdc_gpt_oss_distributed.sh <actor_gpus> <vllm_engines> [model_path] [learning_rate]
#
# Examples:
#   bash scripts/train_grpo_tdc_gpt_oss_distributed.sh 1 3   # 4 GPUs, no AutoTP
#   bash scripts/train_grpo_tdc_gpt_oss_distributed.sh 2 6   # 8 GPUs, AutoTP with TP=2
#

set -euo pipefail
export RAY_TMPDIR=/tmp/jojolee/ray
export MALLOC_TRIM_THRESHOLD_=0
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

### PRIMARY KNOBS — change these two ###
ACTOR_GPUS=${1:?"Usage: $0 <actor_gpus> <vllm_engines> [model_path] [learning_rate]"}
VLLM_NUM_ENGINES=${2:?"Usage: $0 <actor_gpus> <vllm_engines> [model_path] [learning_rate]"}

### ARGS ###
PRETRAIN_PATH=${3:-"openai/gpt-oss-20b"}
LEARNING_RATE=${4:-"1e-6"}
DEBUG_TRACES=${5:-"0"}
NUM_GPUS=$SLURM_GPUS_ON_NODE

### DERIVED GPU LAYOUT ###
VLLM_GPUS=$VLLM_NUM_ENGINES
VLLM_TENSOR_PARALLEL_SIZE=1
TRAIN_BATCH_SIZE=16
ROLLOUT_BATCH_SIZE=4

# Layout tag used in run names and W&B
LAYOUT_TAG="${ACTOR_GPUS}a${VLLM_GPUS}v"

### AUTOTP CONFIG (enabled when ACTOR_GPUS > 1) ###
RING_ATTN_SIZE=1
RING_HEAD_STRIDE=8
if [ "$ACTOR_GPUS" -gt 1 ]; then
    DS_TP_SIZE=$ACTOR_GPUS
else
    DS_TP_SIZE=1
fi

### MULTI-TASK ###
TASK_NAMES=(Bioavailability_Ma HIA_Hou PAMPA_NCATS Pgp_Broccatelli BBB_Martins CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels CYP3A4_Substrate_CarbonMangels SARSCoV2_3CLPro_Diamond SARSCoV2_Vitro_Touret Carcinogens_Lagunin hERG ClinTox DILI Skin_Reaction AMES)
TASK_LABEL="Base"

### GPU CHECK ###
MIN_GPUS=$((ACTOR_GPUS + VLLM_GPUS))
if [ "$NUM_GPUS" -lt "$MIN_GPUS" ]; then
    echo "Error: Need at least $MIN_GPUS GPUs ($ACTOR_GPUS actor + $VLLM_GPUS vLLM), got $NUM_GPUS" >&2
    exit 1
fi

### NCCL / IB / NETWORK CONFIG ###
export OMP_NUM_THREADS=16
export NCCL_NVLS_ENABLE=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_SL=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
export NCCL_SOCKET_IFNAME=bond0
export UCX_TLS=rc

# Add for diagnosis/stability
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

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
N_TASKS=${#TASK_NAMES[@]}
MAX_EPOCHS=1
TOOL_VERSION="${TOOL_VERSION:-v3}"
DATE_TAG=$(date +%m%d)
RUN_NAME="grpo-tdc-gptoss-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-dist-${LAYOUT_TAG}-${DATE_TAG}"
RUN_ID="${RUN_NAME}"
# HF repo name excludes GPU layout so the same model name works across configs
HUB_NAME="grpo-tdc-gptoss-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-${DATE_TAG}"
DATE_STAMP=$(date +%Y%m%d)
RUNS_DIR="$PROJECT_ROOT/runs/${RUN_NAME}/${DATE_STAMP}"
mkdir -p "$RUNS_DIR"
SAVE_PATH="$PROJECT_ROOT/saves/tdc/$RUN_NAME"
HUB_REPO_ID="jiosephlee/${HUB_NAME}"

### TOOL-CALLING CONFIG ###
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"
AGENT_MAX_STEPS=30
CHAT_PROTOCOL="gpt_oss"

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
export OPENRLHF_CHAT_PROTOCOL="$CHAT_PROTOCOL"
export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"
export DEBUG_TRACES="$DEBUG_TRACES"
export OPENRLHF_DEBUG_LOGITS=0
export OPENRLHF_DEBUG_NAN_GUARD=0

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
echo "TDC GRPO Training — GPT-OSS (DISTRIBUTED, ${LAYOUT_TAG})"
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
echo "ROLLOUT_BATCH_SIZE: $ROLLOUT_BATCH_SIZE"
echo "VLLM_NUM_ENGINES: $VLLM_NUM_ENGINES"
if [ "$DS_TP_SIZE" -gt 1 ]; then
    echo "DS Tensor Parallel Size: $DS_TP_SIZE"
    echo "Device Mesh: (dp=$((ACTOR_GPUS / RING_ATTN_SIZE / DS_TP_SIZE)), sp=$RING_ATTN_SIZE, tp=$DS_TP_SIZE)"
fi
echo "----------------------------------------"
echo "Agent Max Steps: $AGENT_MAX_STEPS"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "----------------------------------------"
echo "Runs Dir: $RUNS_DIR"
echo "W&B: project=$WANDB_PROJECT group=TDC-GPTOss-dist-${LAYOUT_TAG}-$TASK_LABEL run=$RUN_ID"
echo "========================================"

### GENERATE PER-TASK TOOLS JSON ###
TDC_TOOLS_JSON="$PROJECT_ROOT/data/tdc/metadata/tools_per_task_${TOOL_VERSION}.json"
python "$PROJECT_ROOT/scripts/generate_tools_json.py" --version "$TOOL_VERSION"

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

### BUILD AUTOTP FLAGS ###
AUTOTP_FLAGS=""
if [ "$DS_TP_SIZE" -gt 1 ]; then
    AUTOTP_FLAGS="--ring_attn_size $RING_ATTN_SIZE --ring_head_stride $RING_HEAD_STRIDE --ds_tensor_parallel_size $DS_TP_SIZE"
fi

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
    --vllm_gpu_memory_utilization 0.975 \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps_ratio 0.5 \
    --save_hf_ckpt \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size 1 \
    --micro_rollout_batch_size 2 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --num_episodes $MAX_EPOCHS \
    --prompt_max_len 12288 \
    --generate_max_len 2048 \
    --max_samples 1000000 \
    --enable_prefix_caching \
    --zero_stage 2 \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps 64 \
    --eval_temperature $TEMPERATURE \
    --eval_n_samples_per_prompt 1 \
    --input_key messages \
    --label_key answer \
    --apply_chat_template \
    --tdc_tools "$TDC_TOOLS_JSON" \
    --tool_version "$TOOL_VERSION" \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend gloo \
    --mxfp4_dequantize \
    --async_train \
    --async_queue_size 1 \
    $([ "$DYNAMIC_FILTERING" = true ] && echo "--dynamic_filtering --dynamic_filtering_reward_range $DYNAMIC_FILTERING_REWARD_RANGE" || echo "") \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    --agent_func_path "$AGENT_FUNC_PATH" \
    --agent_max_steps $AGENT_MAX_STEPS \
    --vllm_stop_strings "<|end|>" \
    --chat_protocol "$CHAT_PROTOCOL" \
    --use_wandb 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group "TDC-GPTOss-dist-${LAYOUT_TAG}-$TASK_LABEL" \
    --wandb_run_name "$RUN_ID" \
    --save_path "$SAVE_PATH" \
    --push_to_hub "$HUB_REPO_ID" \
    --delete_local_after_push \
    --constant_lr_with_warm_up \
    --skip_eval_step_zero \
    --use_dynamic_batch \
    --train_max_tokens_per_gpu 16384 \
    --adam_offload \
    --attn_implementation eager \
    $AUTOTP_FLAGS \
    2>&1 | tee "$RUN_LOG"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
ray stop --force || true
