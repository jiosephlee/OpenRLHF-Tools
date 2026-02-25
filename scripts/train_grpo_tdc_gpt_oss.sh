#!/bin/bash
#
# GPT-OSS GRPO training — unified interactive script.
#
# Supports both colocated and distributed modes via MODE env var.
#
# Uses GPT-OSS Harmony tool-calling format:
#   <|start|>assistant to=functions.<name><|channel|>commentary json<|message|>...
#
# Usage:
#   # Colocated (default — actor and vLLM share GPUs via sleep mode):
#   bash scripts/train_grpo_tdc_gpt_oss.sh
#
#   # Distributed (actor and vLLM on separate GPUs):
#   MODE=distributed ACTOR_GPUS=2 VLLM_NUM_ENGINES=6 bash scripts/train_grpo_tdc_gpt_oss.sh
#
#   # Distributed with extra flags:
#   MODE=distributed ACTOR_GPUS=2 VLLM_NUM_ENGINES=6 SMART_REPLAY=1 \
#     EXTRA_ARGS="--skip_eval_step_zero" bash scripts/train_grpo_tdc_gpt_oss.sh
#
# Feature flags (all env-configurable):
#   MODE=colocated|distributed   # Default: colocated
#   TOOL_VERSION=v3              # Tool schema version (default: v3)
#   SMART_REPLAY=1               # Enable smart replay with max_replay_rounds=2
#   CURRICULUM_BALANCED=1        # Enable curriculum-balanced sampling
#   MAX_EPOCHS=2                 # Training epochs (default: 2)
#   EXTRA_ARGS="..."             # Additional CLI flags
#
module load cuda/13.1.0 # running into issues with gpt-oss with cuda 12.8.1
eval "$(conda shell.bash hook)"
conda activate /vast/projects/myatskar/design-documents/conda_env/open_rlhf_intern # This conda env uses torch 2.9.1, and the corresponding flash-attn for cuda 13.1.0, but torch is compiled for cuda 12.8... torch doesn't have pip wheels for 13.1.0 yet; no problems with this for now except for Adam_offload.
set -euo pipefail
export MALLOC_TRIM_THRESHOLD_=0
export DS_SKIP_CUDA_CHECK=1 # Adam_offload checks CUDA version and which version of torch is compiled for it; this is a workaround to skip the CUDA check.
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

### ARGS ###
PRETRAIN_PATH=${1:-"openai/gpt-oss-20b"}
LEARNING_RATE=${2:-"1e-6"}
NUM_GPUS=$SLURM_GPUS_ON_NODE
DEBUG_TRACES=${3:-"0"}

### FEATURE FLAGS ###
MODE="${MODE:-colocated}"
TOOL_VERSION="${TOOL_VERSION:-v3}"
SMART_REPLAY="${SMART_REPLAY:-0}"
CURRICULUM_BALANCED="${CURRICULUM_BALANCED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

### UNIFIED CONSTANTS ###
AGENT_MAX_STEPS=30
ZERO_STAGE=2
PROMPT_MAX_LEN=12288 # Any responses longer than this will be truncated.
N_SAMPLES_PER_PROMPT=8

### MODE-DEPENDENT DEFAULTS ###
if [ "$MODE" = "colocated" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-$NUM_GPUS}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}" # This decides how many prompts are used for each rollout.
    MINI_GRADIENT_STEPS="${MINI_GRADIENT_STEPS:-8}" # This decides how many mini gradient updates are used per rollout; Rollout_batch_size * N_samples_per_prompt / Mini_gradient_steps = number of trajectories used for each gradient update.
    MICRO_TRAIN_BATCH_SIZE=4 # The larger the micro_train_batch_size, the more memory and less gradient accumulation steps for backwards pass.
    MICRO_ROLLOUT_BATCH_SIZE=8 # ^ but for forwards pass. These two parameters are overridden, however, by default since we use dynamic batching.
    VLLM_GPU_MEM_UTIL=0.5
    VLLM_SYNC_BACKEND=nccl
    EVAL_STEPS="${EVAL_STEPS:-32}"
    TRAIN_MAX_TOKENS_PER_GPU=8192 # Used with dynamic batching; Increasing this will increase the memory usage of the actor, and increase the speed of the training by reducing gradient accumulation steps.
    ROLLOUT_MAX_TOKENS_PER_GPU=$((TRAIN_MAX_TOKENS_PER_GPU*2)) # Rollout max tokens per gpu is set to twice the train max tokens per gpu; safe estimate for memory usage during forwards pass.

elif [ "$MODE" = "distributed" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:?"MODE=distributed requires ACTOR_GPUS"}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:?"MODE=distributed requires VLLM_NUM_ENGINES"}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}" # Decrease rollout batch size so that we can actually be more on-policy when we do async by doing less mini-gradient steps and more "on-policy" gradient updates that are off by only 1 step.
    MINI_GRADIENT_STEPS="${MINI_GRADIENT_STEPS:-2}" # Decreasing rollout batch size 4x but we decrease # of mini-gradient steps by 4x -> same number of gradient steps in total as colocated.
    MICRO_TRAIN_BATCH_SIZE=1
    MICRO_ROLLOUT_BATCH_SIZE=2
    VLLM_GPU_MEM_UTIL=0.985
    VLLM_SYNC_BACKEND=gloo
    COLO_ROLLOUT=32; COLO_EVAL=32
    EVAL_STEPS="${EVAL_STEPS:-$(( COLO_EVAL * COLO_ROLLOUT / ROLLOUT_BATCH_SIZE ))}" # To match the evaluation frequency of the colocated mode.
    TRAIN_MAX_TOKENS_PER_GPU=16384
    ROLLOUT_MAX_TOKENS_PER_GPU=$((TRAIN_MAX_TOKENS_PER_GPU*2))
else
    echo "Error: MODE must be 'colocated' or 'distributed', got '$MODE'" >&2
    exit 1
fi

### BATCH SIZE DERIVATION ###
TRAIN_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / MINI_GRADIENT_STEPS ))

# Assert ratio invariant (cross-multiply to avoid float division)
COLO_ROLLOUT=32; COLO_MINI=8; DIST_ROLLOUT=8; DIST_MINI=2
if [ $(( COLO_ROLLOUT * DIST_MINI )) -ne $(( DIST_ROLLOUT * COLO_MINI )) ]; then
    echo "Error: rollout/mini_gradient_steps ratio mismatch" >&2; exit 1
fi

### GPU CHECK (distributed only) ###
if [ "$MODE" = "distributed" ]; then
    VLLM_GPUS=$VLLM_NUM_ENGINES
    MIN_GPUS=$((ACTOR_GPUS + VLLM_GPUS))
    if [ "$NUM_GPUS" -lt "$MIN_GPUS" ]; then
        echo "Error: Need at least $MIN_GPUS GPUs ($ACTOR_GPUS actor + $VLLM_GPUS vLLM), got $NUM_GPUS" >&2
        exit 1
    fi
    LAYOUT_TAG="${ACTOR_GPUS}a${VLLM_GPUS}v"
fi

### AUTOTP (when distributed and ACTOR_GPUS > 1) ###
AUTOTP_FLAGS=""
if [ "$MODE" = "distributed" ] && [ "$ACTOR_GPUS" -gt 1 ]; then
    AUTOTP_FLAGS="--ring_attn_size 1 --ring_head_stride 8 --ds_tensor_parallel_size $ACTOR_GPUS"
fi

### MODE FLAGS ###
if [ "$MODE" = "colocated" ]; then
    MODE_FLAGS="--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep --adam_offload" #gpt-oss always needs adam_offload as it can't fit on 8 GPUs with colocated mode otherwise.
else
    MODE_FLAGS="--async_train --async_queue_size 1 --adam_offload"
fi

### WARMUP LOGIC (gpt_oss always) ###
WARMUP_STEPS=20
WARM_STEPS_MULTIPLIER=$(python -c "print($ROLLOUT_BATCH_SIZE * $N_SAMPLES_PER_PROMPT // $TRAIN_BATCH_SIZE)"); # This is the multipler to account for mini-gradient steps; currently it should always amount to 8 regardless of mode.
if [ "$WARM_STEPS_MULTIPLIER" -ne 8 ]; then
    echo "Error: WARM_STEPS_MULTIPLIER should amount to 8 currently regardless of mode." >&2
    exit 1
fi

### MULTI-TASK ###
TASK_NAMES=(Bioavailability_Ma HIA_Hou PAMPA_NCATS Pgp_Broccatelli BBB_Martins CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels CYP3A4_Substrate_CarbonMangels SARSCoV2_3CLPro_Diamond SARSCoV2_Vitro_Touret Carcinogens_Lagunin hERG ClinTox DILI Skin_Reaction AMES)
TASK_LABEL="Base"

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
DATE_TAG=$(date +%m%d)
CHAT_PROTOCOL="gpt_oss"
if [ "$MODE" = "colocated" ]; then
    RUN_NAME="grpo-tdc-gptoss-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-colo-${DATE_TAG}"
    WANDB_GROUP="TDC-GPTOss-colo-$TASK_LABEL"
else
    RUN_NAME="grpo-tdc-gptoss-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-dist-${LAYOUT_TAG}-${DATE_TAG}"
    WANDB_GROUP="TDC-GPTOss-dist-${LAYOUT_TAG}-$TASK_LABEL"
fi
RUN_ID="${RUN_NAME}"
HUB_NAME="grpo-tdc-gptoss-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-${DATE_TAG}"
RUNS_DIR="$PROJECT_ROOT/runs/${RUN_NAME}"
mkdir -p "$RUNS_DIR"
SAVE_PATH="$PROJECT_ROOT/saves/tdc/$RUN_NAME"
HUB_REPO_ID="jiosephlee/${HUB_NAME}"

### TOOL-CALLING CONFIG ###
AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"

### GRPO CONFIG ###
ADVANTAGE_ESTIMATOR="group_norm"
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0 1"

WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"
TEMPERATURE=0.7
TOP_P=0.95

### ENVIRONMENT VARIABLES ###
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"

export TRITON_CACHE_DIR="/tmp/triton_${USER}"
mkdir -p "$TRITON_CACHE_DIR"

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
echo "TDC GRPO Training — GPT-OSS (MODE=$MODE)"
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
if [ "$MODE" = "colocated" ]; then
    echo "NUM_GPUS: $NUM_GPUS (colocated — shared between actor and vLLM)"
else
    echo "NUM_GPUS: $NUM_GPUS  ACTOR: $ACTOR_GPUS  VLLM: $VLLM_NUM_ENGINES"
fi
echo "ROLLOUT_BATCH_SIZE: $ROLLOUT_BATCH_SIZE"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "VLLM_NUM_ENGINES: $VLLM_NUM_ENGINES"
echo "EVAL_STEPS: $EVAL_STEPS"
if [ -n "$AUTOTP_FLAGS" ]; then
    echo "AutoTP: $AUTOTP_FLAGS"
fi
echo "----------------------------------------"
echo "Agent Max Steps: $AGENT_MAX_STEPS"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Warmup Steps: $WARMUP_STEPS (multiplier: $WARM_STEPS_MULTIPLIER)"
echo "----------------------------------------"
echo "Smart Replay: $SMART_REPLAY"
echo "Curriculum Balanced: $CURRICULUM_BALANCED"
echo "Tool Version: $TOOL_VERSION"
echo "----------------------------------------"
echo "Runs Dir: $RUNS_DIR"
echo "W&B: project=$WANDB_PROJECT group=$WANDB_GROUP run=$RUN_ID"
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

### OPTIONAL FLAGS ###
OPTIONAL_FLAGS=""
if [ "$DYNAMIC_FILTERING" = true ]; then
    OPTIONAL_FLAGS+=" --dynamic_filtering --dynamic_filtering_reward_range $DYNAMIC_FILTERING_REWARD_RANGE"
fi
if [ "$SMART_REPLAY" = "1" ]; then
    OPTIONAL_FLAGS+=" --smart_replay --max_replay_rounds 2"
fi
if [ "$CURRICULUM_BALANCED" = "1" ]; then
    OPTIONAL_FLAGS+=" --curriculum_balanced"
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
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEM_UTIL \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_steps_ratio 0.5 \
    --save_hf_ckpt \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --micro_train_batch_size $MICRO_TRAIN_BATCH_SIZE \
    --micro_rollout_batch_size $MICRO_ROLLOUT_BATCH_SIZE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --num_episodes $MAX_EPOCHS \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len 2048 \
    --max_samples 1000000 \
    --enable_prefix_caching \
    --zero_stage $ZERO_STAGE \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps $EVAL_STEPS \
    --eval_temperature $TEMPERATURE \
    --eval_n_samples_per_prompt 1 \
    --input_key messages \
    --label_key answer \
    --apply_chat_template \
    --tdc_tools "$TDC_TOOLS_JSON" \
    --tool_version "$TOOL_VERSION" \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend $VLLM_SYNC_BACKEND \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    --agent_func_path "$AGENT_FUNC_PATH" \
    --agent_max_steps $AGENT_MAX_STEPS \
    --vllm_stop_strings "<|return|>" "<|call|>" \
    --chat_protocol "$CHAT_PROTOCOL" \
    --use_wandb 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group "$WANDB_GROUP" \
    --wandb_run_name "$RUN_ID" \
    --save_path "$SAVE_PATH" \
    --push_to_hub "$HUB_REPO_ID" \
    --delete_local_after_push \
    --use_dynamic_batch \
    --train_max_tokens_per_gpu $TRAIN_MAX_TOKENS_PER_GPU \
    --mxfp4_dequantize \
    --constant_lr_with_warm_up \
    --warmup_steps $WARMUP_STEPS \
    --warm_steps_multiplier_for_correction $WARM_STEPS_MULTIPLIER \
    $MODE_FLAGS \
    $AUTOTP_FLAGS \
    $OPTIONAL_FLAGS \
    $EXTRA_ARGS \
    2>&1 | tee "$RUN_LOG"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
ray stop --force || true
