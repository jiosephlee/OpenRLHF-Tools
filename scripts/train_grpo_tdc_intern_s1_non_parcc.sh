#!/bin/bash
#
# Intern-S1 GRPO training — unified interactive script.
#
# Supports both colocated and distributed modes via MODE env var.
#
# Uses the Intern-S1 JSON tool-calling format:
#   <|action_start|><|plugin|>{"name": "...", "parameters": {...}}<|action_end|>
#
# Usage:
#   # Colocated (default — actor and vLLM share GPUs via sleep mode):
#   SMART_REPLAY=1 MAX_REPLAY_ROUNDS=3 REDUCE_OPTIMIZER=adam_8bit bash train_grpo_tdc_intern_s1.sh
#
#   # Distributed (actor and vLLM on separate GPUs):
#   MODE=distributed ACTOR_GPUS=2 VLLM_NUM_ENGINES=6 SMART_REPLAY=1 MAX_REPLAY_ROUNDS=3 bash train_grpo_tdc_intern_s1.sh
#
#   # Distributed with extra flags:
#   MODE=distributed ACTOR_GPUS=2 VLLM_NUM_ENGINES=6 SMART_REPLAY=1 \
#     EXTRA_ARGS="--skip_eval_step_zero" bash scripts/train_grpo_tdc_intern_s1.sh
#
# Feature flags (all env-configurable):
#   MODE=colocated|distributed           # Default: colocated
#   EFFECTIVE_ROLLOUT_BATCH_SIZE=8       # Rollout batch size in distributed/async mode
#   EFFECTIVE_MINI_GRADIENT_STEPS=2      # Mini gradient steps in distributed/async mode
#   ASYNC_ADVANTAGE=4                    # Scale factor: colocated uses ASYNC_ADVANTAGE * EFFECTIVE_* for both
#                                        # ROLLOUT and MINI, keeping ROLLOUT/MINI ratio constant across modes.
#   COLO_EVAL_STEPS=32                   # Eval frequency (global steps) for colocated; distributed scales by ASYNC_ADVANTAGE
#   TOOL_VERSION=v4                      # Tool schema version (default: v4)
#   SMART_REPLAY=1                       # Enable smart replay with max_replay_rounds=2
#   CURRICULUM_BALANCED=1                # Enable curriculum-balanced sampling

#   LIGER_GRPO_LOSS=1                    # Enable Liger fused GRPO loss
#   TIS=1                                # Enable Truncated Importance Sampling (off-policy correction)
#   TIS_TYPE=tis                         # TIS variant: tis (default), icepop, seq-mask-tis
#   TIS_THRESHOLDS="0.5 5.0"            # Low and high clamp thresholds (default: 0.5 5.0)
#   GSPO=1                               # Use GSPO loss (sequence-level IS ratio) instead of PPO
#   REDUCE_OPTIMIZER=adam_offload        # Optimizer: adam_offload (default) or adam_8bit
#   MAX_EPOCHS=2                         # Training epochs (default: 1)
#   EXTRA_ARGS="..."                     # Additional CLI flags
#

set -euo pipefail
export DS_SKIP_CUDA_CHECK=1

### ARGS ###
PRETRAIN_PATH=${1:-"jiosephlee/sft_intern_distillation_Intern-S1-mini-lm_complet_only_chat_think_lr5e-05"}
LEARNING_RATE=${2:-"1e-6"}
NUM_GPUS=8
DEBUG_TRACES=${3:-"0"}

### FEATURE FLAGS ###
MODE="${MODE:-colocated}"
EFFECTIVE_ROLLOUT_BATCH_SIZE="${EFFECTIVE_ROLLOUT_BATCH_SIZE:-8}"
EFFECTIVE_MINI_GRADIENT_STEPS="${EFFECTIVE_MINI_GRADIENT_STEPS:-2}"
ASYNC_ADVANTAGE="${ASYNC_ADVANTAGE:-4}"
TOOL_VERSION="${TOOL_VERSION:-v4}"
SMART_REPLAY="${SMART_REPLAY:-0}"
MAX_REPLAY_ROUNDS="${MAX_REPLAY_ROUNDS:-2}"

LIGER_GRPO_LOSS="${LIGER_GRPO_LOSS:-0}"
CURRICULUM_BALANCED="${CURRICULUM_BALANCED:-0}"
TIS="${TIS:-0}"
TIS_TYPE="${TIS_TYPE:-tis}"
TIS_THRESHOLDS="${TIS_THRESHOLDS:-0.5 5.0}"
GSPO="${GSPO:-0}"
REDUCE_OPTIMIZER="${REDUCE_OPTIMIZER:-adam_offload}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="${VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE:-}"

### UNIFIED CONSTANTS ###
AGENT_MAX_STEPS=30
ZERO_STAGE=2
PROMPT_MAX_LEN=12288 # Any responses longer than this will be truncated.
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
TRAIN_MAX_TOKENS_PER_GPU=8192 # Used with dynamic batching; Increasing this will increase the memory usage of the actor, and increase the speed of the training by reducing gradient accumulation steps.
ROLLOUT_MAX_TOKENS_PER_GPU=$(echo "$TRAIN_MAX_TOKENS_PER_GPU * 1.25" | bc | awk '{print int($1)}')

COLO_EVAL_STEPS="${COLO_EVAL_STEPS:-32}"  # Eval frequency for colocated; distributed multiplies by ASYNC_ADVANTAGE.

### MODE-DEPENDENT DEFAULTS ###
if [ "$MODE" = "colocated" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-$NUM_GPUS}"
    ROLLOUT_BATCH_SIZE=$(( EFFECTIVE_ROLLOUT_BATCH_SIZE * ASYNC_ADVANTAGE ))
    MINI_GRADIENT_STEPS=$(( EFFECTIVE_MINI_GRADIENT_STEPS * ASYNC_ADVANTAGE ))
    VLLM_GPU_MEM_UTIL=0.675
    VLLM_SYNC_BACKEND=nccl
    EVAL_STEPS="${EVAL_STEPS:-$COLO_EVAL_STEPS}"
elif [ "$MODE" = "distributed" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:?"MODE=distributed requires ACTOR_GPUS"}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:?"MODE=distributed requires VLLM_NUM_ENGINES"}"
    ROLLOUT_BATCH_SIZE=$EFFECTIVE_ROLLOUT_BATCH_SIZE  # Distributed uses the effective value directly; smaller than colocated to be more on-policy (only 1-step async lag).
    MINI_GRADIENT_STEPS=$EFFECTIVE_MINI_GRADIENT_STEPS  # Fewer mini gradient steps to match; ROLLOUT/MINI ratio is identical to colocated, same total gradient steps.
    VLLM_GPU_MEM_UTIL=0.975
    VLLM_SYNC_BACKEND=gloo
    EVAL_STEPS="${EVAL_STEPS:-$(( COLO_EVAL_STEPS * ASYNC_ADVANTAGE ))}"  # Distributed takes ASYNC_ADVANTAGE more global steps per colocated step, so scale eval frequency accordingly.
else
    echo "Error: MODE must be 'colocated' or 'distributed', got '$MODE'" >&2
    exit 1
fi

### BATCH SIZE DERIVATION ###
TRAIN_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / MINI_GRADIENT_STEPS ))

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
    MODE_FLAGS="--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep --$REDUCE_OPTIMIZER"
else
    MODE_FLAGS="--async_train --async_queue_size 1 --$REDUCE_OPTIMIZER"
fi
#
### WARMUP LOGIC ###
WARMUP_STEPS=10
WARM_STEPS_MULTIPLIER=$(( EFFECTIVE_MINI_GRADIENT_STEPS * ASYNC_ADVANTAGE ))

### MULTI-TASK ###
#TASK_NAMES=(Bioavailability_Ma HIA_Hou PAMPA_NCATS Pgp_Broccatelli BBB_Martins CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels CYP3A4_Substrate_CarbonMangels SARSCoV2_3CLPro_Diamond SARSCoV2_Vitro_Touret Carcinogens_Lagunin hERG ClinTox DILI Skin_Reaction AMES)
TASK_NAMES=(BBB_Martins)
TASK_LABEL="Base"

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
DATE_TAG=$(date +%m%d_%H%M)
CHAT_PROTOCOL="intern_s1"

# Build suffix tags for active features
SUFFIX=""
[ "$SMART_REPLAY" = "1" ] && SUFFIX+="-sr${MAX_REPLAY_ROUNDS}"
[ "$GSPO" = "1" ] && SUFFIX+="-gspo"
[ "$TIS" = "1" ] && SUFFIX+="-tis"
[ "$USE_LORA" = "1" ] && SUFFIX+="-lora"

if [ "$MODE" = "colocated" ]; then
    RUN_NAME="grpo-tdc-s1-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}${SUFFIX}-${DATE_TAG}"
    WANDB_GROUP="TDC-InternS1-colo-$TASK_LABEL"
else
    RUN_NAME="grpo-tdc-s1-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}${SUFFIX}-dist-${LAYOUT_TAG}-${DATE_TAG}"
    WANDB_GROUP="TDC-InternS1-dist-${LAYOUT_TAG}-$TASK_LABEL"
fi
RUN_ID="${RUN_NAME}"
HUB_NAME="grpo-tdc-s1-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-${DATE_TAG}"
source "$PROJECT_ROOT/scripts/lib/resolve_runs_dir.sh"
LOCAL_SAVE_DIR="${LOCAL_SAVE_DIR:-/vast/projects/myatskar/design-documents/hf_home}"
SAVE_PATH="$LOCAL_SAVE_DIR/$RUN_NAME"
HUB_REPO_ID="jiosephlee/${HUB_NAME}"

### TOOL-CALLING CONFIG ###
AGENT_FUNC_PATH="${AGENT_FUNC_PATH:-$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py}"

### GRPO CONFIG ###
ADVANTAGE_ESTIMATOR="group_norm"
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0 1"

WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"
TEMPERATURE=1.0
TOP_P=0.95

### ENVIRONMENT VARIABLES ###
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"

export TRITON_CACHE_DIR="/tmp/triton_${USER}"
mkdir -p "$TRITON_CACHE_DIR"

export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # vLLM v1 msgspec can't serialize torch.dtype; fall back to pickle

export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
export OPENRLHF_CHAT_PROTOCOL="$CHAT_PROTOCOL"
export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"
export DEBUG_TRACES="$DEBUG_TRACES"
export OPENRLHF_DEBUG_LOGITS=0
export OPENRLHF_DEBUG_NAN_GUARD=0

### RAY ###
export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
ulimit -n 65535 2>/dev/null || true

# Use the conda env's ray to avoid version mismatch with the training script's ray
CONDA_RAY="$(which python) -m ray.scripts.scripts"
echo "Using ray from: $(which python)"

$CONDA_RAY stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/ray/session_* 2>/dev/null || true

echo "Starting Ray head node at $RAY_NODE_IP_ADDRESS"
$CONDA_RAY start --head \
    --node-ip-address "$RAY_NODE_IP_ADDRESS" \
    --num-gpus "$NUM_GPUS" \
    --temp-dir "$RAY_TMPDIR"

# Set explicit address immediately — avoids "multiple active Ray instances" from other users
export RAY_ADDRESS="$RAY_NODE_IP_ADDRESS:6379"

echo "Waiting for Ray..."
RAY_READY=0
for i in {1..90}; do
    if $CONDA_RAY status >/dev/null 2>&1; then
        RAY_READY=1
        break
    fi
    sleep 1
done
if [ "$RAY_READY" -ne 1 ]; then
    echo "Error: Ray never came up after 90 seconds." >&2
    exit 1
fi
$CONDA_RAY status
echo "Ray is ready."

### PRINT CONFIG ###
echo "========================================"
echo "TDC GRPO Training — Intern-S1-mini (MODE=$MODE)"
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

echo "Liger GRPO Loss: $LIGER_GRPO_LOSS"
echo "TIS: $TIS (type=$TIS_TYPE, thresholds=$TIS_THRESHOLDS)"
echo "GSPO: $GSPO"
echo "VLLM_MAX_NUM_SEQS: $VLLM_MAX_NUM_SEQS"
echo "VLLM_MAX_NUM_BATCHED_TOKENS: $VLLM_MAX_NUM_BATCHED_TOKENS"
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
    OPTIONAL_FLAGS+=" --smart_replay --max_replay_rounds $MAX_REPLAY_ROUNDS"
fi
if [ "$CURRICULUM_BALANCED" = "1" ]; then
    OPTIONAL_FLAGS+=" --curriculum_balanced"
fi

if [ "$LIGER_GRPO_LOSS" = "1" ]; then
    OPTIONAL_FLAGS+=" --use_liger_grpo_loss"
fi
if [ "$TIS" = "1" ]; then
    OPTIONAL_FLAGS+=" --enable_vllm_is_correction --vllm_is_correction_type $TIS_TYPE --vllm_is_truncated_threshold $TIS_THRESHOLDS"
fi
if [ "$GSPO" = "1" ]; then
    OPTIONAL_FLAGS+=" --policy_loss_type gspo"
fi
if [ -n "$VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE" ]; then
    OPTIONAL_FLAGS+=" --vllm_cudagraph_max_capture_size $VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE"
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
    --max_num_batched_tokens $VLLM_MAX_NUM_BATCHED_TOKENS \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEM_UTIL \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.282 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --logging_steps 1 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
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
    --eval_temperature 0.1 \
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
    --vllm_stop_strings "<|action_end|>" "<|im_end|>" \
    --vllm_max_num_seqs $VLLM_MAX_NUM_SEQS \
    --chat_protocol "$CHAT_PROTOCOL" \
    --use_wandb 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group "$WANDB_GROUP" \
    --wandb_run_name "$RUN_ID" \
    --save_path "$SAVE_PATH" \
    --push_to_hub "$HUB_REPO_ID" \
    --delete_local_after_push \
    --use_liger_kernel \
    --use_dynamic_batch \
    --train_max_tokens_per_gpu $TRAIN_MAX_TOKENS_PER_GPU \
    --rollout_max_tokens_per_gpu $ROLLOUT_MAX_TOKENS_PER_GPU \
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
