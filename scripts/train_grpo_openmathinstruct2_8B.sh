#!/bin/bash
#
# GRPO training for Llama-3.1-8B-Instruct on OpenMathInstruct-2.
#
# Adapted from train_grpo_tdc_gpt_oss_no_tools_knn3.sh with batch sizes and
# hyperparameters matched to NeMo-RL's grpo_math_8B.yaml config:
#   - 64 prompts/step, 32 generations/prompt  → 2048 rollout sequences
#   - train_batch_size 512 (4 mini gradient steps)
#   - max sequence length 4096, lr 3e-7
#   - temperature 1.0, top_p 1.0
#
# Data source: nvidia/OpenMathInstruct-2  (HuggingFace)
# Convert first: python scripts/convert_openmathinstruct2_to_openrlhf.py
#
# Reward: math answer verification via \boxed{} extraction
#         (examples/python/math_reward_func.py)
#
# Supports colocated and distributed modes via MODE env var.
#
# Usage:
#   # Default (colocated, BF16):
#   DEQUANT=unsloth bash scripts/train_grpo_openmathinstruct2_8B.sh
#
#   # With Liger fused loss + DAPO:
#   DEQUANT=unsloth LIGER_GRPO_LOSS=1 LOSS_TYPE=dapo bash scripts/train_grpo_openmathinstruct2_8B.sh
#
#   # Distributed:
#   MODE=distributed ACTOR_GPUS=4 VLLM_NUM_ENGINES=4 bash scripts/train_grpo_openmathinstruct2_8B.sh
#
# Feature flags (all env-configurable):
#   MODE=colocated|distributed           # Default: colocated
#   PRETRAIN_PATH=...                     # Model (default: meta-llama/Llama-3.1-8B-Instruct)
#   EFFECTIVE_ROLLOUT_BATCH_SIZE=16       # Rollout batch size (×ASYNC_ADVANTAGE=64 actual)
#   N_SAMPLES_PER_PROMPT=32              # Generations per prompt
#   EFFECTIVE_MINI_GRADIENT_STEPS=4      # Mini gradient steps
#   ASYNC_ADVANTAGE=4                    # Colocated multiplier for rollout batch
#   SMART_REPLAY=1                       # Enable smart replay
#   LIGER_GRPO_LOSS=1                    # Enable Liger fused GRPO loss
#   LOSS_TYPE=ppo                        # Loss type: ppo, dapo, bnpo, dr_grpo, gspo, cispo, sapo
#   TIS=1                                # Truncated Importance Sampling
#   REDUCE_OPTIMIZER=adam_offload        # Optimizer: adam_offload, adam_8bit, none
#   MAX_EPOCHS=1                         # Training epochs
#   USE_LORA=1                           # Enable LoRA
#   EXTRA_ARGS="..."                     # Additional CLI flags

### MODEL ###
PRETRAIN_PATH="${PRETRAIN_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

### ENVIRONMENT SETUP ###
eval "$(conda shell.bash hook)"
CONDA_ENV="${CONDA_ENV:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_nightly}"
conda activate "$CONDA_ENV"
set -euo pipefail
export DS_SKIP_CUDA_CHECK=1

# Prevent corrupted torch inductor cache from crashing vLLM compilation.
rm -rf ~/.cache/torch/inductor/ /tmp/torchinductor_${USER}/ ~/.cache/vllm/torch_compile_cache/ 2>/dev/null || true

### ARGS ###
LEARNING_RATE="${LEARNING_RATE:-3e-7}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
DEBUG_TRACES="${DEBUG_TRACES:-0}"

### FEATURE FLAGS ###
MODE="${MODE:-colocated}"
EFFECTIVE_ROLLOUT_BATCH_SIZE="${EFFECTIVE_ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
EFFECTIVE_MINI_GRADIENT_STEPS="${EFFECTIVE_MINI_GRADIENT_STEPS:-4}"
ASYNC_ADVANTAGE="${ASYNC_ADVANTAGE:-4}"
SMART_REPLAY="${SMART_REPLAY:-0}"
MAX_REPLAY_ROUNDS="${MAX_REPLAY_ROUNDS:-2}"

LIGER_GRPO_LOSS="${LIGER_GRPO_LOSS:-0}"
LIGER_GRPO_BACKEND="${LIGER_GRPO_BACKEND:-triton}"
LOSS_TYPE="${LOSS_TYPE:-ppo}"
LIGER_CHUNK_SIZE="${LIGER_CHUNK_SIZE:-1}"
OVERSAMPLE_RATIO="${OVERSAMPLE_RATIO:-1}"
TIS="${TIS:-0}"
TIS_TYPE="${TIS_TYPE:-tis}"
TIS_THRESHOLDS="${TIS_THRESHOLDS:-0.5 5.0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
REDUCE_OPTIMIZER="${REDUCE_OPTIMIZER:-none}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
USE_LORA="${USE_LORA:-0}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="${VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE:-16}"

export TORCH_DYNAMO_CACHE_SIZE_LIMIT=1024
export TORCH_DYNAMO_RECOMPILE_LIMIT=1024

### UNIFIED CONSTANTS ###
ZERO_STAGE=1
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-2048}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-2048}"
TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-8192}"  # NeMo RL: micro_batch(2) × max_seq_len(4096)
ROLLOUT_MAX_TOKENS_PER_GPU="${ROLLOUT_MAX_TOKENS_PER_GPU:-$(echo "$TRAIN_MAX_TOKENS_PER_GPU * 4" | bc | awk '{print int($1)}')}"

COLO_EVAL_STEPS="${COLO_EVAL_STEPS:-10}"

### MODE-DEPENDENT DEFAULTS ###
if [ "$MODE" = "colocated" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-$NUM_GPUS}"
    ROLLOUT_BATCH_SIZE=$(( EFFECTIVE_ROLLOUT_BATCH_SIZE * ASYNC_ADVANTAGE ))
    MINI_GRADIENT_STEPS=$(( EFFECTIVE_MINI_GRADIENT_STEPS ))
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"
    VLLM_SYNC_BACKEND=nccl
    EVAL_STEPS="${EVAL_STEPS:-$COLO_EVAL_STEPS}"
elif [ "$MODE" = "distributed" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:?"MODE=distributed requires ACTOR_GPUS"}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:?"MODE=distributed requires VLLM_NUM_ENGINES"}"
    ROLLOUT_BATCH_SIZE=$EFFECTIVE_ROLLOUT_BATCH_SIZE
    MINI_GRADIENT_STEPS=$EFFECTIVE_MINI_GRADIENT_STEPS
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.96}"
    VLLM_SYNC_BACKEND=gloo
    EVAL_STEPS="${EVAL_STEPS:-$(( COLO_EVAL_STEPS * ASYNC_ADVANTAGE ))}"
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

### MODE FLAGS ###
OPTIMIZER_FLAG=""
if [ "$REDUCE_OPTIMIZER" != "none" ]; then
    OPTIMIZER_FLAG="--$REDUCE_OPTIMIZER"
fi

if [ "$MODE" = "colocated" ]; then
    VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-1}"
    MODE_FLAGS="--colocate_all_models --vllm_enable_sleep --vllm_sleep_level $VLLM_SLEEP_LEVEL --deepspeed_enable_sleep $OPTIMIZER_FLAG"
else
    MODE_FLAGS="--async_train --async_queue_size 1 $OPTIMIZER_FLAG"
fi

### WARMUP LOGIC ###
WARMUP_STEPS=13
WARM_STEPS_MULTIPLIER=$(( MINI_GRADIENT_STEPS ))

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
DATA_DIR="$PROJECT_ROOT/data/math/openmathinstruct2"
TRAIN_DATA="$DATA_DIR/train.jsonl"
EVAL_DATA="$DATA_DIR/val.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found: $TRAIN_DATA"
    echo "Run first: python scripts/convert_openmathinstruct2_to_openrlhf.py"
    exit 1
fi
if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: Validation data not found: $EVAL_DATA"
    echo "Run first: python scripts/convert_openmathinstruct2_to_openrlhf.py"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

### RUN CONFIG ###
DATE_TAG=$(date +%m%d_%H%M)
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

# Build suffix tags for active features
SUFFIX=""
[ "$SMART_REPLAY" = "1" ] && SUFFIX+="-sr${MAX_REPLAY_ROUNDS}"
[ "$LOSS_TYPE" != "ppo" ] && SUFFIX+="-${LOSS_TYPE}"
[ "$TIS" = "1" ] && SUFFIX+="-tis"
[ "$USE_LORA" = "1" ] && SUFFIX+="-lora"

if [ "$MODE" = "colocated" ]; then
    MODE_TAG="colo"
    RUN_NAME="grpo-math-llama8b-openmathinstruct2-ep${MAX_EPOCHS}${SUFFIX}-${MODE_TAG}-${DATE_TAG}"
    WANDB_GROUP="Math-Llama8B-OpenMathInstruct2-colo"
else
    MODE_TAG="dist-${LAYOUT_TAG}"
    RUN_NAME="grpo-math-llama8b-openmathinstruct2-ep${MAX_EPOCHS}${SUFFIX}-${MODE_TAG}-${DATE_TAG}"
    WANDB_GROUP="Math-Llama8B-OpenMathInstruct2-dist-${LAYOUT_TAG}"
fi
RUN_ID="${RUN_NAME}"
HUB_NAME="grpo-math-llama8b-openmathinstruct2-ep${MAX_EPOCHS}-${DATE_TAG}"
RUNS_DIR="$PROJECT_ROOT/runs/${RUN_NAME}"
mkdir -p "$RUNS_DIR"
LOCAL_SAVE_DIR="${LOCAL_SAVE_DIR:-/vast/projects/myatskar/design-documents/hf_home}"
SAVE_PATH="$LOCAL_SAVE_DIR/$RUN_NAME"
HUB_REPO_ID="jiosephlee/${HUB_NAME}"

### GRPO CONFIG ###
ADVANTAGE_ESTIMATOR="group_norm"
DYNAMIC_FILTERING=false
DYNAMIC_FILTERING_REWARD_RANGE="0 1"

WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_math_grpo}"

### ENVIRONMENT VARIABLES ###
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"

export TRITON_CACHE_DIR="/vast/projects/myatskar/design-documents/.cache/triton"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true
export TORCHINDUCTOR_CACHE_DIR="/vast/projects/myatskar/design-documents/.cache/torch_inductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

export TORCH_DYNAMO_RECOMPILE_LIMIT=1024
export TORCH_DYNAMO_CACHE_SIZE_LIMIT=1024

export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
export DEBUG_TRACES="$DEBUG_TRACES"
export OPENRLHF_DEBUG_LOGITS=0
export OPENRLHF_DEBUG_NAN_GUARD=0
export OPENRLHF_VRAM_AUDIT="${OPENRLHF_VRAM_AUDIT:-1}"

### RAY ###
export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
ulimit -n 65535 2>/dev/null || true

CONDA_RAY="$(which python) -m ray.scripts.scripts"
echo "Using ray from: $(which python)"

unset RAY_ADDRESS

$CONDA_RAY stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/ray/session_* 2>/dev/null || true

RAY_PORT=$(( 6379 + (RANDOM % 1000) ))
echo "Starting Ray head node at $RAY_NODE_IP_ADDRESS:$RAY_PORT"
$CONDA_RAY start --head \
    --node-ip-address "$RAY_NODE_IP_ADDRESS" \
    --port "$RAY_PORT" \
    --num-gpus "$NUM_GPUS" \
    --temp-dir "$RAY_TMPDIR"

export RAY_ADDRESS="$RAY_NODE_IP_ADDRESS:$RAY_PORT"

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
echo "Math GRPO Training — Llama-3.1-8B-Instruct on OpenMathInstruct-2 (MODE=$MODE)"
echo "========================================"
echo "Model: $PRETRAIN_PATH"
echo "Learning Rate: $LEARNING_RATE"
echo "Run ID: $RUN_ID"
echo "----------------------------------------"
echo "Training Data: $TRAIN_DATA"
echo "Eval Data: $EVAL_DATA"
echo "Save Path: $SAVE_PATH"
echo "----------------------------------------"
if [ "$MODE" = "colocated" ]; then
    echo "NUM_GPUS: $NUM_GPUS (colocated — shared between actor and vLLM)"
else
    echo "NUM_GPUS: $NUM_GPUS  ACTOR: $ACTOR_GPUS  VLLM: $VLLM_NUM_ENGINES"
fi
echo "ROLLOUT_BATCH_SIZE: $ROLLOUT_BATCH_SIZE"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "N_SAMPLES_PER_PROMPT: $N_SAMPLES_PER_PROMPT"
echo "MINI_GRADIENT_STEPS: $MINI_GRADIENT_STEPS"
echo "VLLM_NUM_ENGINES: $VLLM_NUM_ENGINES"
echo "EVAL_STEPS: $EVAL_STEPS"
echo "TRAIN_MAX_TOKENS_PER_GPU: $TRAIN_MAX_TOKENS_PER_GPU"
echo "ROLLOUT_MAX_TOKENS_PER_GPU: $ROLLOUT_MAX_TOKENS_PER_GPU"
echo "----------------------------------------"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Warmup Steps: $WARMUP_STEPS (multiplier: $WARM_STEPS_MULTIPLIER)"
echo "----------------------------------------"
echo "Smart Replay: $SMART_REPLAY"
echo "Loss Type: $LOSS_TYPE"
echo "Liger GRPO Loss: $LIGER_GRPO_LOSS (backend=$LIGER_GRPO_BACKEND, chunk_size=$LIGER_CHUNK_SIZE)"
echo "LoRA: USE_LORA=$USE_LORA (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
echo "TIS: $TIS (type=$TIS_TYPE, thresholds=$TIS_THRESHOLDS)"
echo "KV Cache Dtype: ${KV_CACHE_DTYPE:-auto}"
echo "VLLM_MAX_NUM_SEQS: $VLLM_MAX_NUM_SEQS"
echo "VLLM_MAX_NUM_BATCHED_TOKENS: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "----------------------------------------"
echo "Runs Dir: $RUNS_DIR"
echo "W&B: project=$WANDB_PROJECT group=$WANDB_GROUP run=$RUN_ID"
echo "========================================"

### OPTIONAL FLAGS ###
OPTIONAL_FLAGS=""
if [ "$DYNAMIC_FILTERING" = true ]; then
    OPTIONAL_FLAGS+=" --dynamic_filtering --dynamic_filtering_reward_range $DYNAMIC_FILTERING_REWARD_RANGE"
fi
if [ "$SMART_REPLAY" = "1" ]; then
    OPTIONAL_FLAGS+=" --smart_replay --max_replay_rounds $MAX_REPLAY_ROUNDS"
fi
OPTIONAL_FLAGS+=" --oversample_ratio $OVERSAMPLE_RATIO"

if [ "$LIGER_GRPO_LOSS" = "1" ]; then
    OPTIONAL_FLAGS+=" --use_liger_grpo_loss"
fi
if [ "$TIS" = "1" ]; then
    OPTIONAL_FLAGS+=" --enable_vllm_is_correction --vllm_is_correction_type $TIS_TYPE --vllm_is_truncated_threshold $TIS_THRESHOLDS"
fi
if [ -n "$KV_CACHE_DTYPE" ]; then
    OPTIONAL_FLAGS+=" --kv_cache_dtype $KV_CACHE_DTYPE"
fi
if [ -n "$VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE" ]; then
    OPTIONAL_FLAGS+=" --vllm_cudagraph_max_capture_size $VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE"
fi
if [ "$USE_LORA" = "1" ]; then
    OPTIONAL_FLAGS+=" --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA"
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
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/examples/python/math_reward_func.py" \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --logging_steps 1 \
    --micro_train_batch_size 2 \
    --micro_rollout_batch_size 4 \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --num_episodes $MAX_EPOCHS \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --max_samples 1000000 \
    --loss_type $LOSS_TYPE \
    --use_dynamic_batch \
    --train_max_tokens_per_gpu $TRAIN_MAX_TOKENS_PER_GPU \
    --rollout_max_tokens_per_gpu $ROLLOUT_MAX_TOKENS_PER_GPU \
    --enable_prefix_caching \
    --zero_stage $ZERO_STAGE \
    --param_dtype bf16 \
    --actor_learning_rate $LEARNING_RATE \
    --prompt_data "$TRAIN_DATA" \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps $EVAL_STEPS \
    --eval_temperature 0.1 \
    --eval_n_samples_per_prompt 1 \
    --input_key text \
    --label_key answer \
    --apply_chat_template \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend $VLLM_SYNC_BACKEND \
    --vllm_max_num_seqs $VLLM_MAX_NUM_SEQS \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    --use_wandb 1 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group "$WANDB_GROUP" \
    --wandb_run_name "$RUN_ID" \
    --save_path "$SAVE_PATH" \
    --push_to_hub "$HUB_REPO_ID" \
    --delete_local_after_push \
    --constant_lr_with_warm_up \
    --skip_eval_step_zero \
    --warmup_steps $WARMUP_STEPS \
    --warm_steps_multiplier_for_correction $WARM_STEPS_MULTIPLIER \
    $MODE_FLAGS \
    $OPTIONAL_FLAGS \
    $EXTRA_ARGS \
    2>&1 | tee "$RUN_LOG"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
$CONDA_RAY stop --force || true
