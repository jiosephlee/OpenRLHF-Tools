#!/bin/bash
#
# KNN-3 no-tools GRPO training for GPT-OSS on TDC binary classification.
#
# Uses KNN-3 augmented prompts (with precomputed RDKit descriptors + 3 nearest
# neighbors per class baked into the text) instead of multi-turn tool calling.
# Input key is "text" (plain string), not "messages".
#
# Data source: Intern-S1-recipe/DataPrepare/TDC_prepended/KNN_3/
# Convert first: python scripts/convert_knn3_to_openrlhf.py
#
# Supports all quantization modes via env vars:
#   QUANT_METHOD=mxfp4 (default) — MXFP4 QAT + FlashInfer MoE kernel
#   QUANT_METHOD=nvfp4            — NVFP4 QAT + NVIDIA kernel backend
#   DEQUANT=unsloth               — Load pre-converted BF16 model (no quant flags)
#
# Supports both colocated and distributed modes via MODE env var.
#
# Usage:
#   # MXFP4 (default):
#   bash scripts/train_grpo_tdc_gpt_oss_no_tools_knn3.sh
#
#   # Unsloth BF16:
#   DEQUANT=unsloth LIGER_GRPO_LOSS=1 LOSS_TYPE=dapo bash scripts/train_grpo_tdc_gpt_oss_no_tools_knn3.sh
#
#   USE_LORA=1 LEARNING_RATE=1e-5 DEQUANT=unsloth EFFECTIVE_ROLLOUT_BATCH_SIZE=8 EFFECTIVE_MINI_GRADIENT_STEPS=2 REDUCE_OPTIMIZER=none TRAIN_MAX_TOKENS_PER_GPU=32768 LIGER_GRPO_LOSS=0 LOSS_TYPE=ppo SMART_REPLAY=1 bash train_grpo_tdc_gpt_oss_no_tools_knn3.sh
#   USE_LORA=1 LEARNING_RATE=2e-5 OVERSAMPLE_RATIO=2 TIS=1 TIS_TYPE=icepop DEQUANT=unsloth EFFECTIVE_ROLLOUT_BATCH_SIZE=8 EFFECTIVE_MINI_GRADIENT_STEPS=2 REDUCE_OPTIMIZER=adam_offload TRAIN_MAX_TOKENS_PER_GPU=32768 LIGER_GRPO_LOSS=0 LOSS_TYPE=cispo bash train_grpo_tdc_gpt_oss_120b_no_tools_knn3.sh
#
#   # With features:
#   EFFECTIVE_ROLLOUT_BATCH_SIZE=8 EFFECTIVE_MINI_GRADIENT_STEPS=2 TIS=1 TIS_TYPE=tis REDUCE_OPTIMIZER=adam_offload TRAIN_MAX_TOKENS_PER_GPU=32768 LIGER_GRPO_LOSS=1 LOSS_TYPE=dapo SMART_REPLAY=1 DEQUANT=unsloth bash scripts/train_grpo_tdc_gpt_oss_no_tools_knn3.sh
#
#   # LoRA:
#   USE_LORA=1 LEARNING_RATE=2e-5 DEQUANT=unsloth bash scripts/train_grpo_tdc_gpt_oss_no_tools_knn3.sh
#
#   # Distributed:
#   MODE=distributed ACTOR_GPUS=1 VLLM_NUM_ENGINES=1 DEQUANT=unsloth bash scripts/train_grpo_tdc_gpt_oss_no_tools_knn3.sh
#
# Feature flags (all env-configurable):
#   MODE=colocated|distributed           # Default: colocated
#   QUANT_METHOD=mxfp4|nvfp4             # FP4 format (default: mxfp4, ignored when DEQUANT set)
#   DEQUANT=unsloth                       # Skip quantization, run in BF16
#   EFFECTIVE_ROLLOUT_BATCH_SIZE=8       # Rollout batch size (distributed/async base)
#   EFFECTIVE_MINI_GRADIENT_STEPS=2      # Mini gradient steps (distributed/async base)
#   ASYNC_ADVANTAGE=4                    # Colocated multiplier for rollout/mini
#   COLO_EVAL_STEPS=8                    # Eval frequency for colocated
#   SMART_REPLAY=1                       # Enable smart replay
#   CURRICULUM_BALANCED=1                # Enable curriculum-balanced sampling
#   OVERSAMPLE_RATIO=1.6                 # Oversample ratio
#   LIGER_GRPO_LOSS=1                    # Enable Liger fused GRPO loss
#   LIGER_GRPO_BACKEND=triton            # Liger backend: triton or chunked
#   LOSS_TYPE=ppo                        # Loss type: ppo, dapo, bnpo, dr_grpo, gspo, cispo, sapo
#   LIGER_CHUNK_SIZE=1                   # Chunk size for chunked backend
#   TIS=1                                # Truncated Importance Sampling
#   TIS_TYPE=tis                         # TIS variant
#   TIS_THRESHOLDS="0.5 5.0"            # Low and high clamp thresholds
#   QAT=fp4_fake_quantize                # QAT method (default: off)
#   KV_CACHE_DTYPE=fp8                   # KV cache dtype for vLLM (default: off)
#   REDUCE_OPTIMIZER=adam_offload        # Optimizer: adam_offload, adam_8bit, none
#   MAX_EPOCHS=1                         # Training epochs
#   USE_LORA=1                           # Enable LoRA
#   LORA_RANK=64                         # LoRA rank
#   LORA_ALPHA=64                        # LoRA alpha
#   UNSLOTH_MOE=1                        # Enable grouped GEMM MoE kernels
#   LENGTH_PENALTY_START=0          # Length penalty (0=off)
#   EXTRA_ARGS="..."                     # Additional CLI flags

### QUANTIZATION MODE RESOLUTION ###
QUANT_METHOD="${QUANT_METHOD:-mxfp4}"
DEQUANT="${DEQUANT:-}"
VLLM_PRETRAIN="${VLLM_PRETRAIN:-}"

if [ -n "$DEQUANT" ]; then
    case "$DEQUANT" in
        unsloth)
            PRETRAIN_PATH="${PRETRAIN_PATH:-unsloth/gpt-oss-120b-BF16}"
            if [ -n "$VLLM_PRETRAIN" ]; then
                # Train BF16 actor, serve MXFP4 vLLM with on-the-fly quantized weight sync
                QUANT_FLAGS="--vllm_pretrain $VLLM_PRETRAIN --vllm_sync_fp4 mxfp4"
                QUANT_LABEL="dequant-unsloth-mxfp4sync"
            else
                QUANT_FLAGS=""
                QUANT_LABEL="dequant-unsloth"
            fi
            ;;
        *)
            echo "Error: DEQUANT must be 'unsloth', got '$DEQUANT'" >&2
            exit 1
            ;;
    esac
    CUDA_MODULE="${CUDA_MODULE:-cuda/13.1.0}"
    CONDA_ENV="${CONDA_ENV:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_nightly}"
else
    case "$QUANT_METHOD" in
        mxfp4)
            PRETRAIN_PATH="${PRETRAIN_PATH:-openai/gpt-oss-120b}"
            QUANT_FLAGS="--mxfp4_dequantize --vllm_sync_fp4 mxfp4"
            QUANT_LABEL="mxfp4"
            ;;
        nvfp4)
            PRETRAIN_PATH="${PRETRAIN_PATH:-jiosephlee/gpt-oss-20B-NVFP4-calibrated}"
            NVFP4_BASE="${NVFP4_BASE:-unsloth/gpt-oss-20b-BF16}"
            QUANT_FLAGS="--vllm_sync_fp4 nvfp4 --nvfp4_dequantize_base_model $NVFP4_BASE"
            QUANT_LABEL="nvfp4"
            export VLLM_USE_FLASHINFER_MOE_FP4=0
            ;;
        *)
            echo "Error: QUANT_METHOD must be 'mxfp4' or 'nvfp4', got '$QUANT_METHOD'" >&2
            exit 1
            ;;
    esac
    CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.1}"
    CONDA_ENV="${CONDA_ENV:-/vast/projects/myatskar/design-documents/conda_env/openrlhf}"
fi

### ENVIRONMENT SETUP ###
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -euo pipefail
export DS_SKIP_CUDA_CHECK=1

# Prevent corrupted torch inductor cache from crashing vLLM compilation.
rm -rf ~/.cache/torch/inductor/ /tmp/torchinductor_${USER}/ ~/.cache/vllm/torch_compile_cache/ 2>/dev/null || true

export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency

### ARGS ###
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
DEBUG_TRACES="${DEBUG_TRACES:-0}"

### FEATURE FLAGS ###
MODE="${MODE:-colocated}"
EFFECTIVE_ROLLOUT_BATCH_SIZE="${EFFECTIVE_ROLLOUT_BATCH_SIZE:-8}"
EFFECTIVE_MINI_GRADIENT_STEPS="${EFFECTIVE_MINI_GRADIENT_STEPS:-2}"
ASYNC_ADVANTAGE="${ASYNC_ADVANTAGE:-4}"
SMART_REPLAY="${SMART_REPLAY:-0}"
MAX_REPLAY_ROUNDS="${MAX_REPLAY_ROUNDS:-2}"

LIGER_GRPO_LOSS="${LIGER_GRPO_LOSS:-0}"
LIGER_GRPO_BACKEND="${LIGER_GRPO_BACKEND:-triton}"
LOSS_TYPE="${LOSS_TYPE:-ppo}"
LIGER_CHUNK_SIZE="${LIGER_CHUNK_SIZE:-1}"
CURRICULUM_BALANCED="${CURRICULUM_BALANCED:-0}"
OVERSAMPLE_RATIO="${OVERSAMPLE_RATIO:-1}"
TIS="${TIS:-0}"
TIS_TYPE="${TIS_TYPE:-tis}"
TIS_THRESHOLDS="${TIS_THRESHOLDS:-0.5 5.0}"
QAT="${QAT:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
REDUCE_OPTIMIZER="${REDUCE_OPTIMIZER:-adam_offload}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
USE_LORA="${USE_LORA:-0}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="${VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE:-}"
LENGTH_PENALTY_START="${LENGTH_PENALTY_START:-0}"

export TORCH_DYNAMO_CACHE_SIZE_LIMIT=1024
export TORCH_DYNAMO_RECOMPILE_LIMIT=1024

### UNIFIED CONSTANTS ###
ZERO_STAGE=2
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-18196}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-16384}"
ROLLOUT_MAX_TOKENS_PER_GPU="${ROLLOUT_MAX_TOKENS_PER_GPU:-$(echo "$TRAIN_MAX_TOKENS_PER_GPU * 1" | bc | awk '{print int($1)}')}"

COLO_EVAL_STEPS="${COLO_EVAL_STEPS:-8}"

### MODE-DEPENDENT DEFAULTS ###
if [ "$MODE" = "colocated" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
    ROLLOUT_BATCH_SIZE=$(( EFFECTIVE_ROLLOUT_BATCH_SIZE * ASYNC_ADVANTAGE ))
    MINI_GRADIENT_STEPS=$(( EFFECTIVE_MINI_GRADIENT_STEPS))
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"
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
WARMUP_STEPS=10
WARM_STEPS_MULTIPLIER=$(( MINI_GRADIENT_STEPS ))

### MULTI-TASK: all 16 TDC tasks ###
TASK_NAMES=(Bioavailability_Ma HIA_Hou PAMPA_NCATS Pgp_Broccatelli BBB_Martins CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels CYP3A4_Substrate_CarbonMangels SARSCoV2_3CLPro_Diamond SARSCoV2_Vitro_Touret Carcinogens_Lagunin hERG ClinTox DILI Skin_Reaction AMES)
#TASK_NAMES=(DILI)
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
DATA_DIR="$PROJECT_ROOT/data/tdc/knn3_format"
mkdir -p "$PROJECT_ROOT/logs"

TRAIN_PARTS=()
for t in "${TASK_NAMES[@]}"; do
    f="$DATA_DIR/${t}_train.jsonl"
    if [ ! -f "$f" ]; then
        echo "Error: Training data not found: $f"
        echo "Have you run: python scripts/convert_knn3_to_openrlhf.py ?"
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

# Build suffix tags for active features
SUFFIX=""
[ "$SMART_REPLAY" = "1" ] && SUFFIX+="-sr${MAX_REPLAY_ROUNDS}"
[ "$LOSS_TYPE" != "ppo" ] && SUFFIX+="-${LOSS_TYPE}"
[ "$TIS" = "1" ] && SUFFIX+="-tis"
[ "$USE_LORA" = "1" ] && SUFFIX+="-lora"

if [ "$MODE" = "colocated" ]; then
    MODE_TAG="colo"
    RUN_NAME="grpo-tdc-gptoss-${QUANT_LABEL}-${N_TASKS}t-notools-knn3-ep${MAX_EPOCHS}${SUFFIX}-${MODE_TAG}-${DATE_TAG}"
    WANDB_GROUP="TDC-GPTOss-${QUANT_LABEL}-no-tools-KNN3-colo-$TASK_LABEL"
else
    MODE_TAG="dist-${LAYOUT_TAG}"
    RUN_NAME="grpo-tdc-gptoss-${QUANT_LABEL}-${N_TASKS}t-notools-knn3-ep${MAX_EPOCHS}${SUFFIX}-${MODE_TAG}-${DATE_TAG}"
    WANDB_GROUP="TDC-GPTOss-${QUANT_LABEL}-no-tools-KNN3-dist-${LAYOUT_TAG}-$TASK_LABEL"
fi
RUN_ID="${RUN_NAME}"
HUB_NAME="grpo-tdc-gptoss-${QUANT_LABEL}-${N_TASKS}t-notools-knn3-ep${MAX_EPOCHS}-${DATE_TAG}"
RUNS_DIR="$PROJECT_ROOT/runs/${RUN_NAME}"
mkdir -p "$RUNS_DIR"
LOCAL_SAVE_DIR="${LOCAL_SAVE_DIR:-/vast/projects/myatskar/design-documents/hf_home}"
SAVE_PATH="$LOCAL_SAVE_DIR/$RUN_NAME"
HUB_REPO_ID="jiosephlee/${HUB_NAME}"

### GRPO CONFIG ###
ADVANTAGE_ESTIMATOR="group_norm"
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0 1"

WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P=0.95

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
export OPENRLHF_SMILES_ERROR_LOG="$RUNS_DIR/smiles_errors.jsonl"

### RAY ###
export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
ulimit -n 65535 2>/dev/null || true

# Use the conda env's ray to avoid version mismatch
CONDA_RAY="$(which python) -m ray.scripts.scripts"
echo "Using ray from: $(which python)"

# Clear any stale RAY_ADDRESS
unset RAY_ADDRESS

$CONDA_RAY stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/ray/session_* 2>/dev/null || true

# Use a unique port to avoid collisions with other users on the same node.
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
echo "TDC GRPO Training — GPT-OSS (No-Tools, KNN-3 Data, MODE=$MODE, QUANT=$QUANT_LABEL)"
echo "========================================"
echo "Tasks: ${TASK_NAMES[*]}"
echo "Model: $PRETRAIN_PATH"
echo "Quantization: $QUANT_LABEL"
echo "Quant Flags: $QUANT_FLAGS"
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
echo "TRAIN_MAX_TOKENS_PER_GPU: $TRAIN_MAX_TOKENS_PER_GPU"
echo "ROLLOUT_MAX_TOKENS_PER_GPU: $ROLLOUT_MAX_TOKENS_PER_GPU"
echo "----------------------------------------"
echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Warmup Steps: $WARMUP_STEPS (multiplier: $WARM_STEPS_MULTIPLIER)"
echo "----------------------------------------"
echo "Smart Replay: $SMART_REPLAY"
echo "Curriculum Balanced: $CURRICULUM_BALANCED"
echo "Oversample Ratio: $OVERSAMPLE_RATIO"
echo "Loss Type: $LOSS_TYPE"
echo "Liger GRPO Loss: $LIGER_GRPO_LOSS (backend=$LIGER_GRPO_BACKEND, chunk_size=$LIGER_CHUNK_SIZE)"
echo "LoRA: USE_LORA=$USE_LORA (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
echo "TIS: $TIS (type=$TIS_TYPE, thresholds=$TIS_THRESHOLDS)"
echo "KV Cache Dtype: ${KV_CACHE_DTYPE:-fp8}"
echo "VLLM_MAX_NUM_SEQS: $VLLM_MAX_NUM_SEQS"
echo "VLLM_MAX_NUM_BATCHED_TOKENS: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "----------------------------------------"
echo "Runs Dir: $RUNS_DIR"
echo "W&B: project=$WANDB_PROJECT group=$WANDB_GROUP run=$RUN_ID"
echo "========================================"

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
OPTIONAL_FLAGS+=" --oversample_ratio $OVERSAMPLE_RATIO"

if [ "$LIGER_GRPO_LOSS" = "1" ]; then
    OPTIONAL_FLAGS+=" --use_liger_grpo_loss"
fi
if [ "$TIS" = "1" ]; then
    OPTIONAL_FLAGS+=" --enable_vllm_is_correction --vllm_is_correction_type $TIS_TYPE --vllm_is_truncated_threshold $TIS_THRESHOLDS"
fi
if [ -n "$QAT" ]; then
    OPTIONAL_FLAGS+=" --qat $QAT"
fi
if [ -n "$KV_CACHE_DTYPE" ]; then
    OPTIONAL_FLAGS+=" --kv_cache_dtype $KV_CACHE_DTYPE"
fi
if [ -n "$VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE" ]; then
    OPTIONAL_FLAGS+=" --vllm_cudagraph_max_capture_size $VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE"
fi
if [ "$LENGTH_PENALTY_START" -gt 0 ]; then
    OPTIONAL_FLAGS+=" --length_penalty_start $LENGTH_PENALTY_START"
fi
if [ "$USE_LORA" = "1" ]; then
    OPTIONAL_FLAGS+=" --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA"
fi
if [ "${UNSLOTH_MOE:-0}" = "1" ]; then
    OPTIONAL_FLAGS+=" --use_unsloth_moe_kernels"
fi

### TRAINING ###
RUN_LOG="$RUNS_DIR/run_${QUANT_LABEL}.log"
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
    --vllm_tensor_parallel_size 2 \
    --optimal_flags_b200_gpt_oss \
    --max_num_batched_tokens $VLLM_MAX_NUM_BATCHED_TOKENS \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEM_UTIL \
    --advantage_estimator $ADVANTAGE_ESTIMATOR \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --eps_clip_low_high 0.2 0.272 \
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
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
    --generate_max_len 8192 \
    --max_samples 1000000 \
    --loss_type $LOSS_TYPE \
    --use_adaptive_batch \
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
    --replace_discarded_prompts_ratio 2.0 \
    --constant_lr_with_warm_up \
    --warmup_steps $WARMUP_STEPS \
    --warm_steps_multiplier_for_correction $WARM_STEPS_MULTIPLIER \
    --attn_implementation "flex_attention" \
    --length_penalty_start 6144 \
    --freeze_router \
    --aux_loss_coef 0 \
    $QUANT_FLAGS \
    $MODE_FLAGS \
    $OPTIONAL_FLAGS \
    $EXTRA_ARGS \
    2>&1 | tee "$RUN_LOG"

### CLEANUP ###
echo "Training complete! Stopping Ray..."
$CONDA_RAY stop --force || true
