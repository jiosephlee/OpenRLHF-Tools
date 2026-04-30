#!/bin/bash
#
# Unified SLURM batch script for GPT-OSS GRPO training.
#
# Supports all quantization modes via env vars:
#   QUANT_METHOD=mxfp4 (default) — MXFP4 QAT + FlashInfer MoE kernel
#   QUANT_METHOD=nvfp4            — NVFP4 QAT + NVIDIA kernel backend
#   DEQUANT=unsloth               — Load pre-converted BF16 model (no quant flags)
#
# When DEQUANT is set, QUANT_METHOD is ignored.
#
# Supports both colocated and distributed modes via MODE env var.
#
# Uses GPT-OSS Harmony tool-calling format:
#   <|start|>assistant to=functions.<name><|channel|>commentary json<|message|>...
#
# Usage:
#   # MXFP4 QAT (default):
#   VLLM_GPU_MEM_UTIL=0.6
#   # MXFP4 QAT (default):
#   TRAIN_MAX_TOKENS_PER_GPU=8192 QAT=fp4_fake_quantize sbatch train_grpo_tdc_gpt_oss_slurm.sh
#
#   # NVFP4 QAT:
#   TRAIN_MAX_TOKENS_PER_GPU=1024 QUANT_METHOD=nvfp4 sbatch train_grpo_tdc_gpt_oss_slurm.sh
#
#   # Unsloth BF16:
#   LEARNING_RATE=1e-5 DEQUANT=unsloth sbatch train_grpo_tdc_gpt_oss_slurm.sh
#
#   # Distributed:
#   MODE=distributed ACTOR_GPUS=1 VLLM_NUM_ENGINES=1 sbatch scripts/train_grpo_tdc_gpt_oss_slurm.sh
#
# Feature flags (set via env before sbatch):
#   MODE=colocated|distributed           # Default: colocated
#   QUANT_METHOD=mxfp4|nvfp4             # FP4 format (default: mxfp4, ignored when DEQUANT set)
#   DEQUANT=unsloth                       # Skip quantization, run in BF16
#   EFFECTIVE_ROLLOUT_BATCH_SIZE=8       # Rollout batch size in distributed/async mode
#   EFFECTIVE_MINI_GRADIENT_STEPS=2      # Mini gradient steps in distributed/async mode
#   ASYNC_ADVANTAGE=4                    # Scale factor: colocated uses ASYNC_ADVANTAGE * EFFECTIVE_* for both
#                                        # ROLLOUT and MINI, keeping ROLLOUT/MINI ratio constant across modes.
#                                        # Reflects that colocated is synchronous and can afford more rollouts
#                                        # before each update without the 1-step off-policy lag of async.
#   COLO_EVAL_STEPS=16                   # Eval frequency (global steps) for colocated; distributed scales by ASYNC_ADVANTAGE
#   TOOL_VERSION=v10                     # Tool schema version
#   SMART_REPLAY=1                       # Enable smart replay with max_replay_rounds=3
#   CURRICULUM_BALANCED=1                # Enable curriculum-balanced sampling
#   OVERSAMPLE_RATIO=2                   # Oversample ratio for dynamic filtering (default: 2)
#   OVERSAMPLE_RATIO_START=1.75          # Optional linear-ramp start ratio (defaults to OVERSAMPLE_RATIO)
#   OVERSAMPLE_RATIO_END=2.75            # Optional linear-ramp end ratio (defaults to OVERSAMPLE_RATIO)
#   OVERSAMPLE_RATIO_RAMP_STEPS=100      # Optional ramp horizon in global steps (defaults to trainer max_steps)

#   LIGER_GRPO_LOSS=1                    # Enable Liger fused GRPO loss
#   LIGER_GRPO_BACKEND=triton             # Liger backend: triton (default) or chunked
#   LOSS_TYPE=ppo                        # Loss type: ppo, dapo, bnpo, dr_grpo, gspo, cispo, sapo (controls ratio+reduction)
#   LIGER_CHUNK_SIZE=1                   # Chunk size for chunked backend (1=max chunking)
#   TIS=1                                # Enable Truncated Importance Sampling (off-policy correction)
#   TIS_TYPE=tis                         # TIS variant: tis (default), icepop, seq-mask-tis
#   TIS_THRESHOLDS="0.5 5.0"            # Low and high clamp thresholds (default: 0.5 5.0)
#   QAT=fp4_fake_quantize                 # QAT method (default: off). fp4_fake_quantize derives format from QUANT_METHOD
#   KV_CACHE_DTYPE=fp8                   # KV cache dtype for vLLM (default: fp8)
#   REDUCE_OPTIMIZER=adam_offload        # Optimizer: adam_offload (default), adam_8bit, or none
#   MAX_EPOCHS=2                         # Training epochs (default: 1)
#   USE_LORA=1                           # Enable LoRA (default: off); tweak LORA_RANK and LORA_ALPHA manually
#   LORA_RANK=256                        # LoRA rank (default: 256, used when USE_LORA=1)
#   LORA_ALPHA=512                       # LoRA alpha (default: 512, used when USE_LORA=1)
#   UNSLOTH_MOE=1                        # Enable grouped GEMM MoE kernels (Triton A100+, grouped_mm H100+)
#   EXTRA_ARGS="..."                     # Additional CLI flags
#

### SLURM PARAMETERS ###
#SBATCH --job-name=grpo-tdc-gptoss
#SBATCH --output=logs/grpo-tdc-gptoss_%j.out
#SBATCH --error=logs/grpo-tdc-gptoss_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1408G
#SBATCH --cpus-per-task=112
#SBATCH --time=00-8:00:00
### PARCC PARAMETERS ###
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

### QUANTIZATION MODE RESOLUTION (before module load — determines conda/CUDA) ###
QUANT_METHOD="${QUANT_METHOD:-mxfp4}"
DEQUANT="${DEQUANT:-}"

if [ -n "$DEQUANT" ]; then
    # BF16 dequantized mode — no FP4 sync/QAT
    case "$DEQUANT" in
        unsloth)
            ;;
        *)
            echo "Error: DEQUANT must be 'unsloth', got '$DEQUANT'" >&2
            exit 1
            ;;
    esac
    CONDA_ENV_PATH="${CONDA_ENV_PATH:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_nightly}"
    CUDA_MODULE="${CUDA_MODULE:-cuda/13.1.0}"
else
    # Quantized mode — FP4 QAT + weight sync
    case "$QUANT_METHOD" in
        mxfp4|nvfp4)
            ;;
        *)
            echo "Error: QUANT_METHOD must be 'mxfp4' or 'nvfp4', got '$QUANT_METHOD'" >&2
            exit 1
            ;;
    esac
    CONDA_ENV_PATH="${CONDA_ENV_PATH:-/vast/projects/myatskar/design-documents/conda_env/openrlhf}"
    CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.1}"
fi

### ENVIRONMENT SETUP ###
module load MAMBA
module load "$CUDA_MODULE"
export CONDA_ENV_PATH

############################
#        TASK SCRIPT       #
############################
run_task() {
    set -euo pipefail
    export DS_SKIP_CUDA_CHECK=1


    ### QUANTIZATION MODE RESOLUTION (inside run_task for srun context) ###
    QUANT_METHOD="${QUANT_METHOD:-mxfp4}"
    DEQUANT="${DEQUANT:-}"

    VLLM_PRETRAIN="${VLLM_PRETRAIN:-}"

    if [ -n "$DEQUANT" ]; then
        case "$DEQUANT" in
            unsloth)
                PRETRAIN_PATH="${PRETRAIN_PATH:-unsloth/gpt-oss-20b-BF16}"
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
    else
        case "$QUANT_METHOD" in
            mxfp4)
                PRETRAIN_PATH="${PRETRAIN_PATH:-openai/gpt-oss-20b}"
                QUANT_FLAGS="--mxfp4_dequantize --vllm_sync_fp4 mxfp4"
                # export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
                QUANT_LABEL="mxfp4"
                ;;
            nvfp4)
                PRETRAIN_PATH="${PRETRAIN_PATH:-jiosephlee/gpt-oss-20B-NVFP4-calibrated}"
                NVFP4_BASE="${NVFP4_BASE:-unsloth/gpt-oss-20b-BF16}"
                QUANT_FLAGS="--vllm_sync_fp4 nvfp4 --nvfp4_dequantize_base_model $NVFP4_BASE"
                QUANT_LABEL="nvfp4"
                # FlashInfer CUTEDSL/CUTLASS hang for hidden_size=2880 (not tile-aligned).
                # Marlin crashes (group_size=16 unsupported).
                # Use VLLM_CUTLASS with calibrated activation scales (w13/w2_input_scale).
                # Run scripts/calibrate_nvfp4_activations.py to regenerate calibrated checkpoint.
                export VLLM_USE_FLASHINFER_MOE_FP4=0
                ;;
            *)
                echo "Error: QUANT_METHOD must be 'mxfp4' or 'nvfp4', got '$QUANT_METHOD'" >&2
                exit 1
                ;;
        esac
    fi

    # Prevent corrupted torch inductor cache from crashing vLLM compilation.
    rm -rf ~/.cache/torch/inductor/ /tmp/torchinductor_${USER}/ ~/.cache/vllm/torch_compile_cache/ 2>/dev/null || true

    # Force Triton for BF16/unquantized GPT-OSS MoE in vLLM by removing the
    # FlashInfer FP16 MoE backends from auto-selection. Quantized FP4 paths use
    # their own backend selectors and are unaffected by this flag.
    VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"
    export VLLM_USE_FLASHINFER_MOE_FP16
    unset VLLM_FLASHINFER_MOE_BACKEND

    ### ARGS (override via env before sbatch) ###
    LEARNING_RATE="${LEARNING_RATE:-1e-6}"
    DEBUG_TRACES="${DEBUG_TRACES:-0}"
    NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"
    if [ -z "$NUM_GPUS" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
        NUM_GPUS=$(python3 - <<'PY'
import os
value = os.environ.get("SLURM_JOB_GPUS", "").strip()
print(len([x for x in value.split(",") if x.strip()]) if value else 0)
PY
)
    fi
    if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" = "0" ]; then
        NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
    fi

    ### FEATURE FLAGS ###
    MODE="${MODE:-colocated}"
    EFFECTIVE_ROLLOUT_BATCH_SIZE="${EFFECTIVE_ROLLOUT_BATCH_SIZE:-8}"
    EFFECTIVE_MINI_GRADIENT_STEPS="${EFFECTIVE_MINI_GRADIENT_STEPS:-2}"
    ASYNC_ADVANTAGE="${ASYNC_ADVANTAGE:-4}"
    TOOL_VERSION="${TOOL_VERSION:-v12}"
    SMART_REPLAY="${SMART_REPLAY:-0}"
    MAX_REPLAY_ROUNDS="${MAX_REPLAY_ROUNDS:-3}"

    LIGER_GRPO_LOSS="${LIGER_GRPO_LOSS:-0}"
    LIGER_GRPO_BACKEND="${LIGER_GRPO_BACKEND:-triton}"
    LOSS_TYPE="${LOSS_TYPE:-ppo}"
    LIGER_CHUNK_SIZE="${LIGER_CHUNK_SIZE:-1}"
    CURRICULUM_BALANCED="${CURRICULUM_BALANCED:-0}"
    OVERSAMPLE_RATIO="${OVERSAMPLE_RATIO:-2}"
    OVERSAMPLE_RATIO_START="${OVERSAMPLE_RATIO_START:-1.75}"
    OVERSAMPLE_RATIO_END="${OVERSAMPLE_RATIO_END:-2.75}"
    OVERSAMPLE_RATIO_RAMP_STEPS="${OVERSAMPLE_RATIO_RAMP_STEPS:-100}"
    TIS="${TIS:-0}"
    TIS_TYPE="${TIS_TYPE:-tis}"
    TIS_THRESHOLDS="${TIS_THRESHOLDS:-0.5 5.0}"
    QAT="${QAT:-}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
    REDUCE_OPTIMIZER="${REDUCE_OPTIMIZER:-adam_offload}"
    MAX_EPOCHS="${MAX_EPOCHS:-1}"
    USE_LORA="${USE_LORA:-0}"
    LORA_RANK="${LORA_RANK:-256}"
    LORA_ALPHA="${LORA_ALPHA:-512}"
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
    EXTRA_ARGS="${EXTRA_ARGS:-}"
    VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="${VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE:-1024}"
    LENGTH_PENALTY_START="${LENGTH_PENALTY_START:-0}"
    MIN_RESPONSE_LEN="${MIN_RESPONSE_LEN:-}"
    UNDERLONG_PENALTY_FACTOR="${UNDERLONG_PENALTY_FACTOR:-1}"
    KNN_CORRECT_REVERSAL_BONUS="${KNN_CORRECT_REVERSAL_BONUS:-0.25}"
    KNN_CORRECT_STICK_DELTA="${KNN_CORRECT_STICK_DELTA:--0.15}"

    if [[ "$TOOL_VERSION" == *no_neighbor* ]]; then
        ENABLE_KNN_TRACKING=0
    else
        ENABLE_KNN_TRACKING=1
    fi

    export TORCH_DYNAMO_CACHE_SIZE_LIMIT=1024
    export TORCH_DYNAMO_RECOMPILE_LIMIT=1024

    ### UNIFIED CONSTANTS ###
    AGENT_MAX_STEPS=30
    ZERO_STAGE=2
    PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-5120}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
    TRAIN_MAX_TOKENS_PER_GPU="${TRAIN_MAX_TOKENS_PER_GPU:-8192}"
    ROLLOUT_MAX_TOKENS_PER_GPU="${ROLLOUT_MAX_TOKENS_PER_GPU:-$(echo "$TRAIN_MAX_TOKENS_PER_GPU * 1.5" | bc | awk '{print int($1)}')}"

    COLO_EVAL_STEPS="${COLO_EVAL_STEPS:-16}"  # Eval frequency for colocated; distributed multiplies by ASYNC_ADVANTAGE.

    ### MODE-DEPENDENT DEFAULTS ###
    if [ "$MODE" = "colocated" ]; then
        ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS}"
        VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-$NUM_GPUS}"
        ROLLOUT_BATCH_SIZE=$(( EFFECTIVE_ROLLOUT_BATCH_SIZE * ASYNC_ADVANTAGE ))
        MINI_GRADIENT_STEPS=$(( EFFECTIVE_MINI_GRADIENT_STEPS ))
        VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.675}"
        VLLM_SYNC_BACKEND=nccl
        EVAL_STEPS="${EVAL_STEPS:-$COLO_EVAL_STEPS}"
    elif [ "$MODE" = "distributed" ]; then
        ACTOR_GPUS="${ACTOR_GPUS:?"MODE=distributed requires ACTOR_GPUS"}"
        VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:?"MODE=distributed requires VLLM_NUM_ENGINES"}"
        ROLLOUT_BATCH_SIZE=$EFFECTIVE_ROLLOUT_BATCH_SIZE  # Distributed uses the effective value directly; smaller than colocated to be more on-policy (only 1-step async lag).
        MINI_GRADIENT_STEPS=$EFFECTIVE_MINI_GRADIENT_STEPS  # Fewer mini gradient steps to match; ROLLOUT/MINI ratio is identical to colocated, same total gradient steps.
        VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.96}"
        VLLM_SYNC_BACKEND=gloo
        EVAL_STEPS="${EVAL_STEPS:-$(( COLO_EVAL_STEPS * ASYNC_ADVANTAGE ))}" # Distributed takes ASYNC_ADVANTAGE more global steps per colocated step, so scale eval frequency accordingly.
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
    WARMUP_STEPS=8
    WARM_STEPS_MULTIPLIER=$(( MINI_GRADIENT_STEPS ))

    ### MULTI-TASK ###
    TASK_NAMES=(Bioavailability_Ma HIA_Hou PAMPA_NCATS Pgp_Broccatelli BBB_Martins CYP2C9_Substrate_CarbonMangels CYP2D6_Substrate_CarbonMangels CYP3A4_Substrate_CarbonMangels SARSCoV2_3CLPro_Diamond SARSCoV2_Vitro_Touret Carcinogens_Lagunin hERG ClinTox DILI Skin_Reaction AMES)
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
    if [ -z "${DATA_DIR:-}" ]; then
        if [ "$TOOL_VERSION" = "v10" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v10"
        elif [ "$TOOL_VERSION" = "v11" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v11"
        elif [ "$TOOL_VERSION" = "v12" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v12_minority_oversampled"
        elif [ "$TOOL_VERSION" = "v13" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v13"
        elif [ "$TOOL_VERSION" = "v14" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v14"
        elif [ "$TOOL_VERSION" = "v14_consolidated" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v14_consolidated"
        elif [ "$TOOL_VERSION" = "v14_no_neighbor" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v14_no_neighbor"
        elif [ "$TOOL_VERSION" = "v14_consolidated_no_neighbor" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v14_consolidated_no_neighbor"
        elif [ -d "$PROJECT_ROOT/data/tdc/openai_format_${TOOL_VERSION}" ]; then
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_${TOOL_VERSION}"
        else
            DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_v10"
        fi
    fi
    if [ ! -d "$DATA_DIR" ] && [ "$TOOL_VERSION" = "v10" ]; then
        LEGACY_V10_DIR="$PROJECT_ROOT/data/tdc/openai_format_gpt_oss"
        if [ -d "$LEGACY_V10_DIR" ]; then
            DATA_DIR="$LEGACY_V10_DIR"
        fi
    fi
    # Short tag derived from dataset dir name for run naming
    DATA_DIR_BASENAME="$(basename "$DATA_DIR")"
    case "$DATA_DIR_BASENAME" in
        openai_format_enriched)          DATA_TAG="enr" ;;
        openai_format_enriched_v7_tools) DATA_TAG="enr-v7t" ;;
        openai_format_enriched_v6_tools) DATA_TAG="enr-v6t" ;;
        openai_format_v6_encourage_tool_use) DATA_TAG="v6etu" ;;
        openai_format_v7_tools)          DATA_TAG="v7t" ;;
        openai_format_v10)               DATA_TAG="v10" ;;
        openai_format_gpt_oss)           DATA_TAG="gptoss" ;;
        openai_format_v11)               DATA_TAG="v11" ;;
        openai_format_v12)               DATA_TAG="v12" ;;
        openai_format_v12_minority_oversampled) DATA_TAG="v12-mos" ;;
        openai_format_v13)               DATA_TAG="v13" ;;
        openai_format_v14)               DATA_TAG="v14" ;;
        openai_format_v14_consolidated)  DATA_TAG="v14c" ;;
        openai_format_v14_no_neighbor)   DATA_TAG="v14nn" ;;
        openai_format_v14_consolidated_no_neighbor) DATA_TAG="v14cnn" ;;
        openai_format_v14_no_neighbor_local_attribution) DATA_TAG="v14nn-localattr" ;;
        openai_format_v14_no_neighbor_local_attribution_pretend) DATA_TAG="v14nn-localattr-pretend" ;;
        openai_format_v16_no_neighbor_local_attribution) DATA_TAG="v16nn-localattr" ;;
        openai_format_v16_no_neighbor_local_attribution_pretend) DATA_TAG="v16nn-localattr-pretend" ;;
        openai_format_v16_no_neighbor_playbook) DATA_TAG="v16nn-playbook" ;;
        openai_format_v16_no_neighbor_playbook_subagent) DATA_TAG="v16nn-playbook-subagent" ;;
        prepended_tools_v6)              DATA_TAG="pre-v6" ;;
        prepended_tools_v7)              DATA_TAG="pre-v7" ;;
        *)                               DATA_TAG="${DATA_DIR_BASENAME#openai_format_}" ;;
    esac
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
    CHAT_PROTOCOL="gpt_oss"

    # Build suffix tags for active features
    SUFFIX=""
    [ "$SMART_REPLAY" = "1" ] && SUFFIX+="-sr${MAX_REPLAY_ROUNDS}"
    [ "$LOSS_TYPE" != "ppo" ] && SUFFIX+="-${LOSS_TYPE}"
    [ "$TIS" = "1" ] && SUFFIX+="-tis"
    [ "$USE_LORA" = "1" ] && SUFFIX+="-lora"

    if [ "$MODE" = "colocated" ]; then
        MODE_TAG="colo"
        RUN_NAME="grpo-tdc-gptoss-${QUANT_LABEL}-${N_TASKS}t-${TOOL_VERSION}-${DATA_TAG}-ep${MAX_EPOCHS}${SUFFIX}-${MODE_TAG}-${DATE_TAG}"
        WANDB_GROUP="TDC-GPTOss-${QUANT_LABEL}-${DATA_TAG}-colo-$TASK_LABEL"
    else
        MODE_TAG="dist-${LAYOUT_TAG}"
        RUN_NAME="grpo-tdc-gptoss-${QUANT_LABEL}-${N_TASKS}t-${TOOL_VERSION}-${DATA_TAG}-ep${MAX_EPOCHS}${SUFFIX}-${MODE_TAG}-${DATE_TAG}"
        WANDB_GROUP="TDC-GPTOss-${QUANT_LABEL}-${DATA_TAG}-dist-${LAYOUT_TAG}-$TASK_LABEL"
    fi
    RUN_ID="${RUN_NAME}"
    HUB_NAME="grpo-tdc-gptoss-${QUANT_LABEL}-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-${DATE_TAG}"
    source "$PROJECT_ROOT/scripts/lib/resolve_runs_dir.sh"
    LOCAL_SAVE_DIR="${LOCAL_SAVE_DIR:-/vast/projects/myatskar/design-documents/hf_home}"
    SAVE_PATH="$LOCAL_SAVE_DIR/$RUN_NAME"
    HUB_REPO_ID="jiosephlee/${HUB_NAME}"

    ### TOOL-CALLING CONFIG ###
    AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"

    ### GRPO CONFIG ###
    ADVANTAGE_ESTIMATOR="group_norm"
    DYNAMIC_FILTERING=true
    DYNAMIC_FILTERING_REWARD_RANGE="0 1"

    WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"
    TEMPERATURE=1.0
    TOP_P=0.95

    ### RAY TMPDIR ###
    export RAY_TMPDIR="/tmp/ray_${USER}/${SLURM_JOB_ID}"
    mkdir -p "$RAY_TMPDIR"
    PERSIST_RAY_DIR="$PROJECT_ROOT/logs/ray_logs/${SLURM_JOB_ID}"

    ### RAY LOG COPY TRAP ###
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

    ### ENVIRONMENT VARIABLES ###
    export TRITON_CACHE_DIR="/vast/projects/myatskar/design-documents/.cache/triton"
    mkdir -p "$TRITON_CACHE_DIR"
    export TORCHINDUCTOR_CACHE_DIR="/vast/projects/myatskar/design-documents/.cache/torch_inductor"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

    export VLLM_NO_USAGE_STATS=1
    export VLLM_DISABLE_TELEMETRY=1
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # vLLM v1 msgspec can't serialize torch.dtype; fall back to pickle

    # Raise torch.compile recompile/cache limits for flex_attention.
    # With adaptive batching, variable sequence lengths create many unique BlockMask
    # shapes. The default limit (8) causes dynamo to fall back to eager, which breaks
    # gradient checkpointing (recomputed tensors have different metadata).
    export TORCH_DYNAMO_RECOMPILE_LIMIT=1024
    export TORCH_DYNAMO_CACHE_SIZE_LIMIT=1024

    export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
    export OPENRLHF_CHAT_PROTOCOL="$CHAT_PROTOCOL"
    export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"
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

    # Clear any stale RAY_ADDRESS from the environment to prevent
    # connecting to another user's cluster on shared nodes.
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

    # Set explicit address immediately — avoids "multiple active Ray instances" from other users
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
    echo "TDC GRPO Training — GPT-OSS (MODE=$MODE, QUANT=$QUANT_LABEL, SLURM BATCH)"
    echo "========================================"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Tasks: ${TASK_NAMES[*]}"
    echo "Model: $PRETRAIN_PATH"
    echo "Chat Protocol: $CHAT_PROTOCOL"
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
    echo "Agent Max Steps: $AGENT_MAX_STEPS"
    echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
    echo "Temperature: $TEMPERATURE"
    echo "Top-p: $TOP_P"
    echo "Warmup Steps: $WARMUP_STEPS (multiplier: $WARM_STEPS_MULTIPLIER)"
    echo "----------------------------------------"
    echo "Smart Replay: $SMART_REPLAY"
    echo "Curriculum Balanced: $CURRICULUM_BALANCED"
    echo "Oversample Ratio: $OVERSAMPLE_RATIO"
    if [ -n "$OVERSAMPLE_RATIO_START" ] || [ -n "$OVERSAMPLE_RATIO_END" ]; then
        echo "Oversample Ramp: start=${OVERSAMPLE_RATIO_START:-$OVERSAMPLE_RATIO} end=${OVERSAMPLE_RATIO_END:-$OVERSAMPLE_RATIO} steps=${OVERSAMPLE_RATIO_RAMP_STEPS:-auto}"
    fi

    echo "Loss Type: $LOSS_TYPE"
    echo "Liger GRPO Loss: $LIGER_GRPO_LOSS (backend=$LIGER_GRPO_BACKEND, chunk_size=$LIGER_CHUNK_SIZE)"
    echo "LoRA: USE_LORA=$USE_LORA (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
    echo "TIS: $TIS (type=$TIS_TYPE, thresholds=$TIS_THRESHOLDS)"
    echo "KV Cache Dtype: ${KV_CACHE_DTYPE:-auto}"
    echo "VLLM_USE_FLASHINFER_MOE_FP16: $VLLM_USE_FLASHINFER_MOE_FP16"
    echo "VLLM_MAX_NUM_SEQS: $VLLM_MAX_NUM_SEQS"
    echo "VLLM_MAX_NUM_BATCHED_TOKENS: $VLLM_MAX_NUM_BATCHED_TOKENS"
    echo "Tool Version: $TOOL_VERSION"
    echo "KNN Reward Shaping: reversal_bonus=$KNN_CORRECT_REVERSAL_BONUS stick_delta=$KNN_CORRECT_STICK_DELTA"
    if [ -n "$MIN_RESPONSE_LEN" ]; then
        echo "Underlong Penalty: min_response_len=$MIN_RESPONSE_LEN factor=$UNDERLONG_PENALTY_FACTOR"
    else
        echo "Underlong Penalty: disabled"
    fi
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
    if [ -n "$MIN_RESPONSE_LEN" ]; then
        OPTIONAL_FLAGS+=" --min_response_len $MIN_RESPONSE_LEN --underlong_penalty_factor $UNDERLONG_PENALTY_FACTOR"
    fi
    OPTIONAL_FLAGS+=" --oversample_ratio $OVERSAMPLE_RATIO"
    if [ -n "$OVERSAMPLE_RATIO_START" ]; then
        OPTIONAL_FLAGS+=" --oversample_ratio_start $OVERSAMPLE_RATIO_START"
    fi
    if [ -n "$OVERSAMPLE_RATIO_END" ]; then
        OPTIONAL_FLAGS+=" --oversample_ratio_end $OVERSAMPLE_RATIO_END"
    fi
    if [ -n "$OVERSAMPLE_RATIO_RAMP_STEPS" ]; then
        OPTIONAL_FLAGS+=" --oversample_ratio_ramp_steps $OVERSAMPLE_RATIO_RAMP_STEPS"
    fi

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
    # KNN pseudo-labels for reversal tracking on tool-calling prompts that do not
    # inline neighbor pseudo-labels. Generate with:
    #   python scripts/build_knn_v10_pseudo_labels.py
    #   python scripts/build_knn_v11_pseudo_labels.py
    if [ "$ENABLE_KNN_TRACKING" = "1" ]; then
        if [ -z "${KNN_PL_PATH:-}" ]; then
            if [ "$TOOL_VERSION" = "v15" ]; then
                KNN_PL_PATH="$PROJECT_ROOT/data/tdc/metadata/knn_v15_pseudo_labels.json"
            elif [ "$TOOL_VERSION" = "v11" ] || [ "$TOOL_VERSION" = "v12" ] || [ "$TOOL_VERSION" = "v13" ] || [ "$TOOL_VERSION" = "v14" ] || [ "$TOOL_VERSION" = "v14_consolidated" ]; then
                KNN_PL_PATH="$PROJECT_ROOT/data/tdc/metadata/knn_v11_pseudo_labels.json"
            else
                KNN_PL_PATH="$PROJECT_ROOT/data/tdc/metadata/knn_v10_pseudo_labels.json"
            fi
        fi
        if [ -f "$KNN_PL_PATH" ]; then
            OPTIONAL_FLAGS+=" --knn_pseudo_labels_path $KNN_PL_PATH"
        fi
    else
        KNN_PL_PATH=""
    fi

    ### RENAME SLURM LOGS ###
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        ln -sf "grpo-tdc-gptoss_${SLURM_JOB_ID}.out" "logs/grpo-tdc-gptoss_${QUANT_LABEL}_${SLURM_JOB_ID}.out" || true
        ln -sf "grpo-tdc-gptoss_${SLURM_JOB_ID}.err" "logs/grpo-tdc-gptoss_${QUANT_LABEL}_${SLURM_JOB_ID}.err" || true
    fi

    ### TRAINING ###
    RUN_LOG="$RUNS_DIR/run_${QUANT_LABEL}.log"
    echo "Logging to: $RUN_LOG"
    KNN_REWARD_FLAGS=""
    if [ "$ENABLE_KNN_TRACKING" = "1" ]; then
        KNN_REWARD_FLAGS+=" --knn_correct_reversal_bonus $KNN_CORRECT_REVERSAL_BONUS"
        KNN_REWARD_FLAGS+=" --knn_correct_stick_delta $KNN_CORRECT_STICK_DELTA"
    fi

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
        --generate_max_len 3072 \
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
        --input_key messages \
        --label_key answer \
        --apply_chat_template \
        --tdc_tools "$TDC_TOOLS_JSON" \
        --tool_version "$TOOL_VERSION" \
        --gradient_checkpointing \
        --vllm_sync_backend $VLLM_SYNC_BACKEND \
        --top_p $TOP_P \
        --temperature $TEMPERATURE \
        --agent_func_path "$AGENT_FUNC_PATH" \
        --agent_max_steps $AGENT_MAX_STEPS \
        --vllm_stop_strings "<|return|>" "<|call|>" \
        --vllm_max_num_seqs $VLLM_MAX_NUM_SEQS \
        --chat_protocol "$CHAT_PROTOCOL" \
        --use_wandb 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_group "$WANDB_GROUP" \
        --wandb_run_name "$RUN_ID" \
        --save_path "$SAVE_PATH" \
        --push_to_hub "$HUB_REPO_ID" \
        --delete_local_after_push \
        --constant_lr_with_warm_up \
        --warmup_steps $WARMUP_STEPS \
        --warm_steps_multiplier_for_correction $WARM_STEPS_MULTIPLIER \
        --attn_implementation "flex_attention" \
        --length_penalty_start 6144 \
        --replace_discarded_prompts_ratio 2.0 \
        --enable_tool_calling_rewards \
        --freeze_router \
        --aux_loss_coef 0 \
        $KNN_REWARD_FLAGS \
        $QUANT_FLAGS \
        $MODE_FLAGS \
        $OPTIONAL_FLAGS \
        $EXTRA_ARGS \
        2>&1 | tee "$RUN_LOG"

    ### CLEANUP ###
    echo "Training complete! Stopping Ray..."
    $CONDA_RAY stop --force || true
}
############################
export -f run_task

mkdir -p logs
srun micromamba run -p $CONDA_ENV_PATH bash -c "run_task"
