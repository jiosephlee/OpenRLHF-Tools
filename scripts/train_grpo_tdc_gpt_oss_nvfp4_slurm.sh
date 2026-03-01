#!/bin/bash
#
# SLURM batch version of the GPT-OSS GRPO training script (NVFP4 variant).
#
# Uses NVFP4 QAT (Quantization-Aware Training) during actor training.
# The model is plain BF16; vLLM loads and serves it as BF16.
# QAT closes the train/inference gap by fake-quantizing MoE expert weights
# to NVFP4 precision (block_size=16, E4M3 scales, per-tensor global scale)
# during forward passes via STE (Straight-Through Estimator).
#
# Key differences from MXFP4 script:
#   - NO VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 (irrelevant for NVFP4)
#   - NO --mxfp4_dequantize (instead, we use --nvfp4_dequantize_base_model)
#   - YES --vllm_sync_fp4 nvfp4 (vLLM loads the packed model, so we must sync in NVFP4)
#   - YES --qat_fp4 nvfp4 (NVFP4 fake-quantization during training via STE)
#
# Supports both colocated and distributed modes via MODE env var.
#
# Usage:
#   # Colocated (default):
#   sbatch scripts/train_grpo_tdc_gpt_oss_nvfp4_slurm.sh
#
#   # Distributed:
#   MODE=distributed ACTOR_GPUS=2 VLLM_NUM_ENGINES=6 sbatch scripts/train_grpo_tdc_gpt_oss_nvfp4_slurm.sh
#
# Feature flags (set via env before sbatch):
#   MODE=colocated|distributed   # Default: colocated
#   TOOL_VERSION=v4              # Tool schema version (default: v4)
#   SMART_REPLAY=1               # Enable smart replay with max_replay_rounds=2
#   CURRICULUM_BALANCED=1        # Enable curriculum-balanced sampling
#   MULTI_STAGE_DISPATCH=1       # Continuous-refill dispatch (best for 2-GPU setups)
#   MAX_EPOCHS=2                 # Training epochs (default: 2)
#   EXTRA_ARGS="..."             # Additional CLI flags
#

### SLURM PARAMETERS ###
#SBATCH --job-name=grpo-tdc-gptoss-nvfp4
#SBATCH --output=logs/grpo-tdc-gptoss-nvfp4_%j.out
#SBATCH --error=logs/grpo-tdc-gptoss-nvfp4_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=768G
#SBATCH --sockets-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=00-1:00:00

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

### ENVIRONMENT SETUP ###
module load MAMBA
module load cuda/12.8.1
export CONDA_ENV_PATH="/vast/projects/myatskar/design-documents/conda_env/openrlhf"

############################
#        TASK SCRIPT       #
############################
run_task() {
    set -euo pipefail
    export DS_SKIP_CUDA_CHECK=1
    # NOTE: No VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 — that flag is MXFP4-only.
    # NVFP4 uses its own kernel backend (select_nvfp4_moe_backend) automatically.

    # Prevent corrupted torch inductor cache from crashing vLLM compilation.
    rm -rf ~/.cache/torch/inductor/ /tmp/torchinductor_${USER}/ ~/.cache/vllm/torch_compile_cache/ 2>/dev/null || true

    ### ARGS (override via env before sbatch) ###
    PRETRAIN_PATH="${PRETRAIN_PATH:-jiosephlee/gpt-oss-20B-NVFP4-packed}"
    LEARNING_RATE="${LEARNING_RATE:-1e-6}"
    DEBUG_TRACES="${DEBUG_TRACES:-0}"
    NUM_GPUS=$SLURM_GPUS_ON_NODE

    ### FEATURE FLAGS ###
    MODE="${MODE:-colocated}"
    TOOL_VERSION="${TOOL_VERSION:-v4}"
    SMART_REPLAY="${SMART_REPLAY:-0}"
    MULTI_STAGE_DISPATCH="${MULTI_STAGE_DISPATCH:-0}"
    LIGER_GRPO_LOSS="${LIGER_GRPO_LOSS:-0}"
    CURRICULUM_BALANCED="${CURRICULUM_BALANCED:-0}"
    MAX_EPOCHS="${MAX_EPOCHS:-1}"
    EXTRA_ARGS="${EXTRA_ARGS:-}"

    ### UNIFIED CONSTANTS ###
    AGENT_MAX_STEPS=30
    ZERO_STAGE=2
    PROMPT_MAX_LEN=8192
    N_SAMPLES_PER_PROMPT=8

    ### MODE-DEPENDENT DEFAULTS ###
    if [ "$MODE" = "colocated" ]; then
        ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS}"
        VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-$NUM_GPUS}"
        ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
        MINI_GRADIENT_STEPS="${MINI_GRADIENT_STEPS:-8}"
        MICRO_TRAIN_BATCH_SIZE=1
        MICRO_ROLLOUT_BATCH_SIZE=2
        VLLM_GPU_MEM_UTIL=0.725
        VLLM_SYNC_BACKEND=nccl
        EVAL_STEPS="${EVAL_STEPS:-32}"
        TRAIN_MAX_TOKENS_PER_GPU=6144
        ROLLOUT_MAX_TOKENS_PER_GPU=$(echo "$TRAIN_MAX_TOKENS_PER_GPU * 3" | bc | awk '{print int($1)}')

    elif [ "$MODE" = "distributed" ]; then
        ACTOR_GPUS="${ACTOR_GPUS:?"MODE=distributed requires ACTOR_GPUS"}"
        VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:?"MODE=distributed requires VLLM_NUM_ENGINES"}"
        ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
        MINI_GRADIENT_STEPS="${MINI_GRADIENT_STEPS:-2}"
        MICRO_TRAIN_BATCH_SIZE=1
        MICRO_ROLLOUT_BATCH_SIZE=2
        VLLM_GPU_MEM_UTIL=0.96
        VLLM_SYNC_BACKEND=gloo
        COLO_ROLLOUT=32; COLO_EVAL=32
        EVAL_STEPS="${EVAL_STEPS:-$(( COLO_EVAL * COLO_ROLLOUT / ROLLOUT_BATCH_SIZE ))}"
        TRAIN_MAX_TOKENS_PER_GPU=8192
        ROLLOUT_MAX_TOKENS_PER_GPU=$(echo "$TRAIN_MAX_TOKENS_PER_GPU * 2" | bc | awk '{print int($1)}')
    else
        echo "Error: MODE must be 'colocated' or 'distributed', got '$MODE'" >&2
        exit 1
    fi

    ### BATCH SIZE DERIVATION ###
    TRAIN_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / MINI_GRADIENT_STEPS ))

    # Assert ratio invariant
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

    ### MODE FLAGS ###
    if [ "$MODE" = "colocated" ]; then
        MODE_FLAGS="--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep --adam_offload"
    else
        MODE_FLAGS="--async_train --async_queue_size 1 --adam_offload"
    fi

    ### WARMUP LOGIC ###
    WARMUP_STEPS=20
    WARM_STEPS_MULTIPLIER=$(( MINI_GRADIENT_STEPS * COLO_ROLLOUT / ROLLOUT_BATCH_SIZE ))
    if [ "$WARM_STEPS_MULTIPLIER" -ne 8 ]; then
        echo "Error: WARM_STEPS_MULTIPLIER should amount to 8 currently regardless of mode." >&2
        exit 1
    fi

    ### MULTI-TASK ###
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
    DATA_DIR="$PROJECT_ROOT/data/tdc/openai_format_gpt_oss"
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
    if [ "$MODE" = "colocated" ]; then
        RUN_NAME="grpo-tdc-gptoss-nvfp4-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-colo-${DATE_TAG}"
        WANDB_GROUP="TDC-GPTOss-NVFP4-colo-$TASK_LABEL"
    else
        RUN_NAME="grpo-tdc-gptoss-nvfp4-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-dist-${LAYOUT_TAG}-${DATE_TAG}"
        WANDB_GROUP="TDC-GPTOss-NVFP4-dist-${LAYOUT_TAG}-$TASK_LABEL"
    fi
    RUN_ID="${RUN_NAME}"
    HUB_NAME="grpo-tdc-gptoss-nvfp4-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-${DATE_TAG}"
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
    export TRITON_CACHE_DIR="/tmp/triton_${USER}"
    mkdir -p "$TRITON_CACHE_DIR"

    export VLLM_NO_USAGE_STATS=1
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1

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
    echo "TDC GRPO Training — GPT-OSS NVFP4 (MODE=$MODE, SLURM BATCH)"
    echo "========================================"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Tasks: ${TASK_NAMES[*]}"
    echo "Model: $PRETRAIN_PATH"
    echo "Chat Protocol: $CHAT_PROTOCOL"
    echo "Quantization: NVFP4 QAT (training only, vLLM serves BF16)"
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
    echo "----------------------------------------"
    echo "Agent Max Steps: $AGENT_MAX_STEPS"
    echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
    echo "Temperature: $TEMPERATURE"
    echo "Top-p: $TOP_P"
    echo "Warmup Steps: $WARMUP_STEPS (multiplier: $WARM_STEPS_MULTIPLIER)"
    echo "----------------------------------------"
    echo "Smart Replay: $SMART_REPLAY"
    echo "Curriculum Balanced: $CURRICULUM_BALANCED"
    echo "Multi Stage Dispatch: $MULTI_STAGE_DISPATCH"
    echo "Liger GRPO Loss: $LIGER_GRPO_LOSS"
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
    if [ "$MULTI_STAGE_DISPATCH" = "1" ]; then
        OPTIONAL_FLAGS+=" --multi_stage_dispatch"
    fi
    if [ "$LIGER_GRPO_LOSS" = "1" ]; then
        OPTIONAL_FLAGS+=" --use_liger_grpo_loss"
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
        --reduce_cuda_graph \
        --kv_cache_dtype fp8 \
        --max_num_batched_tokens 8192 \
        --vllm_gpu_memory_utilization $VLLM_GPU_MEM_UTIL \
        --advantage_estimator $ADVANTAGE_ESTIMATOR \
        --init_kl_coef 0 \
        --kl_estimator k1 \
        --eps_clip_low_high 0.2 0.272 \
        --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py" \
        --save_hf_ckpt \
        --disable_ds_ckpt \
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
        --rollout_max_tokens_per_gpu $ROLLOUT_MAX_TOKENS_PER_GPU \
        --vllm_sync_fp4 nvfp4 \
        --qat_fp4 nvfp4 \
        --nvfp4_dequantize_base_model 2imi9/gpt-oss-20B-NVFP4A16-BF16 \
        --constant_lr_with_warm_up \
        --skip_eval_step_zero \
        --warmup_steps $WARMUP_STEPS \
        --warm_steps_multiplier_for_correction $WARM_STEPS_MULTIPLIER \
        $MODE_FLAGS \
        $OPTIONAL_FLAGS \
        $EXTRA_ARGS

    ### CLEANUP ###
    echo "Training complete! Stopping Ray..."
    ray stop --force || true
}
############################
export -f run_task

mkdir -p logs
srun micromamba run -p $CONDA_ENV_PATH bash -c "run_task"
