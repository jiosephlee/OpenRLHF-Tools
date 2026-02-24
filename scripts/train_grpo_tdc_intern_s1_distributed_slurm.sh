#!/bin/bash
#
# SLURM batch version of the Intern-S1-mini GRPO training script (2 actor + 6 vLLM GPUs).
#
# Distributed (non-colocated) mode — Actor and vLLM run on separate GPU sets.
#
# Uses the Intern-S1 JSON tool-calling format:
#   <|action_start|><|plugin|>{"name": "...", "parameters": {...}}<|action_end|>
#
# Usage:
#   sbatch scripts/train_grpo_tdc_intern_s1_distributed_2a6v_slurm.sh
#
# Override defaults via environment:
#   PRETRAIN_PATH=... LEARNING_RATE=5e-7 sbatch scripts/train_grpo_tdc_intern_s1_distributed_2a6v_slurm.sh
#
# Feature flags (set via env before sbatch):
#   TOOL_VERSION=v3          # Tool schema version (default: v3)
#   SMART_REPLAY=1           # Enable smart replay with max_replay_rounds=2
#   CURRICULUM_BALANCED=1    # Enable curriculum-balanced sampling
#

### SLURM PARAMETERS ###
#SBATCH --job-name=grpo-tdc-s1-2a6v
#SBATCH --output=logs/grpo-tdc-s1-2a6v_%j.out
#SBATCH --error=logs/grpo-tdc-s1-2a6v_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00-18:00:00

### SCHEDULER PARAMETERS ###
export OMP_NUM_THREADS=16
export NCCL_NVLS_ENABLE=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_SL=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
export NCCL_SOCKET_IFNAME=bond0
export UCX_TLS=rc

### BEGIN BATCH SCRIPT ###
module load MAMBA
module load cuda/13.1.0
export ENV_NAME="open_rlhf_intern"

############################
#        TASK SCRIPT       #
############################
run_task() {
    set -euo pipefail
    export MALLOC_TRIM_THRESHOLD_=0

    ### ARGS (override via env before sbatch) ###
    PRETRAIN_PATH="${PRETRAIN_PATH:-jiosephlee/sft_intern_distillation_Intern-S1-mini-lm_complet_only_chat_think_lr5e-05}"
    LEARNING_RATE="${LEARNING_RATE:-1e-6}"
    DEBUG_TRACES="${DEBUG_TRACES:-0}"
    NUM_GPUS=$SLURM_GPUS_ON_NODE

    ### FEATURE FLAGS ###
    TOOL_VERSION="${TOOL_VERSION:-v3}"
    SMART_REPLAY="${SMART_REPLAY:-0}"
    CURRICULUM_BALANCED="${CURRICULUM_BALANCED:-0}"

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
    DATE_TAG=$(date +%m%d)
    RUN_NAME="grpo-tdc-s1-${N_TASKS}t-${TOOL_VERSION}-ep${MAX_EPOCHS}-dist-2a6v-${DATE_TAG}"
    RUN_ID="${RUN_NAME}"
    SAVE_PATH="$PROJECT_ROOT/saves/tdc/$RUN_NAME"
    HUB_REPO_ID="jiosephlee/${RUN_NAME}"

    ### GPU LAYOUT (distributed — separate actor and vLLM GPUs) ###
    ACTOR_GPUS=2
    VLLM_GPUS=6
    VLLM_NUM_ENGINES=6
    VLLM_TENSOR_PARALLEL_SIZE=1
    TRAIN_BATCH_SIZE=4

    MIN_GPUS=$((ACTOR_GPUS + VLLM_GPUS))
    if [ "$NUM_GPUS" -lt "$MIN_GPUS" ]; then
        echo "Error: Need at least $MIN_GPUS GPUs ($ACTOR_GPUS actor + $VLLM_GPUS vLLM), got $NUM_GPUS" >&2
        exit 1
    fi

    ### TOOL-CALLING CONFIG ###
    AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"
    AGENT_MAX_STEPS=35
    CHAT_PROTOCOL="intern_s1"

    ### GRPO CONFIG ###
    N_SAMPLES_PER_PROMPT=8
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
    echo "TDC GRPO Training — Intern-S1-mini (DISTRIBUTED, 1 actor + 7 vLLM GPUs, SLURM BATCH)"
    echo "========================================"
    echo "SLURM Job ID: $SLURM_JOB_ID"
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
    echo "----------------------------------------"
    echo "Agent Max Steps: $AGENT_MAX_STEPS"
    echo "Samples per Prompt: $N_SAMPLES_PER_PROMPT"
    echo "Temperature: $TEMPERATURE"
    echo "Top-p: $TOP_P"
    echo "----------------------------------------"
    echo "Smart Replay: $SMART_REPLAY"
    echo "Curriculum Balanced: $CURRICULUM_BALANCED"
    echo "Tool Version: $TOOL_VERSION"
    echo "----------------------------------------"
    echo "W&B: project=$WANDB_PROJECT group=TDC-InternS1-dist-2a6v-$TASK_LABEL run=$RUN_ID"
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

    ### TRAINING ###
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
        --rollout_batch_size $TRAIN_BATCH_SIZE \
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
        --eval_steps 128 \
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
        --async_train \
        --async_queue_size 1 \
        $([ "$DYNAMIC_FILTERING" = true ] && echo "--dynamic_filtering --dynamic_filtering_reward_range $DYNAMIC_FILTERING_REWARD_RANGE" || echo "") \
        --top_p $TOP_P \
        --temperature $TEMPERATURE \
        --agent_func_path "$AGENT_FUNC_PATH" \
        --agent_max_steps $AGENT_MAX_STEPS \
        --vllm_stop_strings "<|action_end|>" "<|im_end|>" \
        --chat_protocol "$CHAT_PROTOCOL" \
        --use_wandb 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_group "TDC-InternS1-dist-2a6v-$TASK_LABEL" \
        --wandb_run_name "$RUN_ID" \
        --save_path "$SAVE_PATH" \
        --push_to_hub "$HUB_REPO_ID" \
        --delete_local_after_push \
        --use_dynamic_batch \
        --constant_lr_with_warm_up \
        $([ "$SMART_REPLAY" = "1" ] && echo "--smart_replay --max_replay_rounds 2" || echo "") \
        $([ "$CURRICULUM_BALANCED" = "1" ] && echo "--curriculum_balanced" || echo "")

    ### CLEANUP ###
    echo "Training complete! Stopping Ray..."
    ray stop --force || true
}
############################
export -f run_task

mkdir -p logs
srun micromamba run -n $ENV_NAME bash -c "run_task"
