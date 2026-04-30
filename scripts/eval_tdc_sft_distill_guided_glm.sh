#!/bin/bash
#
# Generate SFT distillation traces on the TDC guided/local-attribution dataset
# with a GLM teacher.
#
# This is a standalone GLM analogue of eval_tdc_sft_distill_guided_gpt_oss.sh.
# There is no existing eval_grpo_tdc_glm.sh entrypoint in this repo, so this
# script inlines the eval-only runner setup.
#
# Usage:
#   bash scripts/eval_tdc_sft_distill_guided_glm.sh
#   EVAL_SPLIT=val bash scripts/eval_tdc_sft_distill_guided_glm.sh
#   PRETRAIN_PATH=/path/to/glm bash scripts/eval_tdc_sft_distill_guided_glm.sh
#
set -eo pipefail

PRETRAIN_PATH="${PRETRAIN_PATH:-zai-org/GLM-5.1-FP8}"
MODEL_TAG="${MODEL_TAG:-$(basename "$PRETRAIN_PATH")}"
CONDA_ENV="${CONDA_ENV:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_nightly}"

if ! command -v conda >/dev/null 2>&1; then
    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        export PATH="$(dirname "$CONDA_EXE"):$PATH"
    elif [ -x "/vast/parcc/spack/sw/apps/linux-sapphirerapids/anaconda3-2023.09-0-ieilyrkph5mewqcum3ajc4odlt2vakri/bin/conda" ]; then
        export PATH="/vast/parcc/spack/sw/apps/linux-sapphirerapids/anaconda3-2023.09-0-ieilyrkph5mewqcum3ajc4odlt2vakri/bin:$PATH"
    fi
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -euo pipefail

export DS_SKIP_CUDA_CHECK=1
if [ "${CLEAR_TORCH_COMPILE_CACHE:-0}" = "1" ]; then
    rm -rf ~/.cache/torch/inductor/ /tmp/torchinductor_${USER}/ ~/.cache/vllm/torch_compile_cache/ 2>/dev/null || true
fi

VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"
export VLLM_USE_FLASHINFER_MOE_FP16
unset VLLM_FLASHINFER_MOE_BACKEND

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/openrlhf" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -d "$PROJECT_ROOT/openrlhf" ]; then
    echo "Error: Cannot find project root (no 'openrlhf' directory found)" >&2
    exit 1
fi

NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MODE="${MODE:-colocated}"
TOOL_VERSION="${TOOL_VERSION:-v16_no_neighbor}"
DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-data/tdc/openai_format_v16_no_neighbor_local_attribution_pretend}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
DEBUG_TRACES="${DEBUG_TRACES:-0}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-5120}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-3072}"
TOTAL_MAX_TOKENS_PER_GPU="${TOTAL_MAX_TOKENS_PER_GPU:-$((PROMPT_MAX_LEN + GENERATE_MAX_LEN))}"
MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.1}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
TOP_P="${TOP_P:-0.95}"
TEMPERATURE="${TEMPERATURE:-1.0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="${VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE:-1024}"
AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-30}"
USE_WANDB="${USE_WANDB:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export TORCH_DYNAMO_CACHE_SIZE_LIMIT=1024
export TORCH_DYNAMO_RECOMPILE_LIMIT=1024

case "$EVAL_SPLIT" in
    train|val|test)
        ;;
    *)
        echo "Error: EVAL_SPLIT must be 'train', 'val', or 'test', got '$EVAL_SPLIT'" >&2
        exit 1
        ;;
esac

if [ "$MODE" = "colocated" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-$(( NUM_GPUS / VLLM_TENSOR_PARALLEL_SIZE ))}"
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.675}"
    VLLM_SYNC_BACKEND=nccl
    VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-1}"
    MODE_FLAGS=(--colocate_all_models --vllm_enable_sleep --vllm_sleep_level "$VLLM_SLEEP_LEVEL" --deepspeed_enable_sleep)
    MODE_TAG="colo"
elif [ "$MODE" = "distributed" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:?"MODE=distributed requires ACTOR_GPUS"}"
    VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:?"MODE=distributed requires VLLM_NUM_ENGINES"}"
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.96}"
    VLLM_SYNC_BACKEND=gloo
    MODE_FLAGS=()
    MODE_TAG="dist-${ACTOR_GPUS}a${VLLM_NUM_ENGINES}v"
else
    echo "Error: MODE must be 'colocated' or 'distributed', got '$MODE'" >&2
    exit 1
fi

if [ "$MODE" = "distributed" ]; then
    VLLM_GPUS="$VLLM_NUM_ENGINES"
    MIN_GPUS=$((ACTOR_GPUS + VLLM_GPUS))
    if [ "$NUM_GPUS" -lt "$MIN_GPUS" ]; then
        echo "Error: Need at least $MIN_GPUS GPUs ($ACTOR_GPUS actor + $VLLM_GPUS vLLM), got $NUM_GPUS" >&2
        exit 1
    fi
fi

if [[ "$DATA_DIR_OVERRIDE" = /* ]]; then
    DATA_DIR="$DATA_DIR_OVERRIDE"
else
    DATA_DIR="$PROJECT_ROOT/$DATA_DIR_OVERRIDE"
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR" >&2
    exit 1
fi

DATA_DIR_BASENAME="$(basename "$DATA_DIR")"
case "$DATA_DIR_BASENAME" in
    openai_format_v14_no_neighbor_guided) DATA_TAG="v14nn-guided" ;;
    openai_format_v14_no_neighbor_local_attribution) DATA_TAG="v14nn-localattr" ;;
    openai_format_v14_no_neighbor_local_attribution_pretend) DATA_TAG="v14nn-localattr-pretend" ;;
    openai_format_v16_no_neighbor_guided) DATA_TAG="v16nn-guided" ;;
    openai_format_v16_no_neighbor_local_attribution) DATA_TAG="v16nn-localattr" ;;
    openai_format_v16_no_neighbor_local_attribution_pretend) DATA_TAG="v16nn-localattr-pretend" ;;
    *) DATA_TAG="${DATA_DIR_BASENAME#openai_format_}" ;;
esac

TASK_NAMES=(
    Bioavailability_Ma
    HIA_Hou
    PAMPA_NCATS
    Pgp_Broccatelli
    BBB_Martins
    CYP2C9_Substrate_CarbonMangels
    CYP2D6_Substrate_CarbonMangels
    CYP3A4_Substrate_CarbonMangels
    SARSCoV2_3CLPro_Diamond
    SARSCoV2_Vitro_Touret
    Carcinogens_Lagunin
    hERG
    ClinTox
    DILI
    Skin_Reaction
    AMES
)

for t in "${TASK_NAMES[@]}"; do
    f="$DATA_DIR/${t}_${EVAL_SPLIT}.jsonl"
    if [ ! -f "$f" ]; then
        echo "Error: Eval data not found: $f" >&2
        exit 1
    fi
done

mkdir -p "$PROJECT_ROOT/logs"

N_TASKS=${#TASK_NAMES[@]}
DATE_TAG=$(date +%m%d_%H%M)
CHAT_PROTOCOL="glm51"
RUN_NAME="eval-tdc-glm-${MODEL_TAG}-${N_TASKS}t-${TOOL_VERSION}-${DATA_TAG}-${EVAL_SPLIT}-${MODE_TAG}-${DATE_TAG}"
source "$PROJECT_ROOT/scripts/lib/resolve_runs_dir.sh"
RUN_LOG="$RUNS_DIR/eval_${MODEL_TAG}.log"
SAVE_PATH="${SAVE_PATH:-$RUNS_DIR/eval_only_unused_ckpt}"
WANDB_PROJECT="${WANDB_PROJECT:-openrlhf_tdc_grpo}"
WANDB_GROUP="${WANDB_GROUP:-TDC-GLM-eval-${MODEL_TAG}-${DATA_TAG}-${EVAL_SPLIT}-${MODE_TAG}}"

WANDB_FLAGS=()
if [ "$USE_WANDB" = "1" ]; then
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "Error: WANDB_API_KEY is not set while USE_WANDB=1." >&2
        exit 1
    fi
    WANDB_FLAGS=(--use_wandb 1 --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" --wandb_run_name "$RUN_NAME")
fi

AGENT_FUNC_PATH="$PROJECT_ROOT/openrlhf/utils/tool_calling_turn.py"
TDC_TOOLS_JSON="$PROJECT_ROOT/data/tdc/metadata/tools_per_task_${TOOL_VERSION}.json"
python "$PROJECT_ROOT/scripts/generate_tools_json.py" --version "$TOOL_VERSION"

EVAL_DATA="$DATA_DIR/eval_tdc_${EVAL_SPLIT}.jsonl"
python -c "
import json, sys
tasks = sys.argv[1:]
with open('$EVAL_DATA', 'w') as out:
    for task in tasks:
        with open(f'$DATA_DIR/{task}_${EVAL_SPLIT}.jsonl') as f:
            for line in f:
                rec = json.loads(line)
                rec['datasource'] = task
                rec['eval_split'] = '$EVAL_SPLIT'
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
print(f'Built TDC eval dataset ({\"$EVAL_SPLIT\"}): {sum(1 for _ in open(\"$EVAL_DATA\"))} samples from {len(tasks)} tasks')
" "${TASK_NAMES[@]}"

OPTIONAL_FLAGS=()
if [ -n "$KV_CACHE_DTYPE" ]; then
    OPTIONAL_FLAGS+=(--kv_cache_dtype "$KV_CACHE_DTYPE")
fi
if [ -n "$VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE" ]; then
    OPTIONAL_FLAGS+=(--vllm_cudagraph_max_capture_size "$VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE")
fi
if [ -z "${KNN_PL_PATH:-}" ]; then
    KNN_PL_PATH="$PROJECT_ROOT/data/tdc/metadata/knn_v11_pseudo_labels.json"
fi
if [ -f "$KNN_PL_PATH" ]; then
    OPTIONAL_FLAGS+=(--knn_pseudo_labels_path "$KNN_PL_PATH")
fi
if [ "${SAVE_SFT_DISTILL_TRACES:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS:-} --save_sft_distill_traces"
fi
if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARR=($EXTRA_ARGS)
else
    EXTRA_ARGS_ARR=()
fi

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"
export TRITON_CACHE_DIR="/vast/projects/myatskar/design-documents/.cache/triton"
mkdir -p "$TRITON_CACHE_DIR"
export TORCHINDUCTOR_CACHE_DIR="/vast/projects/myatskar/design-documents/.cache/torch_inductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export OPENRLHF_MODEL_PATH="$PRETRAIN_PATH"
export OPENRLHF_CHAT_PROTOCOL="$CHAT_PROTOCOL"
export OPENRLHF_MAX_STEPS="$AGENT_MAX_STEPS"
export DEBUG_TRACES="$DEBUG_TRACES"
export OPENRLHF_SMILES_ERROR_LOG="$RUNS_DIR/smiles_errors.jsonl"

export RAY_NODE_IP_ADDRESS
RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
ulimit -n 65535 2>/dev/null || true

CONDA_RAY=("$(which python)" -m ray.scripts.scripts)
unset RAY_ADDRESS
"${CONDA_RAY[@]}" stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/ray/session_* 2>/dev/null || true

RAY_PORT=$((6379 + (RANDOM % 1000)))
echo "Starting Ray head node at $RAY_NODE_IP_ADDRESS:$RAY_PORT"
"${CONDA_RAY[@]}" start --head \
    --node-ip-address "$RAY_NODE_IP_ADDRESS" \
    --port "$RAY_PORT" \
    --num-gpus "$NUM_GPUS" \
    --temp-dir "$RAY_TMPDIR"
export RAY_ADDRESS="$RAY_NODE_IP_ADDRESS:$RAY_PORT"

echo "Waiting for Ray..."
RAY_READY=0
for i in {1..90}; do
    if "${CONDA_RAY[@]}" status >/dev/null 2>&1; then
        RAY_READY=1
        break
    fi
    sleep 1
done
if [ "$RAY_READY" -ne 1 ]; then
    echo "Error: Ray never came up after 90 seconds." >&2
    exit 1
fi

echo "========================================"
echo "TDC Eval Only — GLM (MODE=$MODE, MODEL=$MODEL_TAG)"
echo "========================================"
echo "Tasks: ${TASK_NAMES[*]}"
echo "Model: $PRETRAIN_PATH"
echo "Eval Split: $EVAL_SPLIT"
echo "Eval Dataset: $EVAL_DATA"
echo "KNN Pseudo Labels: ${KNN_PL_PATH:-<none>}"
echo "Run Name: $RUN_NAME"
echo "Runs Dir: $RUNS_DIR"
echo "EVAL_TEMPERATURE: $EVAL_TEMPERATURE"
echo "EVAL_N_SAMPLES_PER_PROMPT: $EVAL_N_SAMPLES_PER_PROMPT"
echo "VLLM_NUM_ENGINES: $VLLM_NUM_ENGINES"
echo "========================================"

TRAIN_CMD=(
    python -m openrlhf.cli.train_ppo_ray
    --eval_only
    --pretrain "$PRETRAIN_PATH"
    --ref_num_nodes 0
    --ref_num_gpus_per_node 0
    --reward_num_nodes 0
    --reward_num_gpus_per_node 0
    --actor_num_nodes 1
    --actor_num_gpus_per_node "$ACTOR_GPUS"
    --vllm_num_engines "$VLLM_NUM_ENGINES"
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
    --max_num_batched_tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEM_UTIL"
    --advantage_estimator group_norm
    --init_kl_coef 0
    --kl_estimator k1
    --eps_clip_low_high 0.2 0.272
    --remote_rm_url "$PROJECT_ROOT/openrlhf/utils/tdc_reward_model.py"
    --disable_ds_ckpt
    --logging_steps 1
    --micro_train_batch_size 2
    --micro_rollout_batch_size 2
    --n_samples_per_prompt 2
    --train_batch_size $(( ACTOR_GPUS * 2 ))
    --rollout_batch_size "$ACTOR_GPUS"
    --num_episodes 1
    --prompt_max_len "$PROMPT_MAX_LEN"
    --generate_max_len "$GENERATE_MAX_LEN"
    --max_samples "$MAX_SAMPLES"
    --loss_type ppo
    --use_adaptive_batch
    --train_max_tokens_per_gpu "$TOTAL_MAX_TOKENS_PER_GPU"
    --rollout_max_tokens_per_gpu "$TOTAL_MAX_TOKENS_PER_GPU"
    --enable_prefix_caching
    --zero_stage 2
    --param_dtype bf16
    --actor_learning_rate 1e-6
    --eval_dataset "$EVAL_DATA"
    --eval_steps 1
    --eval_temperature "$EVAL_TEMPERATURE"
    --eval_n_samples_per_prompt "$EVAL_N_SAMPLES_PER_PROMPT"
    --input_key messages
    --label_key answer
    --apply_chat_template
    --tdc_tools "$TDC_TOOLS_JSON"
    --tool_version "$TOOL_VERSION"
    --gradient_checkpointing
    --vllm_sync_backend "$VLLM_SYNC_BACKEND"
    --top_p "$TOP_P"
    --temperature "$TEMPERATURE"
    --agent_func_path "$AGENT_FUNC_PATH"
    --agent_max_steps "$AGENT_MAX_STEPS"
    --vllm_stop_strings "</tool_call>"
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS"
    --chat_protocol "$CHAT_PROTOCOL"
    --wandb_run_name "$RUN_NAME"
    --save_path "$SAVE_PATH"
)

TRAIN_CMD+=("${MODE_FLAGS[@]}")
TRAIN_CMD+=("${OPTIONAL_FLAGS[@]}")
TRAIN_CMD+=("${WANDB_FLAGS[@]}")
TRAIN_CMD+=("${EXTRA_ARGS_ARR[@]}")

"${TRAIN_CMD[@]}" 2>&1 | tee "$RUN_LOG"

echo "Eval complete. Stopping Ray..."
"${CONDA_RAY[@]}" stop --force || true
