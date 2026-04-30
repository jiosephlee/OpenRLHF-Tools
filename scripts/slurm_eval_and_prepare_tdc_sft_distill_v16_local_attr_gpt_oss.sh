#!/bin/bash
#
# SLURM wrapper: generate v16_no_neighbor distill traces for a local-attribution
# variant, convert them into SFT-ready messages with the regular v16_no_neighbor
# prompt, and copy the dataset into therapeutic-tuning.
#
# Usage:
#   PRETRAIN_PATH=openai/gpt-oss-120b DISTILL_VARIANT=local_attribution \
#     sbatch scripts/slurm_eval_and_prepare_tdc_sft_distill_v16_local_attr_gpt_oss.sh
#

#SBATCH --job-name=tdc-distill-prep
#SBATCH --output=logs/tdc-distill-prep_%j.out
#SBATCH --error=logs/tdc-distill-prep_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=01-00:00:00

set -euo pipefail

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

module load MAMBA
CUDA_MODULE="${CUDA_MODULE:-cuda/13.1.0}"
module load "$CUDA_MODULE"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/openrlhf" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -d "$PROJECT_ROOT/openrlhf" ]; then
    echo "Error: Cannot find project root (no 'openrlhf' directory found)" >&2
    exit 1
fi

DISTILL_VARIANT="${DISTILL_VARIANT:-local_attribution}"
case "$DISTILL_VARIANT" in
    local_attribution)
        DATA_TAG="v16nn-localattr"
        ;;
    local_attribution_pretend|pretend)
        DISTILL_VARIANT="local_attribution_pretend"
        DATA_TAG="v16nn-localattr-pretend"
        ;;
    *)
        echo "Error: DISTILL_VARIANT must be local_attribution or local_attribution_pretend" >&2
        exit 1
        ;;
esac

PRETRAIN_PATH="${PRETRAIN_PATH:-openai/gpt-oss-120b}"
THERA_ROOT="${THERA_ROOT:-/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning}"
THERA_DATA_ROOT="${THERA_DATA_ROOT:-$THERA_ROOT/data/sft_distill_v16_no_neighbor}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-$PROJECT_ROOT/data/tdc/sft_distill_v16_no_neighbor}"

export PRETRAIN_PATH
export DISTILL_VARIANT
export TOOL_VERSION="${TOOL_VERSION:-v16_no_neighbor}"
export EVAL_SPLIT="${EVAL_SPLIT:-train}"
export SAVE_SFT_DISTILL_TRACES="${SAVE_SFT_DISTILL_TRACES:-1}"
export USE_WANDB="${USE_WANDB:-0}"
export MODE="${MODE:-colocated}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export CONDA_ENV="${CONDA_ENV:-/vast/projects/myatskar/design-documents/conda_env/openrlhf}"

cd "$PROJECT_ROOT"

echo "========================================"
echo "TDC Distill Eval + Prepare"
echo "========================================"
echo "Project root:      $PROJECT_ROOT"
echo "Variant:           $DISTILL_VARIANT"
echo "Data tag:          $DATA_TAG"
echo "Pretrain path:     $PRETRAIN_PATH"
echo "Tool version:      $TOOL_VERSION"
echo "Eval split:        $EVAL_SPLIT"
echo "Thera data root:   $THERA_DATA_ROOT"
echo "Local output root: $LOCAL_OUTPUT_ROOT"
echo "CUDA module:       $CUDA_MODULE"
echo "========================================"

bash scripts/eval_tdc_sft_distill_v16_local_attr_gpt_oss.sh

TRACE_FILE="$(ls -td "$PROJECT_ROOT"/runs/eval-tdc-gptoss-*-"$DATA_TAG"-"${EVAL_SPLIT}"-*/sft_distill_traces/eval_step_*.jsonl 2>/dev/null | head -n 1 || true)"
if [ -z "$TRACE_FILE" ]; then
    echo "Error: failed to locate saved trace file for data tag $DATA_TAG" >&2
    exit 1
fi

echo "Using trace file: $TRACE_FILE"
TRACE_INPUT="$TRACE_FILE" DISTILL_VARIANT="$DISTILL_VARIANT" OUTPUT_ROOT="$LOCAL_OUTPUT_ROOT" \
    bash scripts/prepare_tdc_sft_distill_v16_local_attr.sh

SRC_DIR="$LOCAL_OUTPUT_ROOT/$DISTILL_VARIANT"
DST_DIR="$THERA_DATA_ROOT/$DISTILL_VARIANT"
mkdir -p "$DST_DIR"
rm -f "$DST_DIR"/*.jsonl
cp -a "$SRC_DIR"/. "$DST_DIR"/

echo "Copied prepared dataset to: $DST_DIR"
