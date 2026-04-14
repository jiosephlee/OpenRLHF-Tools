#!/bin/bash
#
# Single-node SLURM wrapper for scripts/eval_grpo_tdc_gpt_oss.sh.
#
# Defaults are set for merged BF16 checkpoint evaluation on 1 GPU.
#
# Usage:
#   sbatch scripts/eval_grpo_tdc_gpt_oss_slurm.sh
#   PRETRAIN_PATH=/path/to/merged/model TOOL_VERSION=v10 sbatch scripts/eval_grpo_tdc_gpt_oss_slurm.sh
#

### SLURM PARAMETERS ###
#SBATCH --job-name=eval-gptoss
#SBATCH --output=logs/eval-gptoss_%j.out
#SBATCH --error=logs/eval-gptoss_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=00-8:00:00

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

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/openrlhf" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -d "$PROJECT_ROOT/openrlhf" ]; then
    echo "Error: Cannot find project root (no 'openrlhf' directory found)" >&2
    exit 1
fi

PRETRAIN_PATH="${PRETRAIN_PATH:-/vast/projects/myatskar/design-documents/hf_home/grpo-tdc-gptoss-dequant-unsloth-16t-v10-ep1-0411_0207-merged-bf16}"
TOOL_VERSION="${TOOL_VERSION:-v10}"
MODE="${MODE:-colocated}"
ACTOR_GPUS="${ACTOR_GPUS:-1}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"
USE_WANDB="${USE_WANDB:-0}"
DEQUANT="${DEQUANT:-unsloth}"
EVAL_SPLITS="${EVAL_SPLITS:-val}"

if [ ! -d "$PRETRAIN_PATH" ]; then
    echo "Error: PRETRAIN_PATH does not exist: $PRETRAIN_PATH" >&2
    exit 1
fi

export PRETRAIN_PATH
export TOOL_VERSION
export MODE
export ACTOR_GPUS
export VLLM_NUM_ENGINES
export USE_WANDB
export DEQUANT
export EVAL_SPLITS

echo "========================================"
echo "SLURM Eval Wrapper"
echo "========================================"
echo "Project root:  $PROJECT_ROOT"
echo "Pretrain path: $PRETRAIN_PATH"
echo "Tool version:  $TOOL_VERSION"
echo "Mode:          $MODE"
echo "Actor GPUs:    $ACTOR_GPUS"
echo "vLLM engines:  $VLLM_NUM_ENGINES"
echo "Eval splits:   $EVAL_SPLITS"
echo "CUDA module:   $CUDA_MODULE"
echo "========================================"

IFS=',' read -r -a SPLIT_LIST <<< "$EVAL_SPLITS"
for EVAL_SPLIT in "${SPLIT_LIST[@]}"; do
    EVAL_SPLIT="$(echo "$EVAL_SPLIT" | xargs)"
    if [ -z "$EVAL_SPLIT" ]; then
        continue
    fi
    export EVAL_SPLIT
    echo
    echo "----------------------------------------"
    echo "Launching eval for split: $EVAL_SPLIT"
    echo "----------------------------------------"
    bash "$PROJECT_ROOT/scripts/eval_grpo_tdc_gpt_oss.sh"
done
