#!/usr/bin/env bash
# Usage:
#   sbatch scripts/test_vllm_minimal.slurm.sh
#   MODEL=facebook/opt-125m sbatch scripts/test_vllm_minimal.slurm.sh
#   MODEL=unsloth/gpt-oss-20b-BF16 MAX_MODEL_LEN=512 sbatch scripts/test_vllm_minimal.slurm.sh

#SBATCH --job-name=test-vllm-min
#SBATCH --output=logs/test-vllm-min_%j.out
#SBATCH --error=logs/test-vllm-min_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres-flags=enforce-binding
#SBATCH --sockets-per-node=1
#SBATCH --time=02:00:00

set -euo pipefail
export PYTHONUNBUFFERED=1

module load MAMBA
module load cuda/12.8.1

source /vast/parcc/spack/sw/apps/linux-sapphirerapids/anaconda3-2023.09-0-ieilyrkph5mewqcum3ajc4odlt2vakri/etc/profile.d/conda.sh
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/vast/projects/myatskar/design-documents/conda_env/openrlhf}"
conda activate "${CONDA_ENV_PATH}"

cd /vast/home/j/jojolee/OpenRLHF-Tools

MODEL="${MODEL:-unsloth/gpt-oss-20b-BF16}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CONDA_ENV_PATH=${CONDA_ENV_PATH}"
echo "MODEL=${MODEL}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "MOE_BACKEND=${MOE_BACKEND}"
echo

nvidia-smi
echo

export MODEL TEMPERATURE TOP_P MAX_TOKENS MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION TENSOR_PARALLEL_SIZE MOE_BACKEND

srun python -u scripts/test_vllm_minimal.py
