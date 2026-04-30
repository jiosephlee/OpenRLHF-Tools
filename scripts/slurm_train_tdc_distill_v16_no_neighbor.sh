#!/bin/bash
#
# SLURM wrapper: run therapeutic-tuning SFT on one prepared v16_no_neighbor
# distill dataset variant.
#
# Usage:
#   DISTILL_VARIANT=local_attribution \
#     sbatch scripts/slurm_train_tdc_distill_v16_no_neighbor.sh
#

#SBATCH --job-name=tdc-v16nn-sft
#SBATCH --output=logs/tdc-v16nn-sft_%j.out
#SBATCH --error=logs/tdc-v16nn-sft_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64
#SBATCH --time=04-00:00:00

set -euo pipefail

module load MAMBA
CUDA_MODULE="${CUDA_MODULE:-cuda/13.1.0}"
module load "$CUDA_MODULE"

if ! command -v conda >/dev/null 2>&1; then
    export PATH="/vast/parcc/spack/sw/apps/linux-sapphirerapids/anaconda3-2023.09-0-ieilyrkph5mewqcum3ajc4odlt2vakri/bin:$PATH"
fi
eval "$(conda shell.bash hook)"
conda activate /vast/projects/myatskar/design-documents/conda_env/openrlhf

THERA_ROOT="${THERA_ROOT:-/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning}"
DISTILL_VARIANT="${DISTILL_VARIANT:-local_attribution}"

cd "$THERA_ROOT"

echo "========================================"
echo "TDC v16_no_neighbor SFT"
echo "========================================"
echo "Therapeutic root: $THERA_ROOT"
echo "Variant:          $DISTILL_VARIANT"
echo "CUDA module:      $CUDA_MODULE"
echo "Conda env:        ${CONDA_DEFAULT_ENV:-unknown}"
echo "========================================"

bash scripts/Learning/train_tdc_distill_v16_no_neighbor.sh
