#!/bin/bash
#
# Merge a PEFT/LoRA adapter into a full Hugging Face checkpoint under SLURM.
#
# Usage:
#   sbatch scripts/merge_lora_adapter_slurm.sh
#   BASE_MODEL_PATH=unsloth/gpt-oss-20b-BF16 \
#   LORA_PATH=jiosephlee/grpo-tdc-gptoss-dequant-unsloth-16t-v10-ep1-0411_0207 \
#   sbatch scripts/merge_lora_adapter_slurm.sh
#
# Optional env vars:
#   BASE_MODEL_PATH   Base model path/repo. Default: unsloth/gpt-oss-20b-BF16
#   LORA_PATH         Adapter path/repo to merge. Default: requested v10 adapter
#   OUTPUT_PATH       Output directory. Default: merged_checkpoints/<adapter>-merged-bf16
#   PARAM_DTYPE       bf16 or fp16. Default: bf16
#   CONDA_ENV_PATH    Conda env path. Default: openrlhf_nightly
#   CUDA_MODULE       CUDA module. Default: cuda/13.1.0
#

### SLURM PARAMETERS ###
#SBATCH --job-name=merge-lora
#SBATCH --output=logs/merge-lora_%j.out
#SBATCH --error=logs/merge-lora_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=00-4:00:00

module load MAMBA
CUDA_MODULE="${CUDA_MODULE:-cuda/13.1.0}"
module load "$CUDA_MODULE"

run_task() {
    PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
    while [ "$PROJECT_ROOT" != "/" ] && [ ! -d "$PROJECT_ROOT/openrlhf" ]; do
        PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
    done
    if [ ! -d "$PROJECT_ROOT/openrlhf" ]; then
        echo "Error: Cannot find project root (no 'openrlhf' directory found)" >&2
        exit 1
    fi

    mkdir -p "$PROJECT_ROOT/logs"

    BASE_MODEL_PATH="${BASE_MODEL_PATH:-unsloth/gpt-oss-20b-BF16}"
    LORA_PATH="${LORA_PATH:-jiosephlee/grpo-tdc-gptoss-dequant-unsloth-16t-v10-ep1-0411_0207}"
    PARAM_DTYPE="${PARAM_DTYPE:-bf16}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-/vast/projects/myatskar/design-documents/hf_home}"

    LORA_TAG="${LORA_PATH##*/}"
    OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_ROOT/${LORA_TAG}-merged-bf16}"

    CONDA_ENV_PATH="${CONDA_ENV_PATH:-/vast/projects/myatskar/design-documents/conda_env/openrlhf_nightly}"

    if ! command -v conda >/dev/null 2>&1; then
        if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
            export PATH="$(dirname "$CONDA_EXE"):$PATH"
        elif [ -x "/vast/parcc/spack/sw/apps/linux-sapphirerapids/anaconda3-2023.09-0-ieilyrkph5mewqcum3ajc4odlt2vakri/bin/conda" ]; then
            export PATH="/vast/parcc/spack/sw/apps/linux-sapphirerapids/anaconda3-2023.09-0-ieilyrkph5mewqcum3ajc4odlt2vakri/bin:$PATH"
        fi
    fi

    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_PATH"

    set -euo pipefail
    mkdir -p "$OUTPUT_ROOT"

    echo "========================================"
    echo "LoRA Merge Job"
    echo "========================================"
    echo "Base model:  $BASE_MODEL_PATH"
    echo "Adapter:     $LORA_PATH"
    echo "Output path: $OUTPUT_PATH"
    echo "Param dtype: $PARAM_DTYPE"
    echo "Python:      $(which python)"
    echo "CUDA module: $CUDA_MODULE"
    echo "========================================"

    python -m openrlhf.cli.lora_combiner \
        --model_path "$BASE_MODEL_PATH" \
        --lora_path "$LORA_PATH" \
        --output_path "$OUTPUT_PATH" \
        --param_dtype "$PARAM_DTYPE"

    echo "Merge complete: $OUTPUT_PATH"
}

run_task
