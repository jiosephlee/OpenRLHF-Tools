#!/bin/bash
#
# Submit the full two-variant local-attribution distillation pipeline:
#   1. trace generation + prompt-swap preparation
#   2. dependent SFT training in therapeutic-tuning
#
# Prints the submitted SLURM job IDs for both variants.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

submit_variant() {
    local variant="$1"
    local prep_job
    local train_job

    prep_job="$(
        sbatch --parsable \
            --export=ALL,DISTILL_VARIANT="$variant",PRETRAIN_PATH="${PRETRAIN_PATH:-openai/gpt-oss-120b}" \
            "$SCRIPT_DIR/slurm_eval_and_prepare_tdc_sft_distill_v16_local_attr_gpt_oss.sh"
    )"

    train_job="$(
        sbatch --parsable \
            --dependency=afterok:"$prep_job" \
            --export=ALL,DISTILL_VARIANT="$variant" \
            "$SCRIPT_DIR/slurm_train_tdc_distill_v16_no_neighbor.sh"
    )"

    echo "$variant prep_job=$prep_job train_job=$train_job"
}

submit_variant local_attribution
submit_variant local_attribution_pretend
