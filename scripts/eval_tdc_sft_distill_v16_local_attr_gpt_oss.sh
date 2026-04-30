#!/bin/bash
#
# Run GPT-OSS distill-trace eval on the v16_no_neighbor local-attribution
# train split, with inference-time tooling fixed to v16_no_neighbor.
#
# Variants:
#   DISTILL_VARIANT=local_attribution
#   DISTILL_VARIANT=local_attribution_pretend
#
# Example:
#   PRETRAIN_PATH=openai/gpt-oss-120b DISTILL_VARIANT=local_attribution \
#     bash scripts/eval_tdc_sft_distill_v16_local_attr_gpt_oss.sh
#
set -eo pipefail

DISTILL_VARIANT="${DISTILL_VARIANT:-local_attribution}"

case "$DISTILL_VARIANT" in
    local_attribution)
        DATA_DIR="data/tdc/openai_format_v16_no_neighbor_local_attribution"
        ;;
    local_attribution_pretend|pretend)
        DATA_DIR="data/tdc/openai_format_v16_no_neighbor_local_attribution_pretend"
        ;;
    *)
        echo "Error: DISTILL_VARIANT must be one of: local_attribution, local_attribution_pretend, pretend" >&2
        exit 1
        ;;
esac

export TOOL_VERSION="${TOOL_VERSION:-v16_no_neighbor}"
export EVAL_SPLIT="${EVAL_SPLIT:-train}"
export SAVE_SFT_DISTILL_TRACES="${SAVE_SFT_DISTILL_TRACES:-1}"
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-$DATA_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/eval_tdc_sft_distill_guided_gpt_oss.sh" "$@"
