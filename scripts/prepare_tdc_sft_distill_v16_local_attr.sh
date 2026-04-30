#!/bin/bash
#
# Convert a saved distill-trace dump into an SFT-ready messages dataset that
# uses the regular v16_no_neighbor system/user prompt.
#
# Example:
#   TRACE_INPUT=runs/.../sft_distill_traces/eval_step_0.jsonl \
#   DISTILL_VARIANT=local_attribution \
#   bash scripts/prepare_tdc_sft_distill_v16_local_attr.sh
#
set -eo pipefail

TRACE_INPUT="${TRACE_INPUT:?TRACE_INPUT is required}"
DISTILL_VARIANT="${DISTILL_VARIANT:-local_attribution}"
BASE_DATA_DIR="${BASE_DATA_DIR:-data/tdc/openai_format_v16_no_neighbor}"
BASE_SPLIT="${BASE_SPLIT:-train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/tdc/sft_distill_v16_no_neighbor}"

case "$DISTILL_VARIANT" in
    local_attribution)
        OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/local_attribution}"
        ;;
    local_attribution_pretend|pretend)
        OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/local_attribution_pretend}"
        ;;
    *)
        echo "Error: DISTILL_VARIANT must be one of: local_attribution, local_attribution_pretend, pretend" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/convert_eval_sft_distill_traces_to_messages.py" \
    --trace-input "$TRACE_INPUT" \
    --base-data-dir "$BASE_DATA_DIR" \
    --base-split "$BASE_SPLIT" \
    --output-dir "$OUTPUT_DIR"
