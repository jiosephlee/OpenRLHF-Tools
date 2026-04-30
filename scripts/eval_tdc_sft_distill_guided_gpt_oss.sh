#!/bin/bash
#
# Generate SFT distillation traces on the TDC guided dataset with a GPT-OSS teacher.
#
# Thin wrapper around eval_grpo_tdc_gpt_oss.sh that sets the guided dataset,
# v16_no_neighbor tools, and defaults to the train split.
#
# Usage:
#   bash scripts/eval_tdc_sft_distill_guided_gpt_oss.sh
#   EVAL_SPLIT=val bash scripts/eval_tdc_sft_distill_guided_gpt_oss.sh
#   PRETRAIN_PATH=unsloth/gpt-oss-20b-BF16 bash scripts/eval_tdc_sft_distill_guided_gpt_oss.sh
#
set -eo pipefail

export TOOL_VERSION="${TOOL_VERSION:-v16_no_neighbor}"
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-data/tdc/openai_format_v16_no_neighbor_local_attribution_pretend}"
export EVAL_SPLIT="${EVAL_SPLIT:-train}"
export PRETRAIN_PATH="${PRETRAIN_PATH:-openai/gpt-oss-120b}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8="${VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8:-1}"

# Dump full per-sample eval rollouts for distillation. Disable with SAVE_SFT_DISTILL_TRACES=0.
if [ "${SAVE_SFT_DISTILL_TRACES:-0}" = "1" ]; then
    export EXTRA_ARGS="${EXTRA_ARGS:-} --save_sft_distill_traces"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/eval_grpo_tdc_gpt_oss.sh" "$@"
