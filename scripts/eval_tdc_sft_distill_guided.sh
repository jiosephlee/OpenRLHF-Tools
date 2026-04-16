#!/bin/bash
#
# Generate SFT distillation traces on TDC train split with feature-guided prompts.
#
# Thin wrapper around eval_grpo_tdc_gpt_oss.sh that sets the guided dataset,
# v14_no_neighbor tools, and defaults to the train split. All env var overrides
# from the base eval script are supported (EVAL_SPLIT, DEQUANT, MODE, etc.).
#
# Usage:
#   # Train split (default):
#   bash scripts/eval_tdc_sft_distill_guided.sh
#
#   # Val split:
#   EVAL_SPLIT=val bash scripts/eval_tdc_sft_distill_guided.sh
#
#   # With unsloth BF16 model:
#   DEQUANT=unsloth bash scripts/eval_tdc_sft_distill_guided.sh
#
set -eo pipefail

export TOOL_VERSION="${TOOL_VERSION:-v14_no_neighbor}"
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-data/tdc/openai_format_v14_no_neighbor_guided}"
export EVAL_SPLIT="${EVAL_SPLIT:-train}"
export PRETRAIN_PATH="${PRETRAIN_PATH:-openai/gpt-oss-120b}"
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8="${VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/eval_grpo_tdc_gpt_oss.sh" "$@"
