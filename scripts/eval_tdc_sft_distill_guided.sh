#!/bin/bash
#
# Backward-compatible entrypoint for guided SFT distillation evals.
# Defaults to the GPT-OSS teacher wrapper.
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/eval_tdc_sft_distill_guided_gpt_oss.sh" "$@"
