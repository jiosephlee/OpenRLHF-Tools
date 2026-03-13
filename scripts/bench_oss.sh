#!/usr/bin/env bash
set -euo pipefail

MODEL="unsloth/gpt-oss-20b-BF16"
HOST="127.0.0.1"
PORT="8000"

NUM_PROMPTS=256
INPUT_LEN=8192
OUTPUT_LEN=2048
CONCURRENCIES=(16 32 64 128)
# export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
SERVER_LOG="vllm_server.log"

VLLM_CLI=(python -m vllm.entrypoints.cli.main)

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping vLLM server (pid=$SERVER_PID)..."
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Using python: $(which python)"
python -c "import vllm; print('vLLM import OK from:', vllm.__file__)"

echo "Starting vLLM server for ${MODEL}..."
"${VLLM_CLI[@]}" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --max-cudagraph-capture-size 2048 \
  --max-num-batched-tokens 8192 \
  --stream-interval 20 \
  --kv_cache_dtype fp8 \
  --trust-remote-code

