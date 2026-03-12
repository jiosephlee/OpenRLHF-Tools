#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_FLASHINFER_MOE_BACKEND=latency

MODEL="Qwen/Qwen3.5-35B-A3B"
HOST="127.0.0.1"
PORT="8000"

NUM_PROMPTS=256
INPUT_LEN=1024
OUTPUT_LEN=2048
MAX_CONCURRENCY=32

SERVER_LOG="vllm_server.log"
SERVER_PID_FILE="vllm_server.pid"

VLLM_CLI=(python -m vllm.entrypoints.cli.main)

echo "Using python: $(which python)"
python -c "import vllm; print('vLLM import OK from:', vllm.__file__)"

# Only start server if it is not already up.
if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM server already appears to be running at ${HOST}:${PORT}"
else
  echo "Starting vLLM server for ${MODEL}..."
  "${VLLM_CLI[@]}" serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --reasoning-parser qwen3 \
    --language-model-only \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95