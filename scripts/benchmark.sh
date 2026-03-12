#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3.5-35B-A3B"
HOST="127.0.0.1"
PORT="8000"

NUM_PROMPTS=256
INPUT_LEN=8192
OUTPUT_LEN=2048
CONCURRENCIES=(16 32 64 128)

VLLM_CLI=(python -m vllm.entrypoints.cli.main)

echo "Using python: $(which python)"
python -c "import vllm; print('vLLM import OK from:', vllm.__file__)"

echo "Checking server readiness at http://${HOST}:${PORT}/v1/models ..."
if ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "Server is not ready or not reachable."
  exit 1
fi

for MAX_CONCURRENCY in "${CONCURRENCIES[@]}"; do
  echo
  echo "=================================================="
  echo "Running serving benchmark with max_concurrency=${MAX_CONCURRENCY}"
  echo "=================================================="

  "${VLLM_CLI[@]}" bench serve \
    --backend openai \
    --endpoint /v1/completions \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --temperature 0 \
    --request-rate inf \
    --max-concurrency "$MAX_CONCURRENCY" \
    --num-warmups 1 \
    --ready-check-timeout-sec 300
done