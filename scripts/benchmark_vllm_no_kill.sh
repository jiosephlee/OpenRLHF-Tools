#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_FLASHINFER_MOE_BACKEND=latency

MODEL="unsloth/gpt-oss-20b-BF16"
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
    --gpu-memory-utilization 0.95 \
    >"$SERVER_LOG" 2>&1 &

  SERVER_PID=$!
  echo "$SERVER_PID" > "$SERVER_PID_FILE"
  echo "Server pid: $SERVER_PID"
  echo "Logs: $SERVER_LOG"
  echo "PID saved to: $SERVER_PID_FILE"

  echo "Waiting for server to become ready..."
  for i in {1..180}; do
    if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "Server is ready."
      break
    fi
    sleep 2
  done

  if ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server did not become ready. Last 100 log lines:"
    tail -n 100 "$SERVER_LOG" || true
    exit 1
  fi
fi

echo "Running serving benchmark..."
"${VLLM_CLI[@]}" bench serve \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --num-prompts "$NUM_PROMPTS" \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --temperature 0 \
  --ready-check-timeout-sec 300 \
  --request-rate inf \
  --max-concurrency "$MAX_CONCURRENCY"

echo
echo "Server left running."
echo "To kill it later:"
echo "  kill \$(cat $SERVER_PID_FILE)"
echo "Or:"
echo "  pkill -f 'vllm.entrypoints.cli.main serve $MODEL'"