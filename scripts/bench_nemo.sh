#!/usr/bin/env bash
set -euo pipefail

MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
HOST="127.0.0.1"
PORT="8000"

VLLM_CLI=(python -m vllm.entrypoints.cli.main)

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
  --kv-cache-dtype fp8 \
  --stream-interval 20 \
  --trust-remote-code