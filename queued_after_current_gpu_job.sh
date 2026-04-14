#!/usr/bin/env bash

set -euo pipefail

cd /vast/home/j/jojolee/OpenRLHF-Tools

watch_pids=(
  2380214 2404847 2380682 2406342
  2381520 2406344 2382130 2406352
  2379804 2406347 2379434 2406349
  2381667 2406346 2379310 2406350
)

echo "[$(date)] watcher started for PIDs: ${watch_pids[*]}"

while true; do
  active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  still_running=0

  for pid in "${watch_pids[@]}"; do
    if printf '%s\n' "$active_pids" | grep -Fxq "$pid"; then
      still_running=1
      break
    fi
  done

  if [ "$still_running" -eq 0 ]; then
    break
  fi

  sleep 30
done

echo "[$(date)] watched GPU job finished; launching queued command"

LEARNING_RATE=9e-7 \
SMART_REPLAY=1 \
TIS=1 \
TIS_TYPE=icepop \
DEQUANT=unsloth \
EFFECTIVE_ROLLOUT_BATCH_SIZE=8 \
EFFECTIVE_MINI_GRADIENT_STEPS=2 \
REDUCE_OPTIMIZER=none \
TRAIN_MAX_TOKENS_PER_GPU=32768 \
LIGER_GRPO_LOSS=0 \
LOSS_TYPE=cispo \
bash train_grpo_tdc_gpt_oss.sh

status=$?
echo "[$(date)] queued command exited with status $status"
exit "$status"
