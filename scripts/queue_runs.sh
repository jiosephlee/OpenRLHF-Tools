#!/usr/bin/env bash
# Poll until current train_grpo_tdc_gpt_oss_v15.sh run finishes, then chain 3 GRPO runs.
# Run on the same node as the active job (e.g. dgx021), in a detachable shell:
#   nohup bash scripts/queue_runs.sh > runs/queue_runs.out 2>&1 &
#   disown

set -u
cd "$(dirname "$0")/.."

LOG_DIR="runs"
mkdir -p "$LOG_DIR"
STAMP() { date '+%Y-%m-%d %H:%M:%S'; }

WATCH_PATTERN='train_grpo_tdc_gpt_oss_v15\.sh'
PYTHON_PATTERN='openrlhf\.cli\.train_ppo|ray::|openrlhf'

# Robustness: require BOTH the launcher script AND any openrlhf python worker
# to be absent for >= STABLE_SECS continuous seconds. This avoids racing against
# brief gaps between phases (rollout <-> train) or shell wrappers spawning subshells.
STABLE_SECS=60
POLL_SECS=15

echo "[$(STAMP)] queue_runs.sh starting on $(hostname). Watching pattern: $WATCH_PATTERN"
echo "[$(STAMP)] will trigger after no match for ${STABLE_SECS}s of continuous absence."

clear_streak=0
while true; do
  if pgrep -u "$USER" -f "$WATCH_PATTERN" >/dev/null 2>&1 || \
     pgrep -u "$USER" -f "$PYTHON_PATTERN" >/dev/null 2>&1; then
    if (( clear_streak > 0 )); then
      echo "[$(STAMP)] training process reappeared; resetting streak."
    fi
    clear_streak=0
  else
    clear_streak=$((clear_streak + POLL_SECS))
    echo "[$(STAMP)] no training procs (streak=${clear_streak}s / ${STABLE_SECS}s)"
    if (( clear_streak >= STABLE_SECS )); then
      break
    fi
  fi
  sleep "$POLL_SECS"
done

echo "[$(STAMP)] previous run cleared. Launching queued runs sequentially."

COMMON_ENV=(
  TOOL_VERSION=v15_neighbor_only
  DATA_DIR=/vast/home/j/jojolee/OpenRLHF-Tools/data/tdc/openai_format_v15_no_sft_trim_5p4mini_separate_neighbors_only_b100pct_task_smiles_overlap
  PRETRAIN_PATH=jiosephlee/gpt-oss-20b-sft-trim-5.4mini-separate-neighbors-only-b100pct-best-20260425
  TIS=1
  TIS_TYPE=tis
  SMART_REPLAY=1
  DEQUANT=unsloth
  EFFECTIVE_ROLLOUT_BATCH_SIZE=8
  REDUCE_OPTIMIZER=none
  TRAIN_MAX_TOKENS_PER_GPU=32768
  LIGER_GRPO_LOSS=0
  LOSS_TYPE=ppo
)

run_one() {
  local tag="$1"; shift
  local logfile="$LOG_DIR/queued_${tag}.log"
  echo "[$(STAMP)] >>> START $tag -> $logfile"
  env "${COMMON_ENV[@]}" "$@" bash scripts/train_grpo_tdc_gpt_oss_v15.sh \
    > "$logfile" 2>&1
  local rc=$?
  echo "[$(STAMP)] <<< END   $tag rc=$rc"
  return $rc
}

# Run 1: lr 8e-7, mini_grad_steps 4
run_one "lr8e-7_mgs4" LEARNING_RATE=8e-7 EFFECTIVE_MINI_GRADIENT_STEPS=4
rc1=$?

# Run 2: lr 2e-6, mini_grad_steps 2  (always run, even if rc1!=0)
run_one "lr2e-6_mgs2" LEARNING_RATE=2e-6 EFFECTIVE_MINI_GRADIENT_STEPS=2
rc2=$?

# Run 3: lr 4e-7, mini_grad_steps 2
run_one "lr4e-7_mgs2" LEARNING_RATE=4e-7 EFFECTIVE_MINI_GRADIENT_STEPS=2
rc3=$?

echo "[$(STAMP)] all done. rc1=$rc1 rc2=$rc2 rc3=$rc3"
