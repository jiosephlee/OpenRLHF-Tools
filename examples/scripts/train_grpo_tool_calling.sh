#!/bin/bash
# Example: GRPO training with tool-calling support
# This script demonstrates how to train an agent that can use tools via GLM Flash format

set -x

### Configuration ###
NUM_GPUS=4
PRETRAIN_MODEL_PATH=${1:-"PATH_TO_YOUR_MODEL"}  # GLM Flash or compatible model
DATA_PATH=${2:-"PATH_TO_YOUR_DATA"}  # JSON dataset with questions
SAVE_PATH="./checkpoint_grpo_tool_calling"

# Tool-calling configuration
AGENT_SCRIPT="openrlhf/utils/tool_calling_turn.py"
AGENT_MAX_STEPS=40  # Max turns per episode
STOP_STRINGS="</tool_call>"  # Stop generation at tool call boundary
PROMPT_MODE="manual"  # "manual" (fast) or "auto" (robust)

# Environment variables for agent
export OPENRLHF_MODEL_PATH="$PRETRAIN_MODEL_PATH"
export OPENRLHF_PROMPT_CONSTRUCTION_MODE="$PROMPT_MODE"
export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_TELEMETRY=1

### Ray Setup ###
ray stop --force || true
ray start --head --num-gpus=$NUM_GPUS --temp-dir=/tmp/ray_tmp

# Wait for Ray to be ready
for i in {1..60}; do
    curl -fsS http://127.0.0.1:8265/api/version >/dev/null && break
    sleep 1
done

### Generate per-task tools JSON (from Intern-S1-recipe source of truth) ###
TDC_TOOLS_JSON="$PROJECT_ROOT/data/tdc/metadata/tools_per_task.json"
python "$PROJECT_ROOT/scripts/generate_tools_json.py" "$TDC_TOOLS_JSON"

### Training ###
python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 0 \
    --ref_num_gpus_per_node 0 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node $NUM_GPUS \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
    --colocate_all_models \
    --advantage_estimator dr_grpo \
    --init_kl_coef 0 \
    --kl_estimator k1 \
    --pretrain $PRETRAIN_MODEL_PATH \
    --save_path $SAVE_PATH \
    --save_steps 20 \
    --logging_steps 1 \
    --n_samples_per_prompt 8 \
    --micro_train_batch_size 8 \
    --train_batch_size 64 \
    --rollout_batch_size 64 \
    --max_epochs 2 \
    --prompt_max_len 2048 \
    --generate_max_len 1024 \
    --max_samples 10000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --prompt_data $DATA_PATH \
    --input_key "question" \
    --label_key "answer" \
    --apply_chat_template \
    --tdc_tools "$TDC_TOOLS_JSON" \
    --gradient_checkpointing \
    --vllm_sync_backend nccl \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --enforce_eager \
    --dynamic_filtering \
    --dynamic_filtering_reward_range 0.2 0.8 \
    --top_p 0.95 \
    --temperature 1.0 \
    --agent_func_path $AGENT_SCRIPT \
    --agent_max_steps $AGENT_MAX_STEPS \
    --vllm_stop_strings $STOP_STRINGS \
    --prompt_construction_mode $PROMPT_MODE \
    --push_to_hub "" \
    --delete_local_after_push \
    --use_wandb $WANDB_API_KEY \
    --wandb_project "openrlhf_tool_calling" \
    --wandb_run_name "grpo_tool_calling_$(date +%Y%m%d_%H%M%S)"

### Cleanup ###
ray stop --force
