#!/bin/bash

### PROJECT PARAMETERS ###
#SBATCH --job-name=grpo
#SBATCH --output=batch_scripts/logs/grpo_%a.out
#SBATCH --error=batch_scripts/logs/grpo_%a.err
#SBATCH --chdir=/vast/projects/myatskar/lab/cylumn/projects/vlm-reasoning

### JOB PARAMETERS ###
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=7
#SBATCH --time=00-12:00:00

#SBATCH --exclude=dgx011

### SCHEDULER PARAMETERS ###
# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=16
export NCCL_NVLS_ENABLE=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_SL=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
export NCCL_SOCKET_IFNAME=bond0
export UCX_TLS=rc

### BEGIN BATCH SCRIPT ###
module load MAMBA
export ENV_NAME="openrlhf"

############################
#        TASK SCRIPT       #
############################
run_task() {
    set -euo pipefail

    DATASET="vindr-cxr-single-label"
    # BASE 8B MODEL
    # PRETRAIN_PATH="$HF_MODEL_SNAPSHOT_DIR/Qwen_Qwen3-VL-8B-Thinking"
    # SEED_SFT MODEL
    PRETRAIN_PATH="saves/mixes/cxr/vcsl-gts-u82_rit-gts/final/sft-vcsl-gts-u82_rit-gts_2026-01-07_14-38-09_lr1e-6"

    DATA_SPLIT_SWEEP=("ground_truth_sft" "ground_truth_sft_rag90_src-oai_deep_research")
    DATA_SPLIT=${DATA_SPLIT_SWEEP[$SLURM_ARRAY_TASK_ID]}
    LEARNING_RATE="1e-6"

    ### select data split by array index ###
    RUN_ID="grpo-${DATA_SPLIT}_${RUN_ID}_lr${LEARNING_RATE}_r3"
    NUM_GPUS=$SLURM_GPUS_ON_NODE
    TRAIN_BATCH_SIZE=$((NUM_GPUS * 16))
    VLLM_NUM_ENGINES=$((NUM_GPUS / 2))

    ### RAY TMPDIR and SETTINGS ###
    export RAY_TMPDIR="/tmp/ray_cylumn/${SLURM_JOB_ID}"
    mkdir -p "$RAY_TMPDIR"
    PERSIST_RAY_DIR="$PWD/batch_scripts/ray_logs/latest"

    copy_ray_logs () {
        set +e
        local base="$RAY_TMPDIR/ray"
        local latest="$base/session_latest"
        local real="$latest"
        [ -L "$latest" ] && real="$(readlink -f "$latest")"

        echo "Copying Ray logs from: $real"

        # give buffered writers a chance to flush
        sync
        sleep 2

        mkdir -p "$PERSIST_RAY_DIR"
        rm -rf "$PERSIST_RAY_DIR/session_latest" 2>/dev/null
        cp -a "$real" "$PERSIST_RAY_DIR/session_latest" 2>/dev/null || true

        # show sizes so you can confirm non-empty
        echo "Top non-empty logs (source):"
        find "$real/logs" -type f -size +0c 2>/dev/null | head -n 20 || true

        echo "Top non-empty logs (dest):"
        find "$PERSIST_RAY_DIR/session_latest/logs" -type f -size +0c 2>/dev/null | head -n 20 || true
    }
    trap copy_ray_logs EXIT

    # Ensure worker processes use the same python as your run
    export RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')

    PY_EXE=$(micromamba run -n "$ENV_NAME" which python)
    export RAY_PYTHON_EXECUTABLE="$PY_EXE"
    # (Optional) increase fd limit; helps with raylet sockets
    ulimit -n 65535
    # Clean up any previous Ray state
    ray stop --force || true
    ###############################

    ### VLLM SETTINGS ###
    export VLLM_NO_USAGE_STATS=1
    export VLLM_DISABLE_TELEMETRY=1   # if supported by your version
    #####################

    echo "----------------------------------------"
    echo "GRPO Pretrained path: $PRETRAIN_PATH"
    echo "Data split: $DATA_SPLIT"
    echo "Learning Rate: $LEARNING_RATE"
    echo "Run ID: $RUN_ID"
    echo "----------------------------------------"
    echo "Node ID: $SLURM_NODEID"
    echo "SLURM PROC ID: $SLURM_PROCID"
    echo "----------------------------------------"
    echo "RAY_NODE_IP_ADDRESS: $RAY_NODE_IP_ADDRESS"
    echo "----------------------------------------"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
    echo "----------------------------------------"

    ray start --head \
        --node-ip-address $RAY_NODE_IP_ADDRESS \
        --num-gpus $NUM_GPUS \
        --temp-dir $RAY_TMPDIR &
    
    for i in {1..60}; do
        curl -fsS http://127.0.0.1:8265/api/version >/dev/null && break
        sleep 1
    done

    python -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes 0 \
        --ref_num_gpus_per_node 0 \
        --reward_num_nodes 0 \
        --reward_num_gpus_per_node 0 \
        --actor_num_nodes 1 \
        --actor_num_gpus_per_node $NUM_GPUS \
        --vllm_num_engines $VLLM_NUM_ENGINES \
        --vllm_tensor_parallel_size 2 \
        --colocate_all_models \
        --vllm_gpu_memory_utilization 0.8 \
        --advantage_estimator dr_grpo \
        --init_kl_coef 0 \
        --kl_estimator k1 \
        --eps_clip_low_high 0.2 0.272 \
        --pretrain $PRETRAIN_PATH \
        --save_path ./saves/$DATASET/final/$RUN_ID \
        --ckpt_path ./saves/$DATASET/checkpoint/$RUN_ID \
        --remote_rm_url ./models/reward_grpo.py \
        --save_steps 20 \
        --logging_steps 1 \
        --n_samples_per_prompt 8 \
        --micro_train_batch_size 8 \
        --micro_rollout_batch_size 16 \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --rollout_batch_size $TRAIN_BATCH_SIZE \
        --max_epochs 2 \
        --prompt_max_len 4096 \
        --generate_max_len 2048 \
        --max_samples 1_000_000 \
        --zero_stage 3 \
        --bf16 \
        --actor_learning_rate $LEARNING_RATE \
        --prompt_data generations_rollouts/$DATASET/train/rollouts/$DATA_SPLIT.json \
        --input_key question \
        --label_key response_chosen \
        --apply_chat_template \
        --gradient_checkpointing \
        --vllm_sync_backend nccl \
        --vllm_enable_sleep \
        --deepspeed_enable_sleep \
        --enforce_eager \
        --dynamic_filtering \
        --dynamic_filtering_reward_range 0.2 0.8 \
        --top_p 0.95 \
        --temperature 1.0 \
        --use_wandb $WANDB_API_KEY \
        --wandb_group "$DATASET ($DATA_SPLIT)" \
        --wandb_run_name $RUN_ID

    ray stop --force || true
}
############################
export -f run_task

export RUN_ID=$(date +%Y-%m-%d_%H-%M-%S)

srun micromamba run -n $ENV_NAME bash -c "run_task"