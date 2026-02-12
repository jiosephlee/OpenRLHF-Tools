# TDC GRPO Training Guide

## Overview

`train_grpo_tdc.sh` is a production-ready SLURM-compatible script for training models on TDC molecular property prediction datasets using Group Relative Policy Optimization (GRPO) with tool-calling.

**Source**: Adapted from `openrlhf-vlm-fork/batch_scripts/grpo_with_tools.sh`

**Features**:
- ✅ **SLURM native**: Auto-detects GPUs from SLURM allocation
- ✅ **Standalone mode**: Can run without SLURM
- ✅ **Auto-scaling**: Batch sizes scale with GPU count
- ✅ **Dual mode**: Works on clusters and local machines

## Quick Start

### SLURM Cluster (Recommended)

```bash
cd /Users/jlee0/Desktop/research/OpenRLHF-Tools

# Submit to SLURM (auto-detects GPUs from allocation)
sbatch scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Submit with custom learning rate
sbatch scripts/train_grpo_tdc.sh hERG /path/to/glm-flash-model 1e-6

# Override GPU count
sbatch --gpus=8 scripts/train_grpo_tdc.sh ClinTox internlm/internlm2_5-7b-chat

# Specify partition and account
sbatch --partition=gpu --account=mylab scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat
```

### Standalone (No SLURM)

```bash
# Train on AMES dataset with default settings (4 GPUs)
bash scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Train on hERG with custom learning rate
bash scripts/train_grpo_tdc.sh hERG /path/to/glm-flash-model 1e-6

# Train with specific GPU count
bash scripts/train_grpo_tdc.sh ClinTox internlm/internlm2_5-7b-chat 1e-6 4
```

## Usage

### SLURM Mode

```bash
sbatch [slurm-options] scripts/train_grpo_tdc.sh <task_name> <model_path> [learning_rate]
```

**SLURM directives** (edit in script header):
- `--gpus=N` - Number of GPUs (auto-detected from allocation)
- `--partition=NAME` - Partition to use
- `--account=NAME` - Account to charge
- `--time=HH:MM:SS` - Max runtime (default: 12:00:00)
- `--mem-per-gpu=SIZE` - Memory per GPU (default: 128G)

### Standalone Mode

```bash
bash scripts/train_grpo_tdc.sh <task_name> <model_path> [learning_rate] [num_gpus]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `task_name` | TDC dataset name (e.g., AMES, hERG, ClinTox) | AMES |
| `model_path` | Path to pretrained model or HF model ID | internlm/internlm2_5-7b-chat |
| `learning_rate` | Actor learning rate | 1e-6 |
| `num_gpus` | Number of GPUs to use | 4 |

### Available Tasks

See all available tasks:
```bash
ls data/tdc/openai_format/ | grep "_train.jsonl" | sed 's/_train.jsonl//'
```

**23 tasks available:**
- **Toxicity**: AMES, Carcinogens_Lagunin, ClinTox, DILI, Skin_Reaction
- **ADME**: BBB_Martins, Bioavailability_Ma, HIA_Hou, PAMPA_NCATS, Pgp_Broccatelli
- **CYP Inhibition**: CYP1A2/2C9/2C19/2D6/3A4 (multiple variants)
- **Cardiotoxicity**: hERG, hERG_Karim, herg_central
- **Antiviral**: HIV, SARSCoV2_Vitro_Touret

## Configuration

### SLURM Configuration

The script includes SLURM directives in its header. Edit `scripts/train_grpo_tdc.sh` to customize:

```bash
### SLURM DIRECTIVES ###
#SBATCH --job-name=grpo-tdc
#SBATCH --output=logs/grpo_tdc_%j.out      # %j = job ID
#SBATCH --error=logs/grpo_tdc_%j.err
#SBATCH --nodes=1                          # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4                           # Default GPUs (override with --gpus=N)
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00                    # 12 hour limit
# #SBATCH --partition=gpu                  # Uncomment and set your partition
# #SBATCH --account=your_account           # Uncomment and set your account
```

**Auto-Detection:**
- `NUM_GPUS` is automatically set from `$SLURM_GPUS_ON_NODE`
- `RAY_TMPDIR` includes job ID for isolation: `/tmp/ray_${USER}/${SLURM_JOB_ID}`
- Batch sizes scale automatically based on detected GPU count

### Training Hyperparameters

The script uses the following defaults (can be modified in the script):

```bash
# Batch sizes (auto-scaled by GPU count)
TRAIN_BATCH_SIZE = NUM_GPUS * 16
VLLM_NUM_ENGINES = NUM_GPUS / 2

# GRPO settings
N_SAMPLES_PER_PROMPT=8
ADVANTAGE_ESTIMATOR="dr_grpo"
DYNAMIC_FILTERING=true
DYNAMIC_FILTERING_REWARD_RANGE="0.2 0.8"

# Tool-calling settings
AGENT_MAX_STEPS=40
PROMPT_CONSTRUCTION_MODE="manual"  # "manual" or "auto"

# Training loop
MAX_EPOCHS=2
PROMPT_MAX_LEN=4096
GENERATE_MAX_LEN=2048
```

### Tool-Calling Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `AGENT_FUNC_PATH` | `openrlhf/utils/tool_calling_turn.py` | Agent implementation |
| `AGENT_MAX_STEPS` | 40 | Max tool-calling turns per rollout |
| `PROMPT_CONSTRUCTION_MODE` | manual | Prompt reconstruction mode (manual/auto) |

**Prompt Modes:**
- **manual** (default): Fast string concatenation - production use
- **auto**: Robust chat template reconstruction - development/testing

### Environment Variables

Automatically set by the script:

```bash
# vLLM settings
VLLM_NO_USAGE_STATS=1
VLLM_DISABLE_TELEMETRY=1

# OpenRLHF settings (for agent)
OPENRLHF_MODEL_PATH=$PRETRAIN_PATH
OPENRLHF_PROMPT_CONSTRUCTION_MODE=$PROMPT_CONSTRUCTION_MODE
OPENRLHF_MAX_STEPS=$AGENT_MAX_STEPS

# Ray settings
RAY_TMPDIR=/tmp/ray_${USER}
RAY_NODE_IP_ADDRESS=<auto-detected>
```

### W&B Integration

If you have Weights & Biases configured, set your API key:

```bash
export WANDB_API_KEY=your_key_here
bash scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat
```

The script will automatically enable W&B logging if `WANDB_API_KEY` is set.

## Output

### Saved Models

```
saves/tdc/<task_name>/grpo-tdc-<task_name>_<timestamp>_lr<lr>/
├── final/              # Final model checkpoint
└── step_<N>/           # Intermediate checkpoints
```

### Checkpoints

```
checkpoints/tdc/<task_name>/grpo-tdc-<task_name>_<timestamp>_lr<lr>/
├── step_<N>/           # Training checkpoints
└── ...
```

### Logs

```
ray_logs/latest/session_latest/logs/
├── worker-*.out        # Worker stdout
├── worker-*.err        # Worker stderr
└── ...
```

## Monitoring

### SLURM Jobs

```bash
# Check job status
squeue -u $USER

# Monitor SLURM output (replace JOB_ID)
tail -f logs/grpo_tdc_<JOB_ID>.out

# Cancel job
scancel <JOB_ID>

# View job details
scontrol show job <JOB_ID>

# Check resource usage
sacct -j <JOB_ID> --format=JobID,JobName,Elapsed,State,MaxRSS,MaxVMSize
```

### During Training

```bash
# Monitor Ray logs
tail -f logs/ray/latest/session_latest/logs/worker-*.out

# Check training progress
grep "PPO Epoch" logs/ray/latest/session_latest/logs/worker-*.out

# Monitor rewards
grep "reward" logs/ray/latest/session_latest/logs/worker-*.out
```

### After Training

Logs are automatically copied to `ray_logs/latest/session_latest/` on exit.

## Advanced Usage

### Modify Hyperparameters

Edit the script to change defaults:

```bash
# scripts/train_grpo_tdc.sh

# Increase samples per prompt (more diverse rollouts)
N_SAMPLES_PER_PROMPT=16

# Increase max steps (more tool calls)
AGENT_MAX_STEPS=60

# Disable dynamic filtering
DYNAMIC_FILTERING=false

# Use auto prompt mode (more robust)
PROMPT_CONSTRUCTION_MODE="auto"
```

### Multi-Node Training

For distributed training across multiple nodes, modify the script:

```bash
# In the training command section
python -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 2 \           # 2 nodes for actor
    --actor_num_gpus_per_node 4 \   # 4 GPUs per node
    # ... rest of args
```

### Custom Reward Model

Replace the default TDC reward model:

```bash
# In the script configuration
--remote_rm_url "$PROJECT_ROOT/path/to/your_reward_model.py"
```

Your reward model must implement:
```python
class YourRewardModel:
    def get_reward(self, queries, responses, labels):
        # Return list of rewards
        return [compute_reward(r, l) for r, l in zip(responses, labels)]

def get_reward_model():
    return YourRewardModel()
```

## Troubleshooting

### Issue: Data not found

```
Error: Training data not found: data/tdc/openai_format/TASKNAME_train.jsonl
```

**Solution:**
- Check task name spelling
- Verify data conversion completed successfully:
  ```bash
  python scripts/data_conversion/validate_openai_format.py --task <TASK_NAME>
  ```

### Issue: Ray fails to start

```
Error: Ray failed to start or connect
```

**Solution:**
- Clean up previous Ray sessions:
  ```bash
  ray stop --force
  pkill -9 raylet
  rm -rf /tmp/ray_${USER}
  ```
- Check if port 8265 is available:
  ```bash
  lsof -i :8265
  ```

### Issue: Out of memory

```
CUDA out of memory error
```

**Solutions:**
- Reduce batch size:
  ```bash
  # Edit script
  TRAIN_BATCH_SIZE=$((NUM_GPUS * 8))  # Reduced from 16
  ```
- Reduce vLLM memory utilization:
  ```bash
  --vllm_gpu_memory_utilization 0.6  # Reduced from 0.8
  ```
- Use gradient checkpointing (already enabled)
- Reduce max sequence length:
  ```bash
  --prompt_max_len 2048
  --generate_max_len 1024
  ```

### Issue: vLLM generates beyond tool call

```
vLLM continues generating after </tool_call>
```

**Solution:**
- Verify stop strings are set:
  ```bash
  --vllm_stop_strings "</tool_call>"
  ```
- Check model's chat template supports stop strings

### Issue: No rewards / all zeros

```
All rewards are 0.0
```

**Solutions:**
- Check reward model is loading correctly
- Verify answer format matches expectation:
  ```python
  # Test reward model
  python openrlhf/utils/tdc_reward_model.py
  ```
- Check generated text contains final answer:
  ```bash
  grep "Answer:" ray_logs/latest/session_latest/logs/worker-*.out
  ```

### Issue: NCCL errors

```
NCCL timeout or communication error
```

**Solutions:**
- Set NCCL debug mode:
  ```bash
  export NCCL_DEBUG=INFO
  ```
- Check network connectivity between GPUs
- Reduce tensor parallel size:
  ```bash
  --vllm_tensor_parallel_size 1
  ```

## Performance Tips

### For Fast Iteration
- Use small task (e.g., Carcinogens_Lagunin: 344 samples)
- Reduce epochs: `--max_epochs 1`
- Reduce samples per prompt: `N_SAMPLES_PER_PROMPT=4`
- Use manual prompt mode: `PROMPT_CONSTRUCTION_MODE="manual"`

### For Production Training
- Use large task (e.g., herg_central: 306K samples)
- Increase epochs: `--max_epochs 5`
- Increase samples: `N_SAMPLES_PER_PROMPT=16`
- Enable W&B logging
- Save more frequently: `--save_steps 50`

## Examples

### Small-Scale Test

**SLURM:**
```bash
# Quick test on small dataset (2 GPUs)
sbatch --gpus=2 scripts/train_grpo_tdc.sh Carcinogens_Lagunin internlm/internlm2_5-7b-chat 1e-6
```

**Standalone:**
```bash
# Quick test on small dataset
bash scripts/train_grpo_tdc.sh Carcinogens_Lagunin internlm/internlm2_5-7b-chat 1e-6 2
```

### Production Training

**SLURM:**
```bash
# Full AMES training (4 GPUs)
sbatch scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Large-scale hERG training (8 GPUs)
sbatch --gpus=8 scripts/train_grpo_tdc.sh herg_central internlm/internlm2_5-7b-chat
```

**Standalone:**
```bash
# Full AMES training
bash scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat 1e-6 4
```

### Multi-Task Curriculum

**SLURM:**
```bash
# Submit multiple jobs (run in parallel or sequential based on cluster)
for task in Skin_Reaction AMES ClinTox HIV; do
    echo "Submitting $task..."
    sbatch scripts/train_grpo_tdc.sh $task internlm/internlm2_5-7b-chat
done
```

**Standalone:**
```bash
# Train progressively on harder tasks
for task in Skin_Reaction AMES ClinTox HIV; do
    echo "Training on $task..."
    bash scripts/train_grpo_tdc.sh $task internlm/internlm2_5-7b-chat 1e-6 4
done
```

## References

- **Original script**: `openrlhf-vlm-fork/batch_scripts/grpo_with_tools.sh`
- **TDC datasets**: `data/tdc/openai_format/`
- **Tool definitions**: `data/tdc/metadata/tools_*.json`
- **Agent implementation**: `openrlhf/utils/tool_calling_turn.py`
- **Reward model**: `openrlhf/utils/tdc_reward_model.py`

## Related Documentation

- `TDC_QUICKSTART.md` - Quick reference for TDC datasets
- `TDC_INTEGRATION_COMPLETE.md` - Full integration details
- `scripts/data_conversion/README.md` - Data conversion guide
- `openrlhf-vlm-fork/CLAUDE.md` - Tool-calling GRPO architecture

---

**Ready to train!** 🚀
