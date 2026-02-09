# SLURM Integration for TDC Training

**Date:** 2026-02-09

## Summary

Updated `scripts/train_grpo_tdc.sh` to be fully SLURM-compatible while maintaining standalone mode. The script now automatically detects whether it's running under SLURM and adapts accordingly.

## Key Features

### ✅ SLURM Native Support
- **Auto GPU detection**: Reads `$SLURM_GPUS_ON_NODE` automatically
- **Job ID isolation**: Ray tmpdir includes job ID for parallel jobs
- **SBATCH directives**: Full SLURM configuration in script header
- **Dual mode**: Works seamlessly with both `sbatch` and `bash`

### ✅ Automatic Scaling
- Batch sizes scale with detected GPU count
- vLLM engines auto-configured based on GPUs
- Tensor parallel size adapts to hardware

### ✅ Production Features
- NCCL optimizations for multi-GPU training
- Automatic log management with job ID tracking
- Ray session isolation per SLURM job
- Comprehensive job information in output

## Usage

### SLURM Cluster

```bash
# Submit with defaults (4 GPUs)
sbatch scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Override GPU count
sbatch --gpus=8 scripts/train_grpo_tdc.sh herg_central /path/to/model

# Specify partition and account
sbatch --partition=gpu --account=mylab scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Custom learning rate
sbatch scripts/train_grpo_tdc.sh ClinTox internlm/internlm2_5-7b-chat 5e-7
```

### Standalone (No SLURM)

```bash
# Run directly with bash (4 GPUs)
bash scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat 1e-6 4

# Specify GPU count
bash scripts/train_grpo_tdc.sh hERG /path/to/model 1e-6 2
```

## SLURM Configuration

### Default Directives

Located in script header (`scripts/train_grpo_tdc.sh`):

```bash
#SBATCH --job-name=grpo-tdc
#SBATCH --output=logs/grpo_tdc_%j.out      # %j = job ID
#SBATCH --error=logs/grpo_tdc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4                           # Override with --gpus=N
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
```

### Customize for Your Cluster

Uncomment and edit these lines in the script:

```bash
# #SBATCH --partition=gpu                  # Your partition name
# #SBATCH --account=your_account           # Your account/project
```

### Optional NCCL Settings

For InfiniBand clusters, uncomment in script:

```bash
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
# export NCCL_SOCKET_IFNAME=bond0
# export UCX_TLS=rc
```

## Auto-Detection Details

### GPU Count
```bash
# SLURM mode
NUM_GPUS=$SLURM_GPUS_ON_NODE  # Auto-detected from allocation

# Standalone mode
NUM_GPUS=${4:-4}  # 4th argument or default to 4
```

### Ray Temp Directory
```bash
# SLURM mode (isolated per job)
RAY_TMPDIR=/tmp/ray_${USER}/${SLURM_JOB_ID}

# Standalone mode
RAY_TMPDIR=/tmp/ray_${USER}
```

### Batch Size Scaling
```bash
TRAIN_BATCH_SIZE=$((NUM_GPUS * 16))     # Scales with GPUs
VLLM_NUM_ENGINES=$((NUM_GPUS / 2))      # Half GPUs for engines
```

## Monitoring SLURM Jobs

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# Job history
sacct -j <JOB_ID> --format=JobID,JobName,Elapsed,State,MaxRSS
```

### Monitor Output

```bash
# Watch SLURM output (replace JOB_ID)
tail -f logs/grpo_tdc_<JOB_ID>.out

# Watch Ray logs
tail -f logs/ray/latest/session_latest/logs/worker-*.out

# Check for errors
tail -f logs/grpo_tdc_<JOB_ID>.err
```

### Cancel Job

```bash
scancel <JOB_ID>
```

## Output Locations

### SLURM Mode
- **SLURM logs**: `logs/grpo_tdc_<JOB_ID>.out` and `.err`
- **Ray logs**: `logs/ray/latest/session_latest/`
- **Model saves**: `saves/tdc/<task>/<run_id>/`
- **Checkpoints**: `checkpoints/tdc/<task>/<run_id>/`

### Standalone Mode
- **Ray logs**: `logs/ray/latest/session_latest/`
- **Model saves**: `saves/tdc/<task>/<run_id>/`
- **Checkpoints**: `checkpoints/tdc/<task>/<run_id>/`

## Examples

### Quick Test (Small Dataset)

```bash
# SLURM: 2 GPUs, small dataset
sbatch --gpus=2 scripts/train_grpo_tdc.sh Carcinogens_Lagunin internlm/internlm2_5-7b-chat

# Standalone
bash scripts/train_grpo_tdc.sh Carcinogens_Lagunin internlm/internlm2_5-7b-chat 1e-6 2
```

### Production Training

```bash
# SLURM: 4 GPUs, medium dataset
sbatch scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# SLURM: 8 GPUs, large dataset
sbatch --gpus=8 scripts/train_grpo_tdc.sh herg_central /path/to/model
```

### Multi-Task Job Array

Submit multiple tasks as job array:

```bash
# Create job submission script
cat > submit_all_tasks.sh << 'EOF'
#!/bin/bash
for task in AMES ClinTox hERG HIV; do
    sbatch scripts/train_grpo_tdc.sh $task internlm/internlm2_5-7b-chat
    sleep 1  # Avoid overwhelming scheduler
done
EOF

bash submit_all_tasks.sh
```

## Environment Variables Set

### Automatic
- `IS_SLURM` - true/false based on `$SLURM_JOB_ID`
- `NUM_GPUS` - From SLURM or CLI argument
- `RAY_TMPDIR` - Isolated per job
- `RAY_NODE_IP_ADDRESS` - Auto-detected

### NCCL Optimizations
- `NCCL_NVLS_ENABLE=1`
- `NCCL_IB_ADAPTIVE_ROUTING=1`
- `NCCL_IB_SL=1`
- `NCCL_IB_QPS_PER_CONNECTION=2`
- `NCCL_IB_SPLIT_DATA_ON_QPS=0`

### vLLM
- `VLLM_NO_USAGE_STATS=1`
- `VLLM_DISABLE_TELEMETRY=1`

### OpenRLHF
- `OPENRLHF_MODEL_PATH` - Model path
- `OPENRLHF_PROMPT_CONSTRUCTION_MODE` - Prompt mode
- `OPENRLHF_MAX_STEPS` - Max agent steps

## Troubleshooting

### Issue: Job fails immediately

**Check:**
```bash
# View error log
cat logs/grpo_tdc_<JOB_ID>.err

# Check job status
scontrol show job <JOB_ID>
```

### Issue: GPUs not detected

**Verify SLURM allocation:**
```bash
echo $SLURM_GPUS_ON_NODE  # Should show GPU count
nvidia-smi  # Should list allocated GPUs
```

### Issue: Permission denied on logs/

**Create logs directory:**
```bash
mkdir -p logs
```

### Issue: Ray fails to start

**Check tmpdir:**
```bash
ls -la /tmp/ray_${USER}/
# Clean if needed
rm -rf /tmp/ray_${USER}/*
```

## Changes Made

### Modified Files
1. **`scripts/train_grpo_tdc.sh`**
   - Added SBATCH directives
   - Added SLURM/standalone mode detection
   - Auto GPU detection from `$SLURM_GPUS_ON_NODE`
   - Job ID-based Ray tmpdir isolation
   - NCCL optimizations from original script
   - Enhanced logging with SLURM info

2. **`scripts/TRAINING_README.md`**
   - Added SLURM usage section
   - Added SLURM configuration guide
   - Added SLURM monitoring commands
   - Updated examples for both modes

## Compatibility

- ✅ **SLURM clusters**: Full support with auto-detection
- ✅ **Standalone machines**: Works without SLURM
- ✅ **Mixed environments**: Same script for both
- ✅ **Job arrays**: Supports parallel job submission
- ✅ **Multi-node**: Ready for multi-node expansion

## Related Documentation

- **Training guide**: `scripts/TRAINING_README.md`
- **TDC quickstart**: `TDC_QUICKSTART.md`
- **Integration details**: `TDC_INTEGRATION_COMPLETE.md`

---

**Ready for SLURM!** 🚀
