# TDC Integration Updates - Follow-up

**Date:** 2026-02-09 (Follow-up)

## Changes Made

### 1. CoT Instruction Externalization ✓

**Problem:** CoT instruction was hardcoded in `tdc_loader.py` as a class constant, making it hard to modify.

**Solution:** Moved to external file for easy editing.

#### Files Changed

1. **`data/tdc/metadata/cot_instruction.txt`** (NEW)
   - Contains the default CoT instruction
   - Can be edited without touching code
   - Default content:
     ```
     Please think step by step and use tools when necessary (**Don't use the same tool more than once**). Then put your final choice ((A) or (B)) after "Answer:"
     ```

2. **`openrlhf/datasets/tdc_loader.py`** (MODIFIED)
   - Removed hardcoded `DEFAULT_COT_INSTRUCTION` constant
   - Added `cot_instruction_path` parameter to `__init__()`
   - Added `_load_cot_instruction()` method to load from file
   - Maintains backward compatibility (can still override with string)

3. **`scripts/data_conversion/convert_tdc_to_openai.py`** (MODIFIED)
   - Added `--cot_instruction_path` argument
   - Passes path to loader initialization

#### Usage

**Edit the instruction:**
```bash
# Modify the file
vim data/tdc/metadata/cot_instruction.txt

# Re-convert datasets with new instruction
python scripts/data_conversion/convert_tdc_to_openai.py --all
```

**Override at runtime:**
```bash
python scripts/data_conversion/convert_tdc_to_openai.py \
    --task AMES \
    --cot_instruction "Your custom instruction here"
```

### 2. GRPO Training Script ✓

**Problem:** User needed a modified version of the GRPO script from `openrlhf-vlm-fork` for TDC datasets.

**Solution:** Created comprehensive training script adapted for TDC.

#### Files Created

1. **`scripts/train_grpo_tdc.sh`** (NEW)
   - Production-ready GRPO training script
   - Adapted from `openrlhf-vlm-fork/batch_scripts/grpo_with_tools.sh`
   - TDC-specific configurations
   - Automatic Ray setup and cleanup
   - W&B integration (optional)

**Features:**
- ✅ Simple CLI: `bash scripts/train_grpo_tdc.sh <task> <model> [lr] [gpus]`
- ✅ Automatic data path resolution
- ✅ Task validation (checks if data exists)
- ✅ Auto-scaled batch sizes based on GPU count
- ✅ Ray log management with automatic copying
- ✅ Environment variable setup
- ✅ Comprehensive error handling

**Configuration:**
```bash
# Training hyperparameters
TRAIN_BATCH_SIZE = NUM_GPUS * 16
VLLM_NUM_ENGINES = NUM_GPUS / 2
N_SAMPLES_PER_PROMPT = 8
ADVANTAGE_ESTIMATOR = "dr_grpo"

# Tool-calling settings
AGENT_MAX_STEPS = 40

# GRPO settings
DYNAMIC_FILTERING = true
DYNAMIC_FILTERING_REWARD_RANGE = "0.2 0.8"
```

2. **`openrlhf/utils/tdc_reward_model.py`** (NEW)
   - Accuracy-based reward model for TDC tasks
   - Extracts final answer from model output
   - Computes binary reward (1.0 for correct, 0.0 for incorrect)
   - Supports multiple answer formats

**Answer Patterns Recognized:**
- `"Answer: (A)"`
- `"Final answer: (B)"`
- `"(A)"` or `"(B)"` at end of text

**API:**
```python
class TDCRewardModel:
    def get_reward(self, queries, responses, labels):
        return [compute_reward(r, l) for r, l in zip(responses, labels)]
```

3. **`scripts/TRAINING_README.md`** (NEW)
   - Comprehensive training guide
   - Usage examples
   - Configuration details
   - Troubleshooting section
   - Performance tips

#### Usage Examples

**Quick start:**
```bash
# Train on AMES with defaults
bash scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Train on hERG with custom LR and GPUs
bash scripts/train_grpo_tdc.sh hERG /path/to/glm-flash 1e-6 4
```

**Output locations:**
- Models: `saves/tdc/<task>/<run_id>/`
- Checkpoints: `checkpoints/tdc/<task>/<run_id>/`
- Logs: `ray_logs/latest/session_latest/`

### 3. Documentation Updates ✓

**Updated files:**
1. **`TDC_QUICKSTART.md`**
   - Added training script quick start section
   - Added CoT instruction customization guide
   - Updated file locations
   - Added troubleshooting Q&A

## File Summary

### New Files (5)
1. `data/tdc/metadata/cot_instruction.txt` - Editable CoT prompt
2. `scripts/train_grpo_tdc.sh` - TDC GRPO training script
3. `openrlhf/utils/tdc_reward_model.py` - TDC reward model
4. `scripts/TRAINING_README.md` - Training guide
5. `TDC_UPDATES_SUMMARY.md` - This file

### Modified Files (3)
1. `openrlhf/datasets/tdc_loader.py` - Load CoT from file
2. `scripts/data_conversion/convert_tdc_to_openai.py` - Support CoT path arg
3. `TDC_QUICKSTART.md` - Updated with new features

## Verification

### 1. Test CoT Instruction Loading

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('openrlhf/datasets')))
from tdc_loader import TDCDatasetLoader

loader = TDCDatasetLoader()
print('✓ CoT instruction loaded:')
print(repr(loader.cot_instruction))
"
```

**Output:**
```
✓ CoT instruction loaded:
'Please think step by step and use tools when necessary (**Don't use the same tool more than once**). Then put your final choice ((A) or (B)) after "Answer:"'
```

### 2. Test Reward Model

```bash
python openrlhf/utils/tdc_reward_model.py
```

**Output:**
```
Testing TDC Reward Model
============================================================
✓ Test 1: reward=1.0 (expected 1.0)
✓ Test 2: reward=1.0 (expected 1.0)
✓ Test 3: reward=1.0 (expected 1.0)
✓ Test 4: reward=0.0 (expected 0.0)
✓ Test 5: reward=0.0 (expected 0.0)
============================================================
All tests passed!
```

### 3. Validate Training Script

```bash
# Check script is executable
ls -lh scripts/train_grpo_tdc.sh

# Dry-run check (will fail at Ray start, but validates paths)
bash -n scripts/train_grpo_tdc.sh  # Syntax check
```

## Key Improvements

### Before
- ❌ CoT instruction hardcoded in Python class
- ❌ No ready-to-use training script for TDC
- ❌ Users had to manually configure GRPO arguments
- ❌ No TDC-specific reward model

### After
- ✅ CoT instruction in editable text file
- ✅ Production-ready `train_grpo_tdc.sh` script
- ✅ One-line training: `bash scripts/train_grpo_tdc.sh AMES <model>`
- ✅ TDC reward model with comprehensive testing
- ✅ Comprehensive training documentation

## Next Steps

### For Users

1. **Test small-scale training:**
   ```bash
   bash scripts/train_grpo_tdc.sh Carcinogens_Lagunin internlm/internlm2_5-7b-chat 1e-6 2
   ```

2. **Customize CoT instruction:**
   ```bash
   vim data/tdc/metadata/cot_instruction.txt
   python scripts/data_conversion/convert_tdc_to_openai.py --all
   ```

3. **Monitor training:**
   ```bash
   tail -f ray_logs/latest/session_latest/logs/worker-*.out
   ```

### For Development

1. **Implement advanced reward model** - Consider:
   - Partial credit for reasoning quality
   - Tool usage efficiency bonuses
   - Confidence calibration
   - Integration with actual reward model

2. **Add evaluation scripts** - Create:
   - Test set evaluation
   - Tool usage analysis
   - Performance comparison (GRPO vs SFT)

3. **Multi-task training** - Support:
   - Combined dataset loading
   - Task-specific reward weights
   - Curriculum learning

## References

- **Original GRPO script**: `openrlhf-vlm-fork/batch_scripts/grpo_with_tools.sh`
- **OpenRLHF VLM fork docs**: `openrlhf-vlm-fork/CLAUDE.md`
- **TDC integration docs**: `TDC_INTEGRATION_COMPLETE.md`
- **Training guide**: `scripts/TRAINING_README.md`

---

**All follow-up changes complete!** ✓
