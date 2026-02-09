# TDC Dataset Quick Start Guide

## Overview

23 TDC molecular property prediction datasets ready for GRPO tool-calling training.

## Quick Commands

```bash
cd /Users/jlee0/Desktop/research/OpenRLHF-Tools

# Validate a dataset
python scripts/data_conversion/validate_openai_format.py --task AMES

# Run example
python examples/tdc_grpo_example.py

# Convert additional tasks (if needed)
python scripts/data_conversion/convert_tdc_to_openai.py --task <TASK_NAME>
```

## Available Datasets

| Category | Tasks | Records |
|----------|-------|---------|
| **Toxicity** | AMES, Carcinogens_Lagunin, ClinTox, DILI, Skin_Reaction | ~10K |
| **ADME** | BBB_Martins, Bioavailability_Ma, HIA_Hou, PAMPA_NCATS, Pgp_Broccatelli | ~7K |
| **CYP Inhibition** | CYP1A2/2C9/2C19/2D6/3A4 (multiple variants) | ~100K |
| **Cardiotoxicity** | hERG, hERG_Karim, herg_central | ~320K |
| **Antiviral** | HIV, SARSCoV2_Vitro_Touret | ~47K |

**Total: ~458,000 records across 23 tasks**

## Python Usage

```python
from transformers import AutoTokenizer
from openrlhf.datasets.tdc_grpo_dataset import load_tdc_grpo_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat")

# Load dataset
dataset = load_tdc_grpo_dataset(
    data_path="data/tdc/openai_format/AMES_train.jsonl",
    tokenizer=tokenizer,
    tool_mode="TaskSpecific",
    task_name="AMES",
)

# Use in training
for item in dataset:
    question = item["question"]  # Templated prompt with tools
    answer = item["answer"]      # "(A)" or "(B)"
```

## Training with OpenRLHF

### Quick Start (Recommended)

**SLURM Cluster:**
```bash
# Auto-detects GPUs from SLURM allocation
sbatch scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Override GPU count
sbatch --gpus=8 scripts/train_grpo_tdc.sh hERG /path/to/glm-flash-model
```

**Standalone:**
```bash
# Train on AMES dataset (4 GPUs)
bash scripts/train_grpo_tdc.sh AMES internlm/internlm2_5-7b-chat

# Train on hERG with custom settings
bash scripts/train_grpo_tdc.sh hERG /path/to/glm-flash-model 1e-6 4
```

See `SLURM_INTEGRATION.md` for SLURM configuration details.

### Advanced (Direct Python)

```bash
python -m openrlhf.cli.train_ppo_ray \
    --pretrain internlm/internlm2_5-7b-chat \
    --prompt_data data/tdc/openai_format/AMES_train.jsonl \
    --input_key "question" \
    --label_key "answer" \
    --agent_func_path openrlhf/utils/tool_calling_agent.py \
    --agent_max_steps 40 \
    --n_samples_per_prompt 8 \
    --advantage_estimator dr_grpo \
    --dynamic_filtering \
    --dynamic_filtering_reward_range 0.2 0.8 \
    --remote_rm_url openrlhf/utils/tdc_reward_model.py
```

## Data Format

Each record has this structure:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "answer": "(A)",
  "smiles": "CCO",
  "label": 0,
  "task": "AMES"
}
```

At load time, `tokenizer.apply_chat_template()` adds:
- Tool definitions (task-specific or all)
- System prompt
- Generation prompt
- Thinking mode support

## Tool Modes

1. **TaskSpecific** (recommended) - 8-15 curated tools per task
2. **All** - All 47 available tools
3. **None** - No tools (baseline)

## File Locations

- **Raw data:** `data/tdc/raw/`
- **Converted data:** `data/tdc/openai_format/`
- **Tool definitions:** `data/tdc/metadata/tools_*.json`
- **Prompts:** `data/tdc/metadata/prompts.json`
- **CoT instruction:** `data/tdc/metadata/cot_instruction.txt` (editable!)
- **Training script:** `scripts/train_grpo_tdc.sh`
- **Reward model:** `openrlhf/utils/tdc_reward_model.py`

## Recommended Starting Points

1. **Small task:** Carcinogens_Lagunin (344 records)
2. **Medium task:** AMES (7,278 records)
3. **Large task:** HIV (45,710 records)
4. **Extra large:** herg_central (306,893 records)

## Documentation

- **Full details:** `TDC_INTEGRATION_COMPLETE.md`
- **Conversion guide:** `scripts/data_conversion/README.md`
- **Dataset stats:** `data/tdc/CONVERSION_SUMMARY.md`

## Customization

### Modify CoT Instruction

Edit `data/tdc/metadata/cot_instruction.txt` to change the reasoning prompt:

```bash
# Current default:
# "Please think step by step and use tools when necessary (**Don't use the same tool more than once**). Then put your final choice ((A) or (B)) after \"Answer:\""

# Example custom instruction:
echo "Analyze the molecule systematically using available tools. Provide your final answer as (A) or (B) after \"Answer:\"" > data/tdc/metadata/cot_instruction.txt

# Re-convert data with new instruction
python scripts/data_conversion/convert_tdc_to_openai.py --task AMES
```

## Troubleshooting

**Q: Tokenizer not found**
A: Install transformers and download model: `pip install transformers`

**Q: Missing task-specific tools**
A: Falls back to all tools automatically with a warning

**Q: Validation fails**
A: Check JSONL format with: `head -1 <file> | python -m json.tool`

**Q: How to modify training hyperparameters?**
A: Edit `scripts/train_grpo_tdc.sh` directly or see `scripts/TRAINING_README.md`

## Next Steps

1. Choose a task (start with AMES)
2. Test the example script
3. Run a training experiment
4. Compare with SFT baseline
5. Analyze tool usage patterns

---

**Ready to train!** 🚀
