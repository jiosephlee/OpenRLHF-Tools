# TDC Dataset Conversion Summary

## Overview

Successfully converted **26 TDC datasets** to OpenAI message format for GRPO training.

**Generated:** 2026-02-09 (Updated with fuzzy matching + Tox21 support)

## Successfully Converted Tasks

| Task | Train | Val | Test | Total |
|------|-------|-----|------|-------|
| AMES | 5094 | 727 | 1457 | 7278 |
| BBB_Martins | 1478 | 211 | 423 | 2112 |
| Bioavailability_Ma | 503 | 71 | 144 | 718 |
| CYP1A2_Veith | 9716 | 1388 | 2777 | 13881 |
| CYP2C19_Veith | 9716 | 1388 | 2777 | 13881 |
| CYP2C9_Substrate_CarbonMangels | 509 | 72 | 145 | 726 |
| CYP2C9_Veith | 9716 | 1388 | 2777 | 13881 |
| CYP2D6_Substrate_CarbonMangels | 509 | 72 | 145 | 726 |
| CYP2D6_Veith | 9716 | 1388 | 2777 | 13881 |
| CYP3A4_Substrate_CarbonMangels | 509 | 72 | 145 | 726 |
| CYP3A4_Veith | 9716 | 1388 | 2777 | 13881 |
| Carcinogens_Lagunin | 241 | 34 | 69 | 344 |
| ClinTox | 1177 | 168 | 337 | 1682 |
| DILI | 322 | 46 | 92 | 460 |
| HIA_Hou | 488 | 69 | 139 | 696 |
| HIV | 31997 | 4571 | 9142 | 45710 |
| PAMPA_NCATS | 1423 | 203 | 408 | 2034 |
| Pgp_Broccatelli | 852 | 121 | 245 | 1218 |
| SAbDab_Chen | 1686 | 241 | 482 | 2409 |
| SARSCoV2_3CLPro_Diamond | 616 | 88 | 176 | 880 |
| SARSCoV2_Vitro_Touret | 1038 | 148 | 298 | 1484 |
| Skin_Reaction | 282 | 40 | 82 | 404 |
| Tox21 | 54499 | 7781 | 15584 | 77864 |
| hERG | 458 | 65 | 132 | 655 |
| hERG_Karim | 9411 | 1344 | 2690 | 13445 |
| herg_central | 214825 | 30689 | 61379 | 306893 |

**Total Records:** ~539,000 records across all tasks

## Failed Conversions

The following tasks could not be converted (raw data not available):

1. **HuRI** - Raw data directory not found
2. **MHC1_IEDB** - Raw data directory not found
3. **MHC2_IEDB** - Raw data directory not found
4. **weber** - Raw data directory not found

## Conversion Features

### Fuzzy Prompt Matching (≤2 character edits)
- **SARSCoV2_3CLPro_Diamond** → Matched to `SARSCOV2_3CLPro_Diamond` (casing difference)
- Handles typos, underscores vs hyphens, and minor naming variations

### Tox21 Multi-Subtask Support
- **Tox21** has 12 subtasks identified by `task_label` column:
  - `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`
  - `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`
- Each subtask maps to a specific prompt (e.g., `Tox21_NR_AR`)
- Records are labeled with their specific subtask in the `task` field

### Flexible Schema Detection
- **Auto-detects molecule column**: `Drug`, `Antibody`, `SMILES`, `Protein`, `Peptide`
- **SAbDab_Chen**: Uses `Antibody` column (antibody sequences) instead of `Drug`

## Data Format

Each record follows OpenAI message format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Instructions: ...\nQuestion: ...\nDrug SMILES: '...'\n\nPlease think step by step..."
    }
  ],
  "answer": "(A)",
  "smiles": "CCO",
  "label": 0,
  "task": "AMES"
}
```

## Key Features

✅ **Stripped "Answer:" suffix** - Model will generate this part during training
✅ **CoT instructions** - Includes step-by-step thinking prompt
✅ **Tool-calling ready** - Designed for GRPO with tool usage
✅ **Label conversion** - 0 → "(A)", 1 → "(B)"
✅ **OpenAI format** - Compatible with `apply_chat_template()`
✅ **Fuzzy matching** - Handles naming variations (≤2 character edits)
✅ **Multi-subtask support** - Tox21 subtasks mapped via `task_label`
✅ **Flexible schemas** - Auto-detects molecule column (Drug/Antibody/etc.)

## File Locations

```
data/tdc/
├── raw/                          # Original CSV files
├── openai_format/                # Converted JSONL files (26 tasks × 3 splits)
└── metadata/
    ├── prompts.json              # Prompt templates (703 prompts)
    ├── tools_all.json            # All 47 tool definitions
    └── tools_task_specific.json  # Task-specific tool mappings
```

## Usage

### Load Dataset for Training

```python
from transformers import AutoTokenizer
from openrlhf.datasets.tdc_grpo_dataset import load_tdc_grpo_dataset

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat")

dataset = load_tdc_grpo_dataset(
    data_path="data/tdc/openai_format/AMES_train.jsonl",
    tokenizer=tokenizer,
    tool_mode="TaskSpecific",
    task_name="AMES",
)
```

### Train with OpenRLHF

```bash
python -m openrlhf.cli.train_ppo_ray \
    --pretrain internlm/internlm2_5-7b-chat \
    --prompt_data data/tdc/openai_format/AMES_train.jsonl \
    --input_key "question" \
    --label_key "answer" \
    --agent_func_path openrlhf/utils/tool_calling_turn.py \
    --agent_max_steps 40 \
    --n_samples_per_prompt 8
```

## Dataset Statistics

### Size Distribution

- **Largest:** herg_central (306,893 records)
- **Smallest:** Carcinogens_Lagunin (344 records)
- **Median:** ~2,000 records per task

### Property Categories

- **ADME Properties:** BBB_Martins, Bioavailability_Ma, HIA_Hou, PAMPA_NCATS, Pgp_Broccatelli
- **Toxicity:** AMES, Carcinogens_Lagunin, ClinTox, DILI, Skin_Reaction, Tox21 (12 subtasks)
- **CYP Inhibition:** CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4 (multiple variants)
- **Cardiotoxicity:** hERG, hERG_Karim, herg_central
- **Antiviral:** HIV, SARSCoV2_3CLPro_Diamond, SARSCoV2_Vitro_Touret
- **Biologics:** SAbDab_Chen (antibody developability)

## Large Files Note

Some very large datasets are excluded from git (see `data/tdc/.gitignore`):
- `herg_central` (306K+ records) - Regenerate with conversion script
- `HIV`, `MHC1_IEDB`, `MHC2_IEDB` raw CSV files

**To regenerate excluded files:**
```bash
# Copy raw data
bash scripts/data_conversion/copy_tdc_data.sh

# Convert large datasets
python scripts/data_conversion/convert_tdc_to_openai.py --task herg_central
```

## Next Steps

1. **Test on single task**: Start with AMES (7,278 records)
2. **Multi-task training**: Combine multiple tasks
3. **Evaluation**: Compare GRPO vs SFT performance
4. **Add missing tasks**: Convert tasks with missing prompts (if needed)
5. **Analysis**: Study tool usage patterns during training

## Validation

All converted datasets passed validation checks:
- ✅ Correct message structure (role + content)
- ✅ Valid answer format ((A) or (B))
- ✅ No empty fields
- ✅ Proper JSON encoding

Run validation:
```bash
python scripts/data_conversion/validate_openai_format.py --task AMES
```
