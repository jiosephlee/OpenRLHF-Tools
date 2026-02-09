# TDC to OpenRLHF GRPO Conversion

This directory contains scripts to convert TDC molecular property datasets from therapeutic-tuning to OpenRLHF GRPO format.

## Quick Start

```bash
cd /Users/jlee0/Desktop/research/OpenRLHF-Tools

# 1. Copy raw data from therapeutic-tuning
bash scripts/data_conversion/copy_tdc_data.sh

# 2. Build tool definitions
python scripts/data_conversion/build_tool_definitions.py

# 3. Convert single task to OpenAI format
python scripts/data_conversion/convert_tdc_to_openai.py --task AMES

# 4. Validate conversion
python scripts/data_conversion/validate_openai_format.py --task AMES

# 5. Convert all tasks
python scripts/data_conversion/convert_tdc_to_openai.py --all
```

## Data Format Explanation

We store data in **OpenAI message format** (no hardcoded templates):

```json
{
  "messages": [
    {"role": "user", "content": "Instructions: ...\nQuestion: ...\nDrug SMILES: 'CCO'"}
  ],
  "answer": "(A)",
  "smiles": "CCO",
  "label": 0
}
```

At **load time**, the custom dataset loader applies the tokenizer's chat template:
- Adds tool definitions
- Adds system prompt (from tokenizer)
- Enables thinking mode
- Adds generation prompt

This approach:
- Works with any model family (intern, glm, qwen, etc.)
- Uses tokenizer's native formatting
- Supports tool calling via `apply_chat_template`

## Training with OpenRLHF

```python
from transformers import AutoTokenizer
from openrlhf.datasets.tdc_grpo_dataset import load_tdc_grpo_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat")

# Load dataset (applies chat template with tools)
dataset = load_tdc_grpo_dataset(
    data_path="data/tdc/openai_format/AMES_train.jsonl",
    tokenizer=tokenizer,
    tool_mode="TaskSpecific",
    task_name="AMES",
)

# Train with OpenRLHF
python -m openrlhf.cli.train_ppo_ray \
    --pretrain internlm/internlm2_5-7b-chat \
    --prompt_data data/tdc/openai_format/AMES_train.jsonl \
    --input_key "question" \
    --label_key "answer" \
    --agent_func_path openrlhf/utils/tool_calling_agent.py \
    --agent_max_steps 40 \
    --n_samples_per_prompt 8
```

## File Structure

```
data/tdc/
├── raw/              # Raw CSV files from therapeutic-tuning
│   ├── AMES/
│   ├── BBB_Martins/
│   └── ...
├── openai_format/    # OpenAI message format JSONL
│   ├── AMES_train.jsonl
│   ├── AMES_val.jsonl
│   └── ...
└── metadata/
    ├── prompts.json           # TDC prompt templates
    ├── tools_all.json         # All tool definitions
    └── tools_task_specific.json  # Task-specific tool mappings
```

## Script Descriptions

### copy_tdc_data.sh
Copies raw CSV files from therapeutic-tuning and prompt templates to OpenRLHF-Tools structure.

### build_tool_definitions.py
Extracts tool definitions from therapeutic-tuning and saves them as JSON files for runtime loading.

### convert_tdc_to_openai.py
Converts TDC CSV files to OpenAI message format JSONL with CoT instructions.

**Options:**
- `--task TASK_NAME`: Convert a single task
- `--all`: Convert all tasks
- `--raw_dir`: Source directory (default: data/tdc/raw)
- `--output_dir`: Output directory (default: data/tdc/openai_format)
- `--prompts_path`: Path to prompts.json (default: data/tdc/metadata/prompts.json)
- `--cot_instruction`: Custom CoT instruction (optional)

### validate_openai_format.py
Validates OpenAI message format JSONL files for correctness.

**Checks:**
- All records have "messages" and "answer" fields
- Messages have proper role/content structure
- Answers are in "(A)" or "(B)" format
- No empty fields

## Dataset Loading

### TDCDatasetLoader (tdc_loader.py)
Handles CSV → OpenAI format conversion with prompt stripping and CoT instructions.

### TDCGRPODataset (tdc_grpo_dataset.py)
Runtime loader that applies tokenizer's chat template with tools at load time.

**Usage:**
```python
from openrlhf.datasets.tdc_grpo_dataset import load_tdc_grpo_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat")

dataset = load_tdc_grpo_dataset(
    data_path="data/tdc/openai_format/AMES_train.jsonl",
    tokenizer=tokenizer,
    tool_mode="TaskSpecific",  # "TaskSpecific", "All", or "None"
    task_name="AMES",          # Required for TaskSpecific mode
)

print(f"Dataset size: {len(dataset)}")
print(f"First example: {dataset[0]}")
```

## Key Differences from SFT Format

| Aspect | SFT Format | GRPO Format (OpenAI) |
|--------|-----------|----------------------|
| **Storage** | Full conversations with templates | OpenAI message format (no templates) |
| **Responses** | Pre-computed | Generated during training |
| **Tool calls** | Pre-executed | Executed by agent during training |
| **Template** | Hardcoded in data | Applied at load time via tokenizer |
| **System prompt** | Hardcoded | From tokenizer's default |
| **Answer** | Included in conversation | Used for reward computation |
| **Training** | Supervised learning | Reinforcement learning |
| **Flexibility** | Tied to specific model format | Works with any tokenizer |

## Verification

After running the conversion, verify the output:

```bash
# Check single task conversion
python scripts/data_conversion/convert_tdc_to_openai.py --task AMES
python scripts/data_conversion/validate_openai_format.py --task AMES

# Inspect raw format
head -1 data/tdc/openai_format/AMES_train.jsonl | python -m json.tool

# Test dataset loader
python -c "
from transformers import AutoTokenizer
from openrlhf.datasets.tdc_grpo_dataset import load_tdc_grpo_dataset

tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2_5-7b-chat', trust_remote_code=True)
dataset = load_tdc_grpo_dataset(
    'data/tdc/openai_format/AMES_train.jsonl',
    tokenizer,
    tool_mode='TaskSpecific',
    task_name='AMES'
)
print(f'Dataset size: {len(dataset)}')
print(f'Sample keys: {list(dataset[0].keys())}')
print(f'Question preview: {dataset[0][\"question\"][:500]}')
"
```

## Troubleshooting

**Issue: "No prompt template found for task"**
- Ensure prompts.json is copied from therapeutic-tuning
- Check task name spelling matches the keys in prompts.json

**Issue: "task_name required when tool_mode='TaskSpecific'"**
- Pass task_name parameter when loading dataset with TaskSpecific mode

**Issue: "Module 'tools' not found"**
- Ensure therapeutic-tuning is at `/Users/jlee0/Desktop/research/therapeutic-tuning`
- Check that therapeutic-tuning/tools/__init__.py exists

**Issue: Invalid answer format**
- Verify CSV files have "Y" column with 0/1 labels
- Check conversion logic in tdc_loader.py

## Next Steps

After successful conversion:

1. **Single-task training**: Train on individual tasks (e.g., AMES)
2. **Multi-task training**: Combine multiple tasks for joint training
3. **Evaluation**: Test on held-out test sets
4. **Analysis**: Compare GRPO vs SFT performance
