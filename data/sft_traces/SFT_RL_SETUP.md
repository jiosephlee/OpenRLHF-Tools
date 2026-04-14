# TDC-RL-16: SFT + RL Training Setup

## Overview

16 binary drug property prediction tasks (SMILES in, (A)/(B) out). The model reasons step-by-step through molecular features using tool calls, then predicts.

**Pipeline**: SFT on RF-derived reasoning traces → RL with live molecular tools.

**Features**: ~208 RDKit descriptors + 3 KNN similarity features. No external dependencies beyond `rdkit` and `scikit-learn`.

---

## Directory Layout

```
sft_traces/
├── traces_formatted/                    # SFT training data (tool-use format)
│   ├── AMES.jsonl                       # Per-task JSONL
│   ├── ...                              # (16 task files)
│   └── all_tasks_combined.jsonl         # All 21,139 traces
├── traces/                              # Raw traces (plain chat format, for reference)
├── rf_models/                           # Trained RF models per task (.joblib)
├── rdkit_descriptor_descriptions.json   # Semantic descriptions (208 RDKit descriptors)
├── tool_server.py                       # Live tool server for RL rollouts
├── viewer/                              # Interactive HTML trace viewer
└── SFT_RL_SETUP.md                      # This file
```

External dependencies (relative paths from `sft_traces/`):

```
../tdc_eval/reference_splits/           # Train/val CSVs per task
../tdc_eval/base_no_tools/task_formats/ # Task prompts and label mappings
```

---

## Phase 1: SFT

### Data

Use `traces_formatted/all_tasks_combined.jsonl` (21,139 traces, all 16 tasks).

Each line is a complete tool-use conversation:

```
tools: [{get_molecular_descriptors, get_similar_molecules}]  ← function schemas
messages:
  1. system    — task context
  2. user      — SMILES query
  3. assistant — tool_calls (requests descriptors + neighbors)
  4. tool      — RDKit descriptor values with semantic descriptions
  5. tool      — KNN neighbors with labels + stats
  6. assistant — step-by-step reasoning with running log-odds → Answer: (A/B)
```

Every feature value in tool responses carries a semantic description:
```json
{"MolWt": {"value": 292.03, "description": "Molecular weight of the compound."}}
```

The assistant reasoning tracks running probability:
```
Starting from the base rate: log-odds = -0.078, P(B) = 0.480.
Looking at MolWt = 292.0300. Log-odds update: +0.412 (toward positive). Running log-odds: +0.334, P(B) = 0.583.
...
Based on all the evidence, my final prediction is (B) positive / active with 76.1% confidence (P(B) = 0.761).
Answer: (B)
```

### SFT Notes

- The `tools` array is at the top level of each JSON object (not inside `messages`).
- `metadata.task` identifies which of the 16 tasks the trace belongs to.
- Confidence values range from ~50% to ~99% (stratified tree selection, not always 100%).

---

## Phase 2: RL

### Tool Server

`tool_server.py` provides live tool execution for RL rollouts. Computes real molecular features from any SMILES at inference time. No LLM4SD or external feature modules needed.

```python
from tool_server import ToolServer

server = ToolServer(base_dir="/path/to/sft_traces")

# Pre-load tasks (loads RF model + training data for KNN)
server.load_task("AMES")

# Get tool definitions (pass as `tools` param to model)
tools = server.get_tools("AMES")

# Get prompts for a molecule
system_prompt, user_prompt = server.get_prompts("AMES", "c1ccccc1")

# Execute tool calls from model output
descriptor_json = server.call_tool("get_molecular_descriptors", {"smiles": "c1ccccc1"}, task="AMES")
knn_json = server.call_tool("get_similar_molecules", {"smiles": "c1ccccc1", "k": 3}, task="AMES")

# Compute reward from model's final output
reward = server.compute_reward(model_output_text, true_label=1)
# Returns: 1.0 (correct), 0.0 (wrong), -0.1 (no parseable answer)

# Get eval data
val_examples = server.get_val_examples("AMES")  # [(smiles, label), ...]
train_examples = server.get_train_examples("AMES")
```

### RL Rollout Loop (pseudocode)

```python
server = ToolServer(base_dir="path/to/sft_traces")

for task in TASKS:
    server.load_task(task)
    tools = server.get_tools(task)
    examples = server.get_train_examples(task)  # or val for eval

    for smiles, label in examples:
        system, user = server.get_prompts(task, smiles)

        # 1. Model generates initial response (should be tool_calls)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = model.generate(messages, tools=tools)

        # 2. Execute tool calls
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = server.call_tool(tool_call.function.name, args, task=task)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        # 3. Model generates final reasoning + answer
        final = model.generate(messages)

        # 4. Compute reward
        reward = server.compute_reward(final.content, true_label=label)
```

### Reward Function

Binary correctness — extracts the last `(A)` or `(B)` from the model's output:

| Output | True Label | Reward |
|--------|-----------|--------|
| `Answer: (A)` | 0 | **1.0** |
| `Answer: (B)` | 1 | **1.0** |
| `Answer: (A)` | 1 | **0.0** |
| No `(A)`/`(B)` found | any | **-0.1** |

Label mapping: 0 = (A) = negative/inactive, 1 = (B) = positive/active.

---

## Tasks

| Task | Train | Val | Features | Val F1 | Description |
|------|-------|-----|----------|--------|-------------|
| AMES | 5,081 | 721 | 207 | 0.764 | Mutagenicity |
| BBB_Martins | 1,369 | 195 | 203 | 0.820 | Blood-brain barrier penetration |
| Bioavailability_Ma | 448 | 64 | 200 | 0.634 | Oral bioavailability |
| CYP2C9_Substrate | 465 | 66 | 201 | 0.697 | CYP2C9 substrate |
| CYP2D6_Substrate | 464 | 66 | 198 | 0.713 | CYP2D6 substrate |
| CYP3A4_Substrate | 466 | 67 | 202 | 0.726 | CYP3A4 substrate |
| Carcinogens_Lagunin | 194 | 28 | 190 | 0.757 | Carcinogenicity |
| ClinTox | 1,007 | 139 | 203 | 0.633 | Clinical toxicity |
| DILI | 332 | 47 | 200 | 0.790 | Drug-induced liver injury |
| HIA_Hou | 404 | 57 | 195 | 0.870 | Human intestinal absorption |
| PAMPA_NCATS | 1,423 | 203 | 195 | 0.563 | Membrane permeability |
| Pgp_Broccatelli | 846 | 121 | 197 | 0.863 | P-glycoprotein inhibition |
| SARSCoV2_3CLPro | 615 | 88 | 185 | 0.567 | SARS-CoV-2 3CL protease inhibition |
| SARSCoV2_Vitro | 1,032 | 148 | 202 | 0.586 | SARS-CoV-2 in-vitro activity |
| Skin_Reaction | 282 | 40 | 188 | 0.562 | Skin sensitization |
| hERG | 452 | 64 | 196 | 0.770 | hERG channel blockade |

Feature counts = RDKit descriptors (after zero-variance removal) + 3 KNN features.

---

## Environment Setup

```bash
pip install rdkit-pypi scikit-learn joblib pandas numpy
```

### Smoke Test

```bash
cd sft_traces/
python tool_server.py --task AMES --smiles "c1ccccc1"
```

---

## Adding New Features

The pipeline is designed to be extended with additional feature families. Here's how:

### 1. Add feature computation to `build_sft_traces.py`

In `process_task()`, compute your new features alongside RDKit and stack them:

```python
# After RDKit computation:
rdkit_train, rdkit_names = compute_rdkit_descriptors(list(train_smiles))
rdkit_val, _ = compute_rdkit_descriptors(list(val_smiles))

# Add your features:
custom_train, custom_names = your_feature_function(list(train_smiles))
custom_val, _ = your_feature_function(list(val_smiles))

X_train_raw = np.hstack([rdkit_train, custom_train])
X_val_raw = np.hstack([rdkit_val, custom_val])
feature_names = rdkit_names + custom_names
```

The preprocessing (median imputation, zero-variance removal) and KNN computation happen downstream and don't need changes.

### 2. Add semantic descriptions

Create a JSON file mapping feature names to descriptions:

```json
{"my_feature_1": "Description of what this feature measures.", ...}
```

Then update `load_description_maps()` in both `reformat_traces.py` and `tool_server.py` to load and merge it:

```python
def load_description_maps():
    with open('rdkit_descriptor_descriptions.json') as f:
        rdkit_descs = json.load(f)
    with open('my_custom_descriptions.json') as f:
        custom_descs = json.load(f)
    rdkit_descs.update(custom_descs)  # merge into one map
    ...
```

### 3. Update tool_server.py for live inference

Add your feature computation to `_tool_descriptors()`:

```python
def _tool_descriptors(self, smiles, task):
    rdkit_feats = self._compute_rdkit(smiles)
    custom_feats = self._compute_custom(smiles)  # your new function
    all_feats = {**rdkit_feats, **custom_feats}
    ...
```

### 4. Retrain and regenerate

```bash
python build_sft_traces.py      # Retrains RF models + raw traces
python reformat_traces.py        # Reformats into tool-use format
python build_viewer.py           # Rebuilds viewer (optional)
```

### Key constraints

- Feature functions must accept a list of SMILES and return `(np.ndarray, list_of_names)`.
- Feature names must be unique across all families.
- Every feature should have a semantic description (otherwise the tool response will include the value but no description string).
- The RF model is invariant to feature scaling, so raw values are fine.

---

## Notes

- **No feature scaling** — RF models use raw feature values, which are interpretable in the traces.
- **Log-odds clipping** — Individual feature updates are clipped to ±5 to prevent extreme confidence.
- **Tool response format** — Every value carries `{"value": X, "description": "..."}`. The model sees the semantic meaning of each feature alongside its numerical value.
- **16 tasks, shared tools** — The two tool functions are the same across tasks, but the exact RDKit features retained vary per task (~185–207 after zero-variance removal + 3 KNN).
- **No LLM4SD dependency** — Feature computation uses only `rdkit` and `scikit-learn`. The codex_generated_code modules are not required.
