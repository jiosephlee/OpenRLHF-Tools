# TDC Conversion Retry - Success Report

**Date:** 2026-02-09

## Summary

Successfully converted **3 additional TDC datasets** that previously failed, bringing the total from 23 to **26 datasets**.

Total records increased from ~458,000 to **~539,000 records** (+81,000 records, +17.7%).

## Changes Made

### 1. Enhanced `tdc_loader.py`

Added three key features:

#### A. Fuzzy Prompt Matching (≤2 character edits)
```python
def _fuzzy_match_prompt_key(self, task_name: str, max_distance: int = 2)
```
- Uses Levenshtein distance algorithm
- Handles casing differences (e.g., `SARSCoV2` → `SARSCOV2`)
- Handles typos and minor variations
- Falls back to exact match first, then case-insensitive, then fuzzy

#### B. Tox21 Multi-Subtask Support
```python
def _load_tox21_csv(self, csv_path: str, df: pd.DataFrame)
```
- Detects `task_label` column in Tox21 raw CSVs
- Maps each task_label (e.g., `NR-AR`) to prompt key (e.g., `Tox21_NR_AR`)
- Handles hyphen → underscore conversion
- Each record is labeled with its specific subtask

#### C. Flexible Schema Detection
```python
def _detect_molecule_column(self, df: pd.DataFrame)
```
- Auto-detects molecule column: `Drug`, `Antibody`, `SMILES`, `Protein`, `Peptide`
- Enables support for non-drug datasets (antibodies, peptides, etc.)

## Successfully Converted Tasks

### 1. Tox21 (77,864 records)
- **Challenge:** 12 subtasks with different prompts, identified by `task_label` column
- **Solution:** Special handling to map task_label → prompt key
- **Subtasks:**
  - Nuclear receptors: `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`
  - Stress response: `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`
- **Splits:**
  - Train: 54,499 records
  - Val: 7,781 records
  - Test: 15,584 records

### 2. SARSCoV2_3CLPro_Diamond (880 records)
- **Challenge:** Casing mismatch (`SARSCoV2_3CLPro_Diamond` vs `SARSCOV2_3CLPro_Diamond`)
- **Solution:** Fuzzy matching (2 character edits: `o→O`, `o→O`)
- **Splits:**
  - Train: 616 records
  - Val: 88 records
  - Test: 176 records

### 3. SAbDab_Chen (2,409 records)
- **Challenge:** Different schema - uses `Antibody` column instead of `Drug`
- **Solution:** Flexible column detection
- **Data:** Antibody heavy/light chain sequences (not SMILES)
- **Splits:**
  - Train: 1,686 records
  - Val: 241 records
  - Test: 482 records

## Tasks Without Raw Data

These tasks have prompts in `prompts.json` but no corresponding raw CSV files:

1. **HuRI** - Protein-protein interaction dataset
2. **MHC1_IEDB** - MHC Class I binding prediction
3. **MHC2_IEDB** - MHC Class II binding prediction
4. **weber** - Unknown dataset

These likely require separate data downloads or are from different sources.

## Implementation Details

### Code Changes

**File:** `openrlhf/datasets/tdc_loader.py`

**Added:**
- `levenshtein_distance()` function (26 lines)
- `_fuzzy_match_prompt_key()` method (30 lines)
- `_detect_molecule_column()` method (5 lines)
- `_load_tox21_csv()` method (45 lines)

**Modified:**
- `load_csv_to_openai_format()` - Added fuzzy matching + schema detection (15 lines changed)

**Total:** ~120 lines added/modified

### Testing

All conversions validated:
- ✅ Correct message structure (role + content)
- ✅ Valid answer format ((A) or (B))
- ✅ Proper task labeling (Tox21 subtasks)
- ✅ Correct molecule column detection (Drug vs Antibody)
- ✅ Proper JSON encoding

## Files Generated

```
data/tdc/openai_format/
├── Tox21_train.jsonl (54,499 records)
├── Tox21_val.jsonl (7,781 records)
├── Tox21_test.jsonl (15,584 records)
├── SARSCoV2_3CLPro_Diamond_train.jsonl (616 records)
├── SARSCoV2_3CLPro_Diamond_val.jsonl (88 records)
├── SARSCoV2_3CLPro_Diamond_test.jsonl (176 records)
├── SAbDab_Chen_train.jsonl (1,686 records)
├── SAbDab_Chen_val.jsonl (241 records)
└── SAbDab_Chen_test.jsonl (482 records)
```

## Updated Statistics

### Before
- **Tasks:** 23
- **Total Records:** ~458,000
- **Failed:** 7 tasks

### After
- **Tasks:** 26
- **Total Records:** ~539,000
- **Failed:** 4 tasks (no raw data available)
- **Improvement:** +3 tasks, +81,000 records (+17.7%)

### New Dataset Coverage

- **Toxicity:** Added Tox21 (12 subtasks covering nuclear receptors and stress response pathways)
- **Antiviral:** Added SARSCoV2_3CLPro_Diamond (SARS-CoV-2 3CL protease inhibition)
- **Biologics:** Added SAbDab_Chen (antibody developability prediction)

## Usage Example

### Load Tox21 Dataset

```python
from openrlhf.datasets.tdc_loader import TDCDatasetLoader

loader = TDCDatasetLoader(
    prompts_path="data/tdc/metadata/prompts.json",
    cot_instruction_path="data/tdc/metadata/cot_instruction.txt"
)

# Convert Tox21 (automatically handles 12 subtasks)
loader.convert_task("Tox21", "data/tdc/raw", "data/tdc/openai_format")
```

### Load for Training

```bash
python -m openrlhf.cli.train_ppo_ray \
    --pretrain internlm/internlm2_5-7b-chat \
    --prompt_data data/tdc/openai_format/Tox21_train.jsonl \
    --agent_func_path openrlhf/utils/tool_calling_agent.py \
    --agent_max_steps 40 \
    --n_samples_per_prompt 8
```

## Key Insights

1. **Fuzzy matching is essential** - Many TDC dataset names have minor casing/formatting variations
2. **Tox21 is multi-task** - Single dataset with 12 distinct subtasks, each requiring different prompts
3. **Schema flexibility needed** - Biologics datasets use different column names (Antibody, Protein, Peptide)
4. **Large dataset** - Tox21 alone adds 77K records, making it the 2nd largest dataset after herg_central

## Recommendations

1. **Use fuzzy matching by default** - Prevents future failures from naming mismatches
2. **Document multi-task datasets** - Tox21 pattern may apply to other composite datasets
3. **Consider task-specific sampling** - Tox21 subtasks may have imbalanced distributions
4. **Monitor Antibody datasets** - Different input format may need special handling in tokenization

## Next Steps

1. ✅ **Conversion complete** - All available datasets converted
2. ⏭️ **Training validation** - Test GRPO training on new datasets
3. ⏭️ **Multi-task learning** - Combine Tox21 subtasks for better generalization
4. ⏭️ **Antibody-specific tools** - Add tools for antibody sequence analysis (if needed)

---

**Total Time:** ~10 minutes (implementation + testing)
**Code Quality:** Minimal changes, no breaking modifications, backward compatible
