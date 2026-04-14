# TDC Tool Version Transition Checklist

This document captures the full rollout checklist for adding a new TDC tool
version after the v11 and v12 transitions.

The goal is to make version bumps repeatable without missing small but
important follow-up work such as pseudo-label mappings, KNN reversal metrics,
training wrapper defaults, or SLURM submission behavior.


## Scope

Use this checklist whenever adding a new version such as `v13`, `v14`, etc.

These transitions usually involve:

- new tool schemas and callables
- a new dataset build with tool-aware prompts
- a new `tools_per_task_<version>.json`
- updates to training wrappers and CLI validation
- KNN pseudo-label / reversal tracking compatibility
- rollout / eval / W&B metric coverage for any new neighbor tool naming


## Current Reference Versions

Use these as the concrete examples:

- `v11`: generalized `get_features` plus task-specific `get_neighbors_<task>`
- `v12`: same shape as `v11`, but with more granular physicochemical feature names

Reference files:

- [openrlhf/tools/therapeutic_tools/v11.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/tools/therapeutic_tools/v11.py)
- [openrlhf/tools/therapeutic_tools/v12.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/tools/therapeutic_tools/v12.py)
- [data/tdc/build_v11_datasets.py](/vast/home/j/jojolee/OpenRLHF-Tools/data/tdc/build_v11_datasets.py)
- [data/tdc/build_v12_datasets.py](/vast/home/j/jojolee/OpenRLHF-Tools/data/tdc/build_v12_datasets.py)
- [openrlhf/utils/tool_versions.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/utils/tool_versions.py)


## Step 1: Add the New Tool Module

Create a new file:

- `openrlhf/tools/therapeutic_tools/v<NN>.py`

Expected contents:

- the new feature registry
- `FEATURE_NAMES`
- `get_features(smiles, feature_names)`
- `get_neighbors(smiles, task_name, ...)`
- task-specific neighbor wrappers such as `get_neighbors_ames`
- `GET_FEATURES_TOOL`
- `GET_NEIGHBORS_TOOL`
- `TASK_NEIGHBOR_TOOL_SCHEMAS`
- `TASK_NEIGHBOR_CALLABLES`

Rules:

- do not delete or mutate older version modules unless the change is intended to
  fix a shared bug
- preserve backward compatibility for older versions
- if external tool names are reused, keep version-specific callables separate so
  old versions still resolve to their original implementation

Example:

- `v12.py` reuses the public tool names `get_features` and `get_neighbors`, but
  `tool_versions.py` wires `v12` directly to `v12_get_features` and
  `v12_get_neighbors` so `v11` still points at the original v11 functions


## Step 2: Export the New Version From `therapeutic_tools/__init__.py`

Update:

- [openrlhf/tools/therapeutic_tools/__init__.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/tools/therapeutic_tools/__init__.py)

Add imports for:

- the new schemas
- the new feature-name list
- the new task-specific neighbor schema map
- the new task-specific callables

If the new version reuses external tool names already used by older versions:

- alias the imports, for example `v12_get_features`
- do not overwrite older entries in `_FUNCTION_MAP` unless you explicitly want
  all versions to use the new implementation

Notes from v11/v12:

- v11 exports are currently used by `_FUNCTION_MAP`
- v12 is imported with aliases and is wired explicitly in `tool_versions.py`


## Step 3: Register the Version in `tool_versions.py`

Update:

- [openrlhf/utils/tool_versions.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/utils/tool_versions.py)

Checklist:

- import the new schemas and callables
- add `_V<NN>_BASIC_SCHEMAS`
- add `_V<NN>_TASK_MAP`
- add `_V<NN>_CALLABLES`
- add `_V<NN>_VERSION`
- add the version string to `_ALL_VERSIONS`
- add the version to `TOOL_VERSIONS`
- update `get_version()` to include the new version in the fast path

Important:

- if the new version uses the same external tool names as an older version,
  wire the version to explicit version-local callables instead of relying on
  `_FUNCTION_MAP`


## Step 4: Update CLI Validation

Update:

- [openrlhf/cli/train_ppo_ray.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/cli/train_ppo_ray.py)

Checklist:

- add the new version to `--tool_version` choices
- update the help text description

If this is skipped, training fails early with:

- `argument --tool_version: invalid choice`


## Step 5: Generate `tools_per_task_<version>.json`

Use:

- [scripts/generate_tools_json.py](/vast/home/j/jojolee/OpenRLHF-Tools/scripts/generate_tools_json.py)

Output:

- `data/tdc/metadata/tools_per_task_<version>.json`

Expected content:

- `__default__`: base tools
- each task: base tools plus task-specific neighbor tool

For `v11` and `v12`:

- `__default__` should contain `get_features`
- each task should contain `get_features` and `get_neighbors_<alias>`


## Step 6: Build the Dataset Variant

Create a new dataset builder:

- `data/tdc/build_v<NN>_datasets.py`

Create a new output directory:

- `data/tdc/openai_format_v<NN>/`

Checklist:

- keep earlier dataset builders and dataset directories intact
- decide the canonical source directory explicitly
- print the raw source path in the script output
- ensure prompt lookup is tolerant to known prompt-key mismatches

Recommended dataset source for current tool versions:

- `raw_deduplicated`

Do not silently mix:

- `raw`
- `raw_deduplicated`
- `deduplicated_canonicalized`

The dataset builder should state the source clearly so downstream debugging is
unambiguous.


## Step 7: Be Explicit About Raw Source Choice

This was a real source of confusion during the v11 transition.

Current state:

- `v11` and `v12` datasets should be built from `data/tdc/raw_deduplicated`
- `deduplicated_canonicalized` has the same task counts for the 16 core tasks
  but not always the same exact SMILES strings

Why this matters:

- pseudo-label lookup in `PromptDataset` is exact by `task + smiles`
- if the JSON mapping keys are canonicalized but the prompt records preserve the
  original deduplicated SMILES strings, some lookups will miss even when the
  molecule is chemically the same


## Step 8: Build or Rebuild KNN Pseudo-Label Metadata

Used by:

- `PromptDataset`
- KNN reversal tracking
- reward shaping in tool-calling rollouts

Reference files:

- [scripts/build_knn_v10_pseudo_labels.py](/vast/home/j/jojolee/OpenRLHF-Tools/scripts/build_knn_v10_pseudo_labels.py)
- [scripts/build_knn_v11_pseudo_labels.py](/vast/home/j/jojolee/OpenRLHF-Tools/scripts/build_knn_v11_pseudo_labels.py)

Checklist:

- ensure the pseudo-label builder uses the same row source as the dataset
- ensure the output file path is version-appropriate
- ensure the mapping covers every task used by the dataset
- verify coverage is 100% where expected

Current convention:

- `v10` uses `knn_v10_pseudo_labels.json`
- `v11` and `v12` use `knn_v11_pseudo_labels.json`

Reason `v11` and `v12` can share:

- both use `raw_deduplicated`
- both use the same fingerprint-neighbor retrieval logic
- only the exposed feature vocabulary changed in `v12`


## Step 9: Update Training Wrappers

Files:

- [scripts/train_grpo_tdc_gpt_oss.sh](/vast/home/j/jojolee/OpenRLHF-Tools/scripts/train_grpo_tdc_gpt_oss.sh)
- [scripts/train_grpo_tdc_gpt_oss_slurm.sh](/vast/home/j/jojolee/OpenRLHF-Tools/scripts/train_grpo_tdc_gpt_oss_slurm.sh)

Checklist:

- select `openai_format_v<NN>` when `TOOL_VERSION=v<NN>`
- add a `DATA_TAG` for the new dataset directory
- point `KNN_PL_PATH` at the correct pseudo-label file
- decide whether the wrapper default should be changed to the new version

Important:

- if you want true out-of-the-box behavior, update the wrapper defaults
- otherwise the version must be provided explicitly via `TOOL_VERSION=v<NN>`


## Step 10: Update KNN / Neighbor Usage Metrics

This step is easy to miss and is required for clean W&B reporting.

Files:

- [openrlhf/trainer/ppo_utils/experience_maker.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/trainer/ppo_utils/experience_maker.py)
- [openrlhf/trainer/ppo_trainer.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/trainer/ppo_trainer.py)

Checklist:

- update neighbor-tool key detection to cover new tool names
- update molecular-info tool detection if the new version changes the feature tool
- update the regex that detects prepended neighbor context in prompts or traces

This matters for:

- `requested_neighbors_pct`
- `knn_reversal_pct`
- `knn_correct_reversal_pct`
- `knn_incorrect_reversal_pct`
- trace diagnostics such as:
  - `trace_pct_at_least_2_unique_tools`
  - `trace_pct_at_least_3_tool_calls`
  - `trace_pct_at_least_2_molinfo_and_1_neighbor`

Current required prefixes:

- old neighbor tools:
  - `tool_count__find_similar_molecules`
  - `tool_count__get_similar_neighbors`
- new neighbor tools:
  - `tool_count__get_neighbors`

Current required molecular-info prefixes:

- `tool_count__get_molecular_properties`
- `tool_count__get_features`

Current required neighbor-context regex coverage:

- old prepended context:
  - `Nearest Neighbors from Training Set:`
  - `KNN Predicted Label:`
  - `pseudo label from naive Morgan fingerprint KNN prediction is ...`
- new v11/v12 tool output:
  - `Nearest Neighbors for task '<TASK>' (k=...)`


## Step 11: Check `PromptDataset` Assumptions

File:

- [openrlhf/datasets/prompts_dataset.py](/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/datasets/prompts_dataset.py)

Checklist:

- verify pseudo-label extraction still works
- if prompts no longer inline pseudo labels, verify `--knn_pseudo_labels_path`
  is supplied
- verify the JSON lookup keys match the dataset `task` and `smiles` values exactly


## Step 12: Validate the Version End to End

Minimum validation:

1. import / parse the new code
2. generate `tools_per_task_<version>.json`
3. build `openai_format_v<NN>`
4. run a direct tool smoke test in the project conda env
5. verify `train_ppo_ray.py --tool_version v<NN>` no longer rejects the version
6. verify training wrappers point at the right dataset and pseudo-label file
7. verify KNN / neighbor metrics are incremented for the new tool names

Recommended smoke tests:

- `get_features("CCO", ["legacy_alias_or_new_feature"])`
- a task-specific neighbor call such as `get_neighbors_ames`
- `generate_tools_json.py --version v<NN>`
- dataset build command for the new version


## Step 13: Use the Correct Submission Path for SLURM

This caused real failures during the v12 launch.

Do:

- submit the SLURM wrapper with `sbatch`

Do not do:

- run the SLURM wrapper with `bash`

Why:

- `bash train_grpo_tdc_gpt_oss_slurm.sh` ignores the `#SBATCH` resource lines
- this results in tiny CPU-only or incorrect allocations
- the script can then fail immediately and misleadingly

Example:

```bash
TOOL_VERSION=v12 ... sbatch train_grpo_tdc_gpt_oss_slurm.sh
```


## Step 14: Harden SLURM GPU Detection

File:

- [scripts/train_grpo_tdc_gpt_oss_slurm.sh](/vast/home/j/jojolee/OpenRLHF-Tools/scripts/train_grpo_tdc_gpt_oss_slurm.sh)

Checklist:

- do not assume `SLURM_GPUS_ON_NODE` is always populated
- add fallback logic for:
  - `SLURM_JOB_GPUS`
  - `nvidia-smi -L`

This prevents immediate failures under `set -u` when the cluster environment
does not populate `SLURM_GPUS_ON_NODE`.


## Step 15: Verify W&B / Run Metadata Semantics

Checklist:

- confirm `tool_version` in the run config matches the intended version
- confirm `tdc_tools` points at `tools_per_task_<version>.json`
- confirm `prompt_data` and `eval_dataset` point at the correct `openai_format_v<NN>`
- confirm `knn_pseudo_labels_path` points at the intended file

This is especially important when wrapper defaults lag behind the new version.


## v11-Specific Notes

- dataset source should be `raw_deduplicated`
- prompt lookup needed to tolerate `SARSCoV2_3CLPro_Diamond` vs
  `SARSCOV2_3CLPro_Diamond`
- KNN pseudo-label file should come from `raw_deduplicated`, not the older
  canonicalized path, if exact prompt SMILES matching is required


## v12-Specific Notes

- `v12` should keep the same structure as `v11`
- only the available feature names change materially
- replacing `physicochemical` with individual properties requires:
  - new `FEATURE_NAMES`
  - updated dataset system prompt
  - updated `tools_per_task_v12.json`
- keeping a legacy alias for `physicochemical` is useful for forgiving prompts
  and interactive use


## Quick Rollout Checklist

For a new version `v<NN>`:

1. Add `openrlhf/tools/therapeutic_tools/v<NN>.py`.
2. Export it from `therapeutic_tools/__init__.py`.
3. Register it in `openrlhf/utils/tool_versions.py`.
4. Add it to `openrlhf/cli/train_ppo_ray.py`.
5. Generate `data/tdc/metadata/tools_per_task_v<NN>.json`.
6. Add `data/tdc/build_v<NN>_datasets.py`.
7. Build `data/tdc/openai_format_v<NN>/`.
8. Build or point to the correct `knn_v<...>_pseudo_labels.json`.
9. Update `train_grpo_tdc_gpt_oss.sh`.
10. Update `train_grpo_tdc_gpt_oss_slurm.sh`.
11. Update `ppo_utils/experience_maker.py` metric prefix coverage.
12. Update `ppo_trainer.py` eval metric prefix coverage.
13. Verify the regex coverage for prepended neighbor context.
14. Smoke-test tools in the project conda env.
15. Submit via `sbatch`, not `bash`, for the SLURM wrapper.


## Recommended Future Cleanup

These are not required for the current versions, but would reduce future drift:

- centralize neighbor-tool prefix detection in one shared helper
- centralize molecular-info tool detection in one shared helper
- centralize version-to-dataset and version-to-pseudo-label mapping in one place
- canonicalize pseudo-label lookup at runtime so canonicalized and raw-dedup
  sources can share mappings more safely
- add a single integration smoke test that validates:
  - `get_version(v<NN>)`
  - tool JSON generation
  - dataset build
  - pseudo-label lookup
  - KNN metric prefix detection
