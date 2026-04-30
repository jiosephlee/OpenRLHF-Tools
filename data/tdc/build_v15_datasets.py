"""Build v15 TDC datasets in OpenAI chat format.

v15 uses the imported official 16-task scaffold subset under
``data/tdc/official_v15_dataset`` and the refined chain-of-thought prompt
suffix used by the newer tool-calling datasets.

Usage:
    python data/tdc/build_v15_datasets.py \
        [--output-dir data/tdc/openai_format_v15]

    python data/tdc/build_v15_datasets.py \
        --exclude-hf-dataset Kiria-Nozan/TRIM-gpt-5.4-mini-comparison-only \
        --exclude-mode smiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_RAW_DIR = os.path.join(_SCRIPT_DIR, "official_v15_dataset")
PROMPTS_PATH = os.path.join(_SCRIPT_DIR, "metadata", "prompts.json")
COT_INSTRUCTION_PATH = os.path.join(_SCRIPT_DIR, "metadata", "cot_instruction_refined_tools.txt")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "openai_format_v15")
FILTERED_OUTPUT_DIR_TEMPLATE = os.path.join(_SCRIPT_DIR, "openai_format_v15_no_sft_{mode}_overlap")
FILTER_SUMMARY_FILENAME = "overlap_filter_summary.json"

SYSTEM_MESSAGE = "You are a chemist analyzing drug molecules."

TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "Carcinogens_Lagunin", "ClinTox",
    "DILI", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Skin_Reaction", "hERG",
]

EXCLUDE_MODE_SMILES = "smiles"
EXCLUDE_MODE_TASK_SMILES = "task_smiles"
EXCLUDE_MODES = (EXCLUDE_MODE_SMILES, EXCLUDE_MODE_TASK_SMILES)

def load_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    return {k: v.rstrip().removesuffix("Answer:").rstrip() for k, v in prompts.items()}


def load_cot_instruction() -> str:
    with open(COT_INSTRUCTION_PATH) as f:
        return f.read().strip()


def build_user_message(task: str, prompt_template: str, smiles: str, cot_instruction: str) -> str:
    prompt = prompt_template.replace("{Drug SMILES}", smiles)
    return f"{prompt}\n{cot_instruction}"


def resolve_prompt_template(task: str, prompts: dict) -> Optional[str]:
    prompt_template = prompts.get(task)
    if prompt_template is not None:
        return prompt_template

    task_lower = task.lower()
    matches = [value for key, value in prompts.items() if key.lower() == task_lower]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_output_dir(output_dir: Optional[str], exclude_mode: Optional[str]) -> str:
    if output_dir:
        return output_dir
    if exclude_mode:
        return FILTERED_OUTPUT_DIR_TEMPLATE.format(mode=exclude_mode)
    return DEFAULT_OUTPUT_DIR


def iter_local_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".jsonl")
        )
        if not files:
            raise FileNotFoundError(f"No *.jsonl files found under {path}")
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"Exclusion dataset path does not exist: {path}")

    for file_path in files:
        with open(file_path) as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSON in exclusion dataset {file_path}:{line_number}"
                    ) from exc
    return rows


def load_hf_jsonl_rows(repo_id: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    try:
        dataset = load_dataset(repo_id, split="train")
    except ValueError as exc:
        if "Feature type 'Json' not found" not in str(exc):
            raise

        from huggingface_hub import snapshot_download

        local_repo = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                "train/*.jsonl",
                "train/**/*.jsonl",
                "*.jsonl",
                "metadata/*",
                "README.md",
            ],
        )
        train_files = sorted(
            str(path)
            for path in os.scandir(local_repo)
            if path.is_file() and path.name.endswith(".jsonl")
        )
        if not train_files:
            train_files = [
                os.path.join(root, filename)
                for root, _, filenames in os.walk(local_repo)
                for filename in filenames
                if filename.endswith(".jsonl") and "metadata" not in root.split(os.sep)
            ]
        if not train_files:
            raise FileNotFoundError(f"No JSONL files found in downloaded dataset repo {repo_id}")
        dataset = load_dataset("json", data_files=sorted(train_files), split="train")

    return [dict(row) for row in dataset]


def extract_exclusion_key(
    row: dict[str, Any],
    exclude_mode: str,
) -> Optional[str | tuple[str, str]]:
    smiles = row.get("smiles")
    if smiles is None and "drug" in row:
        smiles = row["drug"]
    if smiles is None:
        return None

    smiles = str(smiles).strip()
    if not smiles:
        return None

    if exclude_mode == EXCLUDE_MODE_SMILES:
        return smiles

    task = row.get("task")
    if task is None:
        raise ValueError("Exclusion rows must include a 'task' field for --exclude-mode task_smiles")
    task = str(task).strip()
    if not task:
        raise ValueError("Encountered an exclusion row with an empty 'task' value")
    return (task, smiles)


def load_exclusion_keys(
    exclude_mode: str,
    exclude_hf_dataset: Optional[str],
    exclude_local_data: Optional[str],
) -> tuple[set[str | tuple[str, str]], dict[str, Any]]:
    if bool(exclude_hf_dataset) == bool(exclude_local_data):
        raise ValueError("Pass exactly one of --exclude-hf-dataset or --exclude-local-data")

    if exclude_hf_dataset:
        rows = load_hf_jsonl_rows(exclude_hf_dataset)
        source = {"type": "hf_dataset", "value": exclude_hf_dataset}
    else:
        rows = iter_local_jsonl_rows(exclude_local_data or "")
        source = {"type": "local_jsonl", "value": exclude_local_data}

    keys: set[str | tuple[str, str]] = set()
    skipped_missing_smiles = 0
    for row in rows:
        key = extract_exclusion_key(row, exclude_mode)
        if key is None:
            skipped_missing_smiles += 1
            continue
        keys.add(key)

    return keys, {
        "source": source,
        "rows_loaded": len(rows),
        "rows_skipped_missing_smiles": skipped_missing_smiles,
        "unique_keys": len(keys),
        "exclude_mode": exclude_mode,
    }


def should_exclude_record(
    task: str,
    split: str,
    smiles: str,
    exclusion_keys: Optional[set[str | tuple[str, str]]],
    exclude_mode: Optional[str],
    exclude_splits: set[str],
) -> bool:
    if not exclusion_keys or split not in exclude_splits:
        return False
    if exclude_mode == EXCLUDE_MODE_SMILES:
        return smiles in exclusion_keys
    return (task, smiles) in exclusion_keys


def build_dataset(
    task: str,
    split: str,
    prompts: dict,
    cot_instruction: str,
    output_dir: str,
    raw_dir: str,
    exclusion_keys: Optional[set[str | tuple[str, str]]] = None,
    exclude_mode: Optional[str] = None,
    exclude_splits: Optional[set[str]] = None,
) -> tuple[int, int]:
    raw_path = os.path.join(raw_dir, task, f"{split}.csv")
    if not os.path.exists(raw_path):
        return 0, 0

    df = pd.read_csv(raw_path)
    if "Drug" not in df.columns or "Y" not in df.columns:
        print(f"  Warning: {raw_path} missing Drug/Y columns, skipping")
        return 0, 0

    prompt_template = resolve_prompt_template(task, prompts)
    if prompt_template is None:
        print(f"  Warning: No prompt template for task '{task}', skipping")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task}_{split}.jsonl")

    records = []
    removed = 0
    effective_exclude_splits = exclude_splits or set()
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        label = int(row["Y"])
        answer = "(A)" if label == 0 else "(B)"

        if should_exclude_record(
            task=task,
            split=split,
            smiles=smiles,
            exclusion_keys=exclusion_keys,
            exclude_mode=exclude_mode,
            exclude_splits=effective_exclude_splits,
        ):
            removed += 1
            continue

        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": build_user_message(task, prompt_template, smiles, cot_instruction)},
                ],
                "answer": answer,
                "smiles": smiles,
                "label": label,
                "task": task,
            }
        )

    if not records:
        if os.path.exists(out_path):
            os.remove(out_path)
        return 0, removed

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records), removed


def build_eval_dataset(prompts: dict, cot_instruction: str, output_dir: str, raw_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_tdc.jsonl")

    total = 0
    with open(out_path, "w") as f:
        for task in TASKS:
            raw_path = os.path.join(raw_dir, task, "test.csv")
            if not os.path.exists(raw_path):
                continue

            prompt_template = resolve_prompt_template(task, prompts)
            if prompt_template is None:
                continue

            df = pd.read_csv(raw_path)
            for _, row in df.iterrows():
                smiles = str(row["Drug"])
                label = int(row["Y"])
                answer = "(A)" if label == 0 else "(B)"

                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": build_user_message(task, prompt_template, smiles, cot_instruction)},
                    ],
                    "answer": answer,
                    "smiles": smiles,
                    "label": label,
                    "task": task,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    return total


def main():
    parser = argparse.ArgumentParser(description="Build v15 TDC datasets")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="Input split directory. Defaults to data/tdc/official_v15_dataset.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks (default: all)")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to build")
    parser.add_argument(
        "--exclude-hf-dataset",
        default=None,
        help="HF dataset repo to subtract from the built dataset using SMILES or (task, SMILES) overlap.",
    )
    parser.add_argument(
        "--exclude-local-data",
        default=None,
        help="Local JSONL file or directory to subtract from the built dataset.",
    )
    parser.add_argument(
        "--exclude-mode",
        choices=EXCLUDE_MODES,
        default=EXCLUDE_MODE_SMILES,
        help="Overlap key used for exclusion. 'smiles' is stricter; 'task_smiles' only removes the same task+SMILES rows.",
    )
    parser.add_argument(
        "--exclude-splits",
        nargs="*",
        default=["train"],
        help="Splits that should be filtered when an exclusion dataset is provided. Default: train only.",
    )
    args = parser.parse_args()

    prompts = load_prompts()
    cot_instruction = load_cot_instruction()
    tasks = args.tasks or TASKS
    exclude_splits = set(args.exclude_splits)

    if args.exclude_hf_dataset or args.exclude_local_data:
        exclusion_keys, exclusion_summary = load_exclusion_keys(
            exclude_mode=args.exclude_mode,
            exclude_hf_dataset=args.exclude_hf_dataset,
            exclude_local_data=args.exclude_local_data,
        )
        output_dir = resolve_output_dir(args.output_dir, args.exclude_mode)
    else:
        exclusion_keys = None
        exclusion_summary = None
        output_dir = resolve_output_dir(args.output_dir, None)

    print(f"Raw dir: {args.raw_dir}")
    print(f"Output: {output_dir}")
    print(f"System message prefix: {SYSTEM_MESSAGE[:100]}...")
    if exclusion_summary:
        print(
            "Excluding overlap using "
            f"{exclusion_summary['source']['type']}={exclusion_summary['source']['value']} "
            f"({exclusion_summary['exclude_mode']}, {exclusion_summary['unique_keys']} unique keys)"
        )
    print()

    total = 0
    total_removed = 0
    filter_summary: dict[str, Any] = {
        "exclude": exclusion_summary,
        "exclude_splits": sorted(exclude_splits),
        "tasks": {},
    }
    for task in tasks:
        for split in args.splits:
            n, removed = build_dataset(
                task,
                split,
                prompts,
                cot_instruction,
                output_dir,
                args.raw_dir,
                exclusion_keys=exclusion_keys,
                exclude_mode=args.exclude_mode if exclusion_keys else None,
                exclude_splits=exclude_splits,
            )
            if n > 0 or removed > 0:
                suffix = ""
                if removed:
                    suffix = f" ({removed} removed)"
                print(f"  {task}/{split}: {n} records{suffix}")
                total += n
                total_removed += removed
                filter_summary["tasks"].setdefault(task, {})[split] = {
                    "kept": n,
                    "removed": removed,
                }

    eval_n = build_eval_dataset(prompts, cot_instruction, output_dir, args.raw_dir)
    print(f"\n  eval_tdc.jsonl: {eval_n} records")
    if exclusion_summary:
        filter_summary["total_kept"] = total
        filter_summary["total_removed"] = total_removed
        filter_summary["eval_records"] = eval_n
        summary_path = os.path.join(output_dir, FILTER_SUMMARY_FILENAME)
        with open(summary_path, "w") as handle:
            json.dump(filter_summary, handle, indent=2, sort_keys=True)
        print(f"\nTotal: {total} records + {eval_n} eval ({total_removed} removed)")
        print(f"Filter summary: {summary_path}")
    else:
        print(f"\nTotal: {total} records + {eval_n} eval")


if __name__ == "__main__":
    main()
