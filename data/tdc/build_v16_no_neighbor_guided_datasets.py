"""Build v16_no_neighbor TDC datasets with per-task global guidance prompts."""

import argparse
import json
import os
import sys
from typing import Optional

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.tdc.ml_prompt_artifacts import copy_legacy_v14_global_guides, load_global_guide  # noqa: E402
TOOL_VERSION = "v16_no_neighbor"
DEFAULT_RAW_DIR = os.path.join(_SCRIPT_DIR, "raw_deduplicated")
PROMPTS_PATH = os.path.join(_SCRIPT_DIR, "metadata", "prompts.json")
COT_INSTRUCTION_PATH = os.path.join(_SCRIPT_DIR, "metadata", "cot_instruction_refined_tools.txt")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "openai_format_v16_no_neighbor_guided")

SYSTEM_MESSAGE = "You are a chemist analyzing drug molecules."

TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "Carcinogens_Lagunin", "ClinTox",
    "DILI", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Skin_Reaction", "hERG",
]


def load_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    return {k: v.rstrip().removesuffix("Answer:").rstrip() for k, v in prompts.items()}


def load_cot_instruction() -> str:
    with open(COT_INSTRUCTION_PATH) as f:
        return f.read().strip()


def build_user_message(task: str, prompt_template: str, smiles: str, cot_instruction: str, feature_guide: str) -> str:
    prompt = prompt_template.replace("{Drug SMILES}", smiles)
    if feature_guide:
        return f"{prompt}\n\n{feature_guide}\n\n{cot_instruction}"
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


def build_dataset(task: str, split: str, prompts: dict, cot_instruction: str, feature_guide: str, output_dir: str, raw_dir: str) -> int:
    raw_path = os.path.join(raw_dir, task, f"{split}.csv")
    if not os.path.exists(raw_path):
        return 0
    df = pd.read_csv(raw_path)
    if "Drug" not in df.columns or "Y" not in df.columns:
        print(f"  Warning: {raw_path} missing Drug/Y columns, skipping")
        return 0

    prompt_template = resolve_prompt_template(task, prompts)
    if prompt_template is None:
        print(f"  Warning: No prompt template for task '{task}', skipping")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task}_{split}.jsonl")

    records = []
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        label = int(row["Y"])
        answer = "(A)" if label == 0 else "(B)"
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": build_user_message(task, prompt_template, smiles, cot_instruction, feature_guide)},
                ],
                "answer": answer,
                "smiles": smiles,
                "label": label,
                "task": task,
            }
        )

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def build_eval_dataset(prompts: dict, cot_instruction: str, feature_guides: dict, output_dir: str, raw_dir: str) -> int:
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
            guide = feature_guides.get(task, "")
            df = pd.read_csv(raw_path)
            for _, row in df.iterrows():
                smiles = str(row["Drug"])
                label = int(row["Y"])
                answer = "(A)" if label == 0 else "(B)"
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": build_user_message(task, prompt_template, smiles, cot_instruction, guide)},
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
    parser = argparse.ArgumentParser(description="Build v16_no_neighbor guided TDC datasets")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Input split directory")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks (default: all supported tasks)")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to build")
    args = parser.parse_args()

    prompts = load_prompts()
    cot_instruction = load_cot_instruction()
    tasks = args.tasks or TASKS

    copy_legacy_v14_global_guides(TOOL_VERSION, tasks)

    feature_guides = {}
    for task in tasks:
        guide = load_global_guide(TOOL_VERSION, task)
        if guide:
            feature_guides[task] = guide
            print(f"  Loaded feature guide for {task} ({len(guide)} chars)")
        else:
            print(f"  WARNING: No feature guide for {task}")

    total = 0
    for task in tasks:
        guide = feature_guides.get(task, "")
        for split in args.splits:
            n = build_dataset(task, split, prompts, cot_instruction, guide, args.output_dir, args.raw_dir)
            if n > 0:
                print(f"  {task}/{split}: {n} records")
                total += n

    eval_n = build_eval_dataset(prompts, cot_instruction, feature_guides, args.output_dir, args.raw_dir)
    print(f"\n  eval_tdc.jsonl: {eval_n} records")
    print(f"\nTotal: {total} records + {eval_n} eval")


if __name__ == "__main__":
    main()
