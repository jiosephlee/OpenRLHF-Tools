"""Build v16_no_neighbor TDC datasets from stored per-sample local-attribution prompt artifacts."""

from __future__ import annotations

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

from data.tdc.ml_prompt_artifacts import load_local_attribution_map  # noqa: E402
TOOL_VERSION = "v16_no_neighbor"
DEFAULT_RAW_DIR = os.path.join(_SCRIPT_DIR, "raw_deduplicated")
PROMPTS_PATH = os.path.join(_SCRIPT_DIR, "metadata", "prompts.json")
COT_INSTRUCTION_PATH = os.path.join(_SCRIPT_DIR, "metadata", "cot_instruction_refined_tools.txt")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "openai_format_v16_no_neighbor_local_attribution")
PRETEND_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "openai_format_v16_no_neighbor_local_attribution_pretend")
PRETEND_INSTRUCTION = (
    "When reasoning about this molecule, keep the above evidence in the back of your mind "
    "and treat it as part of your own chemical intuition rather than as model-derived evidence. "
    "Do not mention that it comes from a model; use it implicitly when deciding your answer."
)

SYSTEM_MESSAGE = "You are a chemist analyzing drug molecules."

TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "Carcinogens_Lagunin", "ClinTox",
    "DILI", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Skin_Reaction", "hERG",
]


def load_prompts() -> dict[str, str]:
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    return {k: v.rstrip().removesuffix("Answer:").rstrip() for k, v in prompts.items()}


def load_cot_instruction() -> str:
    with open(COT_INSTRUCTION_PATH) as f:
        return f.read().strip()


def resolve_prompt_template(task: str, prompts: dict[str, str]) -> Optional[str]:
    prompt_template = prompts.get(task)
    if prompt_template is not None:
        return prompt_template
    task_lower = task.lower()
    matches = [value for key, value in prompts.items() if key.lower() == task_lower]
    if len(matches) == 1:
        return matches[0]
    return None


def build_user_message(
    prompt_template: str,
    smiles: str,
    cot_instruction: str,
    evidence_block: str,
    *,
    pretend: bool = False,
) -> str:
    prompt = prompt_template.replace("{Drug SMILES}", smiles)
    if not pretend:
        return f"{prompt}\n\n{evidence_block}\n\n{cot_instruction}"
    return f"{prompt}\n\n{evidence_block}\n\n{PRETEND_INSTRUCTION}\n\n{cot_instruction}"


def build_dataset(
    task: str,
    split: str,
    prompts: dict[str, str],
    cot_instruction: str,
    output_dir: str,
    raw_dir: str,
    *,
    pretend: bool = False,
) -> int:
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

    prompt_map = load_local_attribution_map(TOOL_VERSION, task, split)
    if not prompt_map:
        raise FileNotFoundError(f"No stored local attribution prompts found for {task}/{split} under {TOOL_VERSION}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task}_{split}.jsonl")

    records = []
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        label = int(row["Y"])
        answer = "(A)" if label == 0 else "(B)"
        prompt_row = prompt_map.get(smiles)
        if prompt_row is None:
            raise KeyError(f"Missing stored local attribution prompt for {task}/{split} SMILES={smiles}")
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {
                        "role": "user",
                        "content": build_user_message(
                            prompt_template,
                            smiles,
                            cot_instruction,
                            prompt_row["prompt_block"],
                            pretend=pretend,
                        ),
                    },
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


def build_eval_dataset(
    prompts: dict[str, str],
    cot_instruction: str,
    output_dir: str,
    raw_dir: str,
    *,
    pretend: bool = False,
) -> int:
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
            prompt_map = load_local_attribution_map(TOOL_VERSION, task, "test")
            if not prompt_map:
                raise FileNotFoundError(f"No stored local attribution prompts found for {task}/test under {TOOL_VERSION}")

            df = pd.read_csv(raw_path)
            for _, row in df.iterrows():
                smiles = str(row["Drug"])
                label = int(row["Y"])
                prompt_row = prompt_map.get(smiles)
                if prompt_row is None:
                    raise KeyError(f"Missing stored local attribution prompt for {task}/test SMILES={smiles}")
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {
                            "role": "user",
                            "content": build_user_message(
                                prompt_template,
                                smiles,
                                cot_instruction,
                                prompt_row["prompt_block"],
                                pretend=pretend,
                            ),
                        },
                    ],
                    "answer": "(A)" if label == 0 else "(B)",
                    "smiles": smiles,
                    "label": label,
                    "task": task,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v16_no_neighbor local-attribution TDC datasets")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Input split directory")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks (default: all supported tasks)")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to build")
    parser.add_argument("--pretend", action="store_true", help="Append an instruction to internalize the local-attribution evidence as implicit intuition")
    args = parser.parse_args()

    prompts = load_prompts()
    cot_instruction = load_cot_instruction()
    tasks = args.tasks or TASKS
    output_dir = args.output_dir or (PRETEND_OUTPUT_DIR if args.pretend else DEFAULT_OUTPUT_DIR)

    total = 0
    for task in tasks:
        for split in args.splits:
            n = build_dataset(task, split, prompts, cot_instruction, output_dir, args.raw_dir, pretend=args.pretend)
            if n > 0:
                print(f"  {task}/{split}: {n} records")
                total += n

    eval_n = build_eval_dataset(prompts, cot_instruction, output_dir, args.raw_dir, pretend=args.pretend)
    print(f"\n  eval_tdc.jsonl: {eval_n} records")
    print(f"\nTotal: {total} records + {eval_n} eval")


if __name__ == "__main__":
    main()
