"""Build direct SFT datasets from v17 ridge no-salts local reasoning-trace artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tdc.ml_prompt_artifacts import load_sample_prompt_map  # noqa: E402
from data.tdc.v17_ridge_no_salts_common import (  # noqa: E402
    DEFAULT_RAW_DIR,
    DEFAULT_REASONING_TRACE_OUTPUT_DIR,
    PROMPTS_PATH,
    TASKS,
    TOOL_VERSION,
)

ARTIFACT_VARIANT = "local_reasoning_trace"
SYSTEM_MESSAGE = "You are a chemist analyzing drug molecules."
TRACE_INSTRUCTION = (
    "Use the evidence summary below as a compact readout from a prior tool-informed pass. "
    "Write a short reasoning trace that weighs the evidence for and against the final decision, "
    "then end with `Answer: (A)` or `Answer: (B)`."
)


def load_prompts() -> dict[str, str]:
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    return {k: v.rstrip().removesuffix("Answer:").rstrip() for k, v in prompts.items()}


def resolve_prompt_template(task: str, prompts: dict[str, str]) -> Optional[str]:
    prompt_template = prompts.get(task)
    if prompt_template is not None:
        return prompt_template
    task_lower = task.lower()
    matches = [value for key, value in prompts.items() if key.lower() == task_lower]
    if len(matches) == 1:
        return matches[0]
    return None


def build_user_message(prompt_template: str, smiles: str, prompt_block: str) -> str:
    prompt = prompt_template.replace("{Drug SMILES}", smiles)
    return f"{prompt}\n\n{prompt_block}\n\n{TRACE_INSTRUCTION}"


def build_dataset(
    task: str,
    split: str,
    prompts: dict[str, str],
    output_dir: str,
    raw_dir: str,
    *,
    include_disagreements: bool,
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

    prompt_map = load_sample_prompt_map(TOOL_VERSION, ARTIFACT_VARIANT, task, split)
    if not prompt_map:
        raise FileNotFoundError(f"No stored reasoning-trace artifacts found for {task}/{split} under {TOOL_VERSION}/{ARTIFACT_VARIANT}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task}_{split}.jsonl")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        label = int(row["Y"])
        prompt_row = prompt_map.get(smiles)
        if prompt_row is None:
            raise KeyError(f"Missing stored reasoning-trace artifact for {task}/{split} SMILES={smiles}")
        agreement = bool(prompt_row.get("auxiliary_model", {}).get("agrees_with_target", False))
        if not include_disagreements and not agreement:
            skipped += 1
            continue
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {
                        "role": "user",
                        "content": build_user_message(prompt_template, smiles, prompt_row["prompt_block"]),
                    },
                    {
                        "role": "assistant",
                        "content": prompt_row["reasoning_trace"],
                    },
                ],
                "answer": "(A)" if label == 0 else "(B)",
                "smiles": smiles,
                "label": label,
                "task": task,
                "metadata": {
                    "artifact_variant": ARTIFACT_VARIANT,
                    "auxiliary_model": prompt_row.get("auxiliary_model", {}),
                },
            }
        )

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records), skipped


def build_eval_dataset(
    prompts: dict[str, str],
    output_dir: str,
    raw_dir: str,
    *,
    include_disagreements: bool,
) -> tuple[int, int]:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_tdc.jsonl")

    total = 0
    skipped = 0
    with open(out_path, "w") as f:
        for task in TASKS:
            raw_path = os.path.join(raw_dir, task, "test.csv")
            if not os.path.exists(raw_path):
                continue
            prompt_template = resolve_prompt_template(task, prompts)
            if prompt_template is None:
                continue
            prompt_map = load_sample_prompt_map(TOOL_VERSION, ARTIFACT_VARIANT, task, "test")
            if not prompt_map:
                raise FileNotFoundError(f"No stored reasoning-trace artifacts found for {task}/test under {TOOL_VERSION}/{ARTIFACT_VARIANT}")

            df = pd.read_csv(raw_path)
            for _, row in df.iterrows():
                smiles = str(row["Drug"])
                label = int(row["Y"])
                prompt_row = prompt_map.get(smiles)
                if prompt_row is None:
                    raise KeyError(f"Missing stored reasoning-trace artifact for {task}/test SMILES={smiles}")
                agreement = bool(prompt_row.get("auxiliary_model", {}).get("agrees_with_target", False))
                if not include_disagreements and not agreement:
                    skipped += 1
                    continue
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {
                            "role": "user",
                            "content": build_user_message(prompt_template, smiles, prompt_row["prompt_block"]),
                        },
                        {
                            "role": "assistant",
                            "content": prompt_row["reasoning_trace"],
                        },
                    ],
                    "answer": "(A)" if label == 0 else "(B)",
                    "smiles": smiles,
                    "label": label,
                    "task": task,
                    "metadata": {
                        "artifact_variant": ARTIFACT_VARIANT,
                        "auxiliary_model": prompt_row.get("auxiliary_model", {}),
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    return total, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v17 ridge no-salts local reasoning-trace TDC datasets")
    parser.add_argument("--output-dir", default=str(DEFAULT_REASONING_TRACE_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Input split directory")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to build")
    parser.add_argument(
        "--include-disagreements",
        action="store_true",
        help="Include examples where the auxiliary linear summary disagrees with the training label",
    )
    args = parser.parse_args()

    prompts = load_prompts()
    tasks = args.tasks or TASKS

    total = 0
    total_skipped = 0
    for task in tasks:
        for split in args.splits:
            n, skipped = build_dataset(
                task,
                split,
                prompts,
                args.output_dir,
                args.raw_dir,
                include_disagreements=args.include_disagreements,
            )
            if n > 0 or skipped > 0:
                print(f"  {task}/{split}: {n} records ({skipped} skipped for aux/label disagreement)")
                total += n
                total_skipped += skipped

    eval_n, eval_skipped = build_eval_dataset(
        prompts,
        args.output_dir,
        args.raw_dir,
        include_disagreements=args.include_disagreements,
    )
    print(f"\n  eval_tdc.jsonl: {eval_n} records ({eval_skipped} skipped for aux/label disagreement)")
    print(f"\nTotal: {total} records + {eval_n} eval")
    print(f"Skipped: {total_skipped + eval_skipped}")


if __name__ == "__main__":
    main()
