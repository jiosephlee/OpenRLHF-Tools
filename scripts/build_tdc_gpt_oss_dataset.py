#!/usr/bin/env python3
"""
Build GPT-OSS-style OpenAI-format TDC JSONL files from deduplicated canonicalized CSVs.

This builder:
  1. Reads CSVs from data/tdc/deduplicated_canonicalized/{TASK}/{split}.csv
  2. Uses data/tdc/metadata/prompts.json for prompt templates
  3. Strips the trailing "Answer:" from each template
  4. Appends data/tdc/metadata/cot_instruction_refined_tools.txt verbatim
  5. Writes OpenAI-style JSONL records to data/tdc/openai_format_gpt_oss

Usage:
    python scripts/build_tdc_gpt_oss_dataset.py --all
    python scripts/build_tdc_gpt_oss_dataset.py --tasks AMES hERG
    python scripts/build_tdc_gpt_oss_dataset.py --tasks AMES --splits train val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "tdc" / "deduplicated_canonicalized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "tdc" / "openai_format_gpt_oss"
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "data" / "tdc" / "metadata" / "prompts.json"
DEFAULT_COT_PATH = (
    PROJECT_ROOT / "data" / "tdc" / "metadata" / "cot_instruction_refined_tools.txt"
)

DEFAULT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build OpenAI-format JSONL files from TDC deduplicated canonicalized CSVs."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific TDC tasks to convert. Defaults to every task under --raw-dir.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every task found under --raw-dir.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to convert. Default: train val test",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing per-task CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where JSONL files will be written.",
    )
    parser.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS_PATH),
        help="Path to prompts.json.",
    )
    parser.add_argument(
        "--cot-instruction-path",
        default=str(DEFAULT_COT_PATH),
        help="Path to the CoT instruction text file to append.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cot_instruction(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def strip_answer_suffix(prompt: str) -> str:
    stripped = prompt.rstrip()
    if stripped.endswith("Answer:"):
        stripped = stripped[: -len("Answer:")].rstrip()
    return stripped


def fuzzy_match_prompt_key(task_name: str, prompt_keys: Iterable[str]) -> str | None:
    for key in prompt_keys:
        if key == task_name:
            return key
    lowered = task_name.lower()
    for key in prompt_keys:
        if key.lower() == lowered:
            return key
    return None


def detect_molecule_column(fieldnames: list[str]) -> str | None:
    for candidate in ("Drug", "Antibody", "SMILES", "Protein", "Peptide"):
        if candidate in fieldnames:
            return candidate
    return None


def label_to_answer(label: int) -> str:
    if label not in (0, 1):
        raise ValueError(f"Expected binary label 0/1, got {label}")
    return "(A)" if label == 0 else "(B)"


def build_user_content(prompt_template: str, molecule_value: str, cot_instruction: str) -> str:
    prompt = strip_answer_suffix(prompt_template)
    prompt = prompt.replace("{Drug SMILES}", molecule_value)
    return f"{prompt}\n{cot_instruction}"


def convert_csv(
    csv_path: Path,
    task_name: str,
    prompt_template: str,
    cot_instruction: str,
) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        molecule_column = detect_molecule_column(fieldnames)
        if molecule_column is None:
            raise ValueError(
                f"No supported molecule column found in {csv_path}. Columns: {fieldnames}"
            )
        if "Y" not in fieldnames:
            raise ValueError(f"Missing Y label column in {csv_path}")

        records = []
        for row in reader:
            molecule_value = (row.get(molecule_column) or "").strip()
            label_raw = (row.get("Y") or "").strip()
            if not molecule_value or label_raw == "":
                continue

            label = int(float(label_raw))
            user_content = build_user_content(prompt_template, molecule_value, cot_instruction)
            records.append(
                {
                    "messages": [{"role": "user", "content": user_content}],
                    "answer": label_to_answer(label),
                    "smiles": molecule_value,
                    "label": label,
                    "task": task_name,
                }
            )
    return records


def resolve_tasks(raw_dir: Path, requested_tasks: list[str] | None, convert_all: bool) -> list[str]:
    if requested_tasks:
        return requested_tasks
    if convert_all or not requested_tasks:
        return sorted(path.name for path in raw_dir.iterdir() if path.is_dir())
    raise ValueError("Specify --tasks or --all")


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    prompts_path = Path(args.prompts)
    cot_path = Path(args.cot_instruction_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file does not exist: {prompts_path}")
    if not cot_path.exists():
        raise FileNotFoundError(f"CoT instruction file does not exist: {cot_path}")

    prompts = load_prompts(prompts_path)
    cot_instruction = load_cot_instruction(cot_path)
    tasks = resolve_tasks(raw_dir, args.tasks, args.all)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0
    for task_name in tasks:
        prompt_key = fuzzy_match_prompt_key(task_name, prompts.keys())
        if prompt_key is None:
            print(f"Skipping {task_name}: no prompt template found")
            continue
        prompt_template = prompts[prompt_key]

        print(f"\nProcessing {task_name}...")
        for split in args.splits:
            csv_path = raw_dir / task_name / f"{split}.csv"
            if not csv_path.exists():
                print(f"  {split}: missing {csv_path.name}, skipping")
                continue

            records = convert_csv(csv_path, task_name, prompt_template, cot_instruction)
            output_path = output_dir / f"{task_name}_{split}.jsonl"
            with output_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_records += len(records)
            print(f"  {split}: wrote {len(records)} records to {output_path}")

    print(f"\nDone. Wrote {total_records} records to {output_dir}")


if __name__ == "__main__":
    main()
