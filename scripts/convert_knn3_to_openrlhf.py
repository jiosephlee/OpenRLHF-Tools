#!/usr/bin/env python3
"""
Convert KNN_3 data from Intern-S1-recipe format to OpenRLHF playbooks format.

Input format  (Intern-S1-recipe/DataPrepare/TDC_prepended/KNN_3/{train,valid}/{Task}.jsonl):
    {"text": "...", "drug": "SMILES", "Y": 0|1}

Output format (data/tdc/knn3_format/{Task}_train.jsonl, {Task}_val.jsonl):
    {"text": "...", "answer": "(A)|(B)", "task": "Task", "smiles": "SMILES", "label": 0|1}
"""

import json
import sys
from pathlib import Path


def convert_record(record: dict, task_name: str) -> dict:
    """Convert a single KNN_3 record to OpenRLHF format."""
    y = record["Y"]
    return {
        "text": record["text"],
        "answer": "(B)" if y == 1 else "(A)",
        "task": task_name,
        "smiles": record.get("drug", ""),
        "label": y,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "Intern-S1-recipe" / "DataPrepare" / "TDC_prepended" / "KNN_3"
    out_dir = project_root / "data" / "tdc" / "knn3_format"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "valid": "val"}

    total = 0
    for split_src, split_dst in split_map.items():
        split_dir = src_dir / split_src
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping")
            continue
        for jsonl_file in sorted(split_dir.glob("*.jsonl")):
            task_name = jsonl_file.stem  # e.g. "AMES"
            out_file = out_dir / f"{task_name}_{split_dst}.jsonl"
            count = 0
            with open(jsonl_file) as fin, open(out_file, "w") as fout:
                for line in fin:
                    record = json.loads(line)
                    converted = convert_record(record, task_name)
                    fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
            total += count
            print(f"  {out_file.name}: {count} samples")

    print(f"\nTotal: {total} samples written to {out_dir}")


if __name__ == "__main__":
    main()
