#!/usr/bin/env python3
"""
Convert the HF dataset jiosephlee/tdc-rl-dataset to per-task JSONL files.

Input:  HuggingFace dataset with columns {messages, task, label, drug}
Output: data/tdc/augmented_format/<Task>_train.jsonl and <Task>_val.jsonl
        with schema {messages, answer, task, smiles, label}

The HF dataset has only a train split, so we create a val split by
holding out --val_fraction (default 12.5%) per task with a fixed seed.

Usage:
    python scripts/convert_tdc_hf_dataset.py
    python scripts/convert_tdc_hf_dataset.py --output_dir data/tdc/augmented_format --val_fraction 0.15
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert HF TDC dataset to per-task JSONL")
    parser.add_argument(
        "--dataset",
        default="jiosephlee/tdc-rl-dataset",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <project_root>/data/tdc/augmented_format)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.125,
        help="Fraction of each task to hold out for validation (default: 0.125)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "data" / "tdc" / "augmented_format"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"Loaded {len(ds)} rows, columns: {ds.column_names}")

    # Group rows by task
    task_rows = defaultdict(list)
    for row in ds:
        task_rows[row["task"]].append(row)

    print(f"Found {len(task_rows)} tasks, val_fraction={args.val_fraction}, seed={args.seed}")

    # Split each task into train/val and write
    rng = random.Random(args.seed)
    total_train = 0
    total_val = 0

    for task in sorted(task_rows.keys()):
        rows = task_rows[task]
        rng.shuffle(rows)

        n_val = max(1, int(len(rows) * args.val_fraction))
        val_rows = rows[:n_val]
        train_rows = rows[n_val:]

        for split_name, split_rows in [("train", train_rows), ("val", val_rows)]:
            out_path = output_dir / f"{task}_{split_name}.jsonl"
            with open(out_path, "w") as f:
                for row in split_rows:
                    out_rec = {
                        "messages": row["messages"],
                        "answer": "(A)" if row["label"] == 0 else "(B)",
                        "task": task,
                        "smiles": row["drug"],
                        "label": row["label"],
                    }
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        print(f"  {task}: {len(train_rows)} train, {len(val_rows)} val")
        total_train += len(train_rows)
        total_val += len(val_rows)

    print(f"\nDone. {total_train} train + {total_val} val = {total_train + total_val} total")
    print(f"Written to {output_dir}")


if __name__ == "__main__":
    main()
