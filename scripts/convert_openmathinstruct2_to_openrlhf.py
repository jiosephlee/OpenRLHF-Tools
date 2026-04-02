#!/usr/bin/env python3
"""Convert nvidia/OpenMathInstruct-2 HuggingFace dataset to OpenRLHF JSONL format.

Output format (one JSON object per line):
    {"text": "<CoT-formatted math problem>", "answer": "<expected_answer>"}

The 'text' field wraps each problem with a chain-of-thought prompt so the model
is instructed to produce \\boxed{} answers.  This matches the reward function in
examples/python/math_reward_func.py.

Usage:
    python scripts/convert_openmathinstruct2_to_openrlhf.py [--split train_1M] [--output_dir data/math/openmathinstruct2]
"""

import argparse
import json
import os

from datasets import load_dataset

COT_TEMPLATE = (
    "Think step-by-step to solve the following problem. "
    "Output your answer inside of \\boxed{{}} tags.:\n{problem}\n\n"
    "Let's think step-by-step"
)


def main():
    parser = argparse.ArgumentParser(description="Convert OpenMathInstruct-2 to OpenRLHF JSONL")
    parser.add_argument("--split", default="train_1M", choices=["train", "train_1M", "train_2M", "train_5M"])
    parser.add_argument("--output_dir", default=None, help="Output directory (default: data/math/openmathinstruct2)")
    parser.add_argument("--val_fraction", type=float, default=0.05, help="Fraction to hold out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve output dir relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(project_root, "data", "math", "openmathinstruct2")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading nvidia/OpenMathInstruct-2 split={args.split} ...")
    ds = load_dataset("nvidia/OpenMathInstruct-2", split=args.split)
    print(f"Loaded {len(ds)} samples")

    # Split into train/val
    if args.val_fraction > 0:
        splits = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
        train_ds = splits["train"]
        val_ds = splits["test"]
    else:
        train_ds = ds
        val_ds = None

    def write_jsonl(dataset, path):
        count = 0
        with open(path, "w") as f:
            for row in dataset:
                record = {
                    "text": COT_TEMPLATE.format(problem=row["problem"]),
                    "answer": row["expected_answer"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count

    train_path = os.path.join(output_dir, "train.jsonl")
    n_train = write_jsonl(train_ds, train_path)
    print(f"Wrote {n_train} training samples to {train_path}")

    if val_ds is not None:
        val_path = os.path.join(output_dir, "val.jsonl")
        n_val = write_jsonl(val_ds, val_path)
        print(f"Wrote {n_val} validation samples to {val_path}")

    print("Done!")


if __name__ == "__main__":
    main()
