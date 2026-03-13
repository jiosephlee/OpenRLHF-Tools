#!/usr/bin/env python3
"""
Convert TDC augmented data from {text, Y, drug} format
to the pipeline-expected {text, answer, task, smiles, label} format.

Input:  ~/Downloads/content/tdc_augmented/{train,valid}/<Task>.jsonl
Output: data/tdc/augmented_format/<Task>_{train,val}.jsonl

Usage:
    python scripts/convert_tdc_augmented.py
    python scripts/convert_tdc_augmented.py --input_dir /path/to/tdc_augmented --output_dir data/tdc/augmented_format
"""

import argparse
import json
import os
from pathlib import Path


def convert_file(input_path: Path, output_path: Path, task_name: str) -> int:
    """Convert a single JSONL file. Returns number of records written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            out = {
                "text": rec["text"],
                "answer": "(A)" if rec["Y"] == 0 else "(B)",
                "task": task_name,
                "smiles": rec["drug"],
                "label": rec["Y"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert TDC augmented data to pipeline format")
    parser.add_argument(
        "--input_dir",
        default=os.path.expanduser("~/Downloads/content/tdc_augmented"),
        help="Root of augmented data (contains train/ and valid/ subdirs)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <project_root>/data/tdc/augmented_format)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Derive project root from this script's location
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "data" / "tdc" / "augmented_format"

    split_map = {"train": "train", "valid": "val"}

    total = 0
    for split_in, split_out in split_map.items():
        split_dir = input_dir / split_in
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping")
            continue
        for jsonl_file in sorted(split_dir.glob("*.jsonl")):
            task_name = jsonl_file.stem  # e.g. "AMES"
            out_path = output_dir / f"{task_name}_{split_out}.jsonl"
            n = convert_file(jsonl_file, out_path, task_name)
            total += n
            print(f"  {out_path.name}: {n} records")

    print(f"\nDone. {total} total records written to {output_dir}")


if __name__ == "__main__":
    main()
