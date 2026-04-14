#!/usr/bin/env python3
"""Create a one-time minority-oversampled copy of a TDC dataset directory.

For every ``*_train.jsonl`` in the input directory:
1. Compute the task imbalance ratio = max(class_count) / min(class_count)
2. Linearly map that ratio to a boost factor in ``[1.0, max_boost]``
3. Deterministically duplicate minority-label rows until the minority class
   reaches ``round(boost_factor * original_minority_count)``

All non-train files are copied through unchanged. By default the script writes a
new sibling directory named ``<input_dir>_minority_oversampled``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a minority-oversampled copy of a TDC dataset directory by "
            "deterministically duplicating minority-label train rows."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dataset directory to transform, e.g. data/tdc/openai_format_v12.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Destination directory. Defaults to a sibling directory named "
            "<input_dir>_minority_oversampled."
        ),
    )
    parser.add_argument(
        "--max-boost",
        type=float,
        default=2.0,
        help="Maximum minority duplication factor assigned to the most imbalanced task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic duplicate ordering.",
    )
    return parser.parse_args()


def normalize_label(label: Any) -> str:
    if label in (1, "1", "(B)", "B"):
        return "(B)"
    if label in (0, "0", "(A)", "A"):
        return "(A)"
    raise ValueError(f"Unsupported label value: {label!r}")


def stable_task_seed(base_seed: int, task_name: str) -> int:
    digest = hashlib.md5(task_name.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_train_files(input_dir: Path) -> list[Path]:
    train_files = sorted(path for path in input_dir.rglob("*_train.jsonl") if path.is_file())
    if not train_files:
        raise ValueError(f"No *_train.jsonl files found under {input_dir}")
    return train_files


def compute_boost_factors(task_stats: list[dict[str, Any]], max_boost: float) -> None:
    ratios = [stat["imbalance_ratio"] for stat in task_stats]
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    for stat in task_stats:
        if max_ratio == min_ratio:
            boost = 1.0
        else:
            alpha = (stat["imbalance_ratio"] - min_ratio) / (max_ratio - min_ratio)
            boost = 1.0 + alpha * (max_boost - 1.0)
        stat["boost_factor"] = boost


def build_task_stats(input_dir: Path, train_files: list[Path]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for path in train_files:
        relative_path = path.relative_to(input_dir)
        task_name = path.name.removesuffix("_train.jsonl")
        records = load_records(path)
        label_counts = Counter(normalize_label(record["label"]) for record in records)
        if len(label_counts) != 2:
            raise ValueError(f"Expected exactly two labels in {path}, got {dict(label_counts)}")

        minority_label, minority_count = min(label_counts.items(), key=lambda kv: kv[1])
        majority_label, majority_count = max(label_counts.items(), key=lambda kv: kv[1])
        stats.append(
            {
                "task_name": task_name,
                "input_path": str(path),
                "relative_path": str(relative_path),
                "records": records,
                "label_counts": dict(label_counts),
                "minority_label": minority_label,
                "minority_count": minority_count,
                "majority_label": majority_label,
                "majority_count": majority_count,
                "imbalance_ratio": majority_count / minority_count,
            }
        )
    return stats


def expand_task_records(stat: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = stat["records"]
    minority_label = stat["minority_label"]
    minority_rows = [record for record in records if normalize_label(record["label"]) == minority_label]

    target_minority_count = max(
        stat["minority_count"],
        int(round(stat["minority_count"] * stat["boost_factor"])),
    )
    extra_needed = target_minority_count - stat["minority_count"]

    duplicate_rows: list[dict[str, Any]] = []
    if extra_needed > 0:
        rng = random.Random(stable_task_seed(seed, stat["task_name"]))
        minority_order = list(minority_rows)
        rng.shuffle(minority_order)
        while len(duplicate_rows) < extra_needed:
            for row in minority_order:
                if len(duplicate_rows) >= extra_needed:
                    break
                duplicate_rows.append(copy.deepcopy(row))

    expanded = list(records) + duplicate_rows
    expanded_counts = Counter(normalize_label(record["label"]) for record in expanded)
    summary = {
        "task_name": stat["task_name"],
        "input_path": stat["input_path"],
        "relative_path": stat["relative_path"],
        "minority_label": minority_label,
        "majority_label": stat["majority_label"],
        "original_counts": stat["label_counts"],
        "imbalance_ratio": round(stat["imbalance_ratio"], 6),
        "boost_factor": round(stat["boost_factor"], 6),
        "extra_duplicates_added": extra_needed,
        "expanded_counts": dict(expanded_counts),
    }
    return expanded, summary


def resolve_output_dir(input_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return input_dir.with_name(f"{input_dir.name}_minority_oversampled")


def copy_non_train_files(input_dir: Path, output_dir: Path) -> None:
    for source_path in sorted(input_dir.rglob("*")):
        if not source_path.is_file():
            continue
        if source_path.name.endswith("_train.jsonl"):
            continue
        destination_path = output_dir / source_path.relative_to(input_dir)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir = resolve_output_dir(input_dir, args.output_dir).resolve()
    if output_dir == input_dir:
        raise ValueError("--output-dir must be different from --input-dir")

    train_files = find_train_files(input_dir)
    task_stats = build_task_stats(input_dir, train_files)
    compute_boost_factors(task_stats, max_boost=args.max_boost)

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_non_train_files(input_dir, output_dir)

    summaries: list[dict[str, Any]] = []
    for stat in task_stats:
        expanded_records, summary = expand_task_records(stat, seed=args.seed)
        output_path = output_dir / stat["relative_path"]
        write_records(output_path, expanded_records)
        summary["output_path"] = str(output_path)
        summary["expanded_total_records"] = len(expanded_records)
        summaries.append(summary)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_boost": args.max_boost,
        "seed": args.seed,
        "task_summaries": summaries,
    }
    summary_path = output_dir / "minority_oversampled_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote minority-oversampled dataset to {output_dir}")
    print(f"Wrote summary metadata to {summary_path}")


if __name__ == "__main__":
    main()
