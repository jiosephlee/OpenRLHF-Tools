"""Import the LLM4SD scaffold TDC subset into our local official_v15_dataset schema.

The source directory stores task CSVs as:

    scaffold_datasets/TDC/<task>/<task>_train.csv
    scaffold_datasets/TDC/<task>/<task>_valid.csv

with columns:

    smiles,<task>

This script converts that subset into the schema expected by
``data/tdc/build_v15_datasets.py``:

    official_v15_dataset/<task>/train.csv
    official_v15_dataset/<task>/val.csv

with columns:

    Drug_ID,Drug,Y
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "LLM4SD" / "scaffold_datasets" / "TDC"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "official_v15_dataset"
SUMMARY_NAME = "import_summary.csv"

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
}


def _iter_tasks(source_root: Path) -> list[str]:
    return sorted(path.name for path in source_root.iterdir() if path.is_dir())


def _read_source_rows(path: Path, *, expected_task: str) -> list[tuple[str, int]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if fieldnames != ["smiles", expected_task]:
            raise ValueError(
                f"{path} has unexpected columns {fieldnames}; expected ['smiles', '{expected_task}']"
            )

        rows: list[tuple[str, int]] = []
        for row in reader:
            smiles = str(row["smiles"]).strip()
            if not smiles:
                raise ValueError(f"{path} contains an empty SMILES row")
            label = int(float(row[expected_task]))
            if label not in (0, 1):
                raise ValueError(f"{path} contains non-binary label {label!r}")
            rows.append((smiles, label))
    return rows


def _write_output_rows(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Drug_ID", "Drug", "Y"])
        for index, (smiles, label) in enumerate(rows, start=1):
            writer.writerow([f"official_v15_{index}", smiles, label])


def import_dataset(source_root: Path, output_root: Path) -> list[dict[str, str | int]]:
    summary_rows: list[dict[str, str | int]] = []
    for task in _iter_tasks(source_root):
        for source_split, output_split in SPLIT_MAP.items():
            source_path = source_root / task / f"{task}_{source_split}.csv"
            if not source_path.exists():
                continue
            rows = _read_source_rows(source_path, expected_task=task)
            output_path = output_root / task / f"{output_split}.csv"
            _write_output_rows(output_path, rows)
            summary_rows.append(
                {
                    "task": task,
                    "source_split": source_split,
                    "output_split": output_split,
                    "num_rows": len(rows),
                    "source_file": str(source_path),
                    "output_file": str(output_path),
                }
            )
    return summary_rows


def write_summary(output_root: Path, rows: list[dict[str, str | int]]) -> None:
    summary_path = output_root / SUMMARY_NAME
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "source_split",
                "output_split",
                "num_rows",
                "source_file",
                "output_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = import_dataset(source_root, output_root)
    write_summary(output_root, summary_rows)

    print(f"Imported {len(summary_rows)} task/split files")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Tasks: {len(_iter_tasks(source_root))}")
    print(f"Summary: {output_root / SUMMARY_NAME}")


if __name__ == "__main__":
    main()
