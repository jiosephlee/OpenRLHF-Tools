#!/usr/bin/env python3
"""Deduplicate TDC raw CSVs across train/val/test splits.

For each task, reads data/tdc/raw/{task}/{split}.csv and:
  1. Drops SMILES with conflicting labels across any split
  2. Deduplicates same-label duplicates within each split (keeps first occurrence)
  3. Writes cleaned train/val/test CSVs to data/tdc/raw_deduplicated/{task}/

Usage:
    python scripts/deduplicate_tdc_train.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/tdc/raw")
OUT_DIR = Path("data/tdc/raw_deduplicated")
SPLITS = ["train", "val", "test"]

TASKS = [
    "AMES",
    "BBB_Martins",
    "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "Carcinogens_Lagunin",
    "ClinTox",
    "DILI",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret",
    "Skin_Reaction",
    "hERG",
]


def deduplicate_task(task: str) -> dict:
    raw_task_dir = RAW_DIR / task
    out_task_dir = OUT_DIR / task
    out_task_dir.mkdir(parents=True, exist_ok=True)

    split_dfs: dict[str, pd.DataFrame] = {}
    total_rows: dict[str, int] = {}

    for split in SPLITS:
        split_path = raw_task_dir / f"{split}.csv"
        if not split_path.exists():
            continue
        df = pd.read_csv(split_path)
        split_dfs[split] = df
        total_rows[split] = len(df)

    if not split_dfs:
        print(f"  {task}: no split CSVs found, skipping")
        return {}

    combined = []
    for split, df in split_dfs.items():
        tagged = df.copy()
        tagged["_split"] = split
        combined.append(tagged)
    merged = pd.concat(combined, ignore_index=True)

    label_counts = merged.groupby("Drug")["Y"].nunique()
    conflict_smiles = set(label_counts[label_counts > 1].index)

    dropped_conflicts = {}
    same_label_dupes = {}
    final_rows = {}

    for split, df in split_dfs.items():
        cleaned = df[~df["Drug"].isin(conflict_smiles)].copy()
        before_dedup = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset="Drug", keep="first")

        dropped_conflicts[split] = total_rows[split] - before_dedup
        same_label_dupes[split] = before_dedup - len(cleaned)
        final_rows[split] = len(cleaned)

        dst_path = out_task_dir / f"{split}.csv"
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        cleaned.to_csv(dst_path, index=False)

    return {
        "original": total_rows,
        "conflict_smiles": len(conflict_smiles),
        "conflict_rows": dropped_conflicts,
        "same_label_dupes": same_label_dupes,
        "final": final_rows,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"{'Task':40s} {'Split':>5s} {'Orig':>6s} {'ConfSmiles':>10s} "
        f"{'ConfRows':>8s} {'Dupes':>6s} {'Final':>6s}"
    )
    print("-" * 90)

    for task in TASKS:
        stats = deduplicate_task(task)
        if not stats:
            continue

        for split in SPLITS:
            if split not in stats["original"]:
                continue
            changed = " *" if stats["conflict_rows"][split] or stats["same_label_dupes"][split] else ""
            print(
                f"{task:40s} {split:>5s} {stats['original'][split]:6d} "
                f"{stats['conflict_smiles']:10d} {stats['conflict_rows'][split]:8d} "
                f"{stats['same_label_dupes'][split]:6d} {stats['final'][split]:6d}{changed}"
            )

    print("\nDone. Cleaned data in:", OUT_DIR)


if __name__ == "__main__":
    main()
