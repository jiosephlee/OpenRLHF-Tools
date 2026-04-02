#!/usr/bin/env python3
"""Deduplicate TDC raw training CSVs.

For each task, reads data/tdc/raw/{task}/train.csv and:
  1. Drops SMILES with conflicting labels (same SMILES string, different Y values)
  2. Deduplicates same-label duplicates (keeps first occurrence)
  3. Writes cleaned train.csv to data/tdc/raw_deduplicated/{task}/train.csv
  4. Symlinks val.csv and test.csv (unchanged) from raw/ into raw_deduplicated/

Usage:
    python scripts/deduplicate_tdc_train.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/tdc/raw")
OUT_DIR = Path("data/tdc/raw_deduplicated")

TASKS = [
    "AMES", "BBB_Martins", "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "Carcinogens_Lagunin", "ClinTox",
    "DILI", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Skin_Reaction", "hERG",
]


def deduplicate_train(task: str) -> dict:
    raw_task_dir = RAW_DIR / task
    out_task_dir = OUT_DIR / task
    out_task_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_task_dir / "train.csv"
    if not train_path.exists():
        print(f"  {task}: no train.csv found, skipping")
        return {}

    df = pd.read_csv(train_path)
    n_orig = len(df)

    # Find SMILES with conflicting labels
    label_counts = df.groupby("Drug")["Y"].nunique()
    conflict_smiles = set(label_counts[label_counts > 1].index)
    n_conflicts = len(conflict_smiles)
    n_conflict_rows = df[df["Drug"].isin(conflict_smiles)].shape[0]

    # Drop conflicts
    df_clean = df[~df["Drug"].isin(conflict_smiles)]

    # Deduplicate same-label dupes (keep first)
    n_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset="Drug", keep="first")
    n_same_label_dupes = n_before_dedup - len(df_clean)

    # Write cleaned train
    df_clean.to_csv(out_task_dir / "train.csv", index=False)

    # Symlink val and test (unchanged)
    for split in ["val", "test"]:
        src = raw_task_dir / f"{split}.csv"
        dst = out_task_dir / f"{split}.csv"
        if src.exists():
            dst.unlink(missing_ok=True)
            dst.symlink_to(src.resolve())

    stats = {
        "original": n_orig,
        "conflicts": n_conflicts,
        "conflict_rows": n_conflict_rows,
        "same_label_dupes": n_same_label_dupes,
        "final": len(df_clean),
    }
    return stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'Task':40s} {'Orig':>5s} {'Conflicts':>9s} {'ConfRows':>8s} {'Dupes':>5s} {'Final':>5s}")
    print("-" * 75)

    for task in TASKS:
        stats = deduplicate_train(task)
        if not stats:
            continue
        changed = " *" if stats["conflicts"] or stats["same_label_dupes"] else ""
        print(
            f"{task:40s} {stats['original']:5d} {stats['conflicts']:9d} "
            f"{stats['conflict_rows']:8d} {stats['same_label_dupes']:5d} "
            f"{stats['final']:5d}{changed}"
        )

    print("\nDone. Cleaned data in:", OUT_DIR)


if __name__ == "__main__":
    main()
