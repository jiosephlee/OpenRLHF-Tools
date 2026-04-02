#!/usr/bin/env python3
"""
Canonicalize SMILES in the deduplicated TDC dataset.

Reads from data/tdc/raw_deduplicated/{TASK}/{split}.csv and writes
canonicalized versions to data/tdc/deduplicated_canonicalized/{TASK}/{split}.csv.

Canonicalization: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=True)

Usage:
    python scripts/data_conversion/build_deduplicated_canonicalized.py
    python scripts/data_conversion/build_deduplicated_canonicalized.py --tasks AMES hERG
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TASK_NAMES = [
    "Bioavailability_Ma",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret",
    "Carcinogens_Lagunin",
    "hERG",
    "ClinTox",
    "DILI",
    "Skin_Reaction",
    "AMES",
]

SPLITS = ["train", "val", "test"]


def canonicalize(smiles: str) -> str | None:
    """Canonicalize a SMILES string. Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def process_split(src_path: Path, dst_path: Path) -> dict:
    """Process a single CSV split. Returns stats dict."""
    df = pd.read_csv(src_path)
    stats = {"total": len(df), "changed": 0, "failed": 0}

    canonical_smiles = []
    for smiles in df["Drug"]:
        if pd.isna(smiles):
            canonical_smiles.append(smiles)
            stats["failed"] += 1
            continue
        canon = canonicalize(str(smiles))
        if canon is None:
            print(f"  [WARN] Failed to canonicalize: {smiles!r}")
            canonical_smiles.append(smiles)  # keep original on failure
            stats["failed"] += 1
        else:
            if canon != str(smiles):
                stats["changed"] += 1
            canonical_smiles.append(canon)

    df["Drug"] = canonical_smiles
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Canonicalize SMILES in deduplicated TDC data")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific tasks (default: all)")
    parser.add_argument(
        "--src-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "raw_deduplicated"),
    )
    parser.add_argument(
        "--dst-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "deduplicated_canonicalized"),
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    tasks = args.tasks or TASK_NAMES

    total_rows = 0
    total_changed = 0
    total_failed = 0

    for task in tasks:
        print(f"\n  Processing {task}...")
        for split in SPLITS:
            src_path = src_dir / task / f"{split}.csv"
            if not src_path.exists():
                continue
            dst_path = dst_dir / task / f"{split}.csv"
            stats = process_split(src_path, dst_path)
            total_rows += stats["total"]
            total_changed += stats["changed"]
            total_failed += stats["failed"]
            print(
                f"    {split}: {stats['total']} rows, "
                f"{stats['changed']} changed, {stats['failed']} failed "
                f"-> {dst_path}"
            )

    print(f"\nDone. {total_rows} total rows, {total_changed} SMILES changed, {total_failed} failures.")


if __name__ == "__main__":
    main()
