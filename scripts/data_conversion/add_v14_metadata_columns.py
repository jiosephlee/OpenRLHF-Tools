#!/usr/bin/env python3
"""Add v14-specific columns to ``tdc_metadata_consolidated.csv``.

Populates (when missing):
  - ``f_neutral_7_4`` — neutral microspecies fraction at pH 7.4 from ``most_acidic_pka`` / ``most_basic_pka``
  - ``NOCount`` — RDKit Lipinski NOCount
  - ``NumAliphaticCarbocycles``, ``NumAromaticCarbocycles``, ``NumSaturatedCarbocycles``

``LabuteASA`` and pKa columns are expected to already exist.

Usage:
    python scripts/data_conversion/add_v14_metadata_columns.py
    python scripts/data_conversion/add_v14_metadata_columns.py --dry-run

Backs up the CSV to ``tdc_metadata_consolidated.csv.bak`` before writing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_CSV = (
    PROJECT_ROOT
    / "openrlhf"
    / "tools"
    / "therapeutic_tools"
    / "cache"
    / "tdc_metadata_consolidated.csv"
)


def _f_neutral_row(row: pd.Series) -> float:
    ph = 7.4
    f_neutral = 1.0
    mb = row.get("most_basic_pka")
    ma = row.get("most_acidic_pka")
    if mb is not None and pd.notna(mb):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (float(mb) - ph))
    if ma is not None and pd.notna(ma):
        f_neutral *= 1.0 / (1.0 + 10.0 ** (ph - float(ma)))
    return min(1.0, max(1e-12, f_neutral))


def _rdkit_counts(smiles: str) -> tuple[float, float, float, float]:
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(Lipinski.NOCount(mol)),
        float(Lipinski.NumAliphaticCarbocycles(mol)),
        float(Lipinski.NumAromaticCarbocycles(mol)),
        float(Lipinski.NumSaturatedCarbocycles(mol)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write")
    args = parser.parse_args()

    if not CACHE_CSV.exists():
        print(f"Missing cache CSV: {CACHE_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CACHE_CSV)
    if "Drug" not in df.columns:
        print("CSV missing Drug column", file=sys.stderr)
        sys.exit(1)

    need_f = "f_neutral_7_4" not in df.columns
    need_no = "NOCount" not in df.columns
    need_c1 = "NumAliphaticCarbocycles" not in df.columns
    need_c2 = "NumAromaticCarbocycles" not in df.columns
    need_c3 = "NumSaturatedCarbocycles" not in df.columns

    if not any((need_f, need_no, need_c1, need_c2, need_c3)):
        print("All v14 columns already present; nothing to do.")
        return

    if need_f:
        print("Computing f_neutral_7_4 from pKa columns...")
        df["f_neutral_7_4"] = df.apply(_f_neutral_row, axis=1)

    if need_no or need_c1 or need_c2 or need_c3:
        print("Computing RDKit NOCount / carbocycle counts (may take a few minutes)...")
        n = len(df)
        no_list: list[float] = []
        a_list: list[float] = []
        ar_list: list[float] = []
        s_list: list[float] = []
        for i, smi in enumerate(df["Drug"].astype(str)):
            no, a, ar, s = _rdkit_counts(smi)
            no_list.append(no)
            a_list.append(a)
            ar_list.append(ar)
            s_list.append(s)
            if (i + 1) % 5000 == 0:
                print(f"  ... {i + 1}/{n}")
        if need_no:
            df["NOCount"] = no_list
        if need_c1:
            df["NumAliphaticCarbocycles"] = a_list
        if need_c2:
            df["NumAromaticCarbocycles"] = ar_list
        if need_c3:
            df["NumSaturatedCarbocycles"] = s_list

    if args.dry_run:
        print("Dry run: would write updated CSV with columns:", [c for c in df.columns if c != "Drug"][:20], "...")
        return

    bak = CACHE_CSV.with_suffix(CACHE_CSV.suffix + ".bak")
    shutil.copy2(CACHE_CSV, bak)
    print(f"Backup: {bak}")
    df.to_csv(CACHE_CSV, index=False)
    print(f"Wrote {CACHE_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
