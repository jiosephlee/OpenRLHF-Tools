#!/usr/bin/env python3
"""
Patch [get_3d_properties] error sections in v8 prepended tools dataset.

All 3D sections were built with 'Error: No module named freesasa' because
the openrlhf_nightly env lacks freesasa. This script re-runs get_3d_properties
using the openrlhf env (which has freesasa) and patches in-place.

Usage:
    python scripts/data_conversion/patch_3d_v8.py
    python scripts/data_conversion/patch_3d_v8.py --dry-run
"""

import argparse
import json
import re
import sys
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TASKS_WITH_3D = [
    "Bioavailability_Ma",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
]

SECTION_PATTERN = re.compile(
    r"\[get_3d_properties\]\n[^\[]*",
    re.DOTALL,
)


def patch_file(jsonl_path: Path, predict_fn, dry_run: bool = False) -> dict:
    stats = {"total": 0, "matched": 0, "patched": 0, "failed": 0}

    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    stats["total"] = len(records)

    for rec in records:
        text = rec["text"]
        if "[get_3d_properties]" not in text:
            continue
        if "Error:" not in text.split("[get_3d_properties]")[1].split("[")[0]:
            continue
        stats["matched"] += 1

        smiles = rec["smiles"]
        try:
            new_output = predict_fn(smiles=smiles)
            new_section = f"[get_3d_properties]\n{new_output}"
            patched_text = SECTION_PATTERN.sub(lambda m: new_section, text)
            rec["text"] = patched_text
            stats["patched"] += 1
        except Exception as e:
            print(f"  [FAIL] {smiles}: {e}")
            stats["failed"] += 1

    if not dry_run and stats["patched"] > 0:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=jsonl_path.parent, suffix=".tmp", delete=False
        )
        try:
            for rec in records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.close()
            shutil.move(tmp.name, jsonl_path)
        except Exception:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
            raise

    return stats


def main():
    parser = argparse.ArgumentParser(description="Patch get_3d_properties errors in v8 dataset")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "openai_format_prepended_tools_v8"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tasks = args.tasks or TASKS_WITH_3D

    # Verify freesasa
    try:
        import freesasa
        print("freesasa available")
    except ImportError:
        print("ERROR: freesasa not installed. Run in the openrlhf conda env.")
        sys.exit(1)

    from openrlhf.tools.therapeutic_tools.three_d import get_3d_properties

    total_matched = 0
    total_patched = 0
    total_failed = 0

    for task in tasks:
        for split in args.splits:
            path = data_dir / f"{task}_{split}.jsonl"
            if not path.exists():
                continue

            stats = patch_file(path, get_3d_properties, dry_run=args.dry_run)
            total_matched += stats["matched"]
            total_patched += stats["patched"]
            total_failed += stats["failed"]

            if stats["matched"] > 0:
                action = "would patch" if args.dry_run else "patched"
                print(
                    f"  {path.name}: {stats['matched']} matched, "
                    f"{action} {stats['patched']}, failed {stats['failed']}"
                )

    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{action} {total_patched}/{total_matched} sections (failed={total_failed}).")


if __name__ == "__main__":
    main()
