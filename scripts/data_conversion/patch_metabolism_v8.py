#!/usr/bin/env python3
"""
Patch [predict_metabolites] sections in v8 prepended tools dataset.

Re-runs predict_metabolites for records matching a target pattern (error
sections, SyGMa fallback sections, or all) and replaces the output in-place.
This lets you upgrade SyGMa results to GLORYx after updating the cache.

Must be run in the `openrlhf` conda env which has sygma installed.

Usage:
    python scripts/data_conversion/patch_metabolism_v8.py                  # patch errors only
    python scripts/data_conversion/patch_metabolism_v8.py --target sygma   # upgrade SyGMa → GLORYx
    python scripts/data_conversion/patch_metabolism_v8.py --target all     # re-run everything
    python scripts/data_conversion/patch_metabolism_v8.py --tasks AMES hERG --dry-run
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

TASKS_WITH_METABOLISM = [
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "DILI",
    "Bioavailability_Ma",
    "ClinTox",
    "Carcinogens_Lagunin",
    "AMES",
    "hERG",
]

# Matches the full [predict_metabolites] section up to the next tool section or end
SECTION_PATTERN = re.compile(
    r"\[predict_metabolites\]\n[^\[]*",
    re.DOTALL,
)


def _needs_patch(text: str, target: str) -> bool:
    """Check if a record's predict_metabolites section should be re-run."""
    if "[predict_metabolites]" not in text:
        return False
    if target == "error":
        return "[predict_metabolites]\nError:" in text
    if target == "sygma":
        return "SyGMa" in text and "[predict_metabolites]" in text
    if target == "all":
        return True
    return False


def patch_file(jsonl_path: Path, predict_fn, target: str, dry_run: bool = False) -> dict:
    """Patch a single JSONL file. Returns stats dict."""
    stats = {"total": 0, "matched": 0, "patched": 0, "skipped_same": 0, "failed": 0}

    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    stats["total"] = len(records)

    for rec in records:
        text = rec["text"]
        if not _needs_patch(text, target):
            continue
        stats["matched"] += 1

        smiles = rec["smiles"]
        try:
            new_output = predict_fn(smiles=smiles)
            new_section = f"[predict_metabolites]\n{new_output}"
            patched_text = SECTION_PATTERN.sub(lambda m: new_section, text)
            if patched_text == text:
                stats["skipped_same"] += 1
            else:
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
    parser = argparse.ArgumentParser(description="Patch predict_metabolites sections in v8 dataset")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "openai_format_prepended_tools_v8"),
    )
    parser.add_argument(
        "--target",
        choices=["error", "sygma", "all"],
        default="sygma",
        help="Which sections to re-run: error (Error: only), sygma (SyGMa fallbacks), all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count matches without patching")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tasks = args.tasks or TASKS_WITH_METABOLISM

    # Verify sygma is available
    try:
        import sygma
        print(f"sygma available: {sygma.__file__}")
    except ImportError:
        print("ERROR: sygma not installed. Run in the openrlhf conda env.")
        sys.exit(1)

    # Force reload of gloryx cache (in case it was updated)
    from openrlhf.tools.therapeutic_tools import metabolism as met_module
    met_module._GLORYX_CACHE = None
    from openrlhf.tools.therapeutic_tools.metabolism import predict_metabolites

    total_matched = 0
    total_patched = 0
    total_same = 0
    total_failed = 0

    for task in tasks:
        for split in args.splits:
            path = data_dir / f"{task}_{split}.jsonl"
            if not path.exists():
                continue

            stats = patch_file(path, predict_metabolites, target=args.target, dry_run=args.dry_run)
            total_matched += stats["matched"]
            total_patched += stats["patched"]
            total_same += stats["skipped_same"]
            total_failed += stats["failed"]

            if stats["matched"] > 0:
                action = "would patch" if args.dry_run else "patched"
                print(
                    f"  {path.name}: {stats['matched']} matched, "
                    f"{action} {stats['patched']}, "
                    f"same {stats['skipped_same']}, failed {stats['failed']}"
                )

    action = "Would patch" if args.dry_run else "Patched"
    print(
        f"\n{action} {total_patched}/{total_matched} sections "
        f"(same={total_same}, failed={total_failed}) across {len(tasks)} tasks."
    )


if __name__ == "__main__":
    main()
