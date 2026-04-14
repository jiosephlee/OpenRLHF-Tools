#!/usr/bin/env python3
"""Filter SFT traces to those that only cite v13-supported descriptors.

A trace is kept iff every descriptor referenced by the narrative
(`Looking at <NAME> = ...`) is in the v13 allowed set:

    set(v13.FEATURE_NAMES) | {KNN_mean_label, KNN_min_dist, KNN_mean_dist}

Whole traces are dropped (per plan: preserves narrative coherence). Per-task
counts and top blocking descriptors are reported.

Usage:
    python data/sft_traces/scripts/filter_traces.py                  # write output
    python data/sft_traces/scripts/filter_traces.py --report-only    # no writes
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Make `openrlhf` importable when running this script directly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from openrlhf.tools.therapeutic_tools import v13  # noqa: E402

DEFAULT_INPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "traces")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "traces_v13")
DEFAULT_SUMMARY_PATH = os.path.join(_SCRIPT_DIR, "..", "summary_v13.json")

KNN_NAMES = {"KNN_mean_label", "KNN_min_dist", "KNN_mean_dist"}
ALLOWED = set(v13.FEATURE_NAMES) | KNN_NAMES

DESCRIPTOR_PATTERN = re.compile(r"Looking at ([A-Za-z_][A-Za-z0-9_]*)\s*=")


def cited_descriptors(text: str) -> list[str]:
    return DESCRIPTOR_PATTERN.findall(text)


def is_kept(descriptors: list[str]) -> bool:
    return all(d in ALLOWED for d in descriptors)


def process_file(path: str, out_path: str | None) -> dict:
    """Filter one task file. Returns stats dict."""
    n_total = 0
    n_kept = 0
    blockers: Counter = Counter()
    kept_lines: list[str] = []

    with open(path) as f:
        for line in f:
            n_total += 1
            d = json.loads(line)
            text = d["messages"][-1]["content"]
            cited = cited_descriptors(text)
            if is_kept(cited):
                n_kept += 1
                if out_path is not None:
                    kept_lines.append(line)
            else:
                for name in cited:
                    if name not in ALLOWED:
                        blockers[name] += 1

    if out_path is not None and kept_lines:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.writelines(kept_lines)

    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "blockers": blockers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-only", action="store_true",
                        help="Do not write filtered files; just print stats.")
    parser.add_argument("--top-n-blockers", type=int, default=10)
    args = parser.parse_args()

    files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith(".jsonl") and f != "all_tasks_combined.jsonl"
    )

    print(f"v13 allowed feature names: {len(ALLOWED)}")
    print(f"Input dir : {args.input_dir}")
    print(f"Output dir: {args.output_dir if not args.report_only else '(report-only)'}")
    print()

    per_task_summary = []
    grand_blockers: Counter = Counter()
    total_in = 0
    total_out = 0
    combined_lines: list[str] = []

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = None if args.report_only else os.path.join(args.output_dir, fname)
        stats = process_file(in_path, out_path)

        total_in += stats["n_total"]
        total_out += stats["n_kept"]
        grand_blockers.update(stats["blockers"])

        # Collect for combined output
        if not args.report_only and out_path and os.path.exists(out_path):
            with open(out_path) as f:
                combined_lines.extend(f.readlines())

        pct = (100.0 * stats["n_kept"] / stats["n_total"]) if stats["n_total"] else 0
        print(f"  {fname:40s}  {stats['n_kept']:6d}/{stats['n_total']:6d}  ({pct:5.1f}%)")
        if stats["blockers"]:
            top = ", ".join(f"{n}({c})" for n, c in stats["blockers"].most_common(3))
            print(f"      top blockers: {top}")

        per_task_summary.append({
            "task": fname.replace(".jsonl", ""),
            "n_total": stats["n_total"],
            "n_kept": stats["n_kept"],
            "kept_pct": pct,
            "top_blockers": stats["blockers"].most_common(args.top_n_blockers),
        })

    print()
    print(f"TOTAL: {total_out}/{total_in}  ({100.0 * total_out / total_in:.1f}%)")
    print()
    print(f"Top {args.top_n_blockers} blocking descriptors (corpus-wide):")
    for name, c in grand_blockers.most_common(args.top_n_blockers):
        print(f"  {c:6d}  {name}")

    if not args.report_only:
        # Write combined
        combined_path = os.path.join(args.output_dir, "all_tasks_combined.jsonl")
        with open(combined_path, "w") as f:
            f.writelines(combined_lines)
        print(f"\nWrote combined: {combined_path}")

        # Write summary
        summary = {
            "v13_allowed_feature_count": len(ALLOWED),
            "total_input_traces": total_in,
            "total_kept_traces": total_out,
            "kept_fraction": (total_out / total_in) if total_in else 0,
            "tasks": per_task_summary,
            "top_blockers_corpus_wide": grand_blockers.most_common(50),
        }
        os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)
        with open(args.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary:  {args.summary_path}")


if __name__ == "__main__":
    main()
