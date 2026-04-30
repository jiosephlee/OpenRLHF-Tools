"""One-shot migration: reorganize runs/ into runs/{eval|train}/{Monday-date}/<name>.

Classifies dirs starting with `eval-` as eval, `grpo-`/`ppo_` as train. Other
top-level entries are left in place. Week is the Monday (ISO) of the dir's mtime.

Usage:
    python scripts/migrate_runs_layout.py --dry-run
    python scripts/migrate_runs_layout.py --apply
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def week_folder(when: datetime) -> str:
    d = when.date()
    return (d - timedelta(days=d.weekday())).isoformat()

RUNS_DIR = os.path.join(REPO_ROOT, "runs")


def classify(name: str) -> str | None:
    if name.startswith("eval-"):
        return "eval"
    if name.startswith("grpo-") or name.startswith("ppo_"):
        return "train"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    moves: list[tuple[str, str]] = []
    skipped: list[str] = []

    for name in sorted(os.listdir(RUNS_DIR)):
        src = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(src):
            continue
        # Skip already-migrated category dirs.
        if name in ("eval", "train"):
            continue
        cat = classify(name)
        if cat is None:
            skipped.append(name)
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(src))
        wk = week_folder(mtime)
        dst_parent = os.path.join(RUNS_DIR, cat, wk)
        dst = os.path.join(dst_parent, name)
        if os.path.exists(dst):
            print(f"COLLISION (skip): {src} -> {dst}", file=sys.stderr)
            continue
        moves.append((src, dst))

    print(f"Planned moves: {len(moves)}; skipped (left in place): {len(skipped)}")
    for s in skipped:
        print(f"  SKIP {s}")

    if args.dry_run:
        for src, dst in moves[:20]:
            print(f"  {src} -> {dst}")
        if len(moves) > 20:
            print(f"  ... ({len(moves) - 20} more)")
        return 0

    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    print(f"Moved {len(moves)} run dirs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
