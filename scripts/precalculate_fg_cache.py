#!/usr/bin/env python3
"""Build the FG (functional group) description cache for all TDC SMILES.

Reads SMILES directly from our training data JSONLs in data/tdc/openai_format/
rather than re-downloading from TDC.  Writes the unified cache file that
AccFG.py loads at import time.

Usage (on cluster):
    # All tasks (reads every *.jsonl in data/tdc/openai_format/):
    python scripts/precalculate_fg_cache.py

    # Specific tasks only:
    python scripts/precalculate_fg_cache.py --tasks BBB_Martins AMES

    # Control parallelism (default: cpu_count, capped at 128):
    python scripts/precalculate_fg_cache.py --workers 64

    # Append to an existing cache (skips already-cached SMILES):
    python scripts/precalculate_fg_cache.py --append

Output:
    Intern-S1-recipe/DataPrepare/shared_data/TDC_all_fg_desc_with_attach_points_and_atom_ids.jsonl
"""

import argparse
import json
import logging
import multiprocessing
import multiprocessing.pool
import os
import sys
from pathlib import Path

# Limit per-worker thread overhead before any NumPy/MKL import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tdc" / "openai_format"
INTERN_S1_ROOT = PROJECT_ROOT / "Intern-S1-recipe"
CACHE_PATH = INTERN_S1_ROOT / "DataPrepare" / "shared_data" / "TDC_all_fg_desc_with_attach_points_and_atom_ids.jsonl"

# ---------------------------------------------------------------------------
# Setup: make Intern-S1-recipe importable without triggering tools/__init__.py
# ---------------------------------------------------------------------------
import types

if str(INTERN_S1_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERN_S1_ROOT))
if "tools" not in sys.modules:
    _pkg = types.ModuleType("tools")
    _pkg.__path__ = [str(INTERN_S1_ROOT / "tools")]
    _pkg.__package__ = "tools"
    sys.modules["tools"] = _pkg

# Mock pyPgSQL to prevent RDKit crash on import.
try:
    import pyPgSQL  # noqa: F401
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["pyPgSQL"] = MagicMock()

from tools.AccFG import high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids


# ---------------------------------------------------------------------------
# SMILES collection
# ---------------------------------------------------------------------------
def collect_smiles(tasks: list[str] | None) -> list[str]:
    """Extract unique SMILES from data/tdc/openai_format/ JSONLs."""
    smiles_set: set[str] = set()
    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No JSONL files found in {DATA_DIR}")
        sys.exit(1)

    for path in jsonl_files:
        # Filter by task name if requested.
        if tasks:
            stem = path.stem  # e.g. "BBB_Martins_train"
            if not any(stem.startswith(t + "_") or stem == t for t in tasks):
                continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                smi = rec.get("smiles")
                if isinstance(smi, str) and smi.strip():
                    smiles_set.add(smi.strip())

    return sorted(smiles_set)


# ---------------------------------------------------------------------------
# Existing cache
# ---------------------------------------------------------------------------
def load_existing_cache(path: Path) -> dict[str, str]:
    """Load an existing JSONL cache (one {smiles: desc} per line)."""
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with open(path, "r", encoding="utf-8") as f:
        try:
            # Try single JSON dict first (old format).
            cache = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cache.update(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cache


# ---------------------------------------------------------------------------
# Worker — uses fork-based pool so sys.modules (with mocked tools package)
# is inherited by child processes.
# ---------------------------------------------------------------------------
def _process_one(smiles: str) -> tuple[str, str] | None:
    """Compute FG description for a single SMILES (runs in forked worker)."""
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        desc = high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids(smiles)
        return (smiles, desc)
    except Exception as e:
        return (smiles, f"Error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pre-compute FG cache for TDC SMILES")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task names to include (default: all)")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 128), help="Parallel workers")
    parser.add_argument("--append", action="store_true", help="Skip SMILES already in cache")
    parser.add_argument("--output", type=str, default=str(CACHE_PATH), help="Output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Collect SMILES from our data files.
    logger.info("Collecting SMILES from data/tdc/openai_format/ ...")
    all_smiles = collect_smiles(args.tasks)
    logger.info(f"Found {len(all_smiles)} unique SMILES.")

    # 2. Optionally skip already-cached SMILES.
    existing_cache: dict[str, str] = {}
    if args.append:
        existing_cache = load_existing_cache(output_path)
        before = len(all_smiles)
        all_smiles = [s for s in all_smiles if s not in existing_cache]
        logger.info(f"Append mode: {before - len(all_smiles)} already cached, {len(all_smiles)} remaining.")

    if not all_smiles:
        logger.info("Nothing to compute — cache is complete.")
        return

    # 3. Compute in parallel using fork-based pool.
    # We MUST use fork (not loky/spawn) so child processes inherit the mocked
    # `tools` package in sys.modules — otherwise tools/__init__.py gets
    # imported and fails on missing deps like freesasa.
    logger.info(f"Computing FG descriptions with {args.workers} workers ...")
    computed = 0
    errors = 0

    # Use fork so child processes inherit sys.modules (mocked tools package).
    multiprocessing.set_start_method("fork", force=True)

    mode = "a" if args.append and output_path.exists() else "w"
    with multiprocessing.Pool(processes=args.workers) as pool, \
         open(output_path, mode, encoding="utf-8") as f:
        if mode == "w" and existing_cache:
            for k, v in existing_cache.items():
                f.write(json.dumps({k: v}, ensure_ascii=False) + "\n")

        for res in tqdm(pool.imap_unordered(_process_one, all_smiles, chunksize=8),
                        total=len(all_smiles), desc="Processing"):
            if res is None:
                continue
            smi, desc = res
            if isinstance(desc, str) and desc.startswith("Error:"):
                errors += 1
            else:
                computed += 1
            f.write(json.dumps({smi: desc}, ensure_ascii=False) + "\n")
            if computed % 100 == 0:
                f.flush()

    logger.info(f"Done. Computed {computed} descriptions ({errors} errors). Saved to {output_path}")
    total = len(existing_cache) + computed
    logger.info(f"Total cache entries: {total}")


if __name__ == "__main__":
    main()
