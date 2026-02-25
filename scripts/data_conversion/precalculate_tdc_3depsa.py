#!/usr/bin/env python3
"""Build the 3D-ePSA cache for all TDC SMILES.

Reads SMILES directly from our training data JSONLs in data/tdc/openai_format/
rather than re-downloading from TDC.  Writes the unified cache file that
ePSA_3D.py loads at import time.

Usage:
    # All tasks (reads every *.jsonl in data/tdc/openai_format/):
    python scripts/data_conversion/precalculate_tdc_3depsa.py

    # Specific tasks only:
    python scripts/data_conversion/precalculate_tdc_3depsa.py --tasks BBB_Martins AMES

    # Control parallelism (default: 2):
    python scripts/data_conversion/precalculate_tdc_3depsa.py --workers 4

    # Resume from previous run (skips already-cached SMILES):
    python scripts/data_conversion/precalculate_tdc_3depsa.py --append

Output:
    data/tdc/metadata/TDC_all_3depsa.jsonl
"""

import argparse
import json
import logging
import multiprocessing
import multiprocessing.pool
import os
import sys
import time
from pathlib import Path

# Limit per-worker thread overhead before any NumPy/RDKit/MKL import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("3depsa_cache_gen.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tdc" / "openai_format"
RAW_DIR = PROJECT_ROOT / "data" / "tdc" / "raw"
INTERN_S1_ROOT = PROJECT_ROOT / "Intern-S1-recipe"
CACHE_PATH = PROJECT_ROOT / "data" / "tdc" / "metadata" / "TDC_all_3depsa.jsonl"

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

from tools.ePSA_3D import exposed_polar_sasa_ensemble


# ---------------------------------------------------------------------------
# SMILES collection
# ---------------------------------------------------------------------------
def collect_smiles(tasks: list[str] | None) -> list[str]:
    """Extract unique SMILES from data/tdc/openai_format/ JSONLs and raw CSVs."""
    smiles_set: set[str] = set()

    # 1. Collect from openai_format JSONLs (primary source)
    if DATA_DIR.exists():
        jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
        for path in jsonl_files:
            if tasks:
                stem = path.stem
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
        logger.info(f"Collected {len(smiles_set)} SMILES from openai_format JSONLs.")
    else:
        logger.warning(f"openai_format directory not found: {DATA_DIR}")

    # 2. Also collect from raw CSVs (backup / additional coverage)
    if RAW_DIR.exists():
        import pandas as pd
        csv_files = list(RAW_DIR.rglob("*.csv"))
        before = len(smiles_set)
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if "Drug" in df.columns:
                    smiles_set.update(df["Drug"].dropna().astype(str).unique())
            except Exception as e:
                logger.warning(f"Error reading {csv_file.name}: {e}")
        logger.info(f"Collected {len(smiles_set) - before} additional SMILES from raw CSVs.")

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
    logger.info(f"Loaded {len(cache)} entries from existing cache.")
    return cache


# ---------------------------------------------------------------------------
# Worker — uses fork-based pool so sys.modules (with mocked tools package)
# is inherited by child processes.
# ---------------------------------------------------------------------------
def _process_one(smiles: str) -> tuple[str, str] | None:
    """Compute 3D-ePSA for a single SMILES (runs in forked worker)."""
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None

        rows, stats, boltz = exposed_polar_sasa_ensemble(smiles)

        # Boltzmann-weighted result preferred, fallback to mean
        if (
            boltz is not None
            and "polar_sasa_boltz" in boltz
            and "polar_fraction_boltz" in boltz
            and boltz["polar_sasa_boltz"] is not None
            and boltz["polar_fraction_boltz"] is not None
        ):
            desc = (
                f"3D conformation based estimation of PSA: {boltz['polar_sasa_boltz']:.2f}\n"
                f"3D conformation based estimation of Polar Fraction: {boltz['polar_fraction_boltz']:.2f}"
            )
        else:
            desc = (
                f"3D conformation based estimation of PSA: {stats['polar_sasa_mean']:.2f}\n"
                f"3D conformation based estimation of Polar Fraction: {stats['polar_fraction_mean']:.2f}"
            )
        return (smiles, desc)
    except Exception as e:
        return (smiles, f"ERROR: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pre-compute 3D-ePSA cache for TDC SMILES")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task names to include (default: all)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    parser.add_argument("--append", action="store_true", help="Resume: skip SMILES already in cache")
    parser.add_argument("--output", type=str, default=str(CACHE_PATH), help="Output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("3D-ePSA Cache Generation")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Append/resume mode: {args.append}")
    logger.info("=" * 60)

    # 1. Collect SMILES from our data files.
    logger.info("Collecting SMILES ...")
    all_smiles = collect_smiles(args.tasks)
    logger.info(f"Found {len(all_smiles)} unique SMILES total.")

    # 2. Optionally skip already-cached SMILES (resume support).
    existing_cache: dict[str, str] = {}
    if args.append:
        existing_cache = load_existing_cache(output_path)
        before = len(all_smiles)
        all_smiles = [s for s in all_smiles if s not in existing_cache]
        logger.info(f"Resume mode: {before - len(all_smiles)} already cached, {len(all_smiles)} remaining.")

    if not all_smiles:
        logger.info("Nothing to compute — cache is complete.")
        return

    # 3. Compute in parallel using fork-based pool.
    logger.info(f"Starting 3D-ePSA computation for {len(all_smiles)} SMILES with {args.workers} workers ...")

    # We MUST use fork (not loky/spawn) so child processes inherit the mocked
    # `tools` package in sys.modules.
    multiprocessing.set_start_method("fork", force=True)

    computed = 0
    errors = 0
    start_time = time.time()

    mode = "a" if args.append and output_path.exists() else "w"
    with multiprocessing.Pool(processes=args.workers) as pool, \
         open(output_path, mode, encoding="utf-8") as f:

        # If writing fresh but there was an existing cache loaded, re-write it first.
        if mode == "w" and existing_cache:
            for k, v in existing_cache.items():
                f.write(json.dumps({k: v}, ensure_ascii=False) + "\n")

        for res in tqdm(pool.imap_unordered(_process_one, all_smiles, chunksize=1),
                        total=len(all_smiles), desc="3D-ePSA"):
            if res is None:
                continue
            smi, desc = res
            if isinstance(desc, str) and desc.startswith("ERROR:"):
                errors += 1
                logger.warning(f"Failed: {smi[:60]}... -> {desc}")
            else:
                computed += 1

            f.write(json.dumps({smi: desc}, ensure_ascii=False) + "\n")

            # Flush and log periodically.
            if (computed + errors) % 50 == 0:
                f.flush()
                elapsed = time.time() - start_time
                rate = (computed + errors) / elapsed if elapsed > 0 else 0
                remaining = (len(all_smiles) - computed - errors) / rate if rate > 0 else float("inf")
                logger.info(
                    f"Progress: {computed + errors}/{len(all_smiles)} "
                    f"({computed} ok, {errors} err) | "
                    f"{rate:.1f} mol/s | "
                    f"ETA: {remaining / 60:.1f} min"
                )

    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Finished in {elapsed_total / 60:.1f} min.")
    logger.info(f"Computed: {computed} | Errors: {errors}")
    logger.info(f"Total cache entries: {len(existing_cache) + computed + errors}")
    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
