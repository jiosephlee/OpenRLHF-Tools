#!/usr/bin/env python3
"""
Update universal caches and rebuild fingerprint embeddings for v8-style data.

Three operations:
  1. Append new SMILES to fg_cache.jsonl (functional group descriptions)
  2. Append new SMILES to tdc_metadata_consolidated.csv (RDKit descriptors)
  3. Rebuild fingerprint embeddings from raw_deduplicated/ with stored canonical
     SMILES → cache/fingerprints_with_canonicalized/

Usage:
    python scripts/data_conversion/update_caches_v8.py
    python scripts/data_conversion/update_caches_v8.py --tasks AMES hERG
    python scripts/data_conversion/update_caches_v8.py --skip-fg --skip-metadata  # embeddings only
"""

import argparse
import json
import os
import sys
import logging
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "cache"
DATA_DIR = PROJECT_ROOT / "data" / "tdc" / "deduplicated_canonicalized"
FINGERPRINT_DATA_DIR = PROJECT_ROOT / "data" / "tdc" / "raw_deduplicated"
FINGERPRINT_FALLBACK_DATA_DIR = PROJECT_ROOT / "data" / "tdc" / "raw"

TASK_NAMES = [
    "Bioavailability_Ma", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "BBB_Martins", "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Carcinogens_Lagunin", "hERG", "ClinTox", "DILI", "Skin_Reaction", "AMES",
]

# Fingerprint parameters
FP_RADIUS = 2
FP_NBITS = 2048


# ---------------------------------------------------------------------------
# Collect SMILES from deduplicated_canonicalized
# ---------------------------------------------------------------------------

def collect_all_smiles(data_dir: Path, tasks: list[str]) -> set[str]:
    """Collect all unique SMILES from the canonicalized dataset."""
    all_smiles = set()
    for task in tasks:
        for split in ["train", "val", "test"]:
            path = data_dir / task / f"{split}.csv"
            if path.exists():
                df = pd.read_csv(path)
                if "Drug" in df.columns:
                    all_smiles.update(df["Drug"].dropna().unique())
    return all_smiles


# ---------------------------------------------------------------------------
# 1. FG Cache update
# ---------------------------------------------------------------------------

def load_existing_fg_cache(fg_path: Path) -> set[str]:
    """Load existing SMILES from fg_cache.jsonl."""
    done = set()
    if fg_path.exists():
        with open(fg_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done.add(entry["smiles"])
                except Exception:
                    pass
    return done


def update_fg_cache(all_smiles: set[str], fg_path: Path):
    """Append missing SMILES to fg_cache.jsonl."""
    existing = load_existing_fg_cache(fg_path)
    missing = sorted(all_smiles - existing)
    if not missing:
        logger.info("FG cache: all %d SMILES already cached.", len(all_smiles))
        return

    logger.info("FG cache: %d existing, %d new SMILES to compute.", len(existing), len(missing))

    # Import AccFG for functional group computation
    sys.path.insert(0, str(CACHE_DIR.parent))
    from legacy_tools.AccFG import concise_fg_description

    completed = 0
    start = time.time()
    with open(fg_path, "a") as f:
        for smiles in missing:
            try:
                desc = concise_fg_description(smiles)
                entry = {"smiles": smiles, "fg": desc}
            except Exception as e:
                entry = {"smiles": smiles, "fg": f"Error: {e}"}
            f.write(json.dumps(entry) + "\n")
            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                logger.info("  FG cache: %d/%d (%.0f/s)", completed, len(missing), rate)

    logger.info("FG cache: appended %d new entries.", completed)


# ---------------------------------------------------------------------------
# 2. Metadata consolidated update
# ---------------------------------------------------------------------------

def compute_rdkit_descriptors(smiles: str) -> dict:
    """Compute basic RDKit descriptors for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, GraphDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"Drug": smiles}

    try:
        qed_val = Descriptors.qed(mol)
    except Exception:
        qed_val = float("nan")

    try:
        balaban = GraphDescriptors.BalabanJ(mol)
    except Exception:
        balaban = float("nan")

    try:
        ipc = GraphDescriptors.Ipc(mol)
    except Exception:
        ipc = float("nan")

    aromatic_atoms = sum(1 for a in mol.GetAromaticAtoms())
    heavy = mol.GetNumHeavyAtoms()

    charges = []
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for atom in mol.GetAtoms():
            c = atom.GetDoubleProp("_GasteigerCharge")
            if not (c != c):  # skip NaN
                charges.append(abs(c))
    except Exception:
        pass

    pos_charges = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0)
    neg_charges = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0)

    return {
        "Drug": smiles,
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "MolMR": Crippen.MolMR(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "qed": qed_val,
        "HeavyAtomCount": heavy,
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "MaxAbsPartialCharge": max(charges) if charges else float("nan"),
        "MinAbsPartialCharge": min(charges) if charges else float("nan"),
        "MaxEStateIndex": Descriptors.MaxEStateIndex(mol),
        "MinEStateIndex": Descriptors.MinEStateIndex(mol),
        "NumAromaticAtoms": aromatic_atoms,
        "FractionAromaticAtoms": aromatic_atoms / heavy if heavy else 0.0,
        "NumPositiveCharges": pos_charges,
        "NumNegativeCharges": neg_charges,
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumHeterocycles": rdMolDescriptors.CalcNumHeterocycles(mol),
        "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        "NumAliphaticHeterocycles": rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        "NumSaturatedHeterocycles": rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds(mol),
        "BertzCT": GraphDescriptors.BertzCT(mol),
        "BalabanJ": balaban,
        "Ipc": ipc,
        "HallKierAlpha": GraphDescriptors.HallKierAlpha(mol),
        "Kappa1": GraphDescriptors.Kappa1(mol),
        "Kappa2": GraphDescriptors.Kappa2(mol),
        "Kappa3": GraphDescriptors.Kappa3(mol),
        "NumAtomStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        "NumUnspecifiedAtomStereoCenters": rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
    }


def update_metadata(all_smiles: set[str], meta_path: Path):
    """Append missing SMILES to tdc_metadata_consolidated.csv."""
    if meta_path.exists():
        existing_df = pd.read_csv(meta_path)
        existing_smiles = set(existing_df["Drug"].dropna().unique())
    else:
        existing_df = pd.DataFrame()
        existing_smiles = set()

    missing = sorted(all_smiles - existing_smiles)
    if not missing:
        logger.info("Metadata: all %d SMILES already cached.", len(all_smiles))
        return

    logger.info("Metadata: %d existing, %d new SMILES to compute.", len(existing_smiles), len(missing))

    new_rows = []
    start = time.time()
    for i, smiles in enumerate(missing):
        row = compute_rdkit_descriptors(smiles)
        new_rows.append(row)
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            logger.info("  Metadata: %d/%d (%.0f/s)", i + 1, len(missing), rate)

    new_df = pd.DataFrame(new_rows)

    # Append to existing CSV, preserving all columns
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(meta_path, index=False)
    logger.info("Metadata: appended %d new rows. Total: %d.", len(new_rows), len(combined))


# ---------------------------------------------------------------------------
# 3. Fingerprint embeddings (rebuild from scratch)
# ---------------------------------------------------------------------------

_MORGAN_GEN = None
_FEAT_GEN = None


def _get_generators():
    global _MORGAN_GEN, _FEAT_GEN
    if _MORGAN_GEN is None:
        from rdkit.Chem import rdFingerprintGenerator
        inv_gen = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_NBITS, atomInvariantsGenerator=inv_gen,
        )
        feat_inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        _FEAT_GEN = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_NBITS, atomInvariantsGenerator=feat_inv,
        )
    return _MORGAN_GEN, _FEAT_GEN


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string with RDKit."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_fingerprint(smiles: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute Morgan + FeatureMorgan fingerprints. Returns (morgan, feat) or None."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_gen, feat_gen = _get_generators()
    morgan = morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    feat = feat_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return morgan, feat


def load_fingerprint_split(data_dir: Path, task: str, split: str) -> pd.DataFrame:
    """Load a fingerprint source split, falling back to raw/ if needed."""
    candidates = [
        data_dir / task / f"{split}.csv",
        FINGERPRINT_FALLBACK_DATA_DIR / task / f"{split}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame(columns=["Drug", "Y"])


def build_fingerprint_task(task: str, data_dir: Path, out_dir: Path, overwrite: bool = False):
    """Build fingerprint embeddings for a single task from original dataset SMILES."""
    out_path = out_dir / f"{task}_embeddings.npz"
    if out_path.exists() and not overwrite:
        logger.info("%s: canonicalized fingerprint cache exists, skipping.", task)
        return True

    # Load train + val splits (train for neighbors, val for split tagging)
    all_smiles = []
    all_labels = []
    all_splits = []
    for split in ["train", "val"]:
        df = load_fingerprint_split(data_dir, task, split)
        if df.empty:
            continue
        for _, row in df.iterrows():
            smiles = row.get("Drug")
            label = row.get("Y")
            if pd.isna(smiles) or pd.isna(label):
                continue
            all_smiles.append(str(smiles))
            all_labels.append(int(label))
            all_splits.append(split)

    if not all_smiles:
        logger.warning("%s: no data found, skipping.", task)
        return False

    # Deduplicate by original dataset SMILES while preserving split/label metadata.
    seen = {}
    for smi, lbl, spl in zip(all_smiles, all_labels, all_splits):
        if smi not in seen:
            seen[smi] = (lbl, spl)

    unique_smiles = list(seen.keys())
    unique_labels = [seen[s][0] for s in unique_smiles]
    unique_splits = [seen[s][1] for s in unique_smiles]

    logger.info("%s: computing fingerprints for %d molecules...", task, len(unique_smiles))

    valid_smiles = []
    valid_canonical_smiles = []
    morgan_list = []
    feat_list = []
    labels_list = []
    splits_list = []

    for smi, lbl, spl in zip(unique_smiles, unique_labels, unique_splits):
        canonical_smi = canonicalize_smiles(smi)
        if canonical_smi is None:
            continue
        result = compute_fingerprint(canonical_smi)
        if result is None:
            continue
        morgan, feat = result
        valid_smiles.append(smi)
        valid_canonical_smiles.append(canonical_smi)
        morgan_list.append(morgan)
        feat_list.append(feat)
        labels_list.append(lbl)
        splits_list.append(spl)

    if not valid_smiles:
        logger.warning("%s: no valid fingerprints, skipping.", task)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        smiles=np.array(valid_smiles, dtype=object),
        canonical_smiles=np.array(valid_canonical_smiles, dtype=object),
        morgan_fps=np.stack(morgan_list),
        feat_morgan_fps=np.stack(feat_list),
        labels=np.array(labels_list, dtype=np.int32),
        splits=np.array(splits_list, dtype=object),
    )
    logger.info("%s: saved %d molecules -> %s", task, len(valid_smiles), out_path)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Update caches for v8 (canonicalized data)")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--skip-fg", action="store_true", help="Skip FG cache update")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip metadata update")
    parser.add_argument("--skip-fingerprints", action="store_true", help="Skip fingerprint rebuild")
    parser.add_argument("--overwrite-fingerprints", action="store_true")
    args = parser.parse_args()

    tasks = args.tasks or TASK_NAMES
    data_dir = Path(args.data_dir)

    all_smiles = collect_all_smiles(data_dir, tasks)
    logger.info("Collected %d unique SMILES from %d tasks.", len(all_smiles), len(tasks))

    if not args.skip_fg:
        fg_path = CACHE_DIR / "fg_cache.jsonl"
        update_fg_cache(all_smiles, fg_path)

    if not args.skip_metadata:
        meta_path = CACHE_DIR / "tdc_metadata_consolidated.csv"
        update_metadata(all_smiles, meta_path)

    if not args.skip_fingerprints:
        fp_out_dir = CACHE_DIR / "fingerprints_with_canonicalized"
        for task in tasks:
            build_fingerprint_task(
                task, FINGERPRINT_DATA_DIR, fp_out_dir,
                overwrite=args.overwrite_fingerprints,
            )

    logger.info("All cache updates complete.")


if __name__ == "__main__":
    main()
