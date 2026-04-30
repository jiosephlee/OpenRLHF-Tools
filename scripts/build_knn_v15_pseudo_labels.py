#!/usr/bin/env python3
"""
Build KNN pseudo-label metadata for TDC v15 datasets from official_v15_dataset CSVs.

The v15 OpenAI-format datasets are built from ``data/tdc/official_v15_dataset``,
so the shared KNN pseudo labels should use the same source split and matching
fingerprint cache. This builder therefore defaults to the official v15 CSVs and
to the ``fingerprints_with_canonicalized_official_v15`` cache. The script sets
``THERAPEUTIC_FINGERPRINT_CACHE_SUBDIR`` accordingly before importing the
similarity module.

Usage:
    python scripts/build_knn_v15_pseudo_labels.py
    python scripts/build_knn_v15_pseudo_labels.py --tasks AMES hERG
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "tdc" / "official_v15_dataset"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "tdc" / "metadata" / "knn_v15_pseudo_labels.json"
DEFAULT_FINGERPRINT_SUBDIR = "fingerprints_with_canonicalized_official_v15"
DEFAULT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KNN pseudo-label mapping for TDC v15 tool-calling datasets."
    )
    parser.add_argument("--tasks", nargs="+", help="Specific tasks to process.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to scan. Default: train val test",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing official_v15_dataset TDC CSVs.",
    )
    parser.add_argument(
        "--fingerprint-subdir",
        default=DEFAULT_FINGERPRINT_SUBDIR,
        help="Fingerprint cache subdir under openrlhf/tools/therapeutic_tools/cache.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination JSON path for the pseudo-label mapping.",
    )
    return parser.parse_args()


def label_to_answer(label: int) -> str:
    if label not in (0, 1):
        raise ValueError(f"Expected binary label, got {label}")
    return "(A)" if label == 0 else "(B)"


def detect_molecule_column(fieldnames: list[str]) -> str | None:
    for candidate in ("Drug", "Antibody", "SMILES", "Protein", "Peptide"):
        if candidate in fieldnames:
            return candidate
    return None


def iter_task_rows(task_dir: Path, splits: list[str]):
    for split in splits:
        csv_path = task_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            molecule_column = detect_molecule_column(fieldnames)
            if molecule_column is None:
                raise ValueError(f"No supported molecule column found in {csv_path}")
            if "Y" not in fieldnames:
                raise ValueError(f"Missing Y label column in {csv_path}")
            for row in reader:
                smiles = (row.get(molecule_column) or "").strip()
                label_raw = (row.get("Y") or "").strip()
                if not smiles or label_raw == "":
                    continue
                label = int(float(label_raw))
                yield split, csv_path.name, smiles, label


def _batch_tanimoto(query_fps: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    intersections = query_fps @ ref_fps.T
    query_bits = query_fps.sum(axis=1, keepdims=True)
    ref_bits = ref_fps.sum(axis=1, keepdims=True).T
    denom = query_bits + ref_bits - intersections
    return np.where(denom > 0, intersections / denom, 0.0).astype(np.float32)


def _batch_weighted_tanimoto(
    query_morgan: np.ndarray,
    query_feat: np.ndarray,
    ref_morgan: np.ndarray,
    ref_feat: np.ndarray,
    w_morgan: float = 0.8,
    w_feat: float = 0.2,
) -> np.ndarray:
    return (
        w_morgan * _batch_tanimoto(query_morgan, ref_morgan)
        + w_feat * _batch_tanimoto(query_feat, ref_feat)
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ["THERAPEUTIC_FINGERPRINT_CACHE_SUBDIR"] = args.fingerprint_subdir
    from openrlhf.tools.therapeutic_tools.similarity import (
        _canonicalize_smiles,
        _compute_query_fp,
        _load_split_smiles,
        _load_task_data,
    )

    tasks = args.tasks or sorted(path.name for path in raw_dir.iterdir() if path.is_dir())
    mapping: dict[str, dict[str, dict]] = defaultdict(dict)
    total = 0
    matched = 0
    failures = 0

    for task in tasks:
        task_dir = raw_dir / task
        if not task_dir.is_dir():
            print(f"Skipping {task}: task directory not found")
            continue
        task_data = _load_task_data(task, "fingerprint")
        if task_data is None:
            print(f"Skipping {task}: no fingerprint cache found")
            continue
        splits = task_data.get("splits")
        if splits is not None:
            train_mask = splits == "train"
        else:
            split_data = _load_split_smiles(task)
            if split_data is not None:
                train_smi_set = split_data["train"]
                train_mask = np.array([s in train_smi_set for s in task_data["smiles"]], dtype=bool)
            else:
                train_mask = np.ones(len(task_data["smiles"]), dtype=bool)

        print(f"Processing {task}...")
        task_rows = list(iter_task_rows(task_dir, args.splits))
        unique_smiles = list(dict.fromkeys(smiles for _, _, smiles, _ in task_rows))
        smiles_to_pseudo_label: dict[str, str | None] = {}

        train_smiles = task_data["smiles"]
        train_labels = task_data["labels"]
        canonical_smiles = task_data.get("canonical_smiles")
        morgan_fps = task_data["morgan_fps"]
        feat_fps = task_data["feat_fps"]
        train_indices = np.where(train_mask)[0]
        train_morgan = morgan_fps[train_indices]
        train_feat = feat_fps[train_indices]
        train_neighbor_labels = train_labels[train_indices].astype(np.int8)

        query_morgan_batch = []
        query_feat_batch = []
        query_smiles_batch = []
        query_exact_train_positions = []

        for smiles in unique_smiles:
            exact_mask = train_smiles == smiles
            exact_idx = exact_mask.nonzero()[0]
            chosen_idx = exact_idx[0] if len(exact_idx) > 0 else None
            if chosen_idx is None and canonical_smiles is not None:
                query_canonical_smiles = _canonicalize_smiles(smiles)
                if query_canonical_smiles is not None:
                    canonical_mask = canonical_smiles == query_canonical_smiles
                    canonical_idx = canonical_mask.nonzero()[0]
                    if len(canonical_idx) > 0:
                        chosen_idx = canonical_idx[0]
            if chosen_idx is not None:
                query_morgan = morgan_fps[chosen_idx]
                query_feat = feat_fps[chosen_idx]
            else:
                query_morgan = _compute_query_fp(smiles, use_features=False)
                query_feat = _compute_query_fp(smiles, use_features=True)

            if query_morgan is None or query_feat is None:
                smiles_to_pseudo_label[smiles] = None
                continue

            exact_train_position = None
            if chosen_idx is not None:
                train_pos = np.where(train_indices == chosen_idx)[0]
                if len(train_pos) > 0:
                    exact_train_position = int(train_pos[0])

            query_smiles_batch.append(smiles)
            query_morgan_batch.append(query_morgan)
            query_feat_batch.append(query_feat)
            query_exact_train_positions.append(exact_train_position)

        if query_smiles_batch:
            query_morgan_batch = np.asarray(query_morgan_batch, dtype=np.float32)
            query_feat_batch = np.asarray(query_feat_batch, dtype=np.float32)
            batch_size = 512
            for start in range(0, len(query_smiles_batch), batch_size):
                end = min(start + batch_size, len(query_smiles_batch))
                sims = _batch_weighted_tanimoto(
                    query_morgan_batch[start:end],
                    query_feat_batch[start:end],
                    train_morgan,
                    train_feat,
                )
                for row_idx, smiles in enumerate(query_smiles_batch[start:end]):
                    exact_train_position = query_exact_train_positions[start + row_idx]
                    row_sims = sims[row_idx]
                    if exact_train_position is not None:
                        row_sims[exact_train_position] = float("-inf")
                    display_k = min(3, int(np.isfinite(row_sims).sum()))
                    if display_k <= 0:
                        smiles_to_pseudo_label[smiles] = None
                        continue
                    top_k_idx = np.argpartition(row_sims, -display_k)[-display_k:]
                    top_k_idx = top_k_idx[np.argsort(row_sims[top_k_idx])[::-1]]
                    neighbor_labels = train_neighbor_labels[top_k_idx]
                    mean_label = float(neighbor_labels.mean())
                    smiles_to_pseudo_label[smiles] = "(B)" if mean_label >= 0.5 else "(A)"

        task_total = 0
        for split, source_file, smiles, label in task_rows:
            total += 1
            task_total += 1
            pseudo_label = smiles_to_pseudo_label.get(smiles)
            if pseudo_label is not None:
                matched += 1
            else:
                failures += 1

            # Key by both the raw CSV SMILES and its canonical form so that
            # consumers querying with either representation hit. Some TDC
            # CSVs store Kekulé SMILES while the openai-format eval JSONLs
            # store canonical aromatic SMILES (e.g. PAMPA_NCATS).
            keys = {smiles}
            canonical = _canonicalize_smiles(smiles)
            if canonical:
                keys.add(canonical)

            for key in keys:
                existing = mapping[task].get(key, {})
                source_splits = sorted(set(existing.get("source_splits", [])) | {split})
                source_files = sorted(set(existing.get("source_files", [])) | {source_file})
                mapping[task][key] = {
                    "pseudo_label": pseudo_label,
                    "label": label,
                    "answer": label_to_answer(label),
                    "source_splits": source_splits,
                    "source_files": source_files,
                }
        print(f"  processed {task_total} rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed {total} rows")
    print(f"Extracted pseudo labels from {matched}/{total}")
    if failures:
        print(f"Rows without pseudo label: {failures}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
