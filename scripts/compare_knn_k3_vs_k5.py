#!/usr/bin/env python3
"""Compare KNN pseudo-label quality at k=3 vs k=5 for v11 fingerprint setup.

Reuses the same data pipeline as build_knn_v11_pseudo_labels.py but evaluates
pseudo-label vs ground-truth Y for both k values and reports per-task +
macro accuracy and macro-F1.

Usage:
    python scripts/compare_knn_k3_vs_k5.py
    python scripts/compare_knn_k3_vs_k5.py --tasks AMES hERG
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "tdc" / "raw_deduplicated"
DEFAULT_SPLITS = ("train", "val", "test")
K_VALUES = (3, 5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    return p.parse_args()


def detect_molecule_column(fieldnames):
    for c in ("Drug", "Antibody", "SMILES", "Protein", "Peptide"):
        if c in fieldnames:
            return c
    return None


def iter_task_rows(task_dir: Path, splits):
    for split in splits:
        csv_path = task_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            mol_col = detect_molecule_column(fieldnames)
            if mol_col is None or "Y" not in fieldnames:
                continue
            for row in reader:
                smi = (row.get(mol_col) or "").strip()
                lab = (row.get("Y") or "").strip()
                if not smi or lab == "":
                    continue
                yield split, smi, int(float(lab))


def _batch_tanimoto(q, r):
    inter = q @ r.T
    qb = q.sum(axis=1, keepdims=True)
    rb = r.sum(axis=1, keepdims=True).T
    denom = qb + rb - inter
    return np.where(denom > 0, inter / denom, 0.0).astype(np.float32)


def _batch_weighted(qm, qf, rm, rf, wm=0.8, wf=0.2):
    return (wm * _batch_tanimoto(qm, rm) + wf * _batch_tanimoto(qf, rf)).astype(np.float32)


def macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for cls in (0, 1):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / 2


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from openrlhf.tools.therapeutic_tools.similarity import (
        _canonicalize_smiles,
        _compute_query_fp,
        _load_split_smiles,
        _load_task_data,
    )

    tasks = args.tasks or sorted(p.name for p in raw_dir.iterdir() if p.is_dir())

    per_task_stats = {}  # task -> {k: (acc, f1, n)}
    global_y = []
    global_pred = {k: [] for k in K_VALUES}

    for task in tasks:
        task_dir = raw_dir / task
        if not task_dir.is_dir():
            continue
        td = _load_task_data(task, "fingerprint")
        if td is None:
            print(f"skip {task}: no fingerprint cache")
            continue

        splits = td.get("splits")
        if splits is not None:
            train_mask = splits == "train"
        else:
            sd = _load_split_smiles(task)
            if sd is not None:
                tset = sd["train"]
                train_mask = np.array([s in tset for s in td["smiles"]], dtype=bool)
            else:
                train_mask = np.ones(len(td["smiles"]), dtype=bool)

        train_smiles = td["smiles"]
        train_labels = td["labels"]
        canonical = td.get("canonical_smiles")
        morgan = td["morgan_fps"]
        feat = td["feat_fps"]
        train_idx = np.where(train_mask)[0]
        train_morgan = morgan[train_idx]
        train_feat = feat[train_idx]
        train_neighbor_labels = train_labels[train_idx].astype(np.int8)

        rows = list(iter_task_rows(task_dir, args.splits))
        if not rows:
            continue
        unique_smiles = list(dict.fromkeys(s for _, s, _ in rows))

        q_morgan, q_feat, q_smi, q_exact = [], [], [], []
        smi_to_preds = {}
        for smi in unique_smiles:
            exact = train_smiles == smi
            exact_idx = exact.nonzero()[0]
            chosen = exact_idx[0] if len(exact_idx) > 0 else None
            if chosen is None and canonical is not None:
                c = _canonicalize_smiles(smi)
                if c is not None:
                    cm = canonical == c
                    ci = cm.nonzero()[0]
                    if len(ci) > 0:
                        chosen = ci[0]
            if chosen is not None:
                qm, qf = morgan[chosen], feat[chosen]
            else:
                qm = _compute_query_fp(smi, use_features=False)
                qf = _compute_query_fp(smi, use_features=True)
            if qm is None or qf is None:
                smi_to_preds[smi] = {k: None for k in K_VALUES}
                continue
            ex_pos = None
            if chosen is not None:
                tp = np.where(train_idx == chosen)[0]
                if len(tp) > 0:
                    ex_pos = int(tp[0])
            q_smi.append(smi)
            q_morgan.append(qm)
            q_feat.append(qf)
            q_exact.append(ex_pos)

        if q_smi:
            qm_arr = np.asarray(q_morgan, dtype=np.float32)
            qf_arr = np.asarray(q_feat, dtype=np.float32)
            bs = 512
            for start in range(0, len(q_smi), bs):
                end = min(start + bs, len(q_smi))
                sims = _batch_weighted(qm_arr[start:end], qf_arr[start:end], train_morgan, train_feat)
                for ri, smi in enumerate(q_smi[start:end]):
                    exp = q_exact[start + ri]
                    row_sims = sims[ri]
                    if exp is not None:
                        row_sims[exp] = float("-inf")
                    n_avail = int(np.isfinite(row_sims).sum())
                    preds = {}
                    for k in K_VALUES:
                        dk = min(k, n_avail)
                        if dk <= 0:
                            preds[k] = None
                            continue
                        topk = np.argpartition(row_sims, -dk)[-dk:]
                        mean = float(train_neighbor_labels[topk].mean())
                        preds[k] = 1 if mean >= 0.5 else 0
                    smi_to_preds[smi] = preds

        y_true_task = []
        y_pred_task = {k: [] for k in K_VALUES}
        for _split, smi, y in rows:
            preds = smi_to_preds.get(smi)
            if preds is None:
                continue
            if any(preds[k] is None for k in K_VALUES):
                continue
            y_true_task.append(y)
            for k in K_VALUES:
                y_pred_task[k].append(preds[k])

        if not y_true_task:
            continue
        yt = np.array(y_true_task, dtype=np.int8)
        stats = {}
        for k in K_VALUES:
            yp = np.array(y_pred_task[k], dtype=np.int8)
            acc = float((yp == yt).mean())
            f1 = macro_f1_binary(yt, yp)
            stats[k] = (acc, f1, len(yt))
        per_task_stats[task] = stats
        global_y.extend(y_true_task)
        for k in K_VALUES:
            global_pred[k].extend(y_pred_task[k])

        print(
            f"{task:40s}  n={len(yt):6d}  "
            + "  ".join(f"k={k}: acc={stats[k][0]:.3f} f1={stats[k][1]:.3f}" for k in K_VALUES)
        )

    print("\n=== Micro (pooled across tasks) ===")
    yt = np.array(global_y, dtype=np.int8)
    for k in K_VALUES:
        yp = np.array(global_pred[k], dtype=np.int8)
        acc = float((yp == yt).mean())
        f1 = macro_f1_binary(yt, yp)
        print(f"k={k}: acc={acc:.4f}  macro_f1={f1:.4f}  n={len(yt)}")

    print("\n=== Macro (mean across tasks) ===")
    for k in K_VALUES:
        accs = [s[k][0] for s in per_task_stats.values()]
        f1s = [s[k][1] for s in per_task_stats.values()]
        print(f"k={k}: mean_acc={np.mean(accs):.4f}  mean_macro_f1={np.mean(f1s):.4f}  tasks={len(accs)}")

    # Per-task winner counts
    print("\n=== Per-task winners (macro F1) ===")
    wins = {k: 0 for k in K_VALUES}
    ties = 0
    for task, s in per_task_stats.items():
        f1_3, f1_5 = s[3][1], s[5][1]
        if f1_3 > f1_5:
            wins[3] += 1
        elif f1_5 > f1_3:
            wins[5] += 1
        else:
            ties += 1
    print(f"k=3 wins: {wins[3]}  k=5 wins: {wins[5]}  ties: {ties}")


if __name__ == "__main__":
    main()
