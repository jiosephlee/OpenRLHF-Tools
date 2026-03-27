#!/usr/bin/env python3
"""Extract pseudo labels from knn3_per_label_no_pseudo_label_format prompts.

Since these prompts have 3 neighbors per class (always tied 3-3 by count),
the pseudo label is the class whose neighbors have higher average similarity.

Saves a reusable task -> SMILES -> metadata mapping (JSON).

Usage:
    python scripts/extract_pseudo_labels_per_label.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data/tdc/knn3_per_label_no_pseudo_label_format")
OUTPUT_PATH = Path("data/tdc/metadata/knn3_per_label_pseudo_labels.json")

HEADER_RE = re.compile(r"Top 3 similar molecules with label \(([A-Z])\)")
SIM_RE = re.compile(r"Computed Weighted Similarity:\s*([\d.]+)")


def extract_pseudo_label(text: str) -> str | None:
    """Determine pseudo label by highest average neighbor similarity."""
    avg_sims: dict[str, float] = {}
    headers = list(HEADER_RE.finditer(text))
    for i, m in enumerate(headers):
        label = m.group(1)
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]
        sims = [float(s) for s in SIM_RE.findall(chunk)]
        if sims:
            avg_sims[label] = sum(sims) / len(sims)
    if not avg_sims:
        return None
    winner = max(avg_sims, key=avg_sims.get)
    return f"({winner})"


def main():
    mapping: dict[str, dict[str, dict]] = defaultdict(dict)
    total = 0
    matched = 0

    for fpath in sorted(DATA_DIR.glob("*.jsonl")):
        fname = fpath.name
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                total += 1
                task = row["task"]
                smiles = row["smiles"]
                pseudo_label = extract_pseudo_label(row["text"])
                if pseudo_label:
                    matched += 1
                mapping[task][smiles] = {
                    "pseudo_label": pseudo_label,
                    "label": row.get("label"),
                    "answer": row.get("answer"),
                    "source_file": fname,
                }

    print(f"Processed {total} rows, extracted pseudo labels from {matched}/{total}")
    print(f"Tasks: {sorted(mapping.keys())}")
    for task in sorted(mapping):
        print(f"  {task}: {len(mapping[task])} molecules")

    # Cross-check against original knn3 pseudo labels if available
    orig_path = Path("data/tdc/metadata/knn3_pseudo_labels.json")
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        agree = disagree = missing = 0
        for task in mapping:
            for smiles in mapping[task]:
                orig_entry = orig.get(task, {}).get(smiles, {})
                orig_pl = orig_entry.get("pseudo_label")
                new_pl = mapping[task][smiles]["pseudo_label"]
                if orig_pl is None or new_pl is None:
                    missing += 1
                elif orig_pl == new_pl:
                    agree += 1
                else:
                    disagree += 1
        print(f"\nCross-check vs knn3_format: {agree} agree, {disagree} disagree, {missing} missing")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
