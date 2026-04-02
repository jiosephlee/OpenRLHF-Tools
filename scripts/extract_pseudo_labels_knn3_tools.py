#!/usr/bin/env python3
"""Extract pseudo labels from knn3_tools_format prompts via majority vote
of KNN neighbor labels.

Neighbor format: (similarity: 0.9905, label: A)

Ties broken by sum of similarity scores.

Usage:
    python scripts/extract_pseudo_labels_knn3_tools.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data/tdc/knn3_tools_format")
OUTPUT_PATH = Path("data/tdc/metadata/knn3_tools_pseudo_labels.json")

NEIGHBOR_RE = re.compile(r"\(similarity:\s*([\d.]+),\s*label:\s*([A-Z])\)")


def extract_pseudo_label(text: str) -> str | None:
    matches = NEIGHBOR_RE.findall(text)
    if not matches:
        return None
    counts = Counter(label for _, label in matches)
    if len(counts) == 1:
        winner = counts.most_common(1)[0][0]
    elif counts.most_common(1)[0][1] != counts.most_common(2)[1][1]:
        # Clear majority
        winner = counts.most_common(1)[0][0]
    else:
        # Tie — break by sum of similarity
        sim_sums = defaultdict(float)
        for sim, label in matches:
            sim_sums[label] += float(sim)
        winner = max(sim_sums, key=sim_sums.get)
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
