#!/usr/bin/env python3
"""Extract pseudo labels from knn3_format prompts and save as a reusable
task -> SMILES -> metadata mapping (JSON).

Usage:
    python scripts/extract_pseudo_labels.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

KNN3_DIR = Path("data/tdc/knn3_format")
OUTPUT_PATH = Path("data/tdc/metadata/knn3_pseudo_labels.json")

PSEUDO_LABEL_RE = re.compile(
    r"The pseudo label from naive Morgan fingerprint KNN prediction is \(([A-Z])\)"
)


def main():
    mapping: dict[str, dict[str, dict]] = defaultdict(dict)
    total = 0
    matched = 0

    for fpath in sorted(KNN3_DIR.glob("*.jsonl")):
        fname = fpath.name  # e.g. AMES_train.jsonl
        with open(fpath) as f:
            for line in f:
                row = json.loads(line)
                total += 1
                task = row["task"]
                smiles = row["smiles"]
                text = row["text"]

                m = PSEUDO_LABEL_RE.search(text)
                if m:
                    pseudo_label = f"({m.group(1)})"
                    matched += 1
                else:
                    pseudo_label = None

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
