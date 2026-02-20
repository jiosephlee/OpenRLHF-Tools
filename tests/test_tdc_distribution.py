"""
Self-contained test: visualize how random shuffling distributes TDC tasks across training steps.

Creates two DataLoader instances with different random seeds and plots binary heatmaps
showing which tasks appear at each step (batch).

Usage:
    python tests/test_tdc_distribution.py
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Import the interleaving function from OpenRLHF
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openrlhf.datasets.prompts_dataset import interleave_indices_by_datasource

# ── Config ──────────────────────────────────────────────────────────────────
TASK_NAMES = [
    "Bioavailability_Ma", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "BBB_Martins", "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Carcinogens_Lagunin", "hERG", "ClinTox", "DILI", "Skin_Reaction", "AMES",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tdc" / "openai_format"
BATCH_SIZE = 64  # matches typical GRPO micro-batch


# ── Minimal dataset (no tokenizer/strategy needed) ─────────────────────────
class SimpleTDCDataset(Dataset):
    """Loads pre-converted JSONL files, keeps only task name per sample."""

    def __init__(self, task_names, data_dir, interleave_seed=None):
        self.tasks = []  # task name per sample
        for task in task_names:
            path = data_dir / f"{task}_train.jsonl"
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping {task}")
                continue
            with open(path) as f:
                count = sum(1 for _ in f)
            self.tasks.extend([task] * count)
        print(f"Total samples: {len(self.tasks)} across {len(set(self.tasks))} tasks")

        # Optionally apply curriculum-balanced interleaving
        if interleave_seed is not None:
            order = interleave_indices_by_datasource(
                list(range(len(self.tasks))), self.tasks, interleave_seed
            )
            self.tasks = [self.tasks[i] for i in order]
            print(f"  Applied interleave_indices_by_datasource (seed={interleave_seed})")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_fn(batch):
    return list(batch)


# ── Build step-by-task presence matrix ─────────────────────────────────────
def build_presence_matrix(dataset, seed, task_names_present, sequential=False):
    """Iterate through one epoch and record task presence per step.

    Args:
        sequential: If True, use SequentialSampler (for pre-interleaved datasets).
                    If False, use RandomSampler with given seed.
    """
    if sequential:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)

    n_steps = len(loader)
    task_to_row = {t: i for i, t in enumerate(task_names_present)}
    matrix = np.zeros((len(task_names_present), n_steps), dtype=np.int8)

    for step, batch_tasks in enumerate(loader):
        for t in batch_tasks:
            matrix[task_to_row[t], step] = 1

    return matrix, n_steps


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_heatmaps(matrices, titles, task_labels, n_steps, out_path):
    n = len(matrices)
    fig, axes = plt.subplots(n, 1, figsize=(max(16, n_steps * 0.12), 5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, mat, title in zip(axes, matrices, titles):
        ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        ax.set_yticks(range(len(task_labels)))
        ax.set_yticklabels(task_labels, fontsize=8)
        ax.set_title(f"Task presence per step ({title})", fontsize=12)
        ax.set_xlabel("Step (batch)")
        ax.set_ylabel("Task")

        for y in range(len(task_labels) - 1):
            ax.axhline(y + 0.5, color="gray", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {out_path}")
    plt.close()


def plot_cumulative_proportion(matrices, titles, task_labels, n_steps, out_path):
    """For each task, plot cumulative proportion of samples seen vs steps."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, mat, title in zip(axes, matrices, titles):
        for i, task in enumerate(task_labels):
            row = mat[i]
            cum = np.cumsum(row) / max(row.sum(), 1)
            ax.plot(range(n_steps), cum, label=task, alpha=0.7)
        ax.set_title(f"Cumulative task coverage ({title})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Fraction of steps where task appeared (cumulative)")
        ax.legend(fontsize=6, loc="lower right", ncol=2)
        ax.set_xlim(0, n_steps)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved cumulative plot to {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # 1) Random shuffle (two seeds)
    dataset = SimpleTDCDataset(TASK_NAMES, DATA_DIR)
    task_names_present = sorted(set(dataset.tasks))
    print(f"Tasks present: {task_names_present}")

    counts = defaultdict(int)
    for t in dataset.tasks:
        counts[t] += 1
    print("\nSamples per task:")
    for t in task_names_present:
        print(f"  {t:40s} {counts[t]:5d}")

    mat_rand1, n_steps = build_presence_matrix(dataset, seed=42, task_names_present=task_names_present)
    mat_rand2, _ = build_presence_matrix(dataset, seed=123, task_names_present=task_names_present)

    # 2) Interleaved (curriculum_balanced) — sequential iteration over pre-interleaved order
    dataset_il1 = SimpleTDCDataset(TASK_NAMES, DATA_DIR, interleave_seed=42)
    mat_il1, _ = build_presence_matrix(dataset_il1, seed=0, task_names_present=task_names_present, sequential=True)

    print(f"\nSteps per epoch: {n_steps} (batch_size={BATCH_SIZE})")

    out_dir = PROJECT_ROOT / "tests" / "plots"
    out_dir.mkdir(exist_ok=True)

    # Plot all three: 2 random + 1 interleaved
    all_mats = [mat_rand1, mat_rand2, mat_il1]
    all_titles = [
        "Random Shuffle (seed=42)",
        "Random Shuffle (seed=123)",
        "Interleaved (seed=42)",
    ]

    plot_heatmaps(all_mats, all_titles, task_names_present, n_steps,
                  out_dir / "tdc_task_distribution_heatmap.png")
    plot_cumulative_proportion(all_mats, all_titles, task_names_present, n_steps,
                               out_dir / "tdc_task_distribution_cumulative.png")

    # Summary stats
    print("\n── Coverage summary ──")
    for name, mat in zip(all_titles, all_mats):
        per_task_coverage = mat.sum(axis=1) / n_steps * 100
        print(f"\n{name}:")
        for i, t in enumerate(task_names_present):
            print(f"  {t:40s} appears in {per_task_coverage[i]:5.1f}% of steps")


if __name__ == "__main__":
    main()
