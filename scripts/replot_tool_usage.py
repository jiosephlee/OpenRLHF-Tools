#!/usr/bin/env python3
"""Replot tool_usage_phase_histograms.png from eval_step_*.json files.

Usage:
    python scripts/replot_tool_usage.py <folder_with_json_files> [--output <path.png>] [--raw]

By default uses per_dataset_usage_pct (fraction of prompts that used each tool at least once;
0–1 = 0–100%). Falls back to per_dataset_normalized for older JSONs. Pass --raw for raw counts.
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_jsons(folder: str) -> list[dict]:
    pattern = os.path.join(folder, "eval_step_*.json")
    files = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))
    if not files:
        print(f"No eval_step_*.json files found in {folder}", file=sys.stderr)
        sys.exit(1)
    entries = []
    for f in files:
        with open(f) as fh:
            entries.append(json.load(fh))
    print(f"Loaded {len(entries)} JSON files from {folder}")
    return entries


def plot(entries: list[dict], output_path: str, use_raw: bool):
    if use_raw:
        key = "per_dataset_counts"
    else:
        key = "per_dataset_usage_pct"
    datasets = sorted({ds for entry in entries for ds in entry.get(key, {}).keys()})
    if not datasets:
        print("No datasets found in JSON entries.", file=sys.stderr)
        sys.exit(1)

    nrows = len(datasets)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, max(6, 4.5 * nrows)), squeeze=False)
    fig.subplots_adjust(hspace=0.6)

    for row, ds in enumerate(datasets):
        ax = axes[row][0]
        tools = sorted({tool for entry in entries for tool in entry[key].get(ds, {}).keys()})
        if not tools:
            ax.set_axis_off()
            continue

        matrix = np.array(
            [[entry[key].get(ds, {}).get(tool, 0.0) for tool in tools] for entry in entries],
            dtype=float,
        )

        # Rescale so the largest value becomes 1.0 (approx "fraction of prompts")
        global_max = matrix.max()
        if global_max > 0:
            matrix = matrix / global_max

        # Relative scale: let the colorbar adapt to the actual data range
        vmin = matrix.min()
        vmax = matrix.max()
        if vmin == vmax:
            vmax = vmin + 1.0  # avoid degenerate range

        heatmap = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{ds}  ({'raw counts' if use_raw else '%'})", pad=8)
        ax.set_ylabel("eval step")
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("tool")
        y_labels = [str(entry["global_step"]) for entry in entries]
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.02)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Replot tool usage phase histograms with relative scale.")
    parser.add_argument("folder", help="Folder containing eval_step_*.json files")
    parser.add_argument("--output", "-o", default=None, help="Output PNG path (default: <folder>/tool_usage_phase_histograms.png)")
    parser.add_argument("--raw", action="store_true", help="Plot raw counts instead of normalized fractions")
    args = parser.parse_args()

    entries = load_jsons(args.folder)
    output_path = args.output or os.path.join(args.folder, "tool_usage_phase_histograms.png")
    plot(entries, output_path, use_raw=args.raw)


if __name__ == "__main__":
    main()
