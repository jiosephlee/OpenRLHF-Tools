"""Build local reasoning-trace artifacts from v17 ridge no-salts local-attribution artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tdc.local_reasoning_trace_artifacts_common import (  # noqa: E402
    ARTIFACT_VARIANT,
    render_prompt_block,
    render_reasoning_trace,
    select_summary_items,
)
from data.tdc.ml_prompt_artifacts import load_sample_prompt_map, sample_prompt_variant_dir, write_sample_prompt_records  # noqa: E402
from data.tdc.v17_ridge_no_salts_common import DEFAULT_REASONING_TRACE_DEBUG_DIR, TOOL_VERSION  # noqa: E402

DEFAULT_SPLITS = ["train", "val", "test"]
SOURCE_VARIANT = "local_attribution"

BASE_RE = re.compile(r"base rate before molecule-specific evidence: leans toward \(([AB])\) with P\(B\)=([0-9.]+)")
FINAL_RE = re.compile(r"ultimately points toward \(([AB])\).*?P\(B\)=([0-9.]+)")
ITEM_RE = re.compile(
    r"-\s+(Strong|Moderate|Weak):\s+feature\s+`([^`]+)`\s+\(([^)]*)\)\s+->\s+(.*?)\.\s+contribution=([+-][0-9.]+)"
)


def parse_local_attribution_block(block: str) -> tuple[dict, list[dict]]:
    base_match = BASE_RE.search(block)
    final_match = FINAL_RE.search(block)
    if base_match is None or final_match is None:
        raise ValueError("Could not parse base/final probabilities from local-attribution block")

    decision_summary = {
        "base_prob_pos": float(base_match.group(2).rstrip(".")),
        "base_label": base_match.group(1),
        "final_prob_pos": float(final_match.group(2).rstrip(".")),
        "final_label": final_match.group(1),
    }

    items: list[dict] = []
    for match in ITEM_RE.finditer(block):
        contribution = float(match.group(5))
        items.append(
            {
                "feature": match.group(2),
                "feature_display_name": match.group(3),
                "description": match.group(4),
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "raw_value": None,
            }
        )
    return decision_summary, items


def iter_tasks(source_root: Path) -> list[str]:
    if not source_root.exists():
        return []
    return sorted(path.name for path in source_root.iterdir() if path.is_dir())


def build_records_for_split(task: str, split: str) -> tuple[list[dict], dict]:
    prompt_map = load_sample_prompt_map(TOOL_VERSION, SOURCE_VARIANT, task, split)
    if not prompt_map:
        return [], {}

    records = []
    prompt_block_lengths = []
    trace_lengths = []
    pos_counts = []
    neg_counts = []
    agreement_count = 0

    for smiles, row in prompt_map.items():
        decision_summary, items = parse_local_attribution_block(row["prompt_block"])
        selected_items = select_summary_items(items, decision_summary)
        label = int(row["label"])
        target_label = "B" if label == 1 else "A"
        prompt_block = render_prompt_block(items, decision_summary, selected_items=selected_items)
        reasoning_trace = render_reasoning_trace(selected_items, decision_summary, target_label, task=task)

        if decision_summary["final_label"] == target_label:
            agreement_count += 1
        pos_counts.append(sum(1 for item in selected_items if item["contribution"] > 0))
        neg_counts.append(sum(1 for item in selected_items if item["contribution"] < 0))
        prompt_block_lengths.append(len(prompt_block))
        trace_lengths.append(len(reasoning_trace))

        records.append(
            {
                "task": task,
                "split": split,
                "smiles": smiles,
                "label": label,
                "answer": f"({target_label})",
                "prompt_block": prompt_block,
                "reasoning_trace": reasoning_trace,
                "selected_features": [
                    {
                        "feature": item["feature"],
                        "feature_display_name": item["feature_display_name"],
                        "description": item["description"],
                        "contribution": round(float(item["contribution"]), 4),
                    }
                    for item in selected_items
                ],
                "auxiliary_model": {
                    "base_prob_pos": round(float(decision_summary["base_prob_pos"]), 4),
                    "base_label": decision_summary["base_label"],
                    "final_prob_pos": round(float(decision_summary["final_prob_pos"]), 4),
                    "final_label": decision_summary["final_label"],
                    "agrees_with_target": decision_summary["final_label"] == target_label,
                },
                "source_variant": SOURCE_VARIANT,
            }
        )

    stats = {
        "records": len(records),
        "avg_prompt_block_chars": sum(prompt_block_lengths) / len(prompt_block_lengths) if prompt_block_lengths else 0.0,
        "avg_trace_chars": sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0.0,
        "avg_positive_items": sum(pos_counts) / len(pos_counts) if pos_counts else 0.0,
        "avg_negative_items": sum(neg_counts) / len(neg_counts) if neg_counts else 0.0,
        "agreement_rate": agreement_count / len(records) if records else 0.0,
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert v17 ridge no-salts local-attribution artifacts into reasoning traces")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks to convert")
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS, help="Splits to convert")
    parser.add_argument("--debug-dir", default=str(DEFAULT_REASONING_TRACE_DEBUG_DIR), help="Directory for summary stats")
    args = parser.parse_args()

    source_root = sample_prompt_variant_dir(TOOL_VERSION, SOURCE_VARIANT)
    tasks = args.tasks or iter_tasks(source_root)
    if not tasks:
        raise FileNotFoundError(f"No source tasks found under {source_root}")

    out_root = sample_prompt_variant_dir(TOOL_VERSION, ARTIFACT_VARIANT)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    total = 0
    for task in tasks:
        summary[task] = {}
        for split in args.splits:
            records, stats = build_records_for_split(task, split)
            if not records:
                continue
            out_path = write_sample_prompt_records(TOOL_VERSION, ARTIFACT_VARIANT, task, split, records)
            summary[task][split] = stats
            total += len(records)
            print(f"  {task}/{split}: {len(records)} converted reasoning-trace records -> {out_path}")

    manifest = {
        "tool_version": TOOL_VERSION,
        "artifact_type": ARTIFACT_VARIANT,
        "source_variant": SOURCE_VARIANT,
        "tasks": tasks,
        "total_records": total,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nReasoning-trace artifacts written to {out_root}")
    print(f"Debug summary written to {debug_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
