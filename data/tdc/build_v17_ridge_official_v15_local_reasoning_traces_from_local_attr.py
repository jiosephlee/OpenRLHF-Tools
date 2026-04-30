"""Build local reasoning-trace artifacts from v17 ridge official_v15 local-attribution artifacts."""

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
    TRACE_SCHEMA_VERSION,
    feature_probability_shift,
    render_prompt_block,
    render_reasoning_trace,
    select_summary_items,
    top_feature_probability_shift,
)
from data.tdc.ml_prompt_artifacts import load_sample_prompt_map, sample_prompt_variant_dir, write_sample_prompt_records  # noqa: E402
from data.tdc.v17_ridge_official_v15_common import DEFAULT_REASONING_TRACE_DEBUG_DIR, TOOL_VERSION  # noqa: E402

DEFAULT_SPLITS = ["train", "val"]
SOURCE_VARIANT = "local_attribution"
MODERATE_ABS_SHIFT_FLOOR = 0.025
STRONG_ABS_SHIFT_FLOOR = 0.05
NO_PROBS_ARTIFACT_VARIANT = f"{ARTIFACT_VARIANT}_no_probs"

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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def estimate_task_shift_thresholds(tasks: list[str], splits: list[str]) -> dict[str, dict[str, float]]:
    per_task_shifts: dict[str, list[float]] = {task: [] for task in tasks}
    for task in tasks:
        for split in splits:
            prompt_map = load_sample_prompt_map(TOOL_VERSION, SOURCE_VARIANT, task, split)
            if not prompt_map:
                continue
            for row in prompt_map.values():
                decision_summary, items = parse_local_attribution_block(row["prompt_block"])
                selected_items = select_summary_items(items, decision_summary)
                base_prob_pos = decision_summary["base_prob_pos"]
                per_task_shifts[task].extend(
                    feature_probability_shift(base_prob_pos, float(item["contribution"]))
                    for item in selected_items
                )

    thresholds: dict[str, dict[str, float]] = {}
    for task, shifts in per_task_shifts.items():
        if not shifts:
            thresholds[task] = {"moderate_min": 0.0, "strong_min": 0.0, "count": 0}
            continue
        shifts.sort()
        thresholds[task] = {
            "moderate_min": percentile(shifts, 0.40),
            "strong_min": percentile(shifts, 0.70),
            "moderate_abs_min": MODERATE_ABS_SHIFT_FLOOR,
            "strong_abs_min": STRONG_ABS_SHIFT_FLOOR,
            "count": float(len(shifts)),
        }
    return thresholds


def build_records_for_split(
    task: str,
    split: str,
    *,
    artifact_variant: str = ARTIFACT_VARIANT,
    include_probability_updates: bool = True,
    strength_method: str = "task_probability_shift_percentile",
    task_strength_thresholds: dict[str, float] | None = None,
    top_feature_shift_threshold: float | None = None,
) -> tuple[list[dict], dict]:
    prompt_map = load_sample_prompt_map(TOOL_VERSION, SOURCE_VARIANT, task, split)
    if not prompt_map:
        return [], {}

    records = []
    prompt_block_lengths = []
    trace_lengths = []
    pos_counts = []
    neg_counts = []
    agreement_count = 0
    numeric_filter_dropped = 0

    for smiles, row in prompt_map.items():
        decision_summary, items = parse_local_attribution_block(row["prompt_block"])
        selected_items = select_summary_items(items, decision_summary)
        label = int(row["label"])
        target_label = "B" if label == 1 else "A"
        prompt_block = render_prompt_block(
            items,
            decision_summary,
            selected_items=selected_items,
            task_strength_thresholds=task_strength_thresholds,
            strength_method=strength_method,
            top_feature_shift_threshold=top_feature_shift_threshold,
            include_probability_updates=include_probability_updates,
        )
        reasoning_trace = render_reasoning_trace(
            selected_items,
            decision_summary,
            target_label,
            task=task,
            task_strength_thresholds=task_strength_thresholds,
            strength_method=strength_method,
            top_feature_shift_threshold=top_feature_shift_threshold,
            include_probability_updates=include_probability_updates,
            smiles=smiles,
        )
        if reasoning_trace is None:
            numeric_filter_dropped += 1
            continue

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
                "trace_version": TRACE_SCHEMA_VERSION,
            }
        )

    # Cap kept count at <= 50% of source prompts per task. Deterministic
    # downsampling by sorted SMILES so re-runs are stable.
    source_count = len(prompt_map)
    cap = source_count // 2
    cap_dropped = 0
    if len(records) > cap and cap > 0:
        keep = set(sorted(r["smiles"] for r in records)[:cap])
        cap_dropped = len(records) - len(keep)
        records = [r for r in records if r["smiles"] in keep]

    stats = {
        "records": len(records),
        "avg_prompt_block_chars": sum(prompt_block_lengths) / len(prompt_block_lengths) if prompt_block_lengths else 0.0,
        "avg_trace_chars": sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0.0,
        "avg_positive_items": sum(pos_counts) / len(pos_counts) if pos_counts else 0.0,
        "avg_negative_items": sum(neg_counts) / len(neg_counts) if neg_counts else 0.0,
        "agreement_rate": agreement_count / len(records) if records else 0.0,
        "numeric_filter_dropped": int(numeric_filter_dropped),
        "per_task_cap_dropped": int(cap_dropped),
        "per_task_cap": int(cap),
        "artifact_variant": artifact_variant,
        "include_probability_updates": include_probability_updates,
        "strength_method": strength_method,
        "task_strength_thresholds": task_strength_thresholds,
        "top_feature_shift_threshold": top_feature_shift_threshold,
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert v17 ridge official_v15 local-attribution artifacts into reasoning traces")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks to convert")
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS, help="Splits to convert")
    parser.add_argument(
        "--artifact-variant",
        default=ARTIFACT_VARIANT,
        help=f"Output prompt-artifact variant name. Default: {ARTIFACT_VARIANT}",
    )
    parser.add_argument(
        "--omit-probability-updates",
        action="store_true",
        help=f"Render traces without probability updates. Recommended with --artifact-variant {NO_PROBS_ARTIFACT_VARIANT}.",
    )
    parser.add_argument("--debug-dir", default=None, help="Directory for summary stats")
    args = parser.parse_args()

    source_root = sample_prompt_variant_dir(TOOL_VERSION, SOURCE_VARIANT)
    tasks = args.tasks or iter_tasks(source_root)
    if not tasks:
        raise FileNotFoundError(f"No source tasks found under {source_root}")

    artifact_variant = args.artifact_variant
    include_probability_updates = not args.omit_probability_updates
    out_root = sample_prompt_variant_dir(TOOL_VERSION, artifact_variant)
    out_root.mkdir(parents=True, exist_ok=True)

    strength_method = "task_probability_shift_percentile"
    legacy_strength_method = "legacy_relative_top_feature_downgrade"
    task_shift_thresholds = estimate_task_shift_thresholds(tasks, args.splits)

    summary: dict[str, dict] = {}
    total = 0
    for task in tasks:
        summary[task] = {}
        for split in args.splits:
            records, stats = build_records_for_split(
                task,
                split,
                artifact_variant=artifact_variant,
                include_probability_updates=include_probability_updates,
                strength_method=strength_method,
                task_strength_thresholds=task_shift_thresholds.get(task),
            )
            if not records:
                continue
            out_path = write_sample_prompt_records(TOOL_VERSION, artifact_variant, task, split, records)
            summary[task][split] = stats
            total += len(records)
            print(f"  {task}/{split}: {len(records)} converted reasoning-trace records -> {out_path}")

    manifest = {
        "tool_version": TOOL_VERSION,
        "artifact_type": artifact_variant,
        "source_variant": SOURCE_VARIANT,
        "include_probability_updates": include_probability_updates,
        "tasks": tasks,
        "total_records": total,
        "strength_method": strength_method,
        "legacy_strength_method": legacy_strength_method,
        "task_strength_thresholds": task_shift_thresholds,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.debug_dir is None:
        debug_dir_name = DEFAULT_REASONING_TRACE_DEBUG_DIR.name
        if artifact_variant != ARTIFACT_VARIANT:
            debug_dir_name = f"{debug_dir_name}_{artifact_variant}"
        debug_dir = DEFAULT_REASONING_TRACE_DEBUG_DIR.parent / debug_dir_name
    else:
        debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nReasoning-trace artifacts written to {out_root}")
    print(f"Debug summary written to {debug_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
