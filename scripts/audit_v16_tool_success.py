#!/usr/bin/env python
"""Audit v16 tool-call success over the v16 TDC dataset.

Default mode checks `get_features(smiles, all_feature_groups)` once per unique
SMILES in `data/tdc/openai_format_v16/*.jsonl`, using the same execution helper
that RL uses for tool calls. The script can also audit `get_neighbors` per
dataset row `(smiles, task)`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any


ERROR_MARKERS = (
    "invalid smiles",
    "error -",
    "error:",
    "error screening",
    "error while summarizing",
    "could not compute fingerprint",
)


def _load_v16_rows(dataset_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.jsonl"))):
        if path.endswith("eval_tdc.jsonl"):
            continue
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def _extract_result_text(result_json: str) -> str:
    try:
        payload = json.loads(result_json)
    except Exception:
        return result_json

    if isinstance(payload, dict):
        if isinstance(payload.get("error"), str):
            return payload["error"]
        if isinstance(payload.get("result"), str):
            return payload["result"]
    return result_json


def _classify_failure(error_str: str, result_json: str) -> str | None:
    if error_str:
        return error_str.splitlines()[0][:300]

    text = _extract_result_text(result_json)
    lowered = text.lower()
    for marker in ERROR_MARKERS:
        if marker in lowered:
            for line in text.splitlines():
                if marker in line.lower():
                    return line[:300]
            return marker
    return None


def _run_get_features(smiles: str, feature_names: tuple[str, ...]) -> dict[str, Any]:
    from openrlhf.tools.therapeutic_tools import v16
    from openrlhf.utils.tool_calling_turn import _exec_with_rdkit_log_capture

    started = time.time()
    error_str, result_json = _exec_with_rdkit_log_capture(
        v16.get_features,
        {"smiles": smiles, "feature_names": list(feature_names)},
        "get_features",
    )
    failure = _classify_failure(error_str, result_json)
    return {
        "smiles": smiles,
        "ok": failure is None,
        "failure": failure,
        "elapsed_s": time.time() - started,
    }


def _run_get_neighbors(row: dict[str, Any]) -> dict[str, Any]:
    from openrlhf.tools.therapeutic_tools import v16
    from openrlhf.utils.tool_calling_turn import _exec_with_rdkit_log_capture

    smiles = row["smiles"]
    task = row["task"]
    started = time.time()
    error_str, result_json = _exec_with_rdkit_log_capture(
        v16.get_neighbors,
        {"smiles": smiles, "task_name": task, "feature_names": v16.FEATURE_NAMES},
        "get_neighbors",
    )
    failure = _classify_failure(error_str, result_json)
    return {
        "smiles": smiles,
        "task": task,
        "ok": failure is None,
        "failure": failure,
        "elapsed_s": time.time() - started,
    }


def _summarize_results(
    *,
    tool: str,
    feature_names: list[str] | None,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    failures = [r for r in results if not r["ok"]]
    failure_counts = Counter(r["failure"] for r in failures)
    task_failure_counts = Counter(r.get("task") for r in failures if r.get("task"))

    summary = {
        "tool": tool,
        "feature_names": feature_names,
        "dataset_rows": len(rows),
        "unique_smiles": len({row["smiles"] for row in rows}),
        "audited_items": len(results),
        "success_count": len(results) - len(failures),
        "failure_count": len(failures),
        "success_rate": ((len(results) - len(failures)) / len(results)) if results else None,
        "wall_time_s": time.time() - started,
        "avg_elapsed_s": (sum(r["elapsed_s"] for r in results) / len(results)) if results else None,
        "p95_elapsed_s": None,
        "failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in failure_counts.most_common(20)
        ],
        "failure_tasks": [
            {"task": task, "count": count}
            for task, count in task_failure_counts.most_common()
        ],
        "failure_examples": failures[:25],
    }

    if results:
        elapsed = sorted(r["elapsed_s"] for r in results)
        summary["p95_elapsed_s"] = elapsed[min(len(elapsed) - 1, int(0.95 * len(elapsed)))]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="data/tdc/openai_format_v16",
        help="Directory containing the v16 JSONL dataset shards.",
    )
    parser.add_argument(
        "--tool",
        choices=("get_features", "get_neighbors"),
        default="get_features",
        help="Which v16 tool to audit.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional feature_names override for get_features/get_neighbors. Defaults to all v16 features.",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for quick spot checks.")
    parser.add_argument(
        "--output",
        default="runs/eval/v16_tool_audit_summary.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    rows = _load_v16_rows(args.dataset_dir)
    from openrlhf.tools.therapeutic_tools import v16

    feature_names = tuple(args.features or v16.FEATURE_NAMES)
    if args.tool == "get_features":
        unique_smiles = sorted({row["smiles"] for row in rows})
        items: list[Any] = unique_smiles
        fn = partial(_run_get_features, feature_names=feature_names)
    else:
        items = rows
        fn = _run_get_neighbors

    if args.limit > 0:
        items = items[: args.limit]

    started = time.time()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if idx == total or idx % 250 == 0:
                print(f"[{idx}/{total}] completed", flush=True)

    summary = _summarize_results(
        tool=args.tool,
        feature_names=list(feature_names),
        rows=rows,
        results=results,
        started=started,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
