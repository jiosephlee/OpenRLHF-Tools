#!/usr/bin/env python3
"""Rewrite v13-filtered SFT traces as tool-calling traces.

Input : data/sft_traces/traces_v13/<TASK>.jsonl  (output of filter_traces.py)
Output: data/sft_traces/traces_v13_formatted/<TASK>.jsonl

Each input trace preserves the original trace shape from `data/sft_traces/traces`
and simply prepends tool-calling turns before the existing assistant reasoning:

    [system]   original system message
    [user]     original user message
    [assistant tool_calls=[
        get_features(smiles, feature_names=[<exact set cited downstream>]),
        # optional, only when narrative cites neighbors:
        get_neighbors_<task_alias>(smiles, num_neighbors=3),
    ]]
    [tool name=get_features]   <rendered v13 output>
    [tool name=get_neighbors_<task_alias>]
                               <minimal block: SMILES + similarity + label only>
    [assistant]   original narrative unchanged

The get_features tool content is rendered live via v13.get_features. The
get_neighbors content is rendered minimally from narrative-embedded data
to match the smaller SMILES/similarity/label footprint used in the traces.
"""

import argparse
import hashlib
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Make `openrlhf` importable when running this script directly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from openrlhf.tools.therapeutic_tools import v13  # noqa: E402
from openrlhf.tools.therapeutic_tools.similarity import TASK_ALIASES  # noqa: E402

DEFAULT_INPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "traces_v13")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "traces_v13_formatted")
DESCRIPTOR_PATTERN = re.compile(r"Looking at ([A-Za-z_][A-Za-z0-9_]*)\s*=")
NEIGHBOR_LINE_PATTERN = re.compile(
    r"Looking at the similar molecules:\s*(.+?)\s*Labels:\s*([^.]+)\.",
    re.DOTALL,
)
NEIGHBOR_ENTRY_PATTERN = re.compile(
    r"(\S+)\s*\(similarity:\s*([0-9.]+)\)"
)


def task_alias(task: str) -> str:
    return TASK_ALIASES.get(task, task.lower())


def call_id(seed: str) -> str:
    return "call_" + hashlib.sha1(seed.encode()).hexdigest()[:12]


def cited_descriptors_in_order(text: str) -> list[str]:
    """Return cited descriptors in order of first appearance, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for name in DESCRIPTOR_PATTERN.findall(text):
        # KNN-derived terms come from get_neighbors, not get_features
        if name in {"KNN_mean_label", "KNN_min_dist", "KNN_mean_dist"}:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_neighbor_block(text: str) -> list[dict] | None:
    """Parse the 'Looking at the similar molecules: ... Labels: ...' line.

    Returns a list of {smiles, similarity, label} or None if absent.
    """
    m = NEIGHBOR_LINE_PATTERN.search(text)
    if not m:
        return None
    entries_text = m.group(1)
    labels_text = m.group(2)

    entries = NEIGHBOR_ENTRY_PATTERN.findall(entries_text)
    labels = [s.strip() for s in labels_text.split(",")]
    if len(entries) != len(labels):
        # Fall back: pair only what we can
        n = min(len(entries), len(labels))
        entries = entries[:n]
        labels = labels[:n]

    return [
        {
            "smiles": smi,
            "similarity": float(sim),
            "label": lab,
        }
        for (smi, sim), lab in zip(entries, labels)
    ]


def render_neighbors_minimal(neighbors: list[dict], task: str) -> str:
    """Minimal SMILES/sim/label rendering — matches the narrative's smaller setup."""
    lines = [f"Nearest Neighbors for task '{task}' (k={len(neighbors)}):"]
    for i, nbr in enumerate(neighbors, 1):
        lines.append(
            f"\n{i}. {nbr['smiles']} (similarity: {nbr['similarity']:.4f}, label: {nbr['label']})"
        )
    return "\n".join(lines)


def rewrite_trace(rec: dict) -> dict | None:
    """Transform one trace dict into the v13 multi-turn tool-calling format."""
    msgs = rec["messages"]
    meta = rec.get("metadata", {})
    task = meta.get("task")
    smiles = meta.get("smiles") or ""
    if not task or not smiles:
        return None

    user_msg = next((m for m in msgs if m["role"] == "user"), None)
    assistant_msg = next((m for m in msgs if m["role"] == "assistant"), None)
    if user_msg is None or assistant_msg is None:
        return None

    narrative = assistant_msg["content"]
    feature_names = cited_descriptors_in_order(narrative)
    neighbors = parse_neighbor_block(narrative)

    # Build tool calls
    tool_calls: list[dict] = []
    tool_results: list[dict] = []

    if feature_names:
        tc_id = call_id(f"{task}|{smiles}|get_features")
        tool_calls.append({
            "id": tc_id,
            "type": "function",
            "function": {
                "name": "get_features",
                "arguments": json.dumps(
                    {"smiles": smiles, "feature_names": feature_names},
                    ensure_ascii=False,
                ),
            },
        })
        try:
            content = v13.get_features(smiles, feature_names)
        except Exception as e:
            content = f"Error: {e}"
        tool_results.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": "get_features",
            "content": content,
        })

    if neighbors:
        alias = task_alias(task)
        nbr_tool = f"get_neighbors_{alias}"
        tc_id = call_id(f"{task}|{smiles}|{nbr_tool}")
        tool_calls.append({
            "id": tc_id,
            "type": "function",
            "function": {
                "name": nbr_tool,
                "arguments": json.dumps(
                    {"smiles": smiles, "num_neighbors": len(neighbors)},
                    ensure_ascii=False,
                ),
            },
        })
        tool_results.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": nbr_tool,
            "content": render_neighbors_minimal(neighbors, task),
        })

    if not tool_calls:
        # Nothing to call — drop (shouldn't happen for filtered traces).
        return None

    system_msg = next((m for m in msgs if m["role"] == "system"), None)
    if system_msg is None:
        return None

    new_messages: list[dict] = [
        {"role": "system", "content": system_msg["content"]},
        {"role": "user", "content": user_msg["content"]},
        {"role": "assistant", "tool_calls": tool_calls},
        *tool_results,
        {"role": "assistant", "content": narrative},
    ]

    return {
        "messages": new_messages,
        "metadata": meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith(".jsonl") and f != "all_tasks_combined.jsonl"
    )

    total_in = 0
    total_out = 0
    combined_lines: list[str] = []

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        n_in = 0
        n_out = 0
        with open(in_path) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                n_in += 1
                rec = json.loads(line)
                rewritten = rewrite_trace(rec)
                if rewritten is None:
                    continue
                line_out = json.dumps(rewritten, ensure_ascii=False) + "\n"
                f_out.write(line_out)
                combined_lines.append(line_out)
                n_out += 1

        total_in += n_in
        total_out += n_out
        pct = (100.0 * n_out / n_in) if n_in else 0
        print(f"  {fname:40s}  {n_out:6d}/{n_in:6d}  ({pct:5.1f}%)")

    combined_path = os.path.join(args.output_dir, "all_tasks_combined.jsonl")
    with open(combined_path, "w") as f:
        f.writelines(combined_lines)

    print()
    print(f"TOTAL: {total_out}/{total_in}")
    print(f"Wrote: {args.output_dir}")
    print(f"Combined: {combined_path}")


if __name__ == "__main__":
    main()
