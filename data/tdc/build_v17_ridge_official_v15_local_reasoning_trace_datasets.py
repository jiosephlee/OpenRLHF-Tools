"""Build direct SFT datasets from v17 ridge official_v15 local reasoning-trace artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tdc.ml_prompt_artifacts import load_sample_prompt_map  # noqa: E402
from data.tdc.local_reasoning_trace_artifacts_common import (  # noqa: E402
    V16_CONSOLIDATED_FEATURES,
    map_feature_to_v16_group,
)
from data.tdc.v17_ridge_official_v15_common import (  # noqa: E402
    DEFAULT_RAW_DIR,
    DEFAULT_REASONING_TRACE_OUTPUT_DIR,
    TASKS,
    TOOL_VERSION,
)
from openrlhf.tools.therapeutic_tools.v17 import get_features as v17_get_features  # noqa: E402

ARTIFACT_VARIANT = "local_reasoning_trace"
CHAT_TEMPLATE_DIR = Path("/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/data/openai_format_v15")
ANSWER_SUFFIX_RE = re.compile(r"\s*Answer:\s*(\([AB]\))\s*$", re.IGNORECASE)


def load_chat_template_map(task: str, split: str) -> dict[str, dict]:
    path = CHAT_TEMPLATE_DIR / f"{task}_{split}.jsonl"
    if not path.exists():
        return {}

    payload: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles = str(row["smiles"])
            payload[smiles] = row
    return payload


def derive_feature_names(prompt_row: dict) -> list[str]:
    selected_features = prompt_row.get("selected_features", [])
    seen: set[str] = set()
    feature_names: list[str] = []
    for item in selected_features:
        group = map_feature_to_v16_group(str(item["feature"]))
        if group in seen:
            continue
        seen.add(group)
        feature_names.append(group)
    ordered = [group for group in V16_CONSOLIDATED_FEATURES if group in seen]
    return ordered or list(V16_CONSOLIDATED_FEATURES)


def build_tool_call_id(task: str, split: str, smiles: str) -> str:
    digest = hashlib.sha1(f"{task}|{split}|{smiles}".encode("utf-8")).hexdigest()[:12]
    return f"call_{digest}"


def build_tool_messages(task: str, split: str, smiles: str, prompt_row: dict) -> tuple[list[dict], dict]:
    feature_names = derive_feature_names(prompt_row)
    tool_call_id = build_tool_call_id(task, split, smiles)
    arguments = {"smiles": smiles, "feature_names": feature_names}
    tool_output = v17_get_features(smiles, feature_names)
    return [
        {
            "role": "assistant",
            "content": "",
            "thinking": "Let's call the tool.",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "get_features",
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "get_features",
            "content": tool_output,
        },
    ], {
        "tool_call_id": tool_call_id,
        "feature_names": feature_names,
        "tool_name": "get_features",
    }


def split_reasoning_trace_and_answer(reasoning_trace: str) -> tuple[str, Optional[str]]:
    trace = reasoning_trace.rstrip()
    match = ANSWER_SUFFIX_RE.search(trace)
    if match is None:
        return trace, None
    trimmed = trace[:match.start()].rstrip()
    return trimmed, match.group(1).upper()


def default_output_dir_for_variant(artifact_variant: str) -> Path:
    if artifact_variant == ARTIFACT_VARIANT:
        return DEFAULT_REASONING_TRACE_OUTPUT_DIR
    return DEFAULT_REASONING_TRACE_OUTPUT_DIR.parent / f"openai_format_{TOOL_VERSION}_{artifact_variant}"


def build_dataset(
    task: str,
    split: str,
    output_dir: str,
    raw_dir: str,
    *,
    artifact_variant: str,
    include_disagreements: bool,
) -> tuple[int, int]:
    raw_path = os.path.join(raw_dir, task, f"{split}.csv")
    if not os.path.exists(raw_path):
        return 0, 0

    df = pd.read_csv(raw_path)
    if "Drug" not in df.columns or "Y" not in df.columns:
        print(f"  Warning: {raw_path} missing Drug/Y columns, skipping")
        return 0, 0

    chat_template_map = load_chat_template_map(task, split)
    if not chat_template_map:
        print(f"  Warning: No openai_format_v15 chat templates for {task}/{split}, skipping")
        return 0, 0

    prompt_map = load_sample_prompt_map(TOOL_VERSION, artifact_variant, task, split)
    if not prompt_map:
        raise FileNotFoundError(f"No stored reasoning-trace artifacts found for {task}/{split} under {TOOL_VERSION}/{artifact_variant}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{task}_{split}.jsonl")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        label = int(row["Y"])
        chat_row = chat_template_map.get(smiles)
        if chat_row is None:
            raise KeyError(f"Missing openai_format_v15 chat template for {task}/{split} SMILES={smiles}")
        prompt_row = prompt_map.get(smiles)
        if prompt_row is None:
            # Numeric-fidelity / per-task-cap filters in the trace builder
            # drop rows that don't meet quality thresholds. Skip them here.
            skipped += 1
            continue
        agreement = bool(prompt_row.get("auxiliary_model", {}).get("agrees_with_target", False))
        if not include_disagreements and not agreement:
            skipped += 1
            continue
        base_messages = chat_row.get("messages", [])
        if len(base_messages) < 2:
            raise ValueError(f"Expected system/user messages in openai_format_v15 for {task}/{split} SMILES={smiles}")
        tool_messages, tool_meta = build_tool_messages(task, split, smiles, prompt_row)
        answer_text = "(A)" if label == 0 else "(B)"
        reasoning_trace, trace_answer = split_reasoning_trace_and_answer(prompt_row["reasoning_trace"])
        if trace_answer is not None and trace_answer != answer_text:
            raise ValueError(
                f"Reasoning trace answer mismatch for {task}/{split} SMILES={smiles}: "
                f"trace has {trace_answer}, expected {answer_text}"
            )
        records.append(
            {
                "messages": list(base_messages[:2])
                + tool_messages
                + [
                    {
                        "role": "assistant",
                        "thinking": reasoning_trace,
                        "content": f"Answer: {answer_text}",
                    },
                ],
                "answer": answer_text,
                "smiles": smiles,
                "label": label,
                "task": task,
                "metadata": {
                    "artifact_variant": artifact_variant,
                    "auxiliary_model": prompt_row.get("auxiliary_model", {}),
                    "chat_template_source": str(CHAT_TEMPLATE_DIR),
                    "tool_version": "v17_get_features_only",
                    "tool_call": tool_meta,
                },
            }
        )

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v17 ridge official_v15 local reasoning-trace TDC datasets")
    parser.add_argument(
        "--artifact-variant",
        default=ARTIFACT_VARIANT,
        help=f"Prompt-artifact variant to convert. Default: {ARTIFACT_VARIANT}",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Input split directory")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks")
    parser.add_argument("--splits", nargs="*", default=["train", "val"], help="Splits to build")
    parser.add_argument(
        "--include-disagreements",
        action="store_true",
        help="Include examples where the auxiliary linear summary disagrees with the training label",
    )
    args = parser.parse_args()

    tasks = args.tasks or TASKS
    output_dir = args.output_dir or str(default_output_dir_for_variant(args.artifact_variant))

    total = 0
    total_skipped = 0
    for task in tasks:
        for split in args.splits:
            n, skipped = build_dataset(
                task,
                split,
                output_dir,
                args.raw_dir,
                artifact_variant=args.artifact_variant,
                include_disagreements=args.include_disagreements,
            )
            if n > 0 or skipped > 0:
                print(f"  {task}/{split}: {n} records ({skipped} skipped for aux/label disagreement)")
                total += n
                total_skipped += skipped

    print(f"\nTotal: {total} records")
    print(f"Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
