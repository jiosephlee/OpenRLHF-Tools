#!/usr/bin/env python3
"""Convert saved eval distill traces into SFT-ready messages datasets.

The input traces come from ``--save_sft_distill_traces`` and are expected to
contain:
  - ``task`` / ``smiles`` metadata for lookup
  - ``trace_messages`` reconstructed from the generated rollout

This script swaps the original local-attribution prompt out for the regular
``v16_no_neighbor`` prompt by looking up the matching base record via
``(task, smiles)``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from collections import defaultdict
from pathlib import Path


_GLM_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_GLM_TOOL_RESPONSE_BLOCK_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_GLM_FUNC_DETAIL_RE = re.compile(
    r"<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>",
    re.DOTALL,
)
_GLM_FUNC_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)


def _coerce_arg_value(raw: str):
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return raw
    if s[0] in "[{" or s in ("true", "false", "null"):
        try:
            return json.loads(s)
        except (ValueError, json.JSONDecodeError):
            return raw
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-input",
        required=True,
        help="Path to a distill trace JSONL file or a directory containing JSONL files.",
    )
    parser.add_argument(
        "--base-data-dir",
        default="data/tdc/openai_format_v16_no_neighbor",
        help="Directory containing the regular v16_no_neighbor OpenAI-format datasets.",
    )
    parser.add_argument(
        "--base-split",
        default="train",
        help="Split to read from the base prompt dataset (default: train).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where converted per-task JSONL files will be written.",
    )
    parser.add_argument(
        "--combined-name",
        default="all_tasks_combined.jsonl",
        help="Filename for the combined output JSONL.",
    )
    return parser.parse_args()


def iter_trace_files(trace_input: Path) -> list[Path]:
    if trace_input.is_file():
        return [trace_input]
    if not trace_input.is_dir():
        raise FileNotFoundError(f"Trace input does not exist: {trace_input}")
    return sorted(p for p in trace_input.glob("*.jsonl") if p.is_file())


def load_base_prompt_map(base_data_dir: Path, split: str) -> dict[tuple[str, str], dict]:
    payload: dict[tuple[str, str], dict] = {}
    suffix = f"_{split}.jsonl"
    for path in sorted(base_data_dir.glob(f"*{suffix}")):
        if path.name.startswith("eval_tdc"):
            continue
        task = path.name[: -len(suffix)]
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                smiles = str(record.get("smiles", ""))
                if not smiles:
                    continue
                payload[(task, smiles)] = record
    if not payload:
        raise FileNotFoundError(f"No base records found under {base_data_dir} for split={split!r}")
    return payload


DEFAULT_FEATURE_NAMES = [
    "molecular_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
]


def _parse_raw_get_features_payload(raw: str) -> dict | None:
    if not isinstance(raw, str):
        return None

    smiles_match = re.search(r'"smiles"\s*:\s*"([^"]*)"', raw, flags=re.S)
    feature_match = re.search(r'"feature_names"\s*:\s*(\[[^\]]*\])', raw, flags=re.S)
    if not smiles_match:
        return None

    smiles = smiles_match.group(1)
    if ".replace" in raw:
        smiles = re.sub(r"\s+", "", smiles)

    feature_names = DEFAULT_FEATURE_NAMES
    if feature_match:
        try:
            parsed = ast.literal_eval(feature_match.group(1))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                feature_names = parsed
        except (SyntaxError, ValueError):
            pass

    return {
        "smiles": smiles,
        "feature_names": feature_names,
    }


def _normalize_get_features_args(arguments: dict) -> dict:
    if not isinstance(arguments, dict):
        return arguments

    if "smiles" in arguments and "feature_names" in arguments:
        return {
            "smiles": arguments["smiles"],
            "feature_names": arguments["feature_names"],
        }

    if "raw" in arguments:
        parsed = _parse_raw_get_features_payload(arguments["raw"])
        if parsed is not None:
            return parsed

    if "smiles" in arguments:
        return {
            "smiles": arguments["smiles"],
            "feature_names": DEFAULT_FEATURE_NAMES,
        }

    return arguments


def _sanitize_trace_messages(trace_messages: list[dict]) -> list[dict]:
    cleaned_messages: list[dict] = []

    for message in copy.deepcopy(trace_messages):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if not isinstance(function, dict):
                    continue
                if function.get("name") != "get_features":
                    continue
                function["arguments"] = _normalize_get_features_args(function.get("arguments", {}))

        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            try:
                payload = json.loads(message["content"])
            except json.JSONDecodeError:
                cleaned_messages.append(message)
                continue

            if isinstance(payload, dict) and payload.get("function_name") == "get_features":
                payload["arguments"] = _normalize_get_features_args(payload.get("arguments", {}))
                message["content"] = json.dumps(payload, ensure_ascii=False)

        cleaned_messages.append(message)

    return cleaned_messages


def _parse_glm_tool_call_block(block: str) -> list[dict]:
    tool_call_match = _GLM_FUNC_DETAIL_RE.search(block)
    if not tool_call_match:
        return []

    function_name = tool_call_match.group(1).strip()
    arg_section = tool_call_match.group(2) or ""
    arguments = {}
    for key, value in _GLM_FUNC_ARG_RE.findall(arg_section):
        arguments[key.strip()] = _coerce_arg_value(value.strip())

    if function_name == "get_features":
        arguments = _normalize_get_features_args(arguments)

    return [{"name": function_name, "arguments": arguments}]


def _clean_glm_assistant_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("<|assistant|>", "")
    cleaned = cleaned.replace("<|observation|>", "")
    cleaned = cleaned.replace("<think>", "")
    cleaned = cleaned.replace("</think>", "")
    return cleaned.strip()


def _normalize_tool_response_content(content: str) -> str:
    if not isinstance(content, str):
        return content

    stripped = content.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    if isinstance(payload, dict) and payload.get("function_name") == "get_features":
        payload["arguments"] = _normalize_get_features_args(payload.get("arguments", {}))
        return json.dumps(payload, ensure_ascii=False)
    return stripped


def _reconstruct_glm_trace_messages_from_response(response_text: str) -> list[dict] | None:
    """Rebuild GLM XML traces into OpenAI-style assistant/tool/assistant turns."""
    if not isinstance(response_text, str) or "<tool_call>" not in response_text or "<tool_response>" not in response_text:
        return None

    remaining = response_text
    rebuilt_messages: list[dict] = []
    tool_turn_idx = 1

    while True:
        tool_call_match = _GLM_TOOL_CALL_BLOCK_RE.search(remaining)
        if not tool_call_match:
            final_text = _clean_glm_assistant_text(remaining)
            if final_text:
                rebuilt_messages.append({"role": "assistant", "content": final_text})
            break

        assistant_prefix = _clean_glm_assistant_text(remaining[: tool_call_match.start()])
        tool_calls = _parse_glm_tool_call_block(tool_call_match.group(0))
        if not tool_calls:
            return None

        after_tool_call = remaining[tool_call_match.end() :]
        tool_response_match = _GLM_TOOL_RESPONSE_BLOCK_RE.search(after_tool_call)
        if not tool_response_match:
            return None

        assistant_message = {
            "role": "assistant",
            "tool_calls": [],
        }
        if assistant_prefix:
            assistant_message["thinking"] = assistant_prefix

        for call_idx, tool_call in enumerate(tool_calls, start=1):
            tool_call_id = f"call_s0_t{tool_turn_idx}_{call_idx}"
            assistant_message["tool_calls"].append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
            )
        rebuilt_messages.append(assistant_message)

        # GLM eval traces currently emit a single <tool_response> payload per <tool_call>.
        tool_content = _normalize_tool_response_content(tool_response_match.group(1))
        rebuilt_messages.append(
            {
                "role": "tool",
                "tool_call_id": assistant_message["tool_calls"][0]["id"],
                "name": assistant_message["tool_calls"][0]["function"]["name"],
                "content": tool_content,
            }
        )

        remaining = after_tool_call[tool_response_match.end() :]
        tool_turn_idx += 1

    return rebuilt_messages


def build_output_record(trace_row: dict, base_record: dict, trace_file: Path) -> dict:
    trace_messages = trace_row.get("trace_messages") or []
    if trace_messages:
        trace_messages = _sanitize_trace_messages(trace_messages)

    has_structured_tool_trace = any(
        (msg.get("tool_calls") or msg.get("role") == "tool")
        for msg in trace_messages
        if isinstance(msg, dict)
    )
    if not has_structured_tool_trace:
        reconstructed = _reconstruct_glm_trace_messages_from_response(trace_row.get("response", ""))
        if reconstructed:
            trace_messages = reconstructed

    if not trace_messages:
        response = (trace_row.get("response") or "").strip()
        if response:
            trace_messages = [{"role": "assistant", "content": response}]
        else:
            raise ValueError(
                f"Trace row from {trace_file} is missing both trace_messages and response text "
                f"for task={trace_row.get('task')} smiles={trace_row.get('smiles')}"
            )

    return {
        "messages": copy.deepcopy(base_record["messages"]) + copy.deepcopy(trace_messages),
        "answer": base_record.get("answer", trace_row.get("source_answer")),
        "task": trace_row["task"],
        "smiles": trace_row["smiles"],
        "label": base_record.get("label", trace_row.get("source_label")),
        "metadata": {
            "datasource": trace_row.get("datasource"),
            "source_trace_file": str(trace_file),
            "sample_idx": trace_row.get("sample_idx"),
            "score": trace_row.get("score"),
            "reward": trace_row.get("reward"),
            "pred": trace_row.get("pred"),
            "source_prompt_swapped_to": "v16_no_neighbor",
        },
    }


def main() -> None:
    args = parse_args()

    trace_input = Path(args.trace_input)
    base_data_dir = Path(args.base_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_prompt_map = load_base_prompt_map(base_data_dir, args.base_split)
    per_task_rows: dict[str, list[str]] = defaultdict(list)
    combined_rows: list[str] = []
    total_rows = 0

    for trace_file in iter_trace_files(trace_input):
        with trace_file.open() as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                trace_row = json.loads(line)
                task = trace_row.get("task") or trace_row.get("datasource")
                smiles = str(trace_row.get("smiles", ""))
                if not task or not smiles:
                    raise KeyError(
                        f"Missing task/smiles in {trace_file}:{line_num}. "
                        "Run the patched saver before converting traces."
                    )

                base_record = base_prompt_map.get((task, smiles))
                if base_record is None:
                    raise KeyError(
                        f"No base v16_no_neighbor prompt found for task={task!r} smiles={smiles!r} "
                        f"while processing {trace_file}:{line_num}"
                    )

                trace_row["task"] = task
                trace_row["smiles"] = smiles
                output_record = build_output_record(trace_row, base_record, trace_file)
                output_line = json.dumps(output_record, ensure_ascii=False) + "\n"
                per_task_rows[task].append(output_line)
                combined_rows.append(output_line)
                total_rows += 1

    for task, rows in sorted(per_task_rows.items()):
        out_path = output_dir / f"{task}.jsonl"
        with out_path.open("w") as f:
            f.writelines(rows)

    combined_path = output_dir / args.combined_name
    with combined_path.open("w") as f:
        f.writelines(combined_rows)

    print(f"Converted {total_rows} traces")
    print(f"Wrote per-task outputs to: {output_dir}")
    print(f"Wrote combined output to: {combined_path}")


if __name__ == "__main__":
    main()
