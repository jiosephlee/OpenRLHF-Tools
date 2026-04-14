#!/usr/bin/env python3
"""Validate v13-formatted traces against v13 tool schemas and gpt-oss rendering.

Checks:
  - assistant/tool turn structure is well-formed
  - tool call names match the task-specific v13 tool list
  - tool arguments are valid JSON and include `smiles`
  - every formatted record renders successfully via
    AutoTokenizer.from_pretrained("openai/gpt-oss-20b").apply_chat_template(
        messages, tools=task_tools
      )
  - rendered prompt includes the tool namespace and referenced tool names
"""

import argparse
import json
import os
from collections import Counter

from transformers import AutoTokenizer


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

DEFAULT_TRACE_DIR = os.path.join(_SCRIPT_DIR, "..", "traces_v13_formatted")
DEFAULT_TOOL_MAP = os.path.join(_REPO_ROOT, "data", "tdc", "metadata", "tools_per_task_v13.json")
DEFAULT_MODEL = "openai/gpt-oss-20b"


def validate_record(rec: dict, task_tools: list[dict], tokenizer) -> list[str]:
    errors: list[str] = []
    messages = rec.get("messages", [])
    if len(messages) < 4:
        return ["too_few_messages"]

    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        errors.append("bad_prefix_roles")

    assistant_tool_turn = messages[2]
    if assistant_tool_turn.get("role") != "assistant" or "tool_calls" not in assistant_tool_turn:
        return errors + ["missing_assistant_tool_call_turn"]

    tool_calls = assistant_tool_turn["tool_calls"]
    allowed_names = {tool["function"]["name"] for tool in task_tools}
    call_ids: set[str] = set()

    for tool_call in tool_calls:
        name = tool_call["function"]["name"]
        if name not in allowed_names:
            errors.append(f"unknown_tool:{name}")
        call_ids.add(tool_call["id"])
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            errors.append(f"bad_args_json:{name}")
            continue
        if "smiles" not in arguments:
            errors.append(f"missing_smiles:{name}")

    for tool_msg in messages[3:-1]:
        if tool_msg.get("role") != "tool":
            errors.append("non_tool_between_call_and_final")
            continue
        if tool_msg.get("tool_call_id") not in call_ids:
            errors.append(f"orphan_tool_result:{tool_msg.get('name')}")
        if tool_msg.get("name") not in allowed_names:
            errors.append(f"bad_tool_result_name:{tool_msg.get('name')}")

    if messages[-1].get("role") != "assistant" or "content" not in messages[-1]:
        errors.append("missing_final_assistant")
        return errors

    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=task_tools,
            add_generation_prompt=False,
        )
    except Exception as exc:
        return errors + [f"render_error:{type(exc).__name__}:{exc}"]

    if "# Tools" not in rendered or "namespace functions" not in rendered:
        errors.append("tools_not_rendered")

    for tool_call in tool_calls:
        if tool_call["function"]["name"] not in rendered:
            errors.append(f"tool_name_missing_in_render:{tool_call['function']['name']}")

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--tool-map", default=DEFAULT_TOOL_MAP)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    with open(args.tool_map) as f:
        tool_map = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stats = Counter()
    failures: list[tuple[str, int, list[str]]] = []

    for fname in sorted(os.listdir(args.trace_dir)):
        if not fname.endswith(".jsonl") or fname == "all_tasks_combined.jsonl":
            continue
        path = os.path.join(args.trace_dir, fname)
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                rec = json.loads(line)
                task = rec["metadata"]["task"]
                task_tools = tool_map[task]
                errs = validate_record(rec, task_tools, tokenizer)
                if errs:
                    failures.append((fname, line_no, errs))
                else:
                    stats["validated_records"] += 1
                    stats[f"task::{task}"] += 1
                    stats[f"tool_calls::{len(rec['messages'][2]['tool_calls'])}"] += 1

    print(f"validated_records: {stats['validated_records']}")
    print("tool_call_cardinality:")
    for key, value in sorted(stats.items()):
        if key.startswith("tool_calls::"):
            print(f"  {key.split('::', 1)[1]} -> {value}")

    if failures:
        print(f"failures: {len(failures)}")
        for fname, line_no, errs in failures[:50]:
            print(f"  {fname}:{line_no}: {', '.join(errs)}")
        raise SystemExit(1)

    print("failures: 0")


if __name__ == "__main__":
    main()
