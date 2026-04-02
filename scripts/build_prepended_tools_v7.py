#!/usr/bin/env python3
"""
Build prepended_tools_v7 by upgrading prepended_tools_v6:
  1. Re-run assess_adme_properties and predict_metabolites (updated implementations)
  2. Add find_similar_molecules_{task} output (new in v7)
  3. Replace CoT instruction with v7 SAR-aware version (no tool-calling language)

Input:  data/tdc/prepended_tools_v6/{TASK}_{split}.jsonl
Output: data/tdc/prepended_tools_v7/{TASK}_{split}.jsonl

Usage:
    python scripts/build_prepended_tools_v7.py
    python scripts/build_prepended_tools_v7.py --tasks hERG AMES
"""

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TASK_NAMES = [
    "Bioavailability_Ma", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "BBB_Martins", "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Carcinogens_Lagunin", "hERG", "ClinTox", "DILI", "Skin_Reaction", "AMES",
]

# Tools to re-run (updated implementations)
TOOLS_TO_RERUN = {"assess_adme_properties", "predict_metabolites"}

# Load new CoT instruction
COT_INSTRUCTION_PATH = PROJECT_ROOT / "data" / "tdc" / "metadata" / "cot_instruction_v7_prepended.txt"


def load_cot_instruction() -> str:
    return COT_INSTRUCTION_PATH.read_text().strip()


def parse_tool_sections(text: str):
    """Parse prepended tool output text into (prefix, tool_sections, suffix).

    Returns:
        prefix: everything before "--- Precomputed Tool Outputs ---"
        tool_sections: list of (tool_name, section_text) tuples
        suffix: the CoT instruction at the end
    """
    marker = "--- Precomputed Tool Outputs ---"
    marker_idx = text.find(marker)
    if marker_idx < 0:
        return text, [], ""

    prefix = text[:marker_idx + len(marker)]
    rest = text[marker_idx + len(marker):]

    # Split into tool sections by [tool_name] headers
    # Pattern: \n\n[tool_name]\n
    parts = re.split(r'\n\n\[([a-zA-Z_][a-zA-Z0-9_]*)\]\n', rest)

    # parts[0] is text before first tool (usually just whitespace)
    # Then alternating: tool_name, tool_content, tool_name, tool_content, ...
    tool_sections = []
    i = 1
    while i < len(parts) - 1:
        tool_name = parts[i]
        tool_content = parts[i + 1]
        tool_sections.append((tool_name, tool_content))
        i += 2

    # The suffix (CoT instruction) is embedded in the last tool section's content
    # Find the last "Please analyze..." or "Please think..." instruction
    if tool_sections:
        last_name, last_content = tool_sections[-1]
        # The instruction starts after the last double newline before "Please"
        # Look for the instruction pattern
        instruction_patterns = [
            "\nPlease analyze the molecular structure",
            "\nPlease think step by step",
        ]
        split_idx = -1
        for pat in instruction_patterns:
            idx = last_content.rfind(pat)
            if idx >= 0:
                split_idx = idx
                break

        if split_idx >= 0:
            suffix = last_content[split_idx:].strip()
            tool_sections[-1] = (last_name, last_content[:split_idx])
        else:
            suffix = ""
    else:
        suffix = ""

    return prefix, tool_sections, suffix


def reassemble(prefix: str, tool_sections: list, cot_instruction: str) -> str:
    """Reassemble the full text from parsed components."""
    parts = [prefix]
    for tool_name, content in tool_sections:
        parts.append(f"\n\n[{tool_name}]\n{content}")
    parts.append(f"\n\n{cot_instruction}")
    return "".join(parts)


def process_record(record: dict, function_map: dict, cot_instruction: str) -> dict:
    """Upgrade a single v6 prepended record to v7."""
    text = record["text"]
    task = record["task"]
    smiles = record["smiles"]

    prefix, tool_sections, _old_suffix = parse_tool_sections(text)

    # Re-run updated tools
    new_sections = []
    for tool_name, content in tool_sections:
        if tool_name in TOOLS_TO_RERUN:
            fn = function_map.get(tool_name)
            if fn is not None:
                try:
                    if tool_name == "assess_adme_properties":
                        result = fn(smiles=smiles, ph=7.4)
                    else:
                        result = fn(smiles=smiles)
                    new_sections.append((tool_name, result))
                    continue
                except Exception:
                    pass  # Fall through to keep original
        new_sections.append((tool_name, content))

    # Add find_similar_molecules_{task}
    sim_fn_name = f"find_similar_molecules_{task}"
    sim_fn = function_map.get("find_similar_molecules")
    if sim_fn is not None:
        try:
            sim_result = sim_fn(smiles=smiles, task=task, embedding_type="fingerprint")
            new_sections.append((sim_fn_name, sim_result))
        except Exception as e:
            new_sections.append((sim_fn_name, f"Error: {e}"))

    new_text = reassemble(prefix, new_sections, cot_instruction)

    return {
        "text": new_text,
        "answer": record["answer"],
        "task": task,
        "smiles": smiles,
        "label": record["label"],
    }


def process_split(src_path: Path, dst_path: Path, function_map: dict, cot_instruction: str) -> int:
    records = []
    with open(src_path) as f:
        for line in f:
            records.append(json.loads(line))

    count = 0
    errors = 0
    with open(dst_path, "w") as fout:
        for i, record in enumerate(records):
            try:
                upgraded = process_record(record, function_map, cot_instruction)
                fout.write(json.dumps(upgraded, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                traceback.print_exc()
                errors += 1

            if (i + 1) % 100 == 0:
                print(f"    {src_path.name}: {i+1}/{len(records)} processed", flush=True)

    if errors:
        print(f"    WARNING: {errors} errors in {src_path.name}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Build prepended_tools_v7 from v6")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific tasks (default: all)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process")
    parser.add_argument("--src-dir", default=str(PROJECT_ROOT / "data" / "tdc" / "prepended_tools_v6"))
    parser.add_argument("--dst-dir", default=str(PROJECT_ROOT / "data" / "tdc" / "prepended_tools_v7"))
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASK_NAMES
    cot_instruction = load_cot_instruction()

    # Import therapeutic tools
    from openrlhf.tools.therapeutic_tools import _FUNCTION_MAP

    total = 0
    for task in tasks:
        print(f"\n  Processing {task}...")
        for split in args.splits:
            src_path = src_dir / f"{task}_{split}.jsonl"
            dst_path = dst_dir / f"{task}_{split}.jsonl"

            if not src_path.exists():
                print(f"  [SKIP]  {src_path.name} — not found")
                continue

            count = process_split(src_path, dst_path, _FUNCTION_MAP, cot_instruction)
            total += count
            print(f"  [OK]    {dst_path.name}: {count} samples")

    print(f"\nTotal: {total} samples written to {dst_dir}")


if __name__ == "__main__":
    main()
