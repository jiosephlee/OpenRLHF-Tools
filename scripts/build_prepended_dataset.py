#!/usr/bin/env python3
"""
Build a prepended-tools dataset directly from raw_deduplicated CSVs.

Pipeline:
  1. Read molecules from data/tdc/raw_deduplicated/{TASK}/{split}.csv
  2. Format the prompt using prompts.json template
  3. Run all therapeutic tools specified in the tools-json schema
  4. Assemble: tool_preamble + prompt + precomputed_tool_outputs + cot_instruction
  5. Write to output directory as {TASK}_{split}.jsonl

Usage:
    python scripts/build_prepended_dataset.py --dst-dir data/tdc/prepended_tools_v7
    python scripts/build_prepended_dataset.py --tasks hERG AMES
    python scripts/build_prepended_dataset.py --embedding-type fingerprint
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TASK_NAMES = [
    "Bioavailability_Ma", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "BBB_Martins", "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "Carcinogens_Lagunin", "hERG", "ClinTox", "DILI", "Skin_Reaction", "AMES",
]

# Tasks where 3D conformational properties are informative
TASKS_WITH_3D = {
    "Bioavailability_Ma", "HIA_Hou", "PAMPA_NCATS", "Pgp_Broccatelli",
    "BBB_Martins", "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
}

TOOL_PREAMBLE_TEMPLATE = """\
You have access to the following tools to help analyze the molecule. Use them when necessary (**Don't use the same tool more than once**).

Available tools:
{tool_list}

"""


def format_tool_preamble(tools: list[dict]) -> str:
    """Convert a list of OpenAI-style tool dicts into a human-readable preamble."""
    lines = []
    for i, tool in enumerate(tools, 1):
        fn = tool["function"]
        name = fn["name"]
        description = fn["description"]
        params = fn.get("parameters", {}).get("properties", {})
        param_str = ", ".join(
            f"{p}: {v.get('type', 'any')}" for p, v in params.items()
        )
        lines.append(f"{i}. {name}({param_str})\n   {description}")
    return TOOL_PREAMBLE_TEMPLATE.format(tool_list="\n\n".join(lines))


def load_tools_schema(tools_json_path: Path, task_name: str) -> list[dict]:
    """Load the tool definitions for a given task (falls back to __default__)."""
    with open(tools_json_path) as f:
        mapping = json.load(f)
    return mapping.get(task_name, mapping["__default__"])


def get_tool_names_for_task(tools_schema: list[dict]) -> list[str]:
    """Extract tool function names from the OpenAI-style tool schema list."""
    return [t["function"]["name"] for t in tools_schema]


def run_tools_for_molecule(
    smiles: str, task: str, tool_names: list[str],
    function_map: dict, embedding_type: str,
) -> str:
    """Run the specified therapeutic tools on a molecule and format the results."""
    sections = []
    for tool_name in tool_names:
        # Map per-task find_similar_molecules_* to the generic function
        if tool_name.startswith("find_similar_molecules"):
            fn = function_map.get("find_similar_molecules")
            if fn is None:
                continue
            try:
                result = fn(smiles=smiles, task=task, k=5, embedding_type=embedding_type)
            except Exception as e:
                result = f"Error: {e}"
        else:
            fn = function_map.get(tool_name)
            if fn is None:
                continue
            try:
                if tool_name == "assess_adme_properties":
                    result = fn(smiles=smiles, ph=7.4)
                else:
                    result = fn(smiles=smiles)
            except Exception as e:
                result = f"Error: {e}"

        sections.append(f"[{tool_name}]\n{result}")

    return "\n\n".join(sections)


def build_prompt_text(
    smiles: str,
    prompt_template: str,
    tool_preamble: str,
    precomputed_outputs: str,
    cot_instruction: str,
) -> str:
    """Assemble the full prompt text for a sample."""
    prompt_with_smiles = prompt_template.replace("{Drug SMILES}", smiles)

    # Remove trailing "Answer:" since the CoT instruction adds its own
    if prompt_with_smiles.rstrip().endswith("Answer:"):
        prompt_with_smiles = prompt_with_smiles.rstrip()[: -len("Answer:")].rstrip()

    text = (
        f"{tool_preamble}"
        f"{prompt_with_smiles}\n\n"
        f"--- Precomputed Tool Outputs ---\n\n"
        f"{precomputed_outputs}\n\n"
        f"{cot_instruction}"
    )
    return text


def load_molecules(raw_dir: Path, task: str, split: str) -> pd.DataFrame:
    """Load molecules from raw_deduplicated CSVs."""
    path = raw_dir / task / f"{split}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Keep only valid rows
    df = df.dropna(subset=["Drug", "Y"])
    return df


def label_to_answer(label: int, task: str, prompts: dict) -> str:
    """Convert a binary label to (A)/(B) answer string.

    Convention from prompts.json: label 0 maps to (A), label 1 maps to (B).
    """
    return "(A)" if label == 0 else "(B)"


def process_split(
    raw_dir: Path, task: str, split: str, dst_path: Path,
    prompt_template: str, tool_preamble: str, tool_names: list[str],
    function_map: dict, cot_instruction: str, embedding_type: str,
) -> int:
    """Process a single split: load molecules, run tools, write output."""
    df = load_molecules(raw_dir, task, split)
    if df.empty:
        return 0

    count = 0
    errors = 0
    with open(dst_path, "w") as fout:
        for i, (_, row) in enumerate(df.iterrows()):
            smiles = str(row["Drug"])
            label = int(row["Y"])
            answer = label_to_answer(label, task, {})

            try:
                precomputed = run_tools_for_molecule(
                    smiles, task, tool_names, function_map, embedding_type,
                )
                text = build_prompt_text(
                    smiles, prompt_template, tool_preamble,
                    precomputed, cot_instruction,
                )
                record = {
                    "text": text,
                    "answer": answer,
                    "task": task,
                    "smiles": smiles,
                    "label": label,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                traceback.print_exc()
                errors += 1

            if (i + 1) % 100 == 0:
                print(f"    {task}_{split}: {i+1}/{len(df)} processed", flush=True)

    if errors:
        print(f"    WARNING: {errors} errors in {task}_{split}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build prepended-tools dataset from raw_deduplicated CSVs"
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "raw_deduplicated"),
    )
    parser.add_argument(
        "--dst-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "prepended_tools_v7"),
    )
    parser.add_argument(
        "--prompts",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "prompts.json"),
    )
    parser.add_argument(
        "--tools-json",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "tools_per_task_v7.json"),
    )
    parser.add_argument(
        "--cot-instruction",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "cot_instruction_v7_prepended_simplified.txt"),
    )
    parser.add_argument(
        "--embedding-type", default="fingerprint",
        choices=["fingerprint", "learned"],
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASK_NAMES

    # Load prompts
    with open(args.prompts) as f:
        prompts = json.load(f)

    # Load CoT instruction
    cot_instruction = Path(args.cot_instruction).read_text().strip()

    # Import therapeutic tools
    from openrlhf.tools.therapeutic_tools import _FUNCTION_MAP

    total = 0
    for task in tasks:
        # Resolve prompt template (case-insensitive lookup)
        prompt_key = task
        if task not in prompts:
            for k in prompts:
                if k.lower() == task.lower():
                    prompt_key = k
                    break
        prompt_template = prompts.get(prompt_key, "")
        if not prompt_template:
            print(f"  [WARN]  No prompt template for {task}, skipping")
            continue

        tools_schema = load_tools_schema(Path(args.tools_json), task)
        tool_names = get_tool_names_for_task(tools_schema)

        # Add get_3d_properties for eligible tasks if not already present
        if task in TASKS_WITH_3D and "get_3d_properties" not in tool_names:
            tool_names.append("get_3d_properties")
            from openrlhf.tools.therapeutic_tools.three_d import TOOL_SCHEMA as GET_3D_PROPERTIES_TOOL
            tools_schema = tools_schema + [GET_3D_PROPERTIES_TOOL]

        preamble = format_tool_preamble(tools_schema)

        print(f"\n  Processing {task} ({len(tool_names)} tools: {', '.join(tool_names)})...")

        for split in args.splits:
            dst_path = dst_dir / f"{task}_{split}.jsonl"
            count = process_split(
                raw_dir, task, split, dst_path,
                prompt_template, preamble, tool_names,
                _FUNCTION_MAP, cot_instruction, args.embedding_type,
            )
            total += count
            print(f"  [OK]    {task}_{split}: {count} samples")

    print(f"\nTotal: {total} samples written to {dst_dir}")


if __name__ == "__main__":
    main()
