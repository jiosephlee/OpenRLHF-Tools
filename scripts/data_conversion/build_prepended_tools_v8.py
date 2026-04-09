#!/usr/bin/env python3
"""
Build openai_format_prepended_tools_v8 dataset with precomputed tool outputs.

Builds from scratch using:
  - deduplicated_canonicalized/ as the source dataset
  - prompts.json for prompt templates
  - tools_per_task_v8.json for tool schemas (no remove_salts, yes predict_solubility)
  - cot_instruction_v7_prepended.txt for chain-of-thought instructions
  - fingerprint_v8/ cache for KNN similarity lookups
  - Universal caches (fg_cache.jsonl, tdc_metadata_consolidated.csv) for speedup

Pipeline per record:
  1. Load prompt template, fill {Drug SMILES}
  2. Run each v8 tool on the canonical SMILES
  3. Assemble: prompt + precomputed outputs + CoT instruction
  4. Write JSONL

Usage:
    python scripts/data_conversion/build_prepended_tools_v8.py
    python scripts/data_conversion/build_prepended_tools_v8.py --tasks AMES hERG
    python scripts/data_conversion/build_prepended_tools_v8.py --tasks AMES --splits train
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TASK_NAMES = [
    "Bioavailability_Ma",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret",
    "Carcinogens_Lagunin",
    "hERG",
    "ClinTox",
    "DILI",
    "Skin_Reaction",
    "AMES",
]

TASKS_WITH_3D = {
    "Bioavailability_Ma",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
}

# Known case mismatches between task names and prompts.json keys
PROMPT_KEY_OVERRIDES = {
    "SARSCoV2_3CLPro_Diamond": "SARSCOV2_3CLPro_Diamond",
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


def get_tool_names(tools_schema: list[dict]) -> list[str]:
    """Extract tool function names from schema list."""
    return [t["function"]["name"] for t in tools_schema]


def run_tools_for_molecule(
    smiles: str,
    task: str,
    tool_names: list[str],
    function_map: dict,
) -> str:
    """Run each tool on the molecule and format results."""
    sections = []
    for tool_name in tool_names:
        fn = function_map.get(tool_name)
        if fn is None:
            continue
        try:
            if tool_name.startswith("find_similar_molecules"):
                result = fn(smiles=smiles)
            elif tool_name == "assess_adme_properties":
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
        f"{prompt_with_smiles}\n\n"
        f"--- Precomputed Tool Outputs ---\n\n"
        f"{precomputed_outputs}\n\n"
        f"{cot_instruction}"
    )
    return text


def label_to_answer(label: int, task: str) -> str:
    """Convert numeric label to (A)/(B) answer string."""
    return "(B)" if label == 1 else "(A)"


def process_split(
    task: str,
    split: str,
    src_path: Path,
    dst_path: Path,
    prompt_template: str,
    tool_preamble: str,
    tool_names: list[str],
    function_map: dict,
    cot_instruction: str,
) -> int:
    """Process a single CSV split into JSONL with precomputed tool outputs."""
    import pandas as pd

    df = pd.read_csv(src_path)
    count = 0
    errors = 0

    with open(dst_path, "w") as fout:
        for i, (_, row) in enumerate(df.iterrows()):
            smiles = str(row["Drug"])
            label = int(row["Y"])
            answer = label_to_answer(label, task)

            try:
                precomputed = run_tools_for_molecule(smiles, task, tool_names, function_map)
                text = build_prompt_text(
                    smiles, prompt_template, tool_preamble, precomputed, cot_instruction,
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
        description="Build prepended_tools_v8 dataset from deduplicated_canonicalized data"
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "deduplicated_canonicalized"),
    )
    parser.add_argument(
        "--dst-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "openai_format_prepended_tools_v8"),
    )
    parser.add_argument(
        "--prompts",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "prompts.json"),
    )
    parser.add_argument(
        "--tools-json",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "tools_per_task_v8.json"),
    )
    parser.add_argument(
        "--cot-instruction",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "cot_instruction_v7_prepended.txt"),
    )
    parser.add_argument(
        "--fingerprint-dir",
        default="fingerprints_with_canonicalized",
        help="Fingerprint cache subdirectory name (under therapeutic_tools/cache/)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASK_NAMES

    # Load prompts
    with open(args.prompts) as f:
        prompts = json.load(f)

    # Load CoT instruction
    with open(args.cot_instruction) as f:
        cot_instruction = f.read().strip()

    # Point similarity tool at the configured fingerprint cache
    cache_dir = PROJECT_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "cache"
    fp_v8_dir = cache_dir / args.fingerprint_dir

    # Monkey-patch similarity module to use the configured fingerprint cache and deduplicated_canonicalized
    from openrlhf.tools.therapeutic_tools import similarity as sim_module
    original_embeddings_path = sim_module._embeddings_path
    original_data_dir = sim_module._DATA_DIR

    def _patched_embeddings_path(task: str, embedding_type: str = "learned") -> str:
        if embedding_type == "fingerprint":
            return str(fp_v8_dir / f"{task}_embeddings.npz")
        return original_embeddings_path(task, embedding_type)

    sim_module._embeddings_path = _patched_embeddings_path
    sim_module._DATA_DIR = str(data_dir)
    # Clear cached data so it reloads from new paths
    sim_module._load_task_data.cache_clear()
    sim_module._load_split_smiles.cache_clear()

    # Import function map
    from openrlhf.tools.therapeutic_tools import _FUNCTION_MAP

    total = 0
    for task in tasks:
        # Resolve prompt key (handle case mismatches)
        prompt_key = PROMPT_KEY_OVERRIDES.get(task, task)
        if prompt_key not in prompts:
            for k in prompts:
                if k.lower() == task.lower():
                    prompt_key = k
                    break

        prompt_template = prompts.get(prompt_key, "")
        if not prompt_template:
            print(f"  [WARN]  No prompt template for {task}, skipping")
            continue

        # Load tool schemas for this task
        tools_schema = load_tools_schema(Path(args.tools_json), task)
        tool_names = get_tool_names(tools_schema)

        # Add 3D properties for relevant tasks
        if task in TASKS_WITH_3D and "get_3d_properties" not in tool_names:
            tool_names.append("get_3d_properties")
            from openrlhf.tools.therapeutic_tools.three_d import TOOL_SCHEMA as GET_3D_TOOL
            tools_schema = tools_schema + [GET_3D_TOOL]

        preamble = format_tool_preamble(tools_schema)

        print(f"\n  Processing {task} ({len(tool_names)} tools: {', '.join(tool_names)})...")

        for split in args.splits:
            src_path = data_dir / task / f"{split}.csv"
            dst_path = dst_dir / f"{task}_{split}.jsonl"

            if not src_path.exists():
                print(f"  [SKIP]  {src_path.name} — not found")
                continue

            count = process_split(
                task, split, src_path, dst_path,
                prompt_template, preamble, tool_names, _FUNCTION_MAP,
                cot_instruction,
            )
            total += count
            print(f"  [OK]    {dst_path.name}: {count} samples")

    # Restore original similarity paths
    sim_module._embeddings_path = original_embeddings_path
    sim_module._DATA_DIR = original_data_dir
    sim_module._load_task_data.cache_clear()
    sim_module._load_split_smiles.cache_clear()

    print(f"\nTotal: {total} samples written to {dst_dir}")


if __name__ == "__main__":
    main()
