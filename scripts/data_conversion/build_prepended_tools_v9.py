#!/usr/bin/env python3
"""
Build v9 prepended tools dataset = v8 tools + decision_tree_analysis (RF explanations).

Produces two output directories:
  - openai_format_prepended_tools_v9/          (RF explanation WITHOUT pseudo_label)
  - openai_format_prepended_tools_v9_pseudo/   (RF explanation WITH pseudo_label)

The decision_tree_analysis tool section is built from precomputed RF_explanations
JSONL files (LLM4SD/RF_explanations/{task}/{train,valid}.jsonl). Records without
an RF explanation simply omit that tool section.

Usage:
    python scripts/data_conversion/build_prepended_tools_v9.py
    python scripts/data_conversion/build_prepended_tools_v9.py --tasks AMES hERG
    python scripts/data_conversion/build_prepended_tools_v9.py --tasks AMES --splits train
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

PROMPT_KEY_OVERRIDES = {
    "SARSCoV2_3CLPro_Diamond": "SARSCOV2_3CLPro_Diamond",
}

TOOL_PREAMBLE_TEMPLATE = """\
You have access to the following tools to help analyze the molecule. Use them when necessary (**Don't use the same tool more than once**).

Available tools:
{tool_list}

"""


def format_tool_preamble(tools: list[dict]) -> str:
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
    with open(tools_json_path) as f:
        mapping = json.load(f)
    return mapping.get(task_name, mapping["__default__"])


def get_tool_names(tools_schema: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools_schema]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def run_tools_for_molecule(
    smiles: str,
    task: str,
    tool_names: list[str],
    function_map: dict,
    include_pseudo_label: bool,
) -> str:
    sections = []
    for tool_name in tool_names:
        fn = function_map.get(tool_name)
        if fn is None:
            continue
        try:
            if tool_name == "decision_tree_analysis":
                result = fn(smiles=smiles, task=task, include_pseudo_label=include_pseudo_label)
            elif tool_name.startswith("find_similar_molecules"):
                result = fn(smiles=smiles, include_pseudo_label=include_pseudo_label)
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
    precomputed_outputs: str,
    cot_instruction: str,
) -> str:
    prompt_with_smiles = prompt_template.replace("{Drug SMILES}", smiles)
    if prompt_with_smiles.rstrip().endswith("Answer:"):
        prompt_with_smiles = prompt_with_smiles.rstrip()[: -len("Answer:")].rstrip()
    return (
        f"{prompt_with_smiles}\n\n"
        f"--- Precomputed Tool Outputs ---\n\n"
        f"{precomputed_outputs}\n\n"
        f"{cot_instruction}"
    )


def label_to_answer(label: int, task: str) -> str:
    return "(B)" if label == 1 else "(A)"


def process_split(
    task: str,
    split: str,
    src_path: Path,
    dst_path: Path,
    dst_pseudo_path: Path,
    prompt_template: str,
    tool_names: list[str],
    function_map: dict,
    cot_instruction: str,
) -> int:
    import pandas as pd

    df = pd.read_csv(src_path)
    count = 0
    errors = 0

    with open(dst_path, "w") as fout, open(dst_pseudo_path, "w") as fout_pseudo:
        for i, (_, row) in enumerate(df.iterrows()):
            smiles = str(row["Drug"])
            label = int(row["Y"])
            answer = label_to_answer(label, task)

            try:
                # Build both variants (no pseudo label / with pseudo label)
                precomputed = run_tools_for_molecule(
                    smiles, task, tool_names, function_map,
                    include_pseudo_label=False,
                )
                precomputed_pseudo = run_tools_for_molecule(
                    smiles, task, tool_names, function_map,
                    include_pseudo_label=True,
                )

                text = build_prompt_text(smiles, prompt_template, precomputed, cot_instruction)
                text_pseudo = build_prompt_text(smiles, prompt_template, precomputed_pseudo, cot_instruction)

                record = {
                    "text": text,
                    "answer": answer,
                    "task": task,
                    "smiles": smiles,
                    "label": label,
                }
                record_pseudo = {
                    "text": text_pseudo,
                    "answer": answer,
                    "task": task,
                    "smiles": smiles,
                    "label": label,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout_pseudo.write(json.dumps(record_pseudo, ensure_ascii=False) + "\n")
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
        description="Build prepended_tools_v9 dataset (v8 + decision_tree_analysis)"
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "deduplicated_canonicalized"),
    )
    parser.add_argument(
        "--dst-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "openai_format_prepended_tools_v9"),
    )
    parser.add_argument(
        "--dst-pseudo-dir",
        default=str(PROJECT_ROOT / "data" / "tdc" / "openai_format_prepended_tools_v9_pseudo"),
    )
    parser.add_argument(
        "--prompts",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "prompts.json"),
    )
    parser.add_argument(
        "--tools-json",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "tools_per_task_v9.json"),
    )
    parser.add_argument(
        "--cot-instruction",
        default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "cot_instruction_v9.txt"),
    )
    parser.add_argument(
        "--fingerprint-dir",
        default="fingerprint_v8",
        help="Fingerprint cache subdirectory name (reuses v8 fingerprints)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dst_dir = Path(args.dst_dir)
    dst_pseudo_dir = Path(args.dst_pseudo_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_pseudo_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASK_NAMES

    # Load prompts
    with open(args.prompts) as f:
        prompts = json.load(f)

    # Load CoT instruction
    with open(args.cot_instruction) as f:
        cot_instruction = f.read().strip()

    # Point similarity tool at fingerprint_v8 cache
    cache_dir = PROJECT_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "cache"
    fp_dir = cache_dir / args.fingerprint_dir

    from openrlhf.tools.therapeutic_tools import similarity as sim_module
    original_embeddings_path = sim_module._embeddings_path
    original_data_dir = sim_module._DATA_DIR

    def _patched_embeddings_path(task: str, embedding_type: str = "learned") -> str:
        if embedding_type == "fingerprint":
            return str(fp_dir / f"{task}_embeddings.npz")
        return original_embeddings_path(task, embedding_type)

    sim_module._embeddings_path = _patched_embeddings_path
    sim_module._DATA_DIR = str(data_dir)
    sim_module._load_task_data.cache_clear()
    sim_module._load_split_smiles.cache_clear()

    from openrlhf.tools.therapeutic_tools import _FUNCTION_MAP

    # Preload decision_tree RF caches for all tasks
    from openrlhf.tools.therapeutic_tools.decision_tree import _load_task_cache
    for task in tasks:
        cache = _load_task_cache(task)
        print(f"  RF cache for {task}: {len(cache)} entries")

    total = 0
    for task in tasks:
        # Resolve prompt key
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

        # Load tool schemas
        tools_schema = load_tools_schema(Path(args.tools_json), task)
        tool_names = get_tool_names(tools_schema)

        # Add 3D properties for relevant tasks (if not already in schema)
        if task in TASKS_WITH_3D and "get_3d_properties" not in tool_names:
            tool_names.insert(-1, "get_3d_properties")  # before decision_tree_analysis
            from openrlhf.tools.therapeutic_tools.three_d import TOOL_SCHEMA as GET_3D_TOOL
            tools_schema = tools_schema[:-1] + [GET_3D_TOOL] + [tools_schema[-1]]

        print(f"\n  Processing {task} ({len(tool_names)} tools: {', '.join(tool_names)})...")

        for split in args.splits:
            src_path = data_dir / task / f"{split}.csv"
            dst_path = dst_dir / f"{task}_{split}.jsonl"
            dst_pseudo_path = dst_pseudo_dir / f"{task}_{split}.jsonl"

            if not src_path.exists():
                print(f"  [SKIP]  {src_path.name} — not found")
                continue

            count = process_split(
                task, split, src_path, dst_path, dst_pseudo_path,
                prompt_template, tool_names, _FUNCTION_MAP,
                cot_instruction,
            )
            total += count
            print(f"  [OK]    {dst_path.name}: {count} samples")

    # Restore
    sim_module._embeddings_path = original_embeddings_path
    sim_module._DATA_DIR = original_data_dir
    sim_module._load_task_data.cache_clear()
    sim_module._load_split_smiles.cache_clear()

    print(f"\nTotal: {total} samples written to {dst_dir} and {dst_pseudo_dir}")


if __name__ == "__main__":
    main()
