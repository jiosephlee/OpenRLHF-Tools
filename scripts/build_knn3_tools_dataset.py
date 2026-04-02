#!/usr/bin/env python3
"""
Build knn3_tools_format dataset with precomputed therapeutic tool outputs
baked into each sample's prompt text.

Pipeline:
  1. Load enriched prompts (with target context) as base prompt templates
  2. For each molecule in knn3_format, run all therapeutic tools
  3. Build text: tool_preamble + enriched_prompt + precomputed_tool_outputs + answer_instruction
  4. Write to knn3_tools_format/

Input:  data/tdc/knn3_format/{TASK}_{split}.jsonl       (source records with smiles/label/task)
        data/tdc/metadata/prompts_enriched.json          (enriched prompt templates)
        data/tdc/metadata/tools_per_task_v6.json         (tool schema definitions)
Output: data/tdc/knn3_tools_format/{TASK}_{split}.jsonl

Usage:
    python scripts/build_knn3_tools_dataset.py
    python scripts/build_knn3_tools_dataset.py --tasks hERG AMES   # specific tasks only
    python scripts/build_knn3_tools_dataset.py --workers 4          # parallel workers
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

# No hardcoded tool list — derived from tools_per_task_v6.json at runtime.
# Tasks where 3D conformational properties (ePSA, PMI shape) are informative.
# These involve membrane permeability, oral absorption, or 3D binding-site interactions.
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

# Tool preamble template (same as before)
TOOL_PREAMBLE_TEMPLATE = """\
You have access to the following tools to help analyze the molecule. Use them when necessary (**Don't use the same tool more than once**).

Available tools:
{tool_list}

"""

ANSWER_INSTRUCTION = (
    '\nPlease think step by step and use tools when necessary '
    '(**Don\'t use the same tool more than once**). '
    'Then put your final choice ((A) or (B)) after "Answer:"'
)


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


def get_target_context(task: str, profiles_dir: Path) -> str | None:
    """Load target context .txt for a TDC task. Returns None if no profile exists."""
    # Try exact match first
    exact = profiles_dir / f"{task}.txt"
    if exact.exists():
        return exact.read_text().strip()
    # Case-insensitive fallback
    task_lower = task.lower()
    for f in profiles_dir.glob("*.txt"):
        if f.stem.lower() == task_lower:
            return f.read_text().strip()
    return None


def get_tool_names_for_task(tools_schema: list[dict]) -> list[str]:
    """Extract tool function names from the OpenAI-style tool schema list."""
    return [t["function"]["name"] for t in tools_schema]


def run_tools_for_molecule(smiles: str, task: str, tool_names: list[str], function_map: dict) -> str:
    """Run the specified therapeutic tools on a molecule and format the results."""
    sections = []
    for tool_name in tool_names:
        fn = function_map.get(tool_name)
        if fn is None:
            continue
        try:
            if tool_name == "find_similar_molecules":
                result = fn(smiles=smiles, task=task, k=5)
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
    task: str,
    prompt_template: str,
    tool_preamble: str,
    precomputed_outputs: str,
) -> str:
    """Assemble the full prompt text for a sample."""
    # Fill in SMILES in the prompt template
    # The template ends with "Answer:" — we need to insert tool outputs before that
    prompt_with_smiles = prompt_template.replace("{Drug SMILES}", smiles)

    # Remove the trailing "Answer:" since we'll add our own instruction
    if prompt_with_smiles.rstrip().endswith("Answer:"):
        prompt_with_smiles = prompt_with_smiles.rstrip()[: -len("Answer:")].rstrip()

    # Assemble: preamble + prompt + tool outputs + answer instruction
    text = (
        f"{tool_preamble}"
        f"{prompt_with_smiles}\n\n"
        f"--- Precomputed Tool Outputs ---\n\n"
        f"{precomputed_outputs}\n"
        f"{ANSWER_INSTRUCTION}"
    )
    return text


def process_record(record: dict, prompt_template: str, tool_preamble: str, tool_names: list[str], function_map: dict) -> dict:
    """Process a single record: run tools and build enriched text."""
    smiles = record["smiles"]
    task = record["task"]

    precomputed = run_tools_for_molecule(smiles, task, tool_names, function_map)
    text = build_prompt_text(smiles, task, prompt_template, tool_preamble, precomputed)

    return {
        "text": text,
        "answer": record["answer"],
        "task": task,
        "smiles": smiles,
        "label": record["label"],
    }


def process_split(
    src_path: Path,
    dst_path: Path,
    prompt_template: str,
    tool_preamble: str,
    tool_names: list[str],
    function_map: dict,
) -> int:
    """Process a single split file."""
    records = []
    with open(src_path) as f:
        for line in f:
            records.append(json.loads(line))

    count = 0
    errors = 0
    with open(dst_path, "w") as fout:
        for i, record in enumerate(records):
            try:
                enriched = process_record(record, prompt_template, tool_preamble, tool_names, function_map)
                fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
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
    parser = argparse.ArgumentParser(description="Build knn3_tools_format with precomputed tool outputs")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific tasks to process (default: all)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (not yet implemented for tool calls)")
    parser.add_argument("--src-dir", default=str(PROJECT_ROOT / "data" / "tdc" / "knn3_format"))
    parser.add_argument("--dst-dir", default=str(PROJECT_ROOT / "data" / "tdc" / "knn3_tools_format"))
    parser.add_argument("--prompts", default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "prompts_enriched.json"))
    parser.add_argument("--tools-json", default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "tools_per_task_v6.json"))
    parser.add_argument("--profiles-dir", default=str(PROJECT_ROOT / "data" / "tdc" / "metadata" / "target_profiles"))
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or TASK_NAMES

    # Load enriched prompts
    with open(args.prompts) as f:
        prompts = json.load(f)

    # Fallback: if a task isn't in enriched prompts, try original prompts.json
    original_prompts_path = PROJECT_ROOT / "data" / "tdc" / "metadata" / "prompts.json"
    if original_prompts_path.exists():
        with open(original_prompts_path) as f:
            original_prompts = json.load(f)
    else:
        original_prompts = {}

    profiles_dir = Path(args.profiles_dir)

    # Import therapeutic tools (requires numpy, rdkit, etc.)
    from openrlhf.tools.therapeutic_tools import _FUNCTION_MAP

    total = 0
    missing = []

    for task in tasks:
        # Map task name to prompts.json key
        # knn3_format uses "SARSCoV2_3CLPro_Diamond" but prompts.json uses "SARSCOV2_3CLPro_Diamond"
        prompt_key = task
        if task not in prompts:
            # Try case variations
            for k in prompts:
                if k.lower() == task.lower():
                    prompt_key = k
                    break

        prompt_template = prompts.get(prompt_key, original_prompts.get(prompt_key, original_prompts.get(task, "")))
        if not prompt_template:
            print(f"  [WARN]  No prompt template for {task}, skipping")
            continue

        tools_schema = load_tools_schema(Path(args.tools_json), task)
        tool_names = get_tool_names_for_task(tools_schema)

        # Dynamically add get_3d_properties for tasks where 3D info is relevant
        if task in TASKS_WITH_3D and "get_3d_properties" not in tool_names:
            tool_names.append("get_3d_properties")
            # Also add the schema entry for the preamble
            from openrlhf.tools.therapeutic_tools.three_d import TOOL_SCHEMA as GET_3D_PROPERTIES_TOOL
            tools_schema = tools_schema + [GET_3D_PROPERTIES_TOOL]

        preamble = format_tool_preamble(tools_schema)

        print(f"\n  Processing {task} ({len(tool_names)} tools: {', '.join(tool_names)})...")

        for split in args.splits:
            src_path = src_dir / f"{task}_{split}.jsonl"
            dst_path = dst_dir / f"{task}_{split}.jsonl"

            if not src_path.exists():
                missing.append(str(src_path))
                print(f"  [SKIP]  {src_path.name} — not found")
                continue

            count = process_split(src_path, dst_path, prompt_template, preamble, tool_names, _FUNCTION_MAP)
            total += count
            print(f"  [OK]    {dst_path.name}: {count} samples")

    print(f"\nTotal: {total} samples written to {dst_dir}")
    if missing:
        print(f"\nMissing source files ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
