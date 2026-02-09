#!/usr/bin/env python3
"""
Extract tool definitions from therapeutic-tuning for GRPO training.

Reads tool schemas from therapeutic-tuning/tools/__init__.py and
saves them as JSON files for dataset loading.
"""

import json
import sys
from pathlib import Path

# Add therapeutic-tuning to path
therapeutic_tuning_path = Path("/Users/jlee0/Desktop/research/therapeutic-tuning")
sys.path.insert(0, str(therapeutic_tuning_path))

from tools import ALL_AVAILABLE_TOOLS, TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP


def build_tool_definitions(task_specific: bool = False, task: str = None):
    """
    Build tool definitions for GRPO training.

    Args:
        task_specific: If True, only include tools specific to a task
        task: Task name (e.g., "AMES") - required if task_specific=True

    Returns:
        List of tool schemas in OpenAI format
    """
    if task_specific and task:
        # TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP already contains full tool definitions
        tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task, [])
    else:
        # Use all available tools
        tools = ALL_AVAILABLE_TOOLS

    return tools


def main():
    # Build tool definitions
    all_tools = build_tool_definitions(task_specific=False)

    # Save to metadata directory
    output_dir = Path("/Users/jlee0/Desktop/research/OpenRLHF-Tools/data/tdc/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all tools
    with open(output_dir / "tools_all.json", 'w') as f:
        json.dump(all_tools, f, indent=2, ensure_ascii=False)

    # Save task-specific tool mappings
    task_tool_map = {}
    for task in TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.keys():
        task_tools = build_tool_definitions(task_specific=True, task=task)
        if task_tools:
            task_tool_map[task] = task_tools

    with open(output_dir / "tools_task_specific.json", 'w') as f:
        json.dump(task_tool_map, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(all_tools)} tools to {output_dir}/tools_all.json")
    print(f"✓ Saved task-specific mappings for {len(task_tool_map)} tasks to {output_dir}/tools_task_specific.json")


if __name__ == "__main__":
    main()
