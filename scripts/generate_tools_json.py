#!/usr/bin/env python3
"""Generate tools_per_task.json from the canonical Python source.

Reads BASIC_TOOLS and TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP from
Intern-S1-recipe/tools and writes a merged JSON mapping:
    {task_name: BASIC_TOOLS + task_specific_tools, "__default__": BASIC_TOOLS}

Usage:
    python scripts/generate_tools_json.py [output_path]

Default output: data/tdc/metadata/tools_per_task.json
"""

import json
import sys
from pathlib import Path

# Add Intern-S1-recipe to sys.path so `from tools import ...` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERN_S1_ROOT = PROJECT_ROOT / "Intern-S1-recipe"
assert (INTERN_S1_ROOT / "tools").is_dir(), (
    f"Intern-S1-recipe/tools not found at {INTERN_S1_ROOT}/tools. "
    f"Run: git submodule update --init Intern-S1-recipe"
)
sys.path.insert(0, str(INTERN_S1_ROOT))

from tools import BASIC_TOOLS
from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP

def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else str(
        PROJECT_ROOT / "data" / "tdc" / "metadata" / "tools_per_task.json"
    )

    mapping = {"__default__": BASIC_TOOLS}
    for task, specific_tools in TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.items():
        mapping[task] = BASIC_TOOLS + specific_tools

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(mapping)} tasks ({len(mapping)-1} + __default__) to {output_path}")

if __name__ == "__main__":
    main()
