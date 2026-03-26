#!/usr/bin/env python3
"""Generate versioned tools_per_task JSON files from the tool version registry.

Reads tool schemas from ``openrlhf.utils.tool_versions`` and writes merged
JSON mappings of the form:
    {task_name: basic_schemas + task_specific, "__default__": basic_schemas}

Usage:
    python scripts/generate_tools_json.py --version v1|v2|v3|v4|v5|v6|all

Output: data/tdc/metadata/tools_per_task_<version>.json
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``openrlhf`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrlhf.utils.tool_versions import _ALL_VERSIONS, get_version


def _write_version(ver: str, output_dir: Path) -> None:
    """Write a single version's tools_per_task JSON file."""
    cfg = get_version(ver)
    basic = cfg["basic_schemas"]
    task_map = cfg["task_specific_map"]

    mapping = {"__default__": basic}
    for task, specific_tools in task_map.items():
        mapping[task] = basic + specific_tools

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tools_per_task_{ver}.json"
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    n_default = len(basic)
    print(f"[{ver}] Wrote {len(mapping)} tasks ({len(mapping)-1} + __default__, "
          f"{n_default} base tools) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate versioned tools_per_task JSON files.")
    parser.add_argument(
        "--version",
        required=True,
        choices=sorted(_ALL_VERSIONS) + ["all"],
        help="Tool version to generate (v1, v2, v3, v4, or all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/tdc/metadata/).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "data" / "tdc" / "metadata")

    if args.version == "all":
        for ver in sorted(_ALL_VERSIONS):
            _write_version(ver, output_dir)
    else:
        _write_version(args.version, output_dir)


if __name__ == "__main__":
    main()
