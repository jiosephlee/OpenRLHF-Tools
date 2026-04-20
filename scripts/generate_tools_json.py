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
import importlib.util
import json
import sys
import types
from pathlib import Path

# Ensure the project root is on sys.path so ``openrlhf`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_tool_versions_module():
    """Load ``tool_versions`` without importing ``openrlhf.utils`` package ``__init__``
    (which pulls in torch via ``processor``)."""
    utils_path = PROJECT_ROOT / "openrlhf" / "utils"
    stub = types.ModuleType("openrlhf.utils")
    stub.__path__ = [str(utils_path)]
    sys.modules.setdefault("openrlhf.utils", stub)

    tv_path = utils_path / "tool_versions.py"
    spec = importlib.util.spec_from_file_location(
        "openrlhf.utils.tool_versions",
        tv_path,
        submodule_search_locations=[str(utils_path)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["openrlhf.utils.tool_versions"] = mod
    spec.loader.exec_module(mod)
    return mod


_tv = _load_tool_versions_module()
_ALL_VERSIONS = _tv._ALL_VERSIONS
get_version = _tv.get_version


def _assert_nested(tools, ver: str) -> list:
    """Validate OpenAI-style nesting: {"type":"function","function":{...}}.

    gpt-oss harmony Jinja template accesses tool.function.name; flat schemas
    ({"type":"function","name":...}) fail with 'dict object has no attribute function'.
    Schemas must be wrapped at the source (see therapeutic_tools/v15.py).
    """
    for t in tools:
        if not (isinstance(t, dict) and isinstance(t.get("function"), dict)):
            raise ValueError(
                f"[{ver}] tool schema is not nested OpenAI form "
                f"(expected {{'type':'function','function':{{...}}}}); got keys "
                f"{list(t.keys()) if isinstance(t, dict) else type(t).__name__}"
            )
    return list(tools)


def _write_version(ver: str, output_dir: Path) -> None:
    """Write a single version's tools_per_task JSON file."""
    cfg = get_version(ver)
    basic = _assert_nested(cfg["basic_schemas"], ver)
    task_map = cfg["task_specific_map"]

    mapping = {"__default__": basic}
    for task, specific_tools in task_map.items():
        mapping[task] = basic + _assert_nested(specific_tools, ver)

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
