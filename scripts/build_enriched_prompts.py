#!/usr/bin/env python3
"""
Build prompts_enriched.json by injecting target profiles into prompt Context sections.

Usage:
    python scripts/build_enriched_prompts.py
    python scripts/build_enriched_prompts.py --prompts data/tdc/metadata/prompts.json --profiles data/tdc/metadata/target_profiles --output data/tdc/metadata/prompts_enriched.json
"""

import argparse
import json
import re
from pathlib import Path


def get_target_context(task: str, profiles_dir: str) -> str | None:
    """Load target context .txt for a TDC task. Returns None if no profile exists."""
    profiles_path = Path(profiles_dir)

    # Try exact match first
    exact = profiles_path / f"{task}.txt"
    if exact.exists():
        return exact.read_text().strip()

    # Case-insensitive fallback
    task_lower = task.lower()
    for f in profiles_path.glob("*.txt"):
        if f.stem.lower() == task_lower:
            return f.read_text().strip()

    return None


def build_enriched_prompts(prompts_path: str, profiles_dir: str, output_path: str):
    """For each task in prompts.json, replace Context: section if a profile exists."""
    with open(prompts_path) as f:
        prompts = json.load(f)

    enriched_count = 0
    for task, template in prompts.items():
        context = get_target_context(task, profiles_dir)
        if context:
            # Replace text between "Context: " and "\nQuestion:"
            new_template = re.sub(
                r"(Context: ).*?(\nQuestion:)",
                rf"\1{context}\2",
                template,
                flags=re.DOTALL,
            )
            if new_template != template:
                prompts[task] = new_template
                enriched_count += 1
                print(f"  Enriched: {task}")

    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {output_path} ({enriched_count} tasks enriched, {len(prompts) - enriched_count} unchanged)")


def main():
    parser = argparse.ArgumentParser(description="Build enriched TDC prompts with target context")
    parser.add_argument("--prompts", default="data/tdc/metadata/prompts.json")
    parser.add_argument("--profiles", default="data/tdc/metadata/target_profiles")
    parser.add_argument("--output", default="data/tdc/metadata/prompts_enriched.json")
    args = parser.parse_args()

    build_enriched_prompts(args.prompts, args.profiles, args.output)


if __name__ == "__main__":
    main()
