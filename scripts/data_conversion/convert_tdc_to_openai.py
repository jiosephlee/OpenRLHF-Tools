#!/usr/bin/env python3
"""
Convert TDC CSV datasets to OpenAI message format JSONL.

Usage:
    python convert_tdc_to_openai.py --task AMES
    python convert_tdc_to_openai.py --all
"""

import argparse
import sys
from pathlib import Path

# Add OpenRLHF-Tools datasets to path (avoid importing openrlhf package)
project_root = Path(__file__).parent.parent.parent
datasets_path = project_root / "openrlhf" / "datasets"
sys.path.insert(0, str(datasets_path))

# Import only the loader module (not the whole package)
from tdc_loader import TDCDatasetLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task name (e.g., AMES)")
    parser.add_argument("--all", action="store_true", help="Convert all tasks")
    parser.add_argument("--raw_dir", type=str,
                       default="data/tdc/raw",
                       help="Directory containing raw CSV files")
    parser.add_argument("--output_dir", type=str,
                       default="data/tdc/openai_format",
                       help="Output directory for OpenAI format JSONL files")
    parser.add_argument("--prompts_path", type=str,
                       default="data/tdc/metadata/prompts.json",
                       help="Path to TDC prompts JSON file")
    parser.add_argument("--cot_instruction_path", type=str,
                       default="data/tdc/metadata/cot_instruction.txt",
                       help="Path to CoT instruction text file")
    parser.add_argument("--cot_instruction", type=str,
                       default=None,
                       help="Custom CoT instruction (overrides file)")
    parser.add_argument("--model_type", type=str,
                       default=None,
                       help="Model type (e.g. 'gpt-oss') to apply model-specific string replacements to the CoT instruction.")
    args = parser.parse_args()

    # Initialize loader
    loader = TDCDatasetLoader(
        prompts_path=args.prompts_path,
        cot_instruction_path=args.cot_instruction_path,
        cot_instruction=args.cot_instruction,
        model_type=args.model_type,
    )

    # Convert tasks
    if args.all:
        print("Converting all TDC datasets...")
        loader.convert_all_tasks(args.raw_dir, args.output_dir)
    elif args.task:
        print(f"Converting task: {args.task}")
        loader.convert_task(args.task, args.raw_dir, args.output_dir)
    else:
        parser.error("Must specify either --task or --all")

    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()
