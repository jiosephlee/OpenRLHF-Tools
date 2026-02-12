#!/usr/bin/env python3
"""
TDC GRPO Training Example.

This script demonstrates how to load TDC datasets and prepare them for GRPO training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openrlhf.datasets.tdc_grpo_dataset import load_tdc_grpo_dataset


def main():
    """Demonstrate TDC GRPO dataset loading."""

    print("="*80)
    print("TDC GRPO Dataset Loading Example")
    print("="*80)

    # Configuration
    task_name = "AMES"
    data_path = f"data/tdc/openai_format/{task_name}_train.jsonl"
    model_name = "internlm/internlm2_5-7b-chat"  # Or any model with chat template

    print(f"\nTask: {task_name}")
    print(f"Data: {data_path}")
    print(f"Model: {model_name}\n")

    # Load tokenizer (you need transformers installed)
    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Tokenizer loaded\n")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        print("  Install transformers: pip install transformers")
        print("  Or download model first: huggingface-cli download {model_name}\n")
        return

    # Load dataset with task-specific tools
    print("Loading dataset with task-specific tools...")
    dataset = load_tdc_grpo_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        tool_mode="TaskSpecific",
        task_name=task_name,
    )
    print(f"✓ Loaded {len(dataset)} examples\n")

    # Inspect first example
    print("First example:")
    print(f"  Keys: {list(dataset[0].keys())}")
    print(f"  Answer: {dataset[0]['answer']}")
    print(f"  SMILES: {dataset[0]['smiles']}")
    print(f"  Label: {dataset[0]['label']}\n")

    print("Question (first 500 chars):")
    print("-"*80)
    print(dataset[0]["question"][:500])
    print("...")
    print("-"*80)

    print("\n✓ Dataset ready for GRPO training!")

    # Example training command
    print("\nTo train with OpenRLHF:")
    print("-"*80)
    print(f"""
python -m openrlhf.cli.train_ppo_ray \\
    --pretrain {model_name} \\
    --prompt_data {data_path} \\
    --input_key "question" \\
    --label_key "answer" \\
    --agent_func_path openrlhf/utils/tool_calling_turn.py \\
    --agent_max_steps 40 \\
    --n_samples_per_prompt 8 \\
    --advantage_estimator dr_grpo \\
    --dynamic_filtering \\
    --dynamic_filtering_reward_range 0.2 0.8 \\
    # ... other GRPO args
    """.strip())
    print("-"*80)


if __name__ == "__main__":
    main()
