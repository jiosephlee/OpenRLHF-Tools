#!/usr/bin/env python3
"""Validate OpenAI message format JSONL datasets."""

import json
import argparse
from pathlib import Path


def validate_openai_file(jsonl_path: Path):
    """
    Validate an OpenAI message format JSONL file.

    Checks:
    1. All records have "messages" and "answer" fields
    2. "messages" is a list with role/content structure
    3. "answer" is in format "(A)" or "(B)"
    4. No empty fields
    """
    print(f"\nValidating {jsonl_path.name}...")

    if not jsonl_path.exists():
        print(f"  ⚠️  File not found: {jsonl_path}")
        return False

    # Load JSONL
    records = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"  ✗ Line {line_num}: Invalid JSON - {e}")
                return False

    if not records:
        print(f"  ✗ Empty file")
        return False

    print(f"  Loaded {len(records)} records")

    # Validate each record
    for i, record in enumerate(records):
        # Check required fields
        if "messages" not in record:
            print(f"  ✗ Record {i}: Missing 'messages' field")
            return False
        if "answer" not in record:
            print(f"  ✗ Record {i}: Missing 'answer' field")
            return False

        # Check messages structure
        messages = record["messages"]
        if not isinstance(messages, list):
            print(f"  ✗ Record {i}: 'messages' must be a list")
            return False

        if len(messages) == 0:
            print(f"  ✗ Record {i}: 'messages' is empty")
            return False

        # Validate message structure
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                print(f"  ✗ Record {i}, message {msg_idx}: Message must be a dict")
                return False
            if "role" not in msg:
                print(f"  ✗ Record {i}, message {msg_idx}: Missing 'role' field")
                return False
            if "content" not in msg:
                print(f"  ✗ Record {i}, message {msg_idx}: Missing 'content' field")
                return False
            if not msg["content"] or not msg["content"].strip():
                print(f"  ✗ Record {i}, message {msg_idx}: Empty content")
                return False

        # Check answer format
        answer = record["answer"]
        if answer not in ["(A)", "(B)"]:
            print(f"  ✗ Record {i}: Invalid answer '{answer}' (expected '(A)' or '(B)')")
            return False

    print(f"  ✓ All checks passed")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                       help="Task name (e.g., AMES)")
    parser.add_argument("--data_dir", type=str,
                       default="/Users/jlee0/Desktop/research/OpenRLHF-Tools/data/tdc/openai_format")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Validate all splits
    all_valid = True
    for split in ["train", "val", "test"]:
        jsonl_path = data_dir / f"{args.task}_{split}.jsonl"
        if not validate_openai_file(jsonl_path):
            all_valid = False

    if all_valid:
        print("\n✓ All validations passed!")
    else:
        print("\n✗ Validation failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
