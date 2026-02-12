#!/usr/bin/env python3
"""Decode observation_tokens from rollout traces to understand what the model sees.

Usage:
    python scripts/decode_rollout_trace.py <trace_file_or_json_string> [--model_path MODEL]

Examples:
    # From a trace JSONL file (reads first line)
    python scripts/decode_rollout_trace.py saves/tdc/AMES/rollout_traces/step_1.jsonl

    # From inline JSON
    python scripts/decode_rollout_trace.py '{"trace": {"observation_tokens": [...], "action_ranges": [...]}}'

    # Specify model for tokenizer
    python scripts/decode_rollout_trace.py trace.jsonl --model_path jiosephlee/sft_intern_distillation_Intern-S1-mini-lm
"""

import argparse
import json
import sys
import os


def decode_trace(trace: dict, tokenizer) -> None:
    """Decode and display a single rollout trace."""
    obs_tokens = trace.get("observation_tokens", [])
    action_ranges = trace.get("action_ranges", [])
    prompt = trace.get("prompt", "")
    label = trace.get("label", "")
    reward = trace.get("reward", None)
    scores = trace.get("scores", None)

    print("=" * 80)
    print(f"ROLLOUT TRACE ANALYSIS")
    print("=" * 80)
    print(f"Total observation tokens: {len(obs_tokens)}")
    print(f"Action ranges: {action_ranges}")
    print(f"Label: {label}")
    print(f"Reward: {reward}")
    print(f"Scores: {scores}")
    print(f"rollout_log_probs: {'present' if trace.get('rollout_log_probs') else 'NULL'}")
    print()

    # Decode full sequence
    full_text = tokenizer.decode(obs_tokens, skip_special_tokens=False)

    # Print prompt section
    if action_ranges:
        first_action_start = action_ranges[0][0]
        prompt_tokens = obs_tokens[:first_action_start]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        print(f"--- PROMPT (tokens 0-{first_action_start-1}, {len(prompt_tokens)} tokens) ---")
        print(prompt_text)
        print()

    # Print each action and observation section
    for i, (start, end) in enumerate(action_ranges):
        action_tokens = obs_tokens[start:end]
        action_text = tokenizer.decode(action_tokens, skip_special_tokens=False)
        print(f"--- ACTION {i+1} (tokens {start}-{end-1}, {end-start} tokens) ---")
        print(action_text)
        print()

        # Print observation after this action (if not last action)
        if i + 1 < len(action_ranges):
            next_start = action_ranges[i + 1][0]
            obs_section = obs_tokens[end:next_start]
            obs_text = tokenizer.decode(obs_section, skip_special_tokens=False)
            print(f"--- OBSERVATION {i+1} (tokens {end}-{next_start-1}, {next_start-end} tokens) ---")
            print(obs_text)
            print()
        else:
            # After last action, show remaining tokens if any
            remaining = obs_tokens[end:]
            if remaining:
                rem_text = tokenizer.decode(remaining, skip_special_tokens=False)
                print(f"--- TRAILING TOKENS (tokens {end}-{len(obs_tokens)-1}, {len(remaining)} tokens) ---")
                print(rem_text)
                print()

    # Check for double-wrapping issues
    print("=" * 80)
    print("DIAGNOSTIC CHECKS")
    print("=" * 80)

    # Count special token occurrences
    im_start_count = full_text.count("<|im_start|>")
    system_count = full_text.count("<|im_start|>system")
    user_count = full_text.count("<|im_start|>user")
    assistant_count = full_text.count("<|im_start|>assistant")
    think_count = full_text.count("<think>")

    print(f"<|im_start|> count: {im_start_count}")
    print(f"<|im_start|>system count: {system_count}")
    print(f"<|im_start|>user count: {user_count}")
    print(f"<|im_start|>assistant count: {assistant_count}")
    print(f"<think> count: {think_count}")
    print()

    if system_count > 1:
        print("WARNING: Multiple system messages detected — likely DOUBLE PROMPT WRAPPING!")
    if user_count > 1:
        print("WARNING: Multiple user messages detected — likely DOUBLE PROMPT WRAPPING!")

    # Check for tool call patterns in action text
    for i, (start, end) in enumerate(action_ranges):
        action_text = tokenizer.decode(obs_tokens[start:end], skip_special_tokens=False)
        has_action_start = "<|action_start|>" in action_text
        has_action_end = "<|action_end|>" in action_text
        has_tool_call = "<tool_call>" in action_text
        has_plugin = "<|plugin|>" in action_text

        print(f"Action {i+1} tool call markers:")
        print(f"  <|action_start|>: {has_action_start}")
        print(f"  <|action_end|>:   {has_action_end}")
        print(f"  <|plugin|>:       {has_plugin}")
        print(f"  <tool_call>:      {has_tool_call}")

    # Action mask stats
    total = len(obs_tokens)
    action_count = sum(end - start for start, end in action_ranges)
    prompt_count = action_ranges[0][0] if action_ranges else total
    obs_count = total - action_count - prompt_count
    print(f"\nToken distribution:")
    print(f"  Prompt:      {prompt_count:5d} tokens ({100*prompt_count/total:.1f}%)")
    print(f"  Action:      {action_count:5d} tokens ({100*action_count/total:.1f}%)")
    print(f"  Observation: {obs_count:5d} tokens ({100*obs_count/total:.1f}%)")
    print(f"  Total:       {total:5d} tokens")


def main():
    parser = argparse.ArgumentParser(description="Decode rollout trace observation_tokens")
    parser.add_argument("input", help="Path to trace JSONL file or inline JSON string")
    parser.add_argument("--model_path", default=None,
                        help="Model path for tokenizer (default: OPENRLHF_MODEL_PATH env var)")
    parser.add_argument("--line", type=int, default=0,
                        help="Which line to read from JSONL file (0-indexed)")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model_path or os.environ.get("OPENRLHF_MODEL_PATH")
    if not model_path:
        print("Error: Provide --model_path or set OPENRLHF_MODEL_PATH", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer from: {model_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Parse input
    if os.path.isfile(args.input):
        with open(args.input) as f:
            for i, line in enumerate(f):
                if i == args.line:
                    data = json.loads(line.strip())
                    break
            else:
                print(f"Error: Line {args.line} not found in file", file=sys.stderr)
                sys.exit(1)
    else:
        data = json.loads(args.input)

    # Extract trace
    trace = data.get("trace", data)

    decode_trace(trace, tokenizer)


if __name__ == "__main__":
    main()
