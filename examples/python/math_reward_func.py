"""Math reward function for verifying mathematical answers.

Only supports \\boxed{answer} LaTeX format for answer extraction.
"""

import json
import os
import time
from typing import List

import torch

from openrlhf.utils import extract_boxed_answer, grade_answer

# Directory for logging sample outputs
_TRACE_DIR = os.environ.get("MATH_REWARD_TRACE_DIR", "/tmp/math_reward_traces")
os.makedirs(_TRACE_DIR, exist_ok=True)
_call_count = 0


def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> dict:
    """
    Reward function for verifying math answers.

    Args:
        queries: Complete text sequences containing prompts and responses
        prompts: Input prompt sequences
        labels: Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict with rewards, scores, and extra_logs
    """
    global _call_count
    _call_count += 1

    rewards = []
    n_boxed_found = 0

    # Pick one sample per batch to log fully
    sample_idx = 0
    sample_trace = None

    for i, (query, prompt, label) in enumerate(zip(queries, prompts, labels)):
        # Extract the response part (after the prompt)
        if isinstance(prompt, str) and prompt in query:
            response = query[len(prompt) :]
        else:
            response = query

        # Extract and grade the answer (only boxed format)
        pred_answer = extract_boxed_answer(response)
        is_correct = grade_answer(pred_answer, label)
        rewards.append(1.0 if is_correct else 0.0)

        if pred_answer is not None:
            n_boxed_found += 1

        # Log one sample per batch to stdout
        if i == sample_idx:
            print(f"[Math Reward] Response (last 500 chars): ...{response[-500:]}")
            print(f"[Math Reward] Pred: {pred_answer}, Gold: {label}, Match: {is_correct}")
            sample_trace = {
                "prompt": prompt,
                "response": response,
                "pred_answer": pred_answer,
                "gold": str(label),
                "correct": is_correct,
            }

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    accuracy = rewards_tensor.mean()
    boxed_rate = n_boxed_found / len(queries) if queries else 0.0

    print(f"[Math Reward] Batch: accuracy={accuracy:.3f}, boxed={n_boxed_found}/{len(queries)} ({boxed_rate:.1%})")

    # Write one full trace per batch
    if sample_trace is not None:
        trace_path = os.path.join(_TRACE_DIR, f"batch_{_call_count:04d}_{int(time.time())}.json")
        with open(trace_path, "w") as f:
            json.dump(sample_trace, f, indent=2)

    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {
            "math_accuracy": accuracy,
            "boxed_rate": boxed_rate,
        },
    }
