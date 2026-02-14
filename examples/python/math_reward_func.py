"""Math reward function for verifying mathematical answers.

Only supports \\boxed{answer} LaTeX format for answer extraction.
"""

import json
import os
from typing import List

import torch

from openrlhf.utils import extract_boxed_answer, grade_answer

# Single trace file — one line per batch, appended
_TRACE_FILE = os.environ.get("MATH_REWARD_TRACE_FILE", "/tmp/math_reward_traces.jsonl")


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
    rewards = []
    n_boxed_found = 0
    first_trace = None

    for i, (query, prompt, label) in enumerate(zip(queries, prompts, labels)):
        if isinstance(prompt, str) and prompt in query:
            response = query[len(prompt) :]
        else:
            response = query

        pred_answer = extract_boxed_answer(response)
        is_correct = grade_answer(pred_answer, label)
        r = 1.0 if is_correct else (0.25 if pred_answer is not None else 0.0)
        rewards.append(r)

        if pred_answer is not None:
            n_boxed_found += 1

        # Save first sample per batch
        if i == 0:
            first_trace = {
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

    # Append one trace per batch to single file
    if first_trace is not None:
        with open(_TRACE_FILE, "a") as f:
            f.write(json.dumps(first_trace) + "\n")

    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {
            "math_accuracy": accuracy,
            "boxed_rate": boxed_rate,
        },
    }
