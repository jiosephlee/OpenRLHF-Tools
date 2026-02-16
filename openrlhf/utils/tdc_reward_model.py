"""
TDC Reward Model for GRPO Training.

Computes rewards for TDC molecular property prediction tasks based on
whether the model's final answer matches the ground truth label.

This is a simple accuracy-based reward model:
- Correct answer: reward = 1.0
- Incorrect answer: reward = 0.0

For more sophisticated reward models, consider:
- Partial credit for reasoning quality
- Tool usage efficiency bonuses
- Confidence calibration
"""

import re
from typing import List


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from model output.

    Looks for patterns like:
    - "Answer: (A)"
    - "Final answer: (B)"
    - Just "(A)" or "(B)" at the end

    Args:
        text: Model-generated text

    Returns:
        Extracted answer (e.g., "(A)") or empty string if not found
    """
    # Pattern 1: "Answer: (A)" or "Answer:(A)"
    match = re.search(r"Answer:\s*\(([AB])\)", text, re.IGNORECASE)
    if match:
        return f"({match.group(1)})"

    # Pattern 2: "Final answer: (B)"
    match = re.search(r"Final\s+answer:\s*\(([AB])\)", text, re.IGNORECASE)
    if match:
        return f"({match.group(1)})"

    # Pattern 3: Last occurrence of (A) or (B)
    matches = list(re.finditer(r"\(([AB])\)", text))
    if matches:
        return f"({matches[-1].group(1)})"

    return ""


def compute_reward(generated_text: str, label: str) -> float:
    """
    Compute reward for a single generated response.

    Args:
        generated_text: Model-generated text (full conversation)
        label: Ground truth label (e.g., "(A)" or "(B)")

    Returns:
        Reward score (1.0 if correct, 0.0 if incorrect)
    """
    predicted = extract_final_answer(generated_text)

    if not predicted:
        # No answer found - assign zero reward
        return 0.0

    # Exact match (case-insensitive)
    if predicted.upper() == label.upper():
        return 1.0
    else:
        return 0.0


class TDCRewardModel:
    """
    Reward model for TDC datasets.

    This is a stateless reward model that computes rewards based on
    exact answer matching.
    """

    def __init__(self):
        """Initialize TDC reward model."""
        pass

    def get_reward(
        self,
        queries: List[str],
        responses: List[str],
        labels: List[str],
    ) -> List[float]:
        """
        Compute rewards for a batch of responses.

        Args:
            queries: List of input prompts (not used for TDC)
            responses: List of model-generated responses
            labels: List of ground truth labels

        Returns:
            List of reward scores (one per response)
        """
        rewards = []
        for response, label in zip(responses, labels):
            reward = compute_reward(response, label)
            rewards.append(reward)

        return rewards


# Factory function for OpenRLHF integration
def get_reward_model():
    """
    Factory function to create reward model instance.

    This is the entry point called by OpenRLHF when
    --remote_rm_url points to this file.
    """
    return TDCRewardModel()


# For testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("Let me think... Answer: (A)", "(A)", 1.0),
        ("After analysis, the answer is (B)", "(B)", 1.0),
        ("I conclude (A) is correct", "(A)", 1.0),
        ("Answer: (B)", "(A)", 0.0),  # Wrong answer
        ("No clear answer here", "(A)", 0.0),  # No answer
        ("n\n**Answer: (B)**", "(B)", 1.0),  # Markdown bold format
    ]

    print("Testing TDC Reward Model")
    print("=" * 60)

    for i, (text, label, expected) in enumerate(test_cases, 1):
        reward = compute_reward(text, label)
        status = "✓" if reward == expected else "✗"
        print(f"{status} Test {i}: reward={reward:.1f} (expected {expected:.1f})")
        print(f"  Text: {text[:50]}...")
        print(f"  Label: {label}")
        print()

    print("=" * 60)
    print("All tests passed!" if all(
        compute_reward(text, label) == expected
        for text, label, expected in test_cases
    ) else "Some tests failed!")
