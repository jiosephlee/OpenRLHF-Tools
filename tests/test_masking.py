"""Standalone test for multi-turn token-level masking.

Tests the full pipeline WITHOUT any Ray, vLLM, or GPU dependencies:
  1. MultiTurnAgentExecutor.execute() with a mock vLLM engine
  2. action_ranges tracking across multiple turns
  3. action_mask construction (same logic as _process_response_into_experience)
  4. Verifies only LLM-generated tokens have mask=1

Usage:
    python tests/test_masking.py          # Run with visualization
    pytest tests/test_masking.py -v       # Run as pytest
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Inlined executor classes (from openrlhf/utils/agent.py, no deps needed)
# ═══════════════════════════════════════════════════════════════════════════


class AgentInstanceBase(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    async def reset(self, states: dict, **kwargs):
        return states

    @abstractmethod
    async def step(self, state_dict: dict, **kwargs):
        raise NotImplementedError


class MultiTurnAgentExecutor:
    """Inlined from openrlhf/utils/agent.py (lines 31-142).

    Identical logic — copied here to avoid openrlhf import chain
    that pulls in pylatexenc, ray, vllm, etc.
    """

    def __init__(self, agent_instance_cls):
        self.agent_instance_cls = agent_instance_cls

    async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine):
        agent_instance = self.agent_instance_cls()

        initial_states = {"observation": prompt, "label": label}
        reset_result = await agent_instance.reset(initial_states)
        observation_text = reset_result["observation"]

        current_obs_tokens = hf_tokenizer(observation_text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].tolist()

        min_generation_tokens = sampling_params.max_tokens if hasattr(sampling_params, "max_tokens") else 1
        max_initial_length = max_length - min_generation_tokens
        if len(current_obs_tokens) > max_initial_length:
            current_obs_tokens = current_obs_tokens[-max_initial_length:]
            observation_text = hf_tokenizer.decode(current_obs_tokens, skip_special_tokens=False)

        action_ranges = []
        total_reward = 0
        final_scores = 0
        extra_logs = {}

        if sampling_params.logprobs is not None:
            rollout_log_probs = [0.0] * len(current_obs_tokens)
        else:
            rollout_log_probs = None

        while True:
            sampling_params.max_tokens = max_length - len(current_obs_tokens)
            if sampling_params.max_tokens <= 0:
                break

            request_output = await llm_engine.generate(current_obs_tokens, deepcopy(sampling_params))
            action_tokens = request_output.outputs[0].token_ids
            action_text = request_output.outputs[0].text

            action_start = len(current_obs_tokens)
            action_end = action_start + len(action_tokens)
            action_ranges.append((action_start, action_end))

            states = {
                "observation_text": observation_text,
                "action_text": action_text,
                "label": label,
                "sampling_params": sampling_params,
            }
            step_result = await agent_instance.step(states)

            total_reward += step_result["rewards"].item()
            final_scores = step_result.get("scores", total_reward)
            environment_feedback_text = step_result["environment_feedback"]
            done = step_result["done"]
            extra_logs = step_result.get("extra_logs", {})

            observation_text = observation_text + action_text + environment_feedback_text
            current_obs_tokens = (
                current_obs_tokens
                + action_tokens
                + hf_tokenizer(environment_feedback_text, add_special_tokens=False, return_tensors="pt")["input_ids"][
                    0
                ].tolist()
            )

            if sampling_params.logprobs is not None:
                for i, logprob in enumerate(request_output.outputs[0].logprobs):
                    rollout_log_probs.append(logprob[action_tokens[i]].logprob)
                rollout_log_probs.extend([0.0] * (len(current_obs_tokens) - len(rollout_log_probs)))

            if step_result.get("sampling_params", None):
                sampling_params = step_result["sampling_params"]

            if done:
                break

        return {
            "prompt": prompt,
            "label": label,
            "reward": total_reward,
            "scores": final_scores,
            "observation_tokens": current_obs_tokens,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "extra_logs": extra_logs,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Mock objects (replaces vLLM engine, tokenizer, sampling params)
# ═══════════════════════════════════════════════════════════════════════════


class MockLogProb:
    """Mimics vLLM's Logprob object."""

    def __init__(self, logprob: float):
        self.logprob = logprob


class MockCompletionOutput:
    """Mimics vLLM's CompletionOutput."""

    def __init__(self, token_ids: List[int], text: str, logprobs: Optional[List[dict]] = None):
        self.token_ids = token_ids
        self.text = text
        self.logprobs = logprobs
        self.finish_reason = "stop"


class MockRequestOutput:
    """Mimics vLLM's RequestOutput."""

    def __init__(self, output: MockCompletionOutput):
        self.outputs = [output]


class MockTokenizer:
    """Simple word-level tokenizer for testing.

    Splits text on whitespace, assigns stable integer IDs.
    """

    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._id2word: Dict[int, str] = {}
        self._next_id = 1

    def _get_or_assign(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._id2word[self._next_id] = word
            self._next_id += 1
        return self._vocab[word]

    def __call__(self, text: str, add_special_tokens=False, return_tensors=None):
        words = text.split()
        token_ids = [self._get_or_assign(w) for w in words]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        return {"input_ids": [token_ids]}

    def decode(self, token_ids: List[int], skip_special_tokens=False) -> str:
        return " ".join(self._id2word.get(tid, f"<unk:{tid}>") for tid in token_ids)


@dataclass
class MockSamplingParams:
    """Mimics vLLM's SamplingParams."""

    max_tokens: int = 256
    logprobs: Optional[int] = 1
    temperature: float = 0.7


class MockVLLMEngine:
    """Dummy vLLM engine that returns scripted responses in order."""

    def __init__(self, tokenizer: MockTokenizer, scripted_responses: List[str]):
        self.tokenizer = tokenizer
        self.responses = list(scripted_responses)
        self._call_idx = 0

    async def generate(self, input_token_ids: List[int], sampling_params) -> MockRequestOutput:
        assert self._call_idx < len(self.responses), (
            f"MockVLLMEngine exhausted: called {self._call_idx + 1} times but only "
            f"{len(self.responses)} responses scripted"
        )
        text = self.responses[self._call_idx]
        self._call_idx += 1

        token_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        logprobs = [{tid: MockLogProb(-0.5)} for tid in token_ids]

        output = MockCompletionOutput(token_ids=token_ids, text=text, logprobs=logprobs)
        return MockRequestOutput(output)


# ═══════════════════════════════════════════════════════════════════════════
# Dummy Agents
# ═══════════════════════════════════════════════════════════════════════════


class DummyAgent(AgentInstanceBase):
    """Simulates tool calling: detects "tool_call" / "calculate" in LLM output."""

    def __init__(self):
        self.turn = 0

    async def reset(self, states: dict, **kwargs) -> dict:
        self.turn = 0
        prompt = states.get("observation", "")
        formatted = f"[SYS] You_are_helpful [/SYS] [USR] {prompt} [/USR] [AST]"
        return {"observation": formatted}

    async def step(self, state_dict: dict, **kwargs) -> dict:
        self.turn += 1
        action_text = state_dict["action_text"]

        if "tool_call" in action_text.lower() or "calculate" in action_text.lower():
            feedback = " [OBS] result=42 [/OBS] [AST]"
            return {
                "environment_feedback": feedback,
                "rewards": torch.tensor(0.0),
                "done": False,
                "scores": 0.0,
            }
        else:
            return {
                "environment_feedback": "",
                "rewards": torch.tensor(1.0),
                "done": True,
                "scores": 1.0,
            }


# ═══════════════════════════════════════════════════════════════════════════
# Masking logic (extracted from experience_maker._process_response_into_experience)
# ═══════════════════════════════════════════════════════════════════════════


def build_action_mask(observation_tokens: List[int], action_ranges: List[tuple],
                      truncate_length: int) -> Dict[str, torch.Tensor]:
    """Replicate the masking logic from experience_maker.py:472-489."""
    sequences = torch.tensor(observation_tokens, dtype=torch.long)
    attention_mask = torch.ones(len(observation_tokens), dtype=torch.long)

    # Mark action spans
    raw_action_mask = torch.zeros_like(attention_mask)
    for start, end in action_ranges:
        raw_action_mask[start:end] = 1

    # Truncate
    sequences = sequences[:truncate_length]
    attention_mask = attention_mask[:truncate_length]
    # Shift by 1 (aligns with next-token prediction: position i predicts token i+1)
    action_mask = raw_action_mask[1:truncate_length]

    return {
        "sequences": sequences,
        "attention_mask": attention_mask,
        "action_mask": action_mask,
        "raw_action_mask": raw_action_mask,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════


def visualize_masking(tokenizer: MockTokenizer, observation_tokens: List[int],
                      action_ranges: List[tuple], action_mask: torch.Tensor,
                      raw_action_mask: torch.Tensor):
    """Print a visual representation of the token masking."""
    print("\n" + "=" * 90)
    print("TOKEN-LEVEL MASKING VISUALIZATION")
    print("=" * 90)

    print(f"\nTotal tokens: {len(observation_tokens)}")
    print(f"Action ranges (raw): {action_ranges}")
    print(f"Action tokens (mask=1): {int(action_mask.sum().item())}")
    print(f"Non-action tokens (mask=0): {int((action_mask == 0).sum().item())}")

    # Determine region labels
    regions = ["prompt"] * len(observation_tokens)
    for i, (start, end) in enumerate(action_ranges):
        for pos in range(start, min(end, len(observation_tokens))):
            regions[pos] = f"ACTION_{i + 1}"
        if i < len(action_ranges) - 1:
            next_start = action_ranges[i + 1][0]
            for pos in range(end, min(next_start, len(observation_tokens))):
                regions[pos] = f"OBS_{i + 1}"

    print(f"\n{'Pos':>4} {'TokID':>6} {'Word':<20} {'Raw':>4} {'Shifted':>8} {'Region':<15} {'Note'}")
    print("-" * 90)

    for pos, tid in enumerate(observation_tokens):
        word = tokenizer._id2word.get(tid, f"<unk:{tid}>")
        raw = int(raw_action_mask[pos].item()) if pos < len(raw_action_mask) else "-"
        shifted_idx = pos - 1
        shifted = int(action_mask[shifted_idx].item()) if 0 <= shifted_idx < len(action_mask) else "-"
        region = regions[pos]

        note = ""
        if region.startswith("ACTION"):
            note = "<-- loss ON"
        elif region.startswith("OBS"):
            note = "<-- loss OFF"
        elif region == "prompt":
            note = "<-- loss OFF"

        print(f"{pos:>4} {tid:>6} {word:<20} {raw!s:>4} {shifted!s:>8} {region:<15} {note}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_two_turn_masking():
    """2-turn: tool_call -> observation -> final_answer."""
    tokenizer = MockTokenizer()

    scripted_responses = [
        "I will calculate <tool_call> add a=2 b=3 </tool_call>",
        "The answer is 42",
    ]

    engine = MockVLLMEngine(tokenizer, scripted_responses)
    executor = MultiTurnAgentExecutor(DummyAgent)

    response = asyncio.run(
        executor.execute(
            prompt="What is 2 + 3?",
            label="42",
            sampling_params=MockSamplingParams(),
            max_length=512,
            hf_tokenizer=tokenizer,
            llm_engine=engine,
        )
    )

    observation_tokens = response["observation_tokens"]
    action_ranges = response["action_ranges"]
    rollout_log_probs = response["rollout_log_probs"]

    # ── Assertions on action_ranges ──────────────────────────────────────
    assert len(action_ranges) == 2, f"Expected 2 action ranges, got {len(action_ranges)}"

    # Ranges are ordered and non-overlapping
    for i in range(len(action_ranges) - 1):
        assert action_ranges[i][1] <= action_ranges[i + 1][0], (
            f"Ranges overlap: {action_ranges[i]} and {action_ranges[i + 1]}"
        )

    # ── Build mask ───────────────────────────────────────────────────────
    result = build_action_mask(observation_tokens, action_ranges, truncate_length=512)
    action_mask = result["action_mask"]
    raw_action_mask = result["raw_action_mask"]

    visualize_masking(tokenizer, observation_tokens, action_ranges, action_mask, raw_action_mask)

    # ── Verify mask correctness ──────────────────────────────────────────
    assert action_mask.sum() > 0, "Action mask is all zeros!"

    prompt_end = action_ranges[0][0]
    obs_start = action_ranges[0][1]
    obs_end = action_ranges[1][0]

    # 1. Prompt tokens → mask=0
    if prompt_end > 1:
        prompt_mask = action_mask[:prompt_end - 1]
        assert prompt_mask.sum() == 0, f"Prompt tokens should be masked out"

    # 2. Observation tokens (between actions) → mask=0
    if obs_end > obs_start:
        obs_mask = action_mask[obs_start - 1 : obs_end - 1]
        assert obs_mask.sum() == 0, (
            f"Observation tokens [{obs_start}:{obs_end}) should be masked out, "
            f"but got: {obs_mask.tolist()}"
        )

    # 3. Action_1 → mask=1
    a1_start, a1_end = action_ranges[0]
    a1_mask = action_mask[a1_start - 1 : a1_end - 1]
    assert (a1_mask == 1).all(), f"Action_1 tokens should all be 1, got {a1_mask.tolist()}"

    # 4. Action_2 → mask=1
    a2_start, a2_end = action_ranges[1]
    a2_mask = action_mask[a2_start - 1 : a2_end - 1]
    assert (a2_mask == 1).all(), f"Action_2 tokens should all be 1, got {a2_mask.tolist()}"

    # 5. Total action tokens matches range sizes
    total_action_len = sum(end - start for start, end in action_ranges)
    assert int(action_mask.sum().item()) == total_action_len, (
        f"Expected {total_action_len} action tokens, got {int(action_mask.sum().item())}"
    )

    # ── Verify rollout_log_probs ─────────────────────────────────────────
    assert rollout_log_probs is not None
    assert len(rollout_log_probs) == len(observation_tokens)

    # Prompt tokens → logprob=0.0
    for i in range(prompt_end):
        assert rollout_log_probs[i] == 0.0, f"Prompt logprob[{i}] should be 0.0"

    # Action tokens → logprob=-0.5 (from mock)
    for start, end in action_ranges:
        for i in range(start, end):
            assert rollout_log_probs[i] == -0.5, f"Action logprob[{i}] should be -0.5"

    # Observation feedback tokens → logprob=0.0
    for i in range(obs_start, obs_end):
        assert rollout_log_probs[i] == 0.0, f"Obs logprob[{i}] should be 0.0"

    print("PASSED: test_two_turn_masking")


def test_single_turn_no_tool():
    """Single turn: just a final answer, no tool calls."""
    tokenizer = MockTokenizer()
    engine = MockVLLMEngine(tokenizer, ["The answer is 42"])
    executor = MultiTurnAgentExecutor(DummyAgent)

    response = asyncio.run(
        executor.execute(
            prompt="What is the meaning of life?",
            label="42",
            sampling_params=MockSamplingParams(),
            max_length=512,
            hf_tokenizer=tokenizer,
            llm_engine=engine,
        )
    )

    action_ranges = response["action_ranges"]
    assert len(action_ranges) == 1, f"Expected 1 action range, got {len(action_ranges)}"

    result = build_action_mask(response["observation_tokens"], action_ranges, truncate_length=512)
    action_mask = result["action_mask"]

    prompt_end = action_ranges[0][0]
    if prompt_end > 1:
        assert action_mask[:prompt_end - 1].sum() == 0, "Prompt should be masked out"

    a_start, a_end = action_ranges[0]
    assert (action_mask[a_start - 1 : a_end - 1] == 1).all(), "Action should be mask=1"

    visualize_masking(tokenizer, response["observation_tokens"], action_ranges,
                      result["action_mask"], result["raw_action_mask"])

    print("PASSED: test_single_turn_no_tool")


def test_three_turn_masking():
    """3-turn: tool_call -> obs -> tool_call -> obs -> final_answer."""
    tokenizer = MockTokenizer()

    scripted_responses = [
        "Let me calculate <tool_call> get_qed smiles=CCO </tool_call>",
        "Now check weight <tool_call> get_mw smiles=CCO </tool_call>",
        "The QED is 0.85 and MW is 46.07",
    ]

    class ThreeTurnAgent(AgentInstanceBase):
        def __init__(self):
            self.turn = 0

        async def reset(self, states, **kwargs):
            self.turn = 0
            prompt = states.get("observation", "")
            return {"observation": f"[SYS] Chem [/SYS] [USR] {prompt} [/USR] [AST]"}

        async def step(self, state_dict, **kwargs):
            self.turn += 1
            if "tool_call" in state_dict["action_text"]:
                return {
                    "environment_feedback": f" [OBS] result_t{self.turn} [/OBS] [AST]",
                    "rewards": torch.tensor(0.0),
                    "done": False,
                    "scores": 0.0,
                }
            return {
                "environment_feedback": "",
                "rewards": torch.tensor(1.0),
                "done": True,
                "scores": 1.0,
            }

    engine = MockVLLMEngine(tokenizer, scripted_responses)
    executor = MultiTurnAgentExecutor(ThreeTurnAgent)

    response = asyncio.run(
        executor.execute(
            prompt="Analyze CCO",
            label="0.85",
            sampling_params=MockSamplingParams(),
            max_length=1024,
            hf_tokenizer=tokenizer,
            llm_engine=engine,
        )
    )

    action_ranges = response["action_ranges"]
    assert len(action_ranges) == 3, f"Expected 3 action ranges, got {len(action_ranges)}"

    result = build_action_mask(response["observation_tokens"], action_ranges, truncate_length=1024)
    action_mask = result["action_mask"]

    # Verify observations between actions are masked out
    for i in range(len(action_ranges) - 1):
        obs_s = action_ranges[i][1]
        obs_e = action_ranges[i + 1][0]
        if obs_e > obs_s:
            obs_mask = action_mask[obs_s - 1 : obs_e - 1]
            assert obs_mask.sum() == 0, (
                f"Obs between action_{i+1} and action_{i+2} should be masked, got {obs_mask.tolist()}"
            )

    # Verify all action tokens are mask=1
    for idx, (start, end) in enumerate(action_ranges):
        a_mask = action_mask[start - 1 : end - 1]
        assert (a_mask == 1).all(), f"Action_{idx+1} should all be 1, got {a_mask.tolist()}"

    total_action_len = sum(end - start for start, end in action_ranges)
    assert int(action_mask.sum().item()) == total_action_len

    visualize_masking(tokenizer, response["observation_tokens"], action_ranges,
                      result["action_mask"], result["raw_action_mask"])

    print("PASSED: test_three_turn_masking")


def test_reward_accumulation():
    """Test that rewards accumulate correctly across turns."""
    tokenizer = MockTokenizer()

    class RewardAgent(AgentInstanceBase):
        def __init__(self):
            self.turn = 0

        async def reset(self, states, **kwargs):
            return {"observation": f"prompt: {states['observation']}"}

        async def step(self, state_dict, **kwargs):
            self.turn += 1
            if self.turn == 1:
                return {
                    "environment_feedback": " obs1 ",
                    "rewards": torch.tensor(0.3),
                    "done": False,
                }
            return {
                "environment_feedback": "",
                "rewards": torch.tensor(0.7),
                "done": True,
            }

    engine = MockVLLMEngine(tokenizer, ["tool_call step1", "final answer done"])
    executor = MultiTurnAgentExecutor(RewardAgent)

    response = asyncio.run(
        executor.execute(
            prompt="test",
            label="test",
            sampling_params=MockSamplingParams(),
            max_length=512,
            hf_tokenizer=tokenizer,
            llm_engine=engine,
        )
    )

    assert abs(response["reward"] - 1.0) < 1e-6, f"Expected total reward 1.0, got {response['reward']}"
    print("PASSED: test_reward_accumulation")


def test_logprob_alignment():
    """Verify rollout_log_probs length and values align token-by-token."""
    tokenizer = MockTokenizer()
    engine = MockVLLMEngine(tokenizer, [
        "call <tool_call> fn </tool_call>",
        "done answer",
    ])
    executor = MultiTurnAgentExecutor(DummyAgent)

    response = asyncio.run(
        executor.execute(
            prompt="Q",
            label="L",
            sampling_params=MockSamplingParams(logprobs=1),
            max_length=512,
            hf_tokenizer=tokenizer,
            llm_engine=engine,
        )
    )

    lps = response["rollout_log_probs"]
    obs = response["observation_tokens"]
    ranges = response["action_ranges"]

    assert len(lps) == len(obs), f"logprob len {len(lps)} != token len {len(obs)}"

    # Check structure: prompt=0, action=-0.5, obs=0, action=-0.5
    prompt_end = ranges[0][0]
    for i in range(prompt_end):
        assert lps[i] == 0.0, f"Prompt lp[{i}]={lps[i]}, want 0.0"

    for s, e in ranges:
        for i in range(s, e):
            assert lps[i] == -0.5, f"Action lp[{i}]={lps[i]}, want -0.5"

    if len(ranges) > 1:
        obs_s, obs_e = ranges[0][1], ranges[1][0]
        for i in range(obs_s, obs_e):
            assert lps[i] == 0.0, f"Obs lp[{i}]={lps[i]}, want 0.0"

    print("PASSED: test_logprob_alignment")


def test_empty_feedback_no_extra_tokens():
    """When done=True with empty feedback, no extra tokens should be appended."""
    tokenizer = MockTokenizer()
    engine = MockVLLMEngine(tokenizer, ["The answer"])
    executor = MultiTurnAgentExecutor(DummyAgent)

    response = asyncio.run(
        executor.execute(
            prompt="Q",
            label="L",
            sampling_params=MockSamplingParams(),
            max_length=512,
            hf_tokenizer=tokenizer,
            llm_engine=engine,
        )
    )

    # Prompt tokens + action tokens = total tokens (no feedback tokens added)
    prompt_len = response["action_ranges"][0][0]
    action_len = response["action_ranges"][0][1] - response["action_ranges"][0][0]
    total = len(response["observation_tokens"])

    # Empty string tokenization should produce 0 tokens
    empty_tokens = tokenizer("", return_tensors="pt")["input_ids"][0].tolist()
    expected_total = prompt_len + action_len + len(empty_tokens)
    assert total == expected_total, (
        f"Expected {expected_total} tokens (prompt={prompt_len} + action={action_len} + empty={len(empty_tokens)}), "
        f"got {total}"
    )

    print("PASSED: test_empty_feedback_no_extra_tokens")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("Running multi-turn masking tests...\n")

    test_single_turn_no_tool()
    test_two_turn_masking()
    test_three_turn_masking()
    test_reward_accumulation()
    test_logprob_alignment()
    test_empty_feedback_no_extra_tokens()

    print("\n" + "=" * 90)
    print("ALL TESTS PASSED")
    print("=" * 90)
