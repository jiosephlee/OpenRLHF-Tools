"""Tool-calling turn for GRPO training — single-class agent.

Implements ``AgentInstanceBase`` for multi-turn tool-calling rollouts.

The initial prompt arrives **already formatted** by ``--apply_chat_template``
in the preprocessing step.  ``reset()`` is a passthrough; only ``step()``
does real work (parse tool calls, execute, produce bridge text via
``protocol.render_tool_feedback``).
"""

import asyncio
import json
import os
import re
import time
import torch
from typing import Any, Callable, Dict, Optional

from transformers import AutoTokenizer

from openrlhf.utils.tool_versions import get_version
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from openrlhf.utils.chat_protocol import GLMFlashProtocol, GPTOSSProtocol, InternS1Protocol, Qwen3Protocol


class ToolCallingTurn(AgentInstanceBase):
    """One trajectory of a tool-calling agent.

    Each instance owns its own conversation history and is created fresh by
    ``MultiTurnAgentExecutor.execute`` for every rollout.

    The prompt is expected to arrive already chat-templated (via
    ``--apply_chat_template`` in the data preprocessing).  The protocol is
    only used for two things:
      - ``parse_assistant_text``: extract tool calls from vLLM output
      - ``render_tool_feedback``: produce the bridge text between turns
    """

    def __init__(self, hf_tokenizer=None, reward_fn=None):
        # ---- tokenizer (needed by protocol parsers) ----
        if hf_tokenizer is not None:
            self.tokenizer = hf_tokenizer
        else:
            model_path = os.environ.get("OPENRLHF_MODEL_PATH")
            if not model_path:
                raise ValueError("OPENRLHF_MODEL_PATH environment variable must be set")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ---- optional injectable reward function ----
        # Signature: (generated_text: str, label: str) -> float
        # Falls back to _default_reward_fn (A/B letter extraction) when None.
        self._reward_fn = reward_fn

        # ---- protocol (parse + feedback only, not initial rendering) ----
        protocol_name = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "glm_flash")
        if protocol_name == "intern_s1":
            self.protocol = InternS1Protocol(self.tokenizer)
        elif protocol_name == "gpt_oss":
            self.protocol = GPTOSSProtocol(self.tokenizer)
        elif protocol_name == "qwen3":
            self.protocol = Qwen3Protocol(self.tokenizer)
        elif protocol_name == "glm_flash":
            self.protocol = GLMFlashProtocol(self.tokenizer)
        else:
            raise ValueError(f"Unsupported chat protocol: {protocol_name}")

        # ---- tool callables (from version registry) ----
        tool_version = os.environ.get("OPENRLHF_TOOL_VERSION")
        if not tool_version:
            raise ValueError(
                "OPENRLHF_TOOL_VERSION environment variable must be set "
                "(e.g. v1, v2, v3, v4). Pass --tool_version to train_ppo_ray.py."
            )
        ver_cfg = get_version(tool_version)
        self.tools: Dict[str, Callable] = dict(ver_cfg["callables"])

    # ------------------------------------------------------------------
    # AgentInstanceBase interface
    # ------------------------------------------------------------------

    async def reset(self, states: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Passthrough — prompt is already chat-templated by preprocessing."""
        return {"observation": states.get("observation", "")}

    async def step(self, state_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Parse tool calls, execute, return upstream-contract dict."""
        action_text = state_dict["action_text"]
        action_token_ids = state_dict.get("action_token_ids")
        label = state_dict.get("label", "")
        action = self.protocol.parse_assistant_text(action_text, token_ids=action_token_ids)
        tool_calls = action.get("tool_calls", [])
        base_logs = {
            "tool_call_count": len(tool_calls),
        }
        if "parse_method" in action:
            parse_method = action["parse_method"]
            parse_failed = action.get("parse_failed", False)
            base_logs[f"parse_method__{parse_method}"] = 1
            base_logs["parse_failed"] = 1 if parse_failed else 0
            base_logs["tool_call_attempted"] = 1 if (len(tool_calls) > 0 or parse_failed) else 0

        if tool_calls:
            # Execute all tool calls in parallel, preserving order
            results = await asyncio.gather(*[self._execute_tool(tc) for tc in tool_calls])
            tool_msgs = []
            extra_logs = base_logs.copy()
            # Timing stats: each result carries its own duration
            durations = [r[1] for r in results]
            extra_logs["tool_time_total"] = sum(durations)
            extra_logs["tool_time_max_call"] = max(durations)
            for tc, (result, dur) in zip(tool_calls, results):
                tool_name = tc.get("name", "")
                tool_msgs.append({"name": tool_name, "content": result})
                if tool_name:
                    key = f"tool_count__{tool_name}"
                    extra_logs[key] = extra_logs.get(key, 0) + 1

            # Bridge text: close assistant turn + tool responses + open next turn
            feedback = self.protocol.render_tool_feedback(tool_msgs)
            feedback_token_ids = self.protocol.render_tool_feedback_token_ids(tool_msgs)

            #### small reward for well-formatted tool calls (harmony > regex > unparsed) ####
            # parse_method is only set by GPTOSSProtocol; None means a
            # non-GPT-OSS protocol parsed successfully — no bonus/penalty.
            parse_method = action.get("parse_method")
            if parse_method is None:
                # Non-GPT-OSS protocol: no format shaping
                format_reward = 0
            elif parse_method == "primary":
                # Best case: harmony token-ID parser succeeded on first try
                format_reward = 0.0015
            elif parse_method == "fallback":
                # Harmony succeeded after prepending assistant header — neutral
                format_reward = 0
            elif parse_method == "regex":
                # Had to fall back to regex — mild penalty
                format_reward = -0.0015
            else:
                raise ValueError(f"Unknown parse method: {parse_method}")
            extra_logs["format_reward"] = format_reward
            #### end small reward for well-formatted tool calls ####

            return {
                "environment_feedback": feedback,
                "environment_feedback_token_ids": feedback_token_ids,
                "rewards": torch.tensor(format_reward),
                "done": False,
                "scores": 0.0,
                "extra_logs": extra_logs,
            }

        # Parse failed: model attempted a tool call but mangled the format.
        # Apply an explicit penalty to discourage malformed harmony headers.
        parse_failed = action.get("parse_failed", False)
        if parse_failed:
            base_logs["format_reward"] = -0.0025
            return {
                "environment_feedback": "",
                "rewards": torch.tensor(-0.0025),
                "done": True,
                "scores": 0.0,
                "extra_logs": base_logs,
            }

        # No tool calls → final answer
        generated_text = action.get("content", "")
        if self._reward_fn is not None:
            reward = self._reward_fn(generated_text, label)
        else:
            reward = self._default_reward_fn(generated_text, label)
        return {
            "environment_feedback": "",
            "rewards": torch.tensor(reward),
            "done": True,
            "scores": reward,
            "extra_logs": base_logs,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> tuple:
        """Execute a tool call and return (result_json, duration_seconds)."""
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        t0 = time.monotonic()
        if tool_name not in self.tools:
            result = json.dumps(
                {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": list(self.tools.keys()),
                }
            )
        else:
            try:
                result = json.dumps(
                    {
                        "result": self.tools[tool_name](**arguments),
                        "function_name": tool_name,
                        "arguments": arguments,
                    }
                )
            except Exception as e:
                result = json.dumps(
                    {
                        "error": str(e),
                        "function_name": tool_name,
                        "arguments": arguments,
                    }
                )
        return result, time.monotonic() - t0

    _ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-Za-z])\s*\)?")
    _PAREN_ANSWER_RE = re.compile(r"\(\s*([A-Za-z])\s*\)")

    def _default_reward_fn(self, generated_text: str, label: Optional[str]) -> float:
        """Default reward: 1.0 iff the model's Answer: (X) after </think> matches the label."""
        if not label:
            return 0.0

        # Prefer the post-think region when present; otherwise evaluate full text.
        think_end = generated_text.find("</think>")
        answer_region = generated_text[think_end:] if think_end != -1 else generated_text

        match = self._ANSWER_RE.search(answer_region)
        if match:
            pred = match.group(1).upper()
        else:
            # GPT-OSS frequently emits bare "(A)"/"(B)" without "Answer:" prefix.
            paren_matches = self._PAREN_ANSWER_RE.findall(answer_region)
            if paren_matches:
                pred = paren_matches[-1].upper()
            else:
                stripped = answer_region.strip()
                if len(stripped) == 1 and stripped.isalpha():
                    pred = stripped.upper()
                else:
                    return 0.0

        # Extract letter from label too (handles "A", "(A)", "Answer: (A)", etc.)
        label_match = self._ANSWER_RE.search(label)
        if label_match:
            gold = label_match.group(1).upper()
        else:
            # Bare letter like "A" or "(A)"
            gold = label.strip().strip("()").upper()

        return 1.0 if pred == gold else 0.0


# ---------------------------------------------------------------------------
# Executor (required name for vllm_engine._load_agent_executor)
# ---------------------------------------------------------------------------
class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self, reward_fn=None):
        super().__init__(ToolCallingTurn, reward_fn=reward_fn)


__all__ = ["ToolCallingTurn", "AgentExecutor"]
