"""Tool-calling turn for GRPO training — single-class agent.

Implements ``AgentInstanceBase`` for multi-turn tool-calling rollouts.

The initial prompt arrives **already formatted** by ``--apply_chat_template``
in the preprocessing step.  ``reset()`` is a passthrough; only ``step()``
does real work (parse tool calls, execute, produce bridge text via
``protocol.render_tool_feedback``).
"""

import json
import os
import re
import torch
from typing import Any, Callable, Dict, Optional

from transformers import AutoTokenizer

from openrlhf.utils.tool_versions import get_version
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from openrlhf.utils.chat_protocol import GLMFlashProtocol, GPTOSSProtocol, InternS1Protocol


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

    def __init__(self, hf_tokenizer=None):
        # ---- tokenizer (needed by protocol parsers) ----
        if hf_tokenizer is not None:
            self.tokenizer = hf_tokenizer
        else:
            model_path = os.environ.get("OPENRLHF_MODEL_PATH")
            if not model_path:
                raise ValueError("OPENRLHF_MODEL_PATH environment variable must be set")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ---- protocol (parse + feedback only, not initial rendering) ----
        protocol_name = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "glm_flash")
        if protocol_name == "intern_s1":
            self.protocol = InternS1Protocol(self.tokenizer)
        elif protocol_name == "gpt_oss":
            self.protocol = GPTOSSProtocol(self.tokenizer)
        elif protocol_name in {"glm_flash", "qwen3"}:
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

        if tool_calls:
            # Execute tools and build result dicts
            tool_msgs = []
            for tc in tool_calls:
                result = await self._execute_tool(tc)
                tool_msgs.append({"name": tc.get("name", ""), "content": result})

            # Bridge text: close assistant turn + tool responses + open next turn
            feedback = self.protocol.render_tool_feedback(tool_msgs)
            return {
                "environment_feedback": feedback,
                "rewards": torch.tensor(0.0),
                "done": False,
                "scores": 0.0,
                "extra_logs": {"tool_call_count": len(tool_calls)},
            }

        # No tool calls → final answer
        reward = self._compute_reward(action.get("content", ""), label)
        return {
            "environment_feedback": "",
            "rewards": torch.tensor(reward),
            "done": True,
            "scores": reward,
            "extra_logs": {"tool_call_count": 0},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        if tool_name not in self.tools:
            return json.dumps({
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()),
            })
        try:
            result = self.tools[tool_name](**arguments)
            return json.dumps({
                "result": result,
                "function_name": tool_name,
                "arguments": arguments,
            })
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "function_name": tool_name,
                "arguments": arguments,
            })

    _ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-Za-z])\s*\)?")
    _PAREN_ANSWER_RE = re.compile(r"\(\s*([A-Za-z])\s*\)")

    def _compute_reward(self, generated_text: str, label: Optional[str]) -> float:
        """Reward = 1.0 iff the model's Answer: (X) after </think> matches the label."""
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
    def __init__(self):
        super().__init__(ToolCallingTurn)


__all__ = ["ToolCallingTurn", "AgentExecutor"]
