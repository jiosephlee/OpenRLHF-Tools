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
import sys
import torch
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Extern tools: add Intern-S1-recipe to sys.path so `from tools import ...`
# resolves to Intern-S1-recipe/tools/__init__.py.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_INTERN_S1_ROOT = _PROJECT_ROOT / "Intern-S1-recipe"
assert (_INTERN_S1_ROOT / "tools").is_dir(), (
    f"Intern-S1-recipe/tools not found at {_INTERN_S1_ROOT}/tools. "
    f"Run: git submodule update --init Intern-S1-recipe"
)
if str(_INTERN_S1_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERN_S1_ROOT))

from tools import BASIC_TOOLS, get_function_by_name

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from openrlhf.utils.chat_protocol import GLMFlashProtocol, InternS1Protocol


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

    def __init__(self):
        # ---- tokenizer (needed by protocol parsers) ----
        model_path = os.environ.get("OPENRLHF_MODEL_PATH")
        if not model_path:
            raise ValueError("OPENRLHF_MODEL_PATH environment variable must be set")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ---- protocol (parse + feedback only, not initial rendering) ----
        protocol_name = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "glm_flash")
        if protocol_name == "intern_s1":
            self.protocol = InternS1Protocol(self.tokenizer)
        else:
            self.protocol = GLMFlashProtocol(self.tokenizer)

        # ---- tool callables ----
        self.tools: Dict[str, Callable] = {}
        for tool_spec in BASIC_TOOLS:
            if isinstance(tool_spec, dict) and "function" in tool_spec:
                func_name = tool_spec["function"]["name"]
                func = get_function_by_name(func_name)
                if func:
                    self.tools[func_name] = func

    # ------------------------------------------------------------------
    # AgentInstanceBase interface
    # ------------------------------------------------------------------

    async def reset(self, states: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Passthrough — prompt is already chat-templated by preprocessing."""
        return {"observation": states.get("observation", "")}

    async def step(self, state_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Parse tool calls, execute, return upstream-contract dict."""
        action_text = state_dict["action_text"]
        label = state_dict.get("label", "")

        action = self.protocol.parse_assistant_text(action_text)
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
            "extra_logs": {},
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

    _ANSWER_RE = re.compile(r"Answer\s*:\s*\(\s*([A-Za-z])\s*\)")

    def _compute_reward(self, generated_text: str, label: Optional[str]) -> float:
        """Reward = 1.0 iff the model's Answer: (X) after </think> matches the label."""
        if not label:
            return 0.0

        # Only consider text after the closing </think> tag
        think_end = generated_text.find("</think>")
        if think_end == -1:
            return 0.0
        answer_region = generated_text[think_end:]

        match = self._ANSWER_RE.search(answer_region)
        if not match:
            return 0.0

        pred = match.group(1).upper()

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
