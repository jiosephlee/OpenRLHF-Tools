"""Tool-calling agent for GRPO training with GLM Flash model.

This agent uses AgentSession and ChatProtocol abstractions for clean,
maintainable multi-turn tool-calling interactions.

Features:
- Modular architecture (Session + Protocol)
- GLM Flash XML tool calling format
- RDKit molecular property tools
- Pluggable protocols via environment variable
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer

# Add Intern-S1-recipe to path so `from tools import ...` resolves to
# Intern-S1-recipe/tools/__init__.py (same approach as test_tdc_via_api_F1.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # openrlhf/utils -> openrlhf -> project root
_INTERN_S1_ROOT = _PROJECT_ROOT / "Intern-S1-recipe"
if str(_INTERN_S1_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERN_S1_ROOT))

from tools import BASIC_TOOLS, get_function_by_name

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from openrlhf.utils.agent_session import AgentSession
from openrlhf.utils.chat_protocol import GLMFlashProtocol


class ToolCallAgent(AgentInstanceBase):
    """Tool-calling agent instance using AgentSession abstraction.

    Each instance manages one trajectory with its own conversation history.
    Uses GLMFlashProtocol for format-specific rendering and parsing.
    """

    def __init__(self):
        """Initialize agent with tokenizer, protocol, and session."""
        # Load tokenizer from environment variable
        model_path = os.environ.get("OPENRLHF_MODEL_PATH")
        if not model_path:
            raise ValueError("OPENRLHF_MODEL_PATH environment variable must be set")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Create protocol
        protocol = GLMFlashProtocol(self.tokenizer)

        # Create tool functions dict
        tools = self._create_tools_dict()

        # Create session
        self.session = AgentSession(
            protocol=protocol,
            tools=tools,
            system_prompt=self._get_system_prompt()
        )

    async def reset(self, states: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Reset session and format initial prompt.

        Args:
            states: Dictionary with "observation" (prompt) and "label" keys

        Returns:
            Dictionary with "observation" key containing formatted prompt
        """
        prompt = states.get("observation", "")
        formatted_prompt = await self.session.initialize(prompt)

        return {"observation": formatted_prompt}

    async def step(self, state_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process one agent step.

        Args:
            state_dict: Dictionary with:
                - action_text: Raw LLM output
                - label: Ground truth label for reward

        Returns:
            Dictionary with:
                - environment_feedback: Observation text (empty if done)
                - rewards: Tensor with reward value
                - done: Boolean indicating episode termination
                - scores: Final score (same as reward)
                - extra_logs: Optional extra logging info
        """
        action_text = state_dict["action_text"]
        label = state_dict.get("label", "")

        # Delegate to session
        result = await self.session.step(action_text, label)

        return {
            "environment_feedback": result["feedback"],
            "rewards": torch.tensor(result["reward"]),
            "done": result["done"],
            "scores": result["reward"],
            "extra_logs": result.get("extra_logs", {})
        }

    def _create_tools_dict(self) -> Dict[str, callable]:
        """Create tools dictionary from BASIC_TOOLS registry.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        tools = {}

        for tool_spec in BASIC_TOOLS:
            if isinstance(tool_spec, dict) and "function" in tool_spec:
                func_name = tool_spec["function"]["name"]
                func = get_function_by_name(func_name)
                if func:
                    tools[func_name] = func

        return tools

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent.

        Returns:
            System message describing agent capabilities and tool usage
        """
        return """You are an expert chemist AI assistant with access to molecular property calculation tools.

When asked to analyze a molecule, use the provided tools to calculate relevant properties. Always call tools to get accurate results rather than guessing.

Available tools will be provided in the function definitions. Use them by outputting tool calls in the specified format."""


class AgentExecutor(MultiTurnAgentExecutor):
    """Executor for tool-calling agent.

    Uses factory pattern (zero-arg init) required by MultiTurnAgentExecutor API.
    """

    def __init__(self):
        """Initialize executor with ToolCallAgent class."""
        super().__init__(ToolCallAgent)


# Export public API
__all__ = ["ToolCallAgent", "AgentExecutor"]
