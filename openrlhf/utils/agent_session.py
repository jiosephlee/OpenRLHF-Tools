"""Minimal AgentSession for coordinating tool-calling conversations.

This module provides a simplified session management layer that:
- Maintains conversation history
- Coordinates protocol (format-specific rendering/parsing)
- Handles tool execution inline (no Environment abstraction)
- Computes rewards inline (no RewardPipeline abstraction)

Rationale: Session + Protocol provide 80% of value with 20% of complexity.
"""

import json
from typing import Any, Callable, Dict, List, Optional


class AgentSession:
    """Minimal session manager for tool-calling agents.

    Coordinates conversation flow without heavyweight abstractions.
    Uses inline tool execution and reward computation for simplicity.
    """

    def __init__(
        self,
        protocol: Any,  # ChatProtocol instance
        tools: Dict[str, Callable],  # {"tool_name": callable}
        system_prompt: str,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,  # JSON tool schemas for rendering
    ):
        """Initialize agent session.

        Args:
            protocol: ChatProtocol instance for format-specific rendering/parsing
            tools: Dictionary mapping tool names to callable functions
            system_prompt: System message for the agent
            tool_schemas: JSON-serializable tool definitions for prompt rendering
                          (e.g. BASIC_TOOLS). If None, tools are not included in prompt.
        """
        self.protocol = protocol
        self.tools = tools
        self.tool_schemas = tool_schemas or []
        self.system_prompt = system_prompt
        self.history: List[Dict[str, Any]] = []

    async def initialize(self, prompt: str) -> str:
        """Reset session and format initial prompt with tools.

        Args:
            prompt: Either a plain text question, or a JSON-serialized list of
                    OpenAI-format messages (e.g. [{"role": "user", "content": "..."}]).
                    When JSON messages are provided, they are merged with the
                    agent's system prompt and tool schemas.

        Returns:
            Formatted prompt string ready for LLM generation
        """
        # Try to parse as JSON messages list (from data pipeline without --apply_chat_template)
        input_messages = None
        if prompt.startswith("["):
            try:
                parsed = json.loads(prompt)
                if isinstance(parsed, list) and all(isinstance(m, dict) for m in parsed):
                    input_messages = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        # Build conversation history: agent system prompt + input messages
        self.history = [{"role": "system", "content": self.system_prompt}]

        if input_messages:
            # Merge input messages (skip any system messages from data — agent provides its own)
            for msg in input_messages:
                if msg.get("role") != "system":
                    self.history.append(msg)
        else:
            # Plain text prompt — wrap as user message
            self.history.append({"role": "user", "content": prompt})

        # Render with protocol (includes tool schemas, not callables)
        return self.protocol.render_messages(
            self.history,
            tools=self.tool_schemas,
            add_generation_prompt=True
        )

    async def step(self, action_text: str, label: Optional[str] = None) -> Dict[str, Any]:
        """Process one agent step: parse action, execute tools, compute reward.

        Args:
            action_text: Raw LLM output text
            label: Optional ground truth label for reward computation

        Returns:
            Dictionary with:
                - feedback: str - Observation text to append to prompt (empty if done)
                - reward: float - Reward value (0.0 for intermediate steps)
                - done: bool - Whether episode is terminated
                - extra_logs: dict - Optional extra logging info
        """
        # Parse action using protocol
        action = self.protocol.parse_assistant_text(action_text)

        # Check if agent made tool calls
        tool_calls = action.get("tool_calls", [])

        if tool_calls:
            # Execute tools and collect results
            results = []
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                results.append(result)

            # Update history with assistant message and tool responses
            self.history.append({
                "role": "assistant",
                "content": action.get("content", ""),
                "tool_calls": tool_calls
            })

            for i, result in enumerate(results):
                self.history.append({
                    "role": "tool",
                    "name": tool_calls[i].get("name", ""),
                    "content": result
                })

            # Render only the tool responses as feedback
            # (This preserves action_ranges for GRPO - only action tokens in ranges)
            feedback = self.protocol.render_messages(
                self.history[-len(results):],  # Just tool responses
                add_generation_prompt=True
            )

            return {
                "feedback": feedback,
                "reward": 0.0,  # No reward for intermediate steps
                "done": False,
                "extra_logs": {
                    "tool_call_count": len(tool_calls),
                }
            }
        else:
            # No tool calls - final answer
            feedback = ""
            reward = self._compute_reward(action.get("content", ""), label)
            done = True

            return {
                "feedback": feedback,
                "reward": reward,
                "done": done,
                "extra_logs": {}
            }

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute one tool invocation.

        Args:
            tool_call: Dictionary with "name" and "arguments" keys

        Returns:
            JSON string with tool result
        """
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        if tool_name not in self.tools:
            return json.dumps({
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            })

        try:
            # Call tool function
            tool_func = self.tools[tool_name]
            result = tool_func(**arguments)

            # Format result as JSON
            return json.dumps({
                "result": result,
                "function_name": tool_name,
                "arguments": arguments
            })
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "function_name": tool_name,
                "arguments": arguments
            })

    def _compute_reward(self, generated_text: str, label: Optional[str]) -> float:
        """Compute reward for final answer.

        PLACEHOLDER: Replace with actual reward model.

        Args:
            generated_text: LLM's final answer
            label: Ground truth label

        Returns:
            Reward value in [0, 1]
        """
        if not label:
            return 0.0

        # Simple substring matching (replace with actual reward model)
        if label.lower() in generated_text.lower():
            return 1.0
        else:
            return 0.0
