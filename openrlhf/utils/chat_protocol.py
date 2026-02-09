"""Chat protocol abstractions for format-specific rendering and parsing.

This module provides:
- ChatProtocol ABC: Abstract interface for model-specific formats
- GLMFlashProtocol: Implementation for GLM Flash XML tool calling format
- Extensible design for adding new protocols (Qwen3, Claude, etc.)
"""

import os
import re
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ChatProtocol(ABC):
    """Abstract protocol for format-specific message rendering and parsing."""

    @abstractmethod
    def render_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Dict] = None,
        add_generation_prompt: bool = False
    ) -> str:
        """Render messages to text using format-specific template.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional tool definitions
            add_generation_prompt: Whether to add generation prompt at end

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def parse_assistant_text(self, text: str) -> Dict[str, Any]:
        """Parse assistant text into structured action dict.

        Args:
            text: Raw assistant output text

        Returns:
            Dictionary with:
                - content: str - Final answer text
                - tool_calls: List[Dict] - List of tool calls
                    - name: str - Function name
                    - arguments: Dict - Function arguments
        """
        pass


class GLMFlashProtocol(ChatProtocol):
    """GLM Flash XML format protocol.

    Tool call format: <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
    Observation format: <|observation|>\\n<tool_response>result</tool_response>\\n<|assistant|>\\n
    """

    # Try importing vLLM's official tool parser
    try:
        from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
        VLLM_PARSER_AVAILABLE = True
    except ImportError:
        VLLM_PARSER_AVAILABLE = False

    def __init__(self, tokenizer):
        """Initialize GLM Flash protocol.

        Args:
            tokenizer: HuggingFace tokenizer for the model
        """
        self.tokenizer = tokenizer
        self.mode = os.environ.get("OPENRLHF_PROMPT_CONSTRUCTION_MODE", "manual")

    def render_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Dict] = None,
        add_generation_prompt: bool = False
    ) -> str:
        """Render messages using GLM Flash chat template.

        Supports two modes:
        - auto: Uses tokenizer.apply_chat_template (robust, slower)
        - manual: Manual string construction (fast, brittle)
        """
        if self.mode == "auto":
            # Use tokenizer's chat template with tools parameter
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        else:
            # Manual mode: simple string concatenation
            return self._format_manual(messages, tools, add_generation_prompt)

    def parse_assistant_text(self, text: str) -> Dict[str, Any]:
        """Parse GLM Flash tool call format.

        Uses vLLM's official parser if available, falls back to regex.
        """
        tool_call = self._parse_glm_flash_tool_call(text)

        if tool_call:
            return {
                "content": "",
                "tool_calls": [{
                    "name": tool_call["function_name"],
                    "arguments": tool_call["arguments"]
                }]
            }
        else:
            # No tool call - final answer
            return {
                "content": text,
                "tool_calls": []
            }

    def _parse_glm_flash_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse GLM Flash tool call format using vLLM parser or regex fallback.

        Format: <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

        Returns:
            Dict with 'function_name' and 'arguments', or None if no tool call found
        """
        # Try vLLM parser first (model-agnostic, officially supported)
        if self.VLLM_PARSER_AVAILABLE and self.tokenizer is not None:
            try:
                from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
                parser = Glm47MoeModelToolParser(self.tokenizer)
                parsed = parser.extract_tool_calls(text)
                if parsed:
                    return {
                        "function_name": parsed[0]['name'],
                        "arguments": parsed[0]['arguments']
                    }
            except Exception as e:
                print(f"vLLM parser failed: {e}, falling back to regex")

        # Fallback: Regex parser (using official vLLM patterns)
        func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

        tool_call_match = func_detail_regex.search(text)
        if not tool_call_match:
            return None

        function_name = tool_call_match.group(1).strip()
        arg_section = tool_call_match.group(2)
        arguments = {}

        if arg_section:
            arg_matches = func_arg_regex.findall(arg_section)
            for key, value in arg_matches:
                arguments[key.strip()] = value.strip()

        return {
            "function_name": function_name,
            "arguments": arguments
        }

    def _format_manual(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Dict],
        add_generation_prompt: bool
    ) -> str:
        """Manual string concatenation for GLM Flash format.

        Fast but brittle - format is hardcoded.
        """
        formatted = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "tool":
                # Tool response format
                formatted += f"<|observation|>\n<tool_response>{content}</tool_response>\n"

        if add_generation_prompt:
            formatted += "<|assistant|>\n"

        return formatted


# Export public API
__all__ = ["ChatProtocol", "GLMFlashProtocol"]
