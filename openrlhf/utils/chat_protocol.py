"""Chat protocol abstractions for format-specific rendering and parsing.

This module provides:
- ChatProtocol ABC: Abstract interface for model-specific formats
- GLMFlashProtocol: Implementation for GLM Flash XML tool calling format
- InternS1Protocol: Implementation for Intern-S1-mini JSON tool calling format
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


_VALID_JSON_ESC = set(['"', "\\", "/", "b", "f", "n", "r", "t", "u"])


def _repair_invalid_json_escapes(s: str) -> str:
    r"""Repair invalid escape sequences inside JSON strings (e.g. \C -> \\C).

    Needed for SMILES strings that contain backslash characters which are not
    valid JSON escapes.  Only modifies content inside quoted strings; leaves
    valid escapes (\\, \n, \", \uXXXX, etc.) untouched.
    """
    out = []
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]

        if not in_str:
            if c == '"':
                in_str = True
            out.append(c)
            i += 1
            continue

        # inside a JSON string
        if c == '"':
            in_str = False
            out.append(c)
            i += 1
            continue

        if c == "\\":
            if i + 1 >= len(s):
                out.append("\\\\")
                i += 1
                continue
            nxt = s[i + 1]
            if nxt in _VALID_JSON_ESC:
                out.append("\\")
                out.append(nxt)
                i += 2
            else:
                out.append("\\\\")
                i += 1
            continue

        out.append(c)
        i += 1

    return "".join(out)


class InternS1Protocol(ChatProtocol):
    """Intern-S1-mini JSON tool calling format protocol.

    Tool call format:
        <|action_start|><|plugin|>
        {"name": "tool_name", "parameters": {"key": "value"}}
        <|action_end|>

    Observation format:
        <|im_start|>environment name=<|plugin|>
        {tool_result}
        <|im_end|>

    Generation prompt ends with ``<|im_start|>assistant\\n<think>`` to trigger
    chain-of-thought reasoning before tool use.
    """

    _START = "<|action_start|><|plugin|>"
    _END = "<|action_end|>"

    _TOOL_INSTRUCTION = (
        'Your response should consist of a reasoning step (**thought**) '
        'followed immediately by a function call in valid JSON format. '
        'Wrap each function call using the `<|action_start|><|plugin|>` '
        'and `<|action_end|>` tags.\n'
        '\n'
        '**Format example:**\n'
        '\n'
        '```\n'
        '(Your thought goes here...)\n'
        '\n'
        '<|action_start|><|plugin|>\n'
        '{\n'
        '    "name": "tool_name",\n'
        '    "parameters": {\n'
        '        "parameter1": "value1",\n'
        '        "parameter2": "value2"\n'
        '    }\n'
        '}\n'
        '<|action_end|>\n'
        '```\n'
        '\n'
        '# External Tools\n'
        'You have access to these tools:\n'
    )

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mode = os.environ.get("OPENRLHF_PROMPT_CONSTRUCTION_MODE", "manual")

    def render_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Dict] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        if self.mode == "auto":
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        return self._format_manual(messages, tools, add_generation_prompt)

    def parse_assistant_text(self, text: str) -> Dict[str, Any]:
        """Parse Intern-S1 tool call blocks from assistant output.

        Extracts all ``<|action_start|><|plugin|>...json...<|action_end|>``
        blocks.  Text outside these blocks is treated as content.  Accepts
        both ``parameters`` and ``arguments`` as the key for function args.
        """
        tool_calls: List[Dict[str, Any]] = []
        content_parts: List[str] = []

        pos = 0
        while True:
            s = text.find(self._START, pos)
            if s == -1:
                content_parts.append(text[pos:])
                break

            content_parts.append(text[pos:s])

            e = text.find(self._END, s + len(self._START))
            if e == -1:
                # Incomplete block — treat as plain text
                content_parts.append(text[s:])
                break

            action = text[s + len(self._START):e].strip()
            # Skip to first '{' in case of stray characters
            j = action.find("{")
            if j != -1:
                action = action[j:]

            try:
                action_dict = json.loads(action)
            except json.JSONDecodeError as ex:
                if "Invalid \\escape" in str(ex):
                    try:
                        action_dict = json.loads(_repair_invalid_json_escapes(action))
                    except Exception:
                        content_parts.append(text[s:e + len(self._END)])
                        pos = e + len(self._END)
                        continue
                else:
                    content_parts.append(text[s:e + len(self._END)])
                    pos = e + len(self._END)
                    continue
            except Exception:
                content_parts.append(text[s:e + len(self._END)])
                pos = e + len(self._END)
                continue

            name = action_dict.get("name")
            args = action_dict.get("parameters", action_dict.get("arguments", {}))
            if name:
                tool_calls.append({"name": name, "arguments": args})

            pos = e + len(self._END)

        content = "".join(content_parts).strip()

        if tool_calls:
            return {"content": content, "tool_calls": tool_calls}
        return {"content": content or text, "tool_calls": []}

    # ------------------------------------------------------------------
    # Manual rendering
    # ------------------------------------------------------------------

    def _format_manual(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Dict],
        add_generation_prompt: bool,
    ) -> str:
        """Manual string construction matching the Intern-S1 Jinja template."""
        formatted = ""

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            tool_calls_list = msg.get("tool_calls", [])

            if role == "system":
                header = "<|im_start|>system"
                if tools:
                    header += " name=<|plugin|>"
                    tool_json = json.dumps(tools, indent=2, ensure_ascii=False) if tools else "[]"
                    content = content.rstrip("\n") + "\n\n" + self._TOOL_INSTRUCTION + tool_json
                formatted += f"{header}\n\n{content}<|im_end|>\n"

            elif role == "user":
                formatted += f"<|im_start|>user\n\n{content}<|im_end|>\n"

            elif role == "assistant":
                tc_text = ""
                for tc in tool_calls_list:
                    func = tc.get("function", tc)
                    name = func.get("name", "")
                    args = func.get("arguments", func.get("parameters", {}))
                    if isinstance(args, str):
                        args_str = args
                    else:
                        args_str = json.dumps(args, ensure_ascii=False)
                    tc_text += (
                        f'{self._START}\n'
                        f'{{"name": "{name}", "parameters": {args_str}}}\n'
                        f'{self._END}'
                    )
                formatted += f"<|im_start|>assistant\n\n{content}{tc_text}<|im_end|>\n"

            elif role in ("tool", "environment"):
                formatted += f"<|im_start|>environment name=<|plugin|>\n\n{content}<|im_end|>\n"

        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n\n<think>\n"

        return formatted


# Export public API
__all__ = ["ChatProtocol", "GLMFlashProtocol", "InternS1Protocol"]
