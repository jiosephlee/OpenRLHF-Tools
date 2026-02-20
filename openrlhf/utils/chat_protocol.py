"""Chat protocol abstractions for format-specific parsing and feedback.

This module provides:
- ChatProtocol ABC: Abstract interface for model-specific tool-call handling
- GLMFlashProtocol: Implementation for GLM Flash XML tool calling format
- InternS1Protocol: Implementation for Intern-S1-mini JSON tool calling format
- Extensible design for adding new protocols (Qwen3, Claude, etc.)
"""

import re
import json
import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ChatProtocol(ABC):
    """Abstract protocol for format-specific parsing and feedback."""

    @abstractmethod
    def parse_assistant_text(self, text: str, token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
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

    @abstractmethod
    def render_tool_feedback(self, tool_results: List[Dict[str, str]]) -> str:
        """Render the bridge text between end-of-LLM-generation and next turn.

        This is always manual string construction because the multi-turn loop
        in agent.py works by concatenation::

            observation_text = observation_text + action_text + feedback_text

        ``action_text`` comes straight from vLLM (no closing tags), so the
        feedback must:
          1. Close the assistant turn (if the format requires it)
          2. Render each tool/environment response
          3. Open the next assistant turn (generation prompt)

        Args:
            tool_results: List of dicts with "name" and "content" keys

        Returns:
            Bridge text ready for concatenation
        """
        pass


class GLMFlashProtocol(ChatProtocol):
    """GLM Flash XML format protocol.

    Tool call format: <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
    Observation format: <|observation|>\\n<tool_response>result</tool_response>\\n<|assistant|>\\n
    """

    def __init__(self, tokenizer):
        """Initialize GLM Flash protocol.

        Args:
            tokenizer: HuggingFace tokenizer for the model
        """
        self.tokenizer = tokenizer

    def parse_assistant_text(self, text: str, token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
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

    def render_tool_feedback(self, tool_results: List[Dict[str, str]]) -> str:
        """GLM Flash bridge: no explicit assistant close needed."""
        feedback = ""
        for tr in tool_results:
            feedback += f"<|observation|>\n<tool_response>{tr['content']}</tool_response>\n"
        feedback += "<|assistant|>\n"
        return feedback

    def _parse_glm_flash_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse GLM Flash tool call format using vLLM parser or regex fallback.

        Format: <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

        Returns:
            Dict with 'function_name' and 'arguments', or None if no tool call found
        """
        # Regex parser (based on official vLLM patterns).
        # Note: vLLM's Glm47MoeModelToolParser.extract_tool_calls() now requires
        # a ChatCompletionRequest arg we don't have here, so we use regex directly.
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

    # vLLM's detokenizer may insert whitespace between special tokens,
    # so we match with optional \s* between the two markers.
    _START_RE = re.compile(r"<\|action_start\|>\s*<\|plugin\|>")
    _END_RE = re.compile(r"<\|action_end\|>")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def parse_assistant_text(self, text: str, token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Parse Intern-S1 tool call blocks from assistant output.

        Extracts all ``<|action_start|><|plugin|>...json...<|action_end|>``
        blocks.  Text outside these blocks is treated as content.  Accepts
        both ``parameters`` and ``arguments`` as the key for function args.

        Uses regex to tolerate whitespace that vLLM's detokenizer may insert
        between special tokens.
        """
        tool_calls: List[Dict[str, Any]] = []
        content_parts: List[str] = []

        pos = 0
        while True:
            start_m = self._START_RE.search(text, pos)
            if start_m is None:
                content_parts.append(text[pos:])
                break

            content_parts.append(text[pos:start_m.start()])

            end_m = self._END_RE.search(text, start_m.end())
            if end_m is None:
                # Incomplete block — treat as plain text
                content_parts.append(text[start_m.start():])
                break

            action = text[start_m.end():end_m.start()].strip()
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
                        content_parts.append(text[start_m.start():end_m.end()])
                        pos = end_m.end()
                        continue
                else:
                    content_parts.append(text[start_m.start():end_m.end()])
                    pos = end_m.end()
                    continue
            except Exception:
                content_parts.append(text[start_m.start():end_m.end()])
                pos = end_m.end()
                continue

            name = action_dict.get("name")
            args = action_dict.get("parameters", action_dict.get("arguments", {}))
            if name:
                tool_calls.append({"name": name, "arguments": args})

            pos = end_m.end()

        content = "".join(content_parts).strip()

        if tool_calls:
            return {"content": content, "tool_calls": tool_calls}
        return {"content": content or text, "tool_calls": []}

    def render_tool_feedback(self, tool_results: List[Dict[str, str]]) -> str:
        """Intern-S1 bridge: must close the assistant turn first."""
        feedback = "<|im_end|>\n"  # close the open assistant turn
        for tr in tool_results:
            feedback += f"<|im_start|>environment name=<|plugin|>\n\n{tr['content']}<|im_end|>\n"
        feedback += "<|im_start|>assistant\n\n<think>\n"
        return feedback


class GPTOSSProtocol(ChatProtocol):
    """GPT-OSS Harmony protocol using token-ID parser.

    Mirrors vLLM's ``OpenAIToolParser`` — uses ``parse_output_into_messages``
    from the ``openai_harmony`` / ``harmony_utils`` library.

    Harmony message format (tool call)::

        <|start|>assistant\\nto=functions.tool_name<|channel|>commentary<|message|>{"arg": "val"}<|end|>

    Tool feedback (function → assistant)::

        <|start|>functions.tool_name\\nto=assistant<|channel|>commentary<|message|>result<|end|>

    Generation prompt::

        <|start|>assistant

    **Stop-string semantics** (different from InternS1):

    InternS1 uses *two* stop tokens — ``<|action_end|>`` (tool call) vs
    ``<|im_end|>`` (final answer) — so the stop token itself tells us what
    happened.

    Harmony uses a *single* ``<|end|>`` token for **both** tool calls and
    final answers.  The distinction is made by parsing the message structure:

    - ``msg.recipient.startswith("functions.")`` → tool call → continue
    - otherwise → final answer → done

    This is exactly how vLLM's ``OpenAIToolParser.extract_tool_calls()``
    works: it inspects ``msg.recipient``, not the stop token.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Cache <|start|>assistant token IDs for the fallback path.
        self._cached_header_ids: Optional[List[int]] = None

    @property
    def _assistant_header_ids(self) -> List[int]:
        """Token IDs for ``<|start|>assistant`` (computed once, cached)."""
        if self._cached_header_ids is None:
            self._cached_header_ids = self.tokenizer.encode(
                "<|start|>assistant", add_special_tokens=False
            )
        return self._cached_header_ids

    def parse_assistant_text(self, text: str, token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Parse assistant output using the Harmony token-ID parser.

        Matches vLLM's ``OpenAIToolParser.extract_tool_calls()``: passes
        output token IDs directly to ``parse_output_into_messages``.

        If the direct parse fails (e.g. the parser expects a complete
        ``<|start|>``-prefixed message), falls back to prepending the
        ``<|start|>assistant`` header that was part of the generation prompt.
        """
        if token_ids is None:
            raise NotImplementedError("GPT-OSS parsing requires generated token IDs.")

        harmony_utils = importlib.import_module("vllm.entrypoints.openai.parser.harmony_utils")

        # Primary path: pass output tokens directly (matches vLLM serving).
        try:
            parser = harmony_utils.parse_output_into_messages(token_ids)
        except Exception:
            # Fallback: prepend <|start|>assistant header that was part of
            # the prompt/feedback and retry.
            parser = harmony_utils.parse_output_into_messages(
                self._assistant_header_ids + list(token_ids)
            )

        return self._extract_from_parser(parser, text)

    def _extract_from_parser(self, parser, raw_text: str) -> Dict[str, Any]:
        """Extract tool calls / content from a parsed Harmony message.

        Closely mirrors the extraction logic in vLLM's
        ``OpenAIToolParser.extract_tool_calls()``.
        """
        tool_calls: List[Dict[str, Any]] = []
        final_content = None
        commentary_content = None

        for msg in parser.messages:
            if not msg.content:
                continue
            msg_text = msg.content[0].text

            if msg.recipient and msg.recipient.startswith("functions."):
                name = msg.recipient.split("functions.", 1)[1]
                # Parse JSON arguments; tolerate malformed model output.
                if not getattr(msg, "content_type", None) or "json" in (getattr(msg, "content_type", "") or ""):
                    try:
                        args: Any = json.loads(msg_text)
                    except json.JSONDecodeError:
                        args = msg_text
                else:
                    args = msg_text
                # Double-encoded JSON strings (model quirk)
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {"raw": args}
                tool_calls.append({"name": name, "arguments": args})
            elif msg.channel == "final":
                final_content = msg_text
            elif msg.channel == "commentary" and not msg.recipient:
                commentary_content = msg_text

        # Handle truncated output (model hit max_tokens before <|end|>).
        if parser.current_content:
            if parser.current_channel == "final":
                final_content = parser.current_content
            elif parser.current_channel == "commentary" and not parser.current_recipient:
                commentary_content = parser.current_content

        return {
            "content": final_content or commentary_content or raw_text,
            "tool_calls": tool_calls,
        }

    def render_tool_feedback(self, tool_results: List[Dict[str, str]]) -> str:
        """Build bridge text: tool responses + generation prompt.

        The Harmony wire format for function→assistant messages uses ``\\n``
        between the source role and the ``to=`` directive::

            <|start|>functions.tool_name\\nto=assistant<|channel|>commentary<|message|>content<|end|>

        The model's output already includes ``<|end|>`` (via
        ``include_stop_str_in_output=True``), so the feedback just appends
        the function response(s) and a new generation prompt.

        ``tool_content`` is already a JSON string produced by
        ``json.dumps(...)`` in the tool executor — embedded directly
        (no second ``json.dumps``).
        """
        feedback = ""
        for tr in tool_results:
            tool_name = tr["name"]
            tool_content = tr["content"]
            feedback += (
                f"<|start|>functions.{tool_name}\nto=assistant"
                f"<|channel|>commentary<|message|>{tool_content}<|end|>"
            )
        # Generation prompt for the next assistant turn
        feedback += "<|start|>assistant"
        return feedback

# Export public API
__all__ = ["ChatProtocol", "GLMFlashProtocol", "InternS1Protocol", "GPTOSSProtocol"]
