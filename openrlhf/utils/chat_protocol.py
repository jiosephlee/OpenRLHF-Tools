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

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


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

        <|start|>assistant<|channel|>analysis<|message|>reasoning...<|end|>
        <|start|>assistant<|channel|>commentary to=functions.tool_name <|constrain|>json<|message|>{"arg": "val"}<|call|>

    Tool feedback (function → assistant)::

        <|start|>functions.tool_name to=assistant<|channel|>commentary<|message|>result<|end|>

    Generation prompt::

        <|start|>assistant

    **Stop tokens:** ``<|return|>`` (200002) and ``<|call|>`` (200012).

    ``<|end|>`` (200007) is a **message boundary**, not a stop token — the
    model emits multiple ``<|end|>``-separated messages (e.g. analysis then
    tool call).  Stopping at ``<|end|>`` truncates the output before the
    tool call or final answer is produced.

    ``<|call|>`` signals the model wants to invoke a tool; ``<|return|>``
    signals the model is done (final answer).

    After parsing, the distinction is made by inspecting the message:

    - ``msg.recipient.startswith("functions.")`` → tool call → continue
    - ``msg.channel == "final"`` → final answer → done
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
        except Exception as primary_err:
            # Fallback: prepend <|start|>assistant header that was part of
            # the prompt/feedback and retry.
            try:
                parser = harmony_utils.parse_output_into_messages(
                    self._assistant_header_ids + list(token_ids)
                )
            except Exception as fallback_err:
                # Last resort: regex-based fallback on decoded special tokens.
                logger.warning(
                    "GPT-OSS Harmony token-ID parser failed on both paths "
                    "(primary: %s | fallback: %s). Using regex fallback.",
                    primary_err, fallback_err,
                )
                return self._regex_fallback_parse(token_ids, text)

        return self._extract_from_parser(parser, text)

    # ------------------------------------------------------------------
    # Regex fallback parser
    # ------------------------------------------------------------------

    # Patterns for harmony special tokens rendered as text
    _RE_TOOL_CALL = re.compile(
        r'to=functions\.(\S+?)'           # recipient: functions.TOOL_NAME
        r'(?:<\|channel\|>\w+)?'           # optional channel (json, commentary, …)
        r'(?:\s*<\|constrain\|>\w+)?'      # optional constrain tag
        r'<\|message\|>(.*?)'             # message body (args)
        r'(?:<\|call\|>|<\|end\|>|$)',     # terminator
        re.DOTALL,
    )
    _RE_FINAL = re.compile(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)',
        re.DOTALL,
    )
    _RE_COMMENTARY = re.compile(
        r'<\|channel\|>(?:commentary|analysis)<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>|<\|return\|>|$)',
        re.DOTALL,
    )

    def _regex_fallback_parse(self, token_ids: List[int], raw_text: str) -> Dict[str, Any]:
        """Regex-based fallback when the Harmony token-ID parser fails.

        Decodes the full token stream (with special tokens visible) and
        extracts tool calls / content via pattern matching on harmony markers.
        """
        full_text = self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

        tool_calls: List[Dict[str, Any]] = []
        for m in self._RE_TOOL_CALL.finditer(full_text):
            name = m.group(1)
            args_text = m.group(2).strip()
            try:
                args: Any = json.loads(args_text)
            except json.JSONDecodeError:
                args = args_text
            # Unwrap double-encoded JSON strings
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"raw": args}
            tool_calls.append({"name": name, "arguments": args})

        final_content = None
        fm = self._RE_FINAL.search(full_text)
        if fm:
            final_content = fm.group(1).strip()

        commentary_content = None
        for cm in self._RE_COMMENTARY.finditer(full_text):
            # Skip matches that are part of a tool call (preceded by to=functions.)
            preceding = full_text[max(0, cm.start() - 120):cm.start()]
            if "to=functions." not in preceding:
                commentary_content = cm.group(1).strip()

        return {
            "content": final_content or commentary_content or raw_text,
            "tool_calls": tool_calls,
        }

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

        The Harmony wire format for function→assistant messages uses a space
        between the source role and the ``to=`` directive::

            <|start|>functions.tool_name to=assistant<|channel|>commentary<|message|>content<|end|>

        The model's output already includes ``<|call|>`` (via
        ``include_stop_str_in_output=True``), so the feedback just appends
        the function response(s) and a new generation prompt.

        ``tool_content`` is embedded as raw JSON, matching the canonical
        format from the harmony docs and the ``openai_harmony`` renderer.

        Note: the HF chat template applies ``|tojson`` which double-encodes
        string content. That is a template quirk — the model was trained
        with the harmony renderer which embeds raw JSON.
        """
        feedback = ""
        for tr in tool_results:
            tool_name = tr["name"]
            tool_content = tr["content"]
            feedback += (
                f"<|start|>functions.{tool_name} to=assistant"
                f"<|channel|>commentary<|message|>{tool_content}<|end|>"
            )
        # Generation prompt for the next assistant turn
        feedback += "<|start|>assistant"
        return feedback

# Export public API
__all__ = ["ChatProtocol", "GLMFlashProtocol", "InternS1Protocol", "GPTOSSProtocol"]
