import importlib.util
from pathlib import Path
import sys
import types
import logging


def _load_chat_protocol_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "openrlhf" / "utils" / "chat_protocol.py"

    # Load chat_protocol.py without importing the full openrlhf.utils package,
    # which pulls in heavyweight training deps not required for parser tests.
    openrlhf_pkg = types.ModuleType("openrlhf")
    utils_pkg = types.ModuleType("openrlhf.utils")
    logging_utils_mod = types.ModuleType("openrlhf.utils.logging_utils")
    logging_utils_mod.init_logger = logging.getLogger

    sys.modules.setdefault("openrlhf", openrlhf_pkg)
    sys.modules.setdefault("openrlhf.utils", utils_pkg)
    sys.modules["openrlhf.utils.logging_utils"] = logging_utils_mod

    spec = importlib.util.spec_from_file_location("openrlhf_glm_chat_protocol", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


chat_protocol = _load_chat_protocol_module()
GLM51Protocol = chat_protocol.GLM51Protocol


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1]


def test_glm51_parser_handles_direct_arg_key_without_newline():
    protocol = GLM51Protocol(DummyTokenizer())
    text = (
        "<tool_call>search<arg_key>query</arg_key>"
        "<arg_value>GLM 5.1</arg_value></tool_call>"
    )

    parsed = protocol.parse_assistant_text(text)

    assert parsed["content"] == ""
    assert parsed["tool_calls"] == [{"name": "search", "arguments": {"query": "GLM 5.1"}}]


def test_glm51_parser_handles_zero_argument_calls():
    protocol = GLM51Protocol(DummyTokenizer())
    text = "<tool_call>ping</tool_call>"

    parsed = protocol.parse_assistant_text(text)

    assert parsed["content"] == ""
    assert parsed["tool_calls"] == [{"name": "ping", "arguments": {}}]


def test_glm51_parser_collects_multiple_tool_calls_and_preserves_content():
    protocol = GLM51Protocol(DummyTokenizer())
    text = (
        "I will inspect two tools.\n"
        "<tool_call>tool_a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n"
        "<tool_call>tool_b</tool_call>\n"
    )

    parsed = protocol.parse_assistant_text(text)

    assert parsed["content"] == "I will inspect two tools."
    assert parsed["tool_calls"] == [
        {"name": "tool_a", "arguments": {"x": "1"}},
        {"name": "tool_b", "arguments": {}},
    ]
