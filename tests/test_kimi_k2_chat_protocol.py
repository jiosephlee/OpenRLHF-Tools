import importlib.util
from pathlib import Path
import sys
import types
import logging


def _load_chat_protocol_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "openrlhf" / "utils" / "chat_protocol.py"

    openrlhf_pkg = types.ModuleType("openrlhf")
    utils_pkg = types.ModuleType("openrlhf.utils")
    logging_utils_mod = types.ModuleType("openrlhf.utils.logging_utils")
    logging_utils_mod.init_logger = logging.getLogger

    sys.modules.setdefault("openrlhf", openrlhf_pkg)
    sys.modules.setdefault("openrlhf.utils", utils_pkg)
    sys.modules["openrlhf.utils.logging_utils"] = logging_utils_mod

    spec = importlib.util.spec_from_file_location("openrlhf_kimi_chat_protocol", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


chat_protocol = _load_chat_protocol_module()
KimiK2Protocol = chat_protocol.KimiK2Protocol


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1]


def test_kimi_k2_parser_extracts_tool_calls_and_strips_reasoning():
    protocol = KimiK2Protocol(DummyTokenizer())
    text = (
        "<think>I should call a tool.</think>"
        "<|tool_calls_section_begin|>"
        '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city":"Beijing"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )

    parsed = protocol.parse_assistant_text(text)

    assert parsed["content"] == ""
    assert parsed["tool_calls"] == [
        {
            "id": "functions.get_weather:0",
            "name": "get_weather",
            "arguments": {"city": "Beijing"},
        }
    ]


def test_kimi_k2_parser_handles_multiple_calls_and_leading_content():
    protocol = KimiK2Protocol(DummyTokenizer())
    text = (
        "<think>Reasoning</think>Using tools now.\n"
        "<|tool_calls_section_begin|>"
        '<|tool_call_begin|>functions.tool_a:0<|tool_call_argument_begin|>{"x":1}<|tool_call_end|>'
        '<|tool_call_begin|>functions.tool_b:1<|tool_call_argument_begin|>{"y":"z"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )

    parsed = protocol.parse_assistant_text(text)

    assert parsed["content"] == "Using tools now."
    assert parsed["tool_calls"] == [
        {"id": "functions.tool_a:0", "name": "tool_a", "arguments": {"x": 1}},
        {"id": "functions.tool_b:1", "name": "tool_b", "arguments": {"y": "z"}},
    ]


def test_kimi_k2_render_tool_feedback_preserves_tool_call_ids():
    protocol = KimiK2Protocol(DummyTokenizer())
    feedback = protocol.render_tool_feedback(
        [
            {
                "name": "get_weather",
                "tool_call_id": "functions.get_weather:0",
                "content": '{"weather":"sunny"}',
            }
        ]
    )

    assert feedback.startswith("</think><|im_end|><|im_system|>get_weather<|im_middle|>")
    assert "## Return of functions.get_weather:0\n" in feedback
    assert feedback.endswith('<|im_end|><|im_assistant|>assistant<|im_middle|><think>')
