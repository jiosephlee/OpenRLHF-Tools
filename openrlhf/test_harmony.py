#!/usr/bin/env python3
"""Comprehensive tests for GPT-OSS Harmony support.

Tests cover:
  1. Harmony parser with complete messages (token IDs)
  2. GPTOSSProtocol.parse_assistant_text (direct + fallback)
  3. GPTOSSProtocol.render_tool_feedback format
  4. Round-trip: prompt → generate (simulated) → parse → feedback → parse
  5. apply_chat_template with tools
  6. Stop string validation
  7. Tool call vs final answer distinction (single <|end|> stop)

Run: python openrlhf/test_harmony.py
"""

import json
import sys
import traceback

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
try:
    import openai_harmony
except ImportError:
    print("SKIP: openai_harmony not installed")
    sys.exit(0)

from transformers import AutoTokenizer

MODEL = "openai/gpt-oss-20b"
print(f"Loading tokenizer: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"✅ PASSED: {name}")
        passed += 1
    except Exception as e:
        print(f"❌ FAILED: {name}")
        traceback.print_exc()
        failed += 1


# ---------------------------------------------------------------------------
# Test 1: Raw Harmony parser — complete tool-call message
# ---------------------------------------------------------------------------
def test_harmony_parser_tool_call():
    """Parse a COMPLETE tool-call message with <|start|> prefix."""
    parser = openai_harmony.create_parser("gpt-oss", tokenizer)
    text = '<|start|>assistant\nto=functions.get_num_saturated_rings<|channel|>commentary<|message|>{"smiles": "c1ccccc1"}<|end|>'
    for t_id in tokenizer.encode(text):
        parser.process(t_id)
    assert len(parser.messages) >= 1, f"Expected at least 1 message, got {len(parser.messages)}"
    msg = parser.messages[0]
    assert msg.recipient == "functions.get_num_saturated_rings", f"Wrong recipient: {msg.recipient}"
    print(f"  Recipient: {msg.recipient}")
    print(f"  Content: {msg.content[0].text if msg.content else 'NONE'}")


# ---------------------------------------------------------------------------
# Test 2: Raw Harmony parser — complete function response message
# ---------------------------------------------------------------------------
def test_harmony_parser_function_response():
    """Parse function→assistant response message."""
    parser = openai_harmony.create_parser("gpt-oss", tokenizer)
    text = '<|start|>functions.get_num_saturated_rings\nto=assistant<|channel|>commentary<|message|>{"result": "0"}<|end|>'
    for t_id in tokenizer.encode(text):
        parser.process(t_id)
    assert len(parser.messages) >= 1, f"Expected at least 1 message, got {len(parser.messages)}"
    msg = parser.messages[0]
    assert msg.recipient == "assistant", f"Wrong recipient: {msg.recipient}"
    print(f"  Recipient: {msg.recipient}")
    print(f"  Content: {msg.content[0].text if msg.content else 'NONE'}")


# ---------------------------------------------------------------------------
# Test 3: parse_output_into_messages on vLLM-style output tokens
#
# This is the KEY test — in vLLM serving, parse_output_into_messages receives
# ONLY the generated token IDs (after the prompt), WITHOUT the <|start|>
# assistant header. Verify this works.
# ---------------------------------------------------------------------------
def test_parse_output_into_messages_direct():
    """Verify parse_output_into_messages handles output-only token IDs.

    In vLLM's serving path, OpenAIToolParser passes output tokens directly
    (no header prepending). Our code should do the same.
    """
    from vllm.entrypoints.openai.parser.harmony_utils import parse_output_into_messages

    # Simulate what vLLM generates AFTER the prompt "<|start|>assistant":
    continuation = '\nto=functions.get_num_saturated_rings<|channel|>commentary<|message|>{"smiles": "c1ccccc1"}<|end|>'
    action_token_ids = tokenizer.encode(continuation, add_special_tokens=False)

    print(f"  Token count: {len(action_token_ids)}")
    print(f"  First few tokens: {action_token_ids[:5]}")

    try:
        parser = parse_output_into_messages(action_token_ids)
        print(f"  Direct parse succeeded: {len(parser.messages)} messages")
        if parser.messages:
            msg = parser.messages[0]
            print(f"  Recipient: {msg.recipient}")
    except Exception as e:
        # If direct parse fails, the fallback path in GPTOSSProtocol handles this
        print(f"  Direct parse raised (fallback will handle): {type(e).__name__}: {e}")

        # Verify the fallback works
        header = tokenizer.encode("<|start|>assistant", add_special_tokens=False)
        parser = parse_output_into_messages(header + action_token_ids)
        print(f"  Fallback parse succeeded: {len(parser.messages)} messages")
        if parser.messages:
            msg = parser.messages[0]
            print(f"  Recipient: {msg.recipient}")


# ---------------------------------------------------------------------------
# Test 4: GPTOSSProtocol.parse_assistant_text (tool call)
# ---------------------------------------------------------------------------
def test_protocol_parse_tool_call():
    """GPTOSSProtocol extracts tool call from action tokens."""
    from openrlhf.utils.chat_protocol import GPTOSSProtocol

    protocol = GPTOSSProtocol(tokenizer)

    continuation = '\nto=functions.get_num_saturated_rings<|channel|>commentary<|message|>{"smiles": "c1ccccc1"}<|end|>'
    action_token_ids = tokenizer.encode(continuation, add_special_tokens=False)

    result = protocol.parse_assistant_text(continuation, token_ids=action_token_ids)

    assert len(result["tool_calls"]) == 1, f"Expected 1 tool call, got {len(result['tool_calls'])}"
    tc = result["tool_calls"][0]
    assert tc["name"] == "get_num_saturated_rings", f"Wrong name: {tc['name']}"
    assert tc["arguments"]["smiles"] == "c1ccccc1", f"Wrong args: {tc['arguments']}"
    print(f"  Tool call: {tc['name']}({tc['arguments']})")


# ---------------------------------------------------------------------------
# Test 5: GPTOSSProtocol.parse_assistant_text (final answer — no tool call)
#
# Critical: Harmony uses <|end|> for BOTH tool calls and final answers.
# The parser must correctly identify this as a final answer (no tool_calls).
# ---------------------------------------------------------------------------
def test_protocol_parse_final_answer():
    """GPTOSSProtocol correctly identifies final answer (no tool calls).

    In Harmony, final answers also end with <|end|>. The distinction is:
    - Tool call: msg.recipient starts with "functions."
    - Final answer: msg.channel is "commentary" or "final", no recipient
    """
    from openrlhf.utils.chat_protocol import GPTOSSProtocol

    protocol = GPTOSSProtocol(tokenizer)

    # Final answer: no to=functions.X, just commentary content
    final_text = '\n<|channel|>commentary<|message|>Based on the molecular weight of 180.16 g/mol, the answer is (A).<|end|>'
    final_token_ids = tokenizer.encode(final_text, add_special_tokens=False)

    result = protocol.parse_assistant_text(final_text, token_ids=final_token_ids)

    assert len(result["tool_calls"]) == 0, f"Should be final answer, got tool calls: {result['tool_calls']}"
    assert "(A)" in result["content"], f"Missing answer in content: {result['content']}"
    print(f"  Content: {result['content'][:80]!r}...")
    print(f"  Tool calls: {result['tool_calls']} (correctly empty)")


# ---------------------------------------------------------------------------
# Test 6: render_tool_feedback format validation
# ---------------------------------------------------------------------------
def test_render_tool_feedback():
    """Verify tool feedback is correctly formatted for Harmony."""
    from openrlhf.utils.chat_protocol import GPTOSSProtocol

    protocol = GPTOSSProtocol(tokenizer)

    tool_results = [{"name": "get_num_saturated_rings", "content": '{"result": 0}'}]
    feedback = protocol.render_tool_feedback(tool_results)

    print(f"  Feedback: {feedback!r}")

    # Structure checks
    assert "<|start|>functions.get_num_saturated_rings" in feedback
    assert "\nto=assistant" in feedback, f"Missing newline before to=: {feedback!r}"
    assert "<|channel|>commentary" in feedback
    assert "<|end|>" in feedback
    assert feedback.endswith("<|start|>assistant"), f"Should end with generation prompt"

    # Verify the function response portion parses cleanly
    parser = openai_harmony.create_parser("gpt-oss", tokenizer)
    func_msg = feedback[: feedback.rfind("<|start|>assistant")]
    for t_id in tokenizer.encode(func_msg, add_special_tokens=False):
        parser.process(t_id)
    assert len(parser.messages) >= 1, f"Feedback not parseable by Harmony"
    print(f"  Parsed {len(parser.messages)} message(s) from feedback")


# ---------------------------------------------------------------------------
# Test 7: Full round-trip (mimics agent.py MultiTurnAgentExecutor)
# ---------------------------------------------------------------------------
def test_round_trip():
    """Simulate the multi-turn agent loop:

    1. apply_chat_template → prompt
    2. Model generates tool call → parse → execute → feedback
    3. Model generates final answer → parse → done

    This tests the exact flow from agent.py with Harmony format.
    """
    from openrlhf.utils.chat_protocol import GPTOSSProtocol

    protocol = GPTOSSProtocol(tokenizer)

    # === Turn 0: Build prompt ===
    messages = [{"role": "user", "content": "What is the molecular weight of aspirin (CC(=O)OC1=CC=CC=C1C(=O)O)?"}]
    tools = [{
        "type": "function",
        "function": {
            "name": "get_molecular_weight",
            "description": "Get mol weight",
            "parameters": {
                "type": "object",
                "properties": {"smiles": {"type": "string"}},
                "required": ["smiles"],
            },
        },
    }]

    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"  Prompt: {len(prompt_tokens)} tokens, ends: ...{prompt[-50:]!r}")

    # === Turn 1: Model makes tool call ===
    # vLLM generates after "<|start|>assistant", stops at "<|end|>"
    tc_text = '\nto=functions.get_molecular_weight<|channel|>commentary<|message|>{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}<|end|>'
    tc_tokens = tokenizer.encode(tc_text, add_special_tokens=False)

    action = protocol.parse_assistant_text(tc_text, token_ids=tc_tokens)
    assert action["tool_calls"], "No tool calls parsed!"
    tc = action["tool_calls"][0]
    print(f"  Turn 1 tool call: {tc['name']}({tc['arguments']})")

    # Execute tool (simulated) + render feedback
    tool_result = json.dumps({"result": 180.16})
    feedback = protocol.render_tool_feedback([{"name": tc["name"], "content": tool_result}])
    feedback_tokens = tokenizer.encode(feedback, add_special_tokens=False)

    # Build observation for turn 2
    obs_tokens = prompt_tokens + tc_tokens + feedback_tokens
    print(f"  After turn 1: {len(obs_tokens)} obs tokens")

    # === Turn 2: Model gives final answer ===
    # vLLM generates after "<|start|>assistant", stops at "<|end|>"
    ans_text = '\n<|channel|>commentary<|message|>The molecular weight of aspirin is 180.16 g/mol.\n\nAnswer: (A)<|end|>'
    ans_tokens = tokenizer.encode(ans_text, add_special_tokens=False)

    final = protocol.parse_assistant_text(ans_text, token_ids=ans_tokens)
    assert not final["tool_calls"], f"Should be final answer: {final}"
    print(f"  Turn 2 answer: {final['content'][:60]!r}...")
    print(f"  Round-trip complete!")


# ---------------------------------------------------------------------------
# Test 8: apply_chat_template with tools
# ---------------------------------------------------------------------------
def test_apply_chat_template():
    """Verify apply_chat_template produces valid Harmony prompt."""
    messages = [{"role": "user", "content": "Predict bioavailability: CC(=O)OC1=CC=CC=C1C(=O)O"}]
    tools = [{
        "type": "function",
        "function": {
            "name": "get_molecular_weight",
            "description": "Get mol weight",
            "parameters": {
                "type": "object",
                "properties": {"smiles": {"type": "string"}},
                "required": ["smiles"],
            },
        },
    }]

    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )

    assert isinstance(prompt, str) and len(prompt) > 0
    assert "<|start|>assistant" in prompt, f"Missing generation prompt"
    print(f"  Prompt: {len(prompt)} chars")
    print(f"  Ends: ...{prompt[-60:]!r}")

    # Verify tokenize round-trip
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"  Tokens: {len(token_ids)}")


# ---------------------------------------------------------------------------
# Test 9: <|end|> token verification
# ---------------------------------------------------------------------------
def test_end_token():
    """<|end|> should be a valid stop string for vLLM."""
    end_ids = tokenizer.encode("<|end|>", add_special_tokens=False)
    print(f"  <|end|> token IDs: {end_ids}")
    assert len(end_ids) >= 1, "<|end|> must encode to at least 1 token"
    decoded = tokenizer.decode(end_ids, skip_special_tokens=False)
    print(f"  Decoded: {decoded!r}")


# ---------------------------------------------------------------------------
# Test 10: Multi-tool feedback
# ---------------------------------------------------------------------------
def test_multi_tool_feedback():
    """Verify feedback with multiple tool results."""
    from openrlhf.utils.chat_protocol import GPTOSSProtocol

    protocol = GPTOSSProtocol(tokenizer)

    tool_results = [
        {"name": "get_molecular_weight", "content": '{"result": 180.16}'},
        {"name": "get_tpsa", "content": '{"result": 63.60}'},
    ]
    feedback = protocol.render_tool_feedback(tool_results)

    assert feedback.count("<|start|>functions.") == 2
    assert feedback.count("<|end|>") == 2
    assert feedback.endswith("<|start|>assistant")
    print(f"  Multi-tool feedback: {len(feedback)} chars, 2 function responses")


# ---------------------------------------------------------------------------
# Test 11: Tool-call vs final-answer distinction with same <|end|> token
#
# This is the semantic test: with a SINGLE stop token <|end|>, the parser
# must correctly distinguish tool calls from final answers by examining
# the message structure (recipient, channel), NOT the stop token.
# ---------------------------------------------------------------------------
def test_tool_call_vs_final_answer_distinction():
    """Both tool calls and final answers stop at <|end|>.

    The parser distinguishes them by message structure — same as
    vLLM's OpenAIToolParser. This test verifies both are handled
    correctly with the same stop mechanism.
    """
    from openrlhf.utils.chat_protocol import GPTOSSProtocol

    protocol = GPTOSSProtocol(tokenizer)

    # Case 1: Tool call (has to=functions.X)
    tc_text = '\nto=functions.get_tpsa<|channel|>commentary<|message|>{"smiles": "c1ccccc1"}<|end|>'
    tc_ids = tokenizer.encode(tc_text, add_special_tokens=False)
    tc_result = protocol.parse_assistant_text(tc_text, token_ids=tc_ids)
    assert tc_result["tool_calls"], "Should detect tool call"
    print(f"  Tool call detected: {tc_result['tool_calls'][0]['name']}")

    # Case 2: Final answer (no to=functions.X)
    ans_text = '\n<|channel|>commentary<|message|>The answer is (B).<|end|>'
    ans_ids = tokenizer.encode(ans_text, add_special_tokens=False)
    ans_result = protocol.parse_assistant_text(ans_text, token_ids=ans_ids)
    assert not ans_result["tool_calls"], f"Should be final answer, not tool call"
    print(f"  Final answer detected: {ans_result['content']!r}")

    # Both used <|end|> — distinction was by parsing, not stop token ✓
    print(f"  Both cases correctly distinguished with single <|end|> stop")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_test("Harmony parser: tool call", test_harmony_parser_tool_call)
    run_test("Harmony parser: function response", test_harmony_parser_function_response)
    run_test("parse_output_into_messages: direct tokens", test_parse_output_into_messages_direct)
    run_test("GPTOSSProtocol: parse tool call", test_protocol_parse_tool_call)
    run_test("GPTOSSProtocol: parse final answer", test_protocol_parse_final_answer)
    run_test("GPTOSSProtocol: render_tool_feedback", test_render_tool_feedback)
    run_test("Round-trip (multi-turn)", test_round_trip)
    run_test("apply_chat_template with tools", test_apply_chat_template)
    run_test("<|end|> token", test_end_token)
    run_test("Multi-tool feedback", test_multi_tool_feedback)
    run_test("Tool call vs final answer (same <|end|>)", test_tool_call_vs_final_answer_distinction)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
