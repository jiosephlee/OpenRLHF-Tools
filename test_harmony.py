"""Compare HF tokenizer vs harmony encoding token IDs for the same prompt.

This checks if `apply_chat_template()` → HF tokenizer produces the same
token IDs as `render_conversation_for_completion()` from the harmony library.

Also tests `GPTOSSProtocol.render_tool_feedback_token_ids()` to verify that
canonical harmony token IDs are produced for tool feedback messages.
"""
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from openai_harmony import (
    HarmonyEncodingName, Role, Author, Message, Conversation,
    SystemContent, DeveloperContent, ToolDescription,
    load_harmony_encoding, ReasoningEffort,
)
from transformers import AutoTokenizer

MODEL = "openai/gpt-oss-20b"

print("Loading harmony encoding...")
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

print(f"Loading HF tokenizer ({MODEL})...")
hf_tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Build a simple conversation
messages_for_hf = [
    {"role": "user", "content": "What is logP of aspirin?"},
]
tools = [{
    "type": "function",
    "function": {
        "name": "get_mol_logp",
        "description": "Return Wildman-Crippen cLogP",
        "parameters": {
            "type": "object",
            "properties": {"smiles": {"type": "string"}},
            "required": ["smiles"],
        },
    },
}]

# --- Path 1: HF apply_chat_template ---
# First try without tools to avoid template format issues
hf_prompt_no_tools = hf_tok.apply_chat_template(
    messages_for_hf, tokenize=False, add_generation_prompt=True
)
hf_token_ids_no_tools = hf_tok(hf_prompt_no_tools, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()

print(f"\n=== HF Path (no tools) ===")
print(f"  Prompt text (last 100): ...{hf_prompt_no_tools[-100:]!r}")
print(f"  Token count: {len(hf_token_ids_no_tools)}")
print(f"  Last 10 tokens: {hf_token_ids_no_tools[-10:]}")

# With tools
try:
    hf_prompt = hf_tok.apply_chat_template(
        messages_for_hf, tools=tools, tokenize=False, add_generation_prompt=True
    )
    hf_token_ids = hf_tok(hf_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    print(f"\n=== HF Path (with tools) ===")
    print(f"  Prompt text (last 100): ...{hf_prompt[-100:]!r}")
    print(f"  Token count: {len(hf_token_ids)}")
    print(f"  Last 10 tokens: {hf_token_ids[-10:]}")
except Exception as e:
    print(f"\n=== HF Path (with tools) FAILED: {e} ===")
    hf_token_ids = hf_token_ids_no_tools
    hf_prompt = hf_prompt_no_tools

# --- Path 2: Harmony render_conversation_for_completion ---
convo = Conversation.from_messages([
    Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent.new().with_reasoning_effort(ReasoningEffort.MEDIUM),
    ),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_function_tools([
            ToolDescription.new(
                "get_mol_logp",
                "Return Wildman-Crippen cLogP",
                parameters={
                    "type": "object",
                    "properties": {"smiles": {"type": "string"}},
                    "required": ["smiles"],
                },
            ),
        ]),
    ),
    Message.from_role_and_content(Role.USER, "What is logP of aspirin?"),
])
harmony_token_ids = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

print(f"\n=== Harmony Path ===")
harmony_text = enc.decode(harmony_token_ids)
print(f"  Prompt text (last 100): ...{harmony_text[-100:]!r}")
print(f"  Token count: {len(harmony_token_ids)}")
print(f"  Last 10 tokens: {harmony_token_ids[-10:]}")

# --- Compare ---
print(f"\n=== Comparison ===")
print(f"  HF tokens:      {len(hf_token_ids)}")
print(f"  Harmony tokens:  {len(harmony_token_ids)}")
print(f"  Token IDs match: {hf_token_ids == harmony_token_ids}")

# If they don't match, find where they diverge
if hf_token_ids != harmony_token_ids:
    min_len = min(len(hf_token_ids), len(harmony_token_ids))
    for i in range(min_len):
        if hf_token_ids[i] != harmony_token_ids[i]:
            print(f"\n  First divergence at position {i}:")
            print(f"    HF:      {hf_token_ids[max(0,i-3):i+5]}")
            print(f"    Harmony: {harmony_token_ids[max(0,i-3):i+5]}")
            # Decode around divergence
            print(f"    HF decoded:      {hf_tok.decode(hf_token_ids[max(0,i-3):i+5])!r}")
            print(f"    Harmony decoded: {enc.decode(harmony_token_ids[max(0,i-3):i+5])!r}")
            break

    # Show last tokens of each
    print(f"\n  HF last 20 decoded:      {hf_tok.decode(hf_token_ids[-20:])!r}")
    print(f"  Harmony last 20 decoded: {enc.decode(harmony_token_ids[-20:])!r}")


# ===================================================================
# Test: render_tool_feedback_token_ids (GPTOSSProtocol)
# ===================================================================
print(f"\n{'='*60}")
print("=== Test: render_tool_feedback_token_ids ===")
print(f"{'='*60}")

from openrlhf.utils.chat_protocol import GPTOSSProtocol

protocol = GPTOSSProtocol(hf_tok)

tool_results = [
    {"name": "get_mol_logp", "content": json.dumps({"result": 1.31, "function_name": "get_mol_logp"})},
]

# Path A: Canonical token IDs from harmony encoder
canonical_ids = protocol.render_tool_feedback_token_ids(tool_results)
print(f"\n  Canonical token IDs ({len(canonical_ids)} tokens): {canonical_ids}")
print(f"  Decoded: {enc.decode(canonical_ids)!r}")

# Path B: Text → HF tokenizer (the old path we're replacing)
feedback_text = protocol.render_tool_feedback(tool_results)
hf_feedback_ids = hf_tok(feedback_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
print(f"\n  HF text feedback ({len(hf_feedback_ids)} tokens): {hf_feedback_ids}")
print(f"  Feedback text: {feedback_text!r}")

# Compare
match = canonical_ids == hf_feedback_ids
print(f"\n  Token IDs match: {match}")
if not match:
    print(f"  ** MISMATCH — this is the bug we are fixing! **")
    min_len = min(len(canonical_ids), len(hf_feedback_ids))
    for i in range(min_len):
        if canonical_ids[i] != hf_feedback_ids[i]:
            print(f"  First divergence at position {i}:")
            print(f"    Canonical: {canonical_ids[max(0,i-2):i+3]}")
            print(f"    HF:        {hf_feedback_ids[max(0,i-2):i+3]}")
            print(f"    Canonical decoded: {enc.decode(canonical_ids[max(0,i-2):i+3])!r}")
            print(f"    HF decoded:        {hf_tok.decode(hf_feedback_ids[max(0,i-2):i+3])!r}")
            break
    if len(canonical_ids) != len(hf_feedback_ids):
        print(f"  Length difference: canonical={len(canonical_ids)} vs HF={len(hf_feedback_ids)}")

# Verify the canonical IDs match what harmony's enc.render() produces directly
print(f"\n--- Verification: canonical IDs match enc.render() ---")
expected_ids = []
for tr in tool_results:
    msg = Message.from_author_and_content(
        Author.new(Role.TOOL, f"functions.{tr['name']}"),
        tr["content"],
    ).with_channel("commentary").with_recipient("assistant")
    expected_ids.extend(enc.render(msg))
# Add <|start|>assistant header
expected_ids.extend(protocol._assistant_header_ids)

ids_match_expected = canonical_ids == expected_ids
print(f"  Canonical matches enc.render(): {ids_match_expected}")
if not ids_match_expected:
    print(f"  ** MISMATCH **")
    min_len = min(len(canonical_ids), len(expected_ids))
    for i in range(min_len):
        if canonical_ids[i] != expected_ids[i]:
            print(f"  First divergence at position {i}:")
            print(f"    Canonical: {canonical_ids[max(0,i-2):i+3]}")
            print(f"    Expected:  {expected_ids[max(0,i-2):i+3]}")
            print(f"    Canonical decoded: {enc.decode(canonical_ids[max(0,i-2):i+3])!r}")
            print(f"    Expected decoded:  {enc.decode(expected_ids[max(0,i-2):i+3])!r}")
            break
assert ids_match_expected, "render_tool_feedback_token_ids must match enc.render() output!"

# Test with multiple tool results
print(f"\n--- Multi-tool test ---")
multi_tool_results = [
    {"name": "get_mol_logp", "content": json.dumps({"result": 1.31})},
    {"name": "get_mol_weight", "content": json.dumps({"result": 180.16})},
]
multi_ids = protocol.render_tool_feedback_token_ids(multi_tool_results)
multi_text = protocol.render_tool_feedback(multi_tool_results)
print(f"  Multi-tool canonical: {len(multi_ids)} tokens")
print(f"  Multi-tool decoded: {enc.decode(multi_ids)!r}")
print(f"  Multi-tool text:    {multi_text!r}")

# Verify base class default returns None
from openrlhf.utils.chat_protocol import GLMFlashProtocol
glm = GLMFlashProtocol(hf_tok)
assert glm.render_tool_feedback_token_ids(tool_results) is None, "Base class default should return None"
print(f"\n  GLMFlashProtocol.render_tool_feedback_token_ids() returns None: PASS")

print(f"\n{'='*60}")
print("All tests passed!")
print(f"{'='*60}")
