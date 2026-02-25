"""Check what format the Harmony renderer library actually produces for tool calls."""
from openai_harmony import (
    Author, Conversation, DeveloperContent, HarmonyEncodingName,
    Message, Role, SystemContent, ToolDescription, load_harmony_encoding,
    ReasoningEffort
)
from vllm.entrypoints.openai.parser.harmony_utils import parse_output_into_messages

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build a conversation with a tool call
system_message = (
    SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date("2025-06-28")
)

developer_message = (
    DeveloperContent.new()
        .with_instructions("You are a chemistry assistant")
        .with_function_tools([
            ToolDescription.new(
                "get_mol_logp",
                "Gets the logP of a molecule",
                parameters={
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string", "description": "SMILES string"},
                    },
                    "required": ["smiles"],
                },
            ),
        ])
)

convo = Conversation.from_messages([
    Message.from_role_and_content(Role.SYSTEM, system_message),
    Message.from_role_and_content(Role.DEVELOPER, developer_message),
    Message.from_role_and_content(Role.USER, "Predict bioavailability of CC1=CC(=NC=C1)NC(=S)N2CCN(CC2)"),
    # Analysis (CoT)
    Message.from_role_and_content(
        Role.ASSISTANT, 'We need to compute logP.'
    ).with_channel("analysis"),
    # Tool call
    Message.from_role_and_content(Role.ASSISTANT, '{"smiles": "CC1=CC(=NC=C1)NC(=S)N2CCN(CC2)"}')
    .with_channel("commentary")
    .with_recipient("functions.get_mol_logp")
    .with_content_type("json"),
    # Tool response
    Message.from_author_and_content(
        Author.new(Role.TOOL, "functions.get_mol_logp"),
        '{"result": 2.5}',
    ).with_channel("commentary"),
])

# Render with generation prompt
tokens_for_completion = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# Decode the full rendered prompt to see the format
full_text = encoding.decode(tokens_for_completion)
print("=== Full rendered prompt (text) ===")
print(full_text)
print()

# Now look at the token structure around the tool call
print("=== Token analysis around tool call ===")
for i, t in enumerate(tokens_for_completion):
    decoded = encoding.decode([t])
    if t >= 200000:
        print(f"  [{i:4d}] {t:6d}  SPECIAL: {decoded!r}")
    elif "functions" in decoded or "to=" in decoded or "commentary" in decoded or "assistant" in decoded:
        print(f"  [{i:4d}] {t:6d}  >>> {decoded!r}")

# Now test: what does the model generate?
# After the generation prompt "<|start|>assistant", the model produces output tokens.
# Let's find where the assistant tool call starts and extract those tokens.
print("\n=== Looking for assistant tool call tokens in rendered prompt ===")
# Find <|start|>assistant segments
start_token = 200006  # <|start|>
assistant_word = encoding.encode("assistant")[0] if len(encoding.encode("assistant")) == 1 else None
for i, t in enumerate(tokens_for_completion):
    if t == start_token:
        # Check what follows
        next_few = tokens_for_completion[i:i+10]
        decoded_segment = encoding.decode(next_few)
        print(f"  <|start|> at {i}: {decoded_segment!r}")

# The key: extract just the assistant tool call message and parse it
# Find the analysis + tool call messages (after the generation prompt position in a real scenario)
print("\n=== Simulating model output parsing ===")
# Model would generate everything after the prompt's "<|start|>assistant"
# In this case, the analysis + tool call from the convo
analysis_msg = Message.from_role_and_content(Role.ASSISTANT, 'We need to compute logP.').with_channel("analysis")
tool_call_msg = (
    Message.from_role_and_content(Role.ASSISTANT, '{"smiles": "CC1=CC(=NC=C1)NC(=S)N2CCN(CC2)"}')
    .with_channel("commentary")
    .with_recipient("functions.get_mol_logp")
    .with_content_type("json")
)

# Render these messages to see exactly what tokens the model would output
output_convo = Conversation.from_messages([analysis_msg, tool_call_msg])
# Can't easily render just the output, so let's find in the full render
# Instead, let's try to find the generated portion

# Actually, let's look at the specific format by rendering individual messages
print("\n=== Individual message rendering ===")
for msg_name, msg in [("analysis", analysis_msg), ("tool_call", tool_call_msg)]:
    single = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, "test"),
        msg,
    ])
    toks = encoding.render_conversation_for_completion(single, Role.ASSISTANT)
    # Find the assistant message portion
    text = encoding.decode(toks)
    # Find the last <|start|>assistant and show what's around it
    idx = text.rfind("<|start|>assistant")
    if idx >= 0:
        before_gen = text[:idx]
        # Find the previous <|start|>assistant  
        prev_idx = before_gen.rfind("<|start|>assistant")
        if prev_idx >= 0:
            msg_text = text[prev_idx:idx]
            print(f"  {msg_name}: {msg_text!r}")
