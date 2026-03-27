import json
import logging
import re
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def clean_message(msg: dict) -> dict:
    cleaned = {}
    for k, v in msg.items():
        if v is None:
            continue
        cleaned[k] = v
    if "tool_calls" in cleaned:
        for tc in cleaned["tool_calls"]:
            if "function" in tc and isinstance(tc["function"].get("arguments"), str):
                try:
                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    pass
    return cleaned

def transform_conversation(messages: list[dict]) -> list[dict]:
    transformed = []
    for msg in messages:
        cleaned = clean_message(msg)
        cleaned.pop("reasoning", None)
        if "reasoning_content" in cleaned:
            cleaned["thinking"] = cleaned.pop("reasoning_content")
            
        if cleaned.get("content") == "":
            del cleaned["content"]
        if "tool_calls" in cleaned and len(cleaned["tool_calls"]) == 0:
            del cleaned["tool_calls"]
            
        # Merge consecutive assistant messages
        if transformed and transformed[-1]["role"] == "assistant" and cleaned["role"] == "assistant":
            prev = transformed[-1]
            if "tool_calls" not in prev and "tool_calls" not in cleaned:
                # Merge logic: older content becomes thinking, newer content remains content
                if "content" in prev:
                    prev["thinking"] = prev.pop("content")
                if "content" in cleaned:
                    prev["content"] = cleaned["content"]
            else:
                # If there are tool calls, we just keep them separate unless they're both tool calls
                transformed.append(cleaned)
        else:
            transformed.append(cleaned)
            
    return transformed

def patch_chat_template(template: str) -> str:
    # Use re.sub to loosely match whitespace to avoid failed replaces due to \r or indent changes
    # Analysis & content for tool calls
    template = re.sub(
        r'({%- elif message\.content and not future_final_message\.found %}\s*)(\{\{- "<\|start\|>assistant<\|channel\|>analysis<\|message\|>" \+ message\.content \+ "<\|end\|>" \}\})(\s*{%- elif message\.thinking and not future_final_message\.found %}\s*)(\{\{- "<\|start\|>assistant<\|channel\|>analysis<\|message\|>" \+ message\.thinking \+ "<\|end\|>" \}\})',
        r'{%- elif message.content %}\n                {% generation %}\2{% endgeneration %}\3{% generation %}\4{% endgeneration %}',
        template
    )
    
    # Tool call formatting
    template = re.sub(
        r'(\{\{- "<\|start\|>assistant to=" \}\}\s*\{\{- "functions\." \+ tool_call\.name \+ "<\|channel\|>commentary " \}\}\s*\{\{- \(tool_call\.content_type if tool_call\.content_type is defined else "json"\) \+ "<\|message\|>" \}\}\s*{%- if tool_call\.arguments is string %}\s*\{\{- tool_call\.arguments \}\}\s*{%- else %}\s*\{\{- tool_call\.arguments\|tojson \}\}\s*{%- endif %}\s*\{\{- "<\|call\|>" \}\})',
        r'{% generation %}\1{% endgeneration %}',
        template
    )
    
    # End of turn thinking
    template = re.sub(
        r'({#- This is a situation that should only occur in training, never in inference\. #}\s*{%- if "thinking" in message %}\s*)(\{\{- "<\|start\|>assistant<\|channel\|>analysis<\|message\|>" \+ message\.thinking \+ "<\|end\|>" \}\})(\s*{%- endif %}\s*{#- <\|return\|>)',
        r'\1{% generation %}\2{% endgeneration %}\3',
        template
    )
    
    # End of turn final content
    template = re.sub(
        r'(\{\{- "<\|start\|>assistant<\|channel\|>final<\|message\|>" \+ message\.content \+ "<\|end\|>" \}\})(\s*{%- elif "thinking" in message %}\s*{#- CoT is dropped during all previous turns)',
        r'{% generation %}\1{% endgeneration %}\2',
        template
    )
    
    # Mid-turn thinking
    template = re.sub(
        r'({#- CoT is dropped during all previous turns, so we never render it for inference #}\s*)(\{\{- "<\|start\|>assistant<\|channel\|>analysis<\|message\|>" \+ message\.thinking \+ "<\|end\|>" \}\})(\s*{%- set last_tool_call\.name = none %})',
        r'\1{% generation %}\2{% endgeneration %}\3',
        template
    )
    
    # Mid-turn final content
    template = re.sub(
        r'({#- CoT is dropped during all previous turns, so we never render it for inference #}\s*)(\{\{- "<\|start\|>assistant<\|channel\|>final<\|message\|>" \+ message\.content \+ "<\|end\|>" \}\})(\s*{%- set last_tool_call\.name = none %})',
        r'\1{% generation %}\2{% endgeneration %}\3',
        template
    )
    
    return template

def main():
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b", trust_remote_code=True)
    
    patched = patch_chat_template(tokenizer.chat_template)
    if "{% generation %}" not in patched:
        logger.error("Failed to patch chat template!")
    tokenizer.chat_template = patched
    
    trace_path = "/vast/home/j/jojolee/OpenRLHF-Tools/data/tdc/sft_traces/AMES_trace_24dichloroaniline.json"
    with open(trace_path, "r") as f:
        trace = json.load(f)
        
    messages = transform_conversation(trace["messages"])
    tools = [{
        "type": "function",
        "function": {
            "name": "get_molecule_profile",
            "description": "mock tool",
            "parameters": {"type": "object", "properties": {"smiles": {"type": "string"}}}
        }
    }]
    
    logger.info("Applying chat template and fetching masks")
    try:
        outputs = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = outputs["input_ids"]
        mask = outputs["assistant_masks"]
        
        print(f"Total tokens: {len(input_ids)}")
        print(f"Masked tokens (loss calculation): {sum(mask)}")
        
        # Let's print the first few masked segments to verify what is being trained on
        masked_tokens = []
        is_masked = False
        print("\n=== Masked Segments ===")
        for token, m in zip(input_ids, mask):
            if m:
                if not is_masked:
                    print("\n[START MASKED SEGMENT]")
                    is_masked = True
                print(tokenizer.decode([token]), end="")
                masked_tokens.append(token)
            else:
                if is_masked:
                    print("\n[END MASKED SEGMENT]")
                    is_masked = False
                    
        if is_masked:
            print("\n[END MASKED SEGMENT]")
            
    except Exception as e:
        logger.error(f"Error getting masks: {e}")
        # fallback
        rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
        print("Fallback Render:", rendered)

if __name__ == "__main__":
    main()
