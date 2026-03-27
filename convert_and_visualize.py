import json
import logging
import re
from pathlib import Path
from copy import deepcopy
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ANSI color codes
BLUE = '\033[94m'
RESET = '\033[0m'

ANSWER_ONLY_RE = re.compile(r"^\s*Answer:\s*\([A-Z]\)\s*$")
FIXED_TEMPLATE_PATH = Path(
    "/vast/home/j/jojolee/OpenRLHF-Tools/openrlhf/models/fixed_gpt_oss_template.txt"
)

def split_multi_tool_call_turns(messages: list[dict]) -> list[dict]:
    """
    Split assistant messages with multiple tool calls into multiple assistant turns,
    each containing exactly one tool call.

    Tool messages are attached to the matching split assistant turn by tool_call_id.
    Reasoning/thinking is kept only on the first split assistant message.
    """
    result = []
    i = 0

    while i < len(messages):
        msg = deepcopy(messages[i])

        if (
            msg.get("role") == "assistant"
            and isinstance(msg.get("tool_calls"), list)
            and len(msg["tool_calls"]) > 1
        ):
            tool_calls = msg["tool_calls"]

            # Gather following tool messages that belong to this assistant turn
            tool_msgs = []
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_msgs.append(deepcopy(messages[j]))
                j += 1

            tool_msg_by_id = {
                tm.get("tool_call_id"): tm
                for tm in tool_msgs
                if tm.get("tool_call_id") is not None
            }

            base = deepcopy(msg)
            base.pop("tool_calls", None)

            for idx, tc in enumerate(tool_calls):
                split_assistant = deepcopy(base)
                split_assistant["tool_calls"] = [deepcopy(tc)]

                # Keep reasoning/content only on the first split assistant turn
                if idx > 0:
                    split_assistant.pop("thinking", None)
                    split_assistant.pop("content", None)

                result.append(split_assistant)

                tc_id = tc.get("id")
                if tc_id in tool_msg_by_id:
                    result.append(tool_msg_by_id[tc_id])

            i = j
            continue

        result.append(msg)
        i += 1

    return result

def parse_json_maybe(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def load_fixed_chat_template(template_path: Path = FIXED_TEMPLATE_PATH) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Chat template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def clean_message(msg: dict) -> dict:
    cleaned = {}
    for k, v in msg.items():
        if v is None:
            continue
        if k == "channel":
            continue
        cleaned[k] = deepcopy(v)

    if "tool_calls" in cleaned:
        for tc in cleaned["tool_calls"]:
            if "function" in tc and isinstance(tc["function"].get("arguments"), str):
                tc["function"]["arguments"] = parse_json_maybe(tc["function"]["arguments"])

    return cleaned


def remove_empty_fields(msg: dict) -> dict:
    if msg.get("content") == "":
        del msg["content"]

    if "thinking" in msg and isinstance(msg["thinking"], str) and not msg["thinking"].strip():
        del msg["thinking"]

    if "tool_calls" in msg and len(msg["tool_calls"]) == 0:
        del msg["tool_calls"]

    return msg


def normalise_assistant_tool_call_message(msg: dict) -> dict:
    """
    If an assistant message has tool_calls and visible prose in `content`,
    move that prose into `thinking`.
    """
    if msg.get("role") != "assistant":
        return msg
    if "tool_calls" not in msg:
        return msg

    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        if "thinking" in msg and isinstance(msg["thinking"], str) and msg["thinking"].strip():
            msg["thinking"] = msg["thinking"].rstrip() + "\n\n" + content.strip()
        else:
            msg["thinking"] = content.strip()
        del msg["content"]

    return msg


def convert_reasoning_before_final_answer(messages: list[dict]) -> list[dict]:
    """
    Convert:
      assistant(content=reasoning trace)
      assistant(content="Answer: (X)")
    into:
      assistant(thinking=reasoning trace)
      assistant(content="Answer: (X)")
    """
    if not messages:
        return messages

    repaired = []
    i = 0

    while i < len(messages):
        current = deepcopy(messages[i])

        if (
            i + 1 < len(messages)
            and current.get("role") == "assistant"
            and "tool_calls" not in current
            and messages[i + 1].get("role") == "assistant"
            and "tool_calls" not in messages[i + 1]
        ):
            nxt = deepcopy(messages[i + 1])
            nxt_content = nxt.get("content", "")

            if isinstance(nxt_content, str) and ANSWER_ONLY_RE.match(nxt_content):
                current_content = current.get("content", "")
                if isinstance(current_content, str) and current_content.strip():
                    if "thinking" in current and isinstance(current["thinking"], str) and current["thinking"].strip():
                        current["thinking"] = current["thinking"].rstrip() + "\n\n" + current_content.strip()
                    else:
                        current["thinking"] = current_content.strip()
                    current.pop("content", None)

                repaired.append(remove_empty_fields(current))
                repaired.append(remove_empty_fields(nxt))
                i += 2
                continue

        repaired.append(remove_empty_fields(current))
        i += 1

    return repaired


def transform_conversation(messages: list[dict]) -> list[dict]:
    transformed = []

    for msg in messages:
        cleaned = clean_message(msg)

        cleaned.pop("reasoning", None)

        if "reasoning_content" in cleaned:
            cleaned["thinking"] = cleaned.pop("reasoning_content")

        cleaned = normalise_assistant_tool_call_message(cleaned)
        cleaned = remove_empty_fields(cleaned)
        transformed.append(cleaned)

    transformed = split_multi_tool_call_turns(transformed)
    transformed = convert_reasoning_before_final_answer(transformed)
    transformed = [remove_empty_fields(m) for m in transformed]
    return transformed


def main():
    input_dir = Path("/vast/home/j/jojolee/OpenRLHF-Tools/data/tdc/sft_traces")
    output_dir = Path("/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/data/converted_sft_traces")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting traces...")
    trace_files = list(input_dir.glob("*.json"))
    sample_trace = None

    for file_path in trace_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                trace = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue

        if "messages" in trace:
            trace["messages"] = transform_conversation(trace["messages"])

        if sample_trace is None:
            sample_trace = trace

        out_path = output_dir / file_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

    logger.info(f"Converted {len(trace_files)} traces and saved to {output_dir}")

    logger.info("Generating colored visualization for a sample trace...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b", trust_remote_code=True)
    tokenizer.chat_template = load_fixed_chat_template()

    logger.info("Loaded chat template from %s", FIXED_TEMPLATE_PATH)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_molecule_profile",
            "description": "mock tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"}
                }
            }
        }
    }]

    outputs = tokenizer.apply_chat_template(
        sample_trace["messages"],
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )

    input_ids = outputs["input_ids"]
    masks = outputs["assistant_masks"]

    colored_output = []

    for token, mask in zip(input_ids, masks):
        decoded = tokenizer.decode([token])
        if mask == 1:
            colored_output.append(f"{BLUE}{decoded}{RESET}")
        else:
            colored_output.append(decoded)

    vis_file = output_dir / "visualization.txt"
    with open(vis_file, "w", encoding="utf-8") as f:
        f.write(f"{BLUE}--- THIS TEXT IS BLUE: THESE TOKENS ARE MASK=1 (TRAINED ON) ---{RESET}\n")
        f.write("--- NORMAL TEXT: THESE TOKENS ARE MASK=0 (IGNORED IN LOSS) ---\n\n")
        f.write("".join(colored_output))

    logger.info(f"Visualization saved to {vis_file}")
    print(f"To view the colored output, run: cat {vis_file}")


if __name__ == "__main__":
    main()