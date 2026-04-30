"""Reconstruct OpenAI-style message lists from gpt-oss training samples.

Pulled out of ppo_trainer so the rollout-side full-trace writer can import it
without dragging in the trainer module.
"""

from __future__ import annotations

import re
from typing import Optional

import torch

from openrlhf.utils.chat_protocol import GPTOSSProtocol


_GPT_OSS_ANALYSIS_BLOCK_RE = re.compile(
    r"(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)
_GPT_OSS_FINAL_BLOCK_RE = re.compile(
    r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)
_GPT_OSS_TOOL_FEEDBACK_RE = re.compile(
    r"<\|start\|>functions\.(?P<name>[^ <|]+)\s+to=assistant<\|channel\|>commentary<\|message\|>(?P<content>.*?)<\|end\|>",
    re.DOTALL,
)


def _join_nonempty_blocks(blocks) -> Optional[str]:
    cleaned = [block.strip() for block in blocks if isinstance(block, str) and block.strip()]
    return "\n\n".join(cleaned) if cleaned else None


def _assistant_ranges_from_action_mask(action_mask: torch.Tensor) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if action_mask is None:
        return ranges

    ones = torch.where(action_mask.flatten().bool())[0].tolist()
    if not ones:
        return ranges

    start = ones[0]
    prev = ones[0]
    for idx in ones[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start + 1, prev + 2))
        start = idx
        prev = idx
    ranges.append((start + 1, prev + 2))
    return ranges


def _parse_gpt_oss_assistant_message(tokenizer, token_ids: list[int], call_id_prefix: str) -> tuple[dict, list[str]]:
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    protocol = GPTOSSProtocol(tokenizer)
    parsed = protocol.parse_assistant_text(raw_text, token_ids=token_ids)
    thinking = _join_nonempty_blocks(_GPT_OSS_ANALYSIS_BLOCK_RE.findall(raw_text))
    tool_calls = parsed.get("tool_calls") or []

    if tool_calls:
        message = {"role": "assistant", "tool_calls": []}
        if thinking:
            message["thinking"] = thinking
        tool_call_ids: list[str] = []
        for idx, tool_call in enumerate(tool_calls, start=1):
            tool_call_id = f"{call_id_prefix}_{idx}"
            tool_call_ids.append(tool_call_id)
            message["tool_calls"].append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": tool_call.get("arguments", {}),
                    },
                }
            )
        return message, tool_call_ids

    content = _join_nonempty_blocks(_GPT_OSS_FINAL_BLOCK_RE.findall(raw_text))
    if not content:
        content = (tokenizer.decode(token_ids, skip_special_tokens=True) or "").strip()

    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    return message, []


def _parse_gpt_oss_tool_messages(tokenizer, token_ids: list[int], tool_call_ids: list[str]) -> list[dict]:
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    messages: list[dict] = []
    for idx, match in enumerate(_GPT_OSS_TOOL_FEEDBACK_RE.finditer(raw_text)):
        tool_call_id = tool_call_ids[idx] if idx < len(tool_call_ids) else f"tool_{idx + 1}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": match.group("name"),
                "content": match.group("content").strip(),
            }
        )
    return messages


def reconstruct_gpt_oss_trace_messages(tokenizer, sample, sample_idx: int) -> list[dict]:
    """Walk an Experience-like sample's action ranges and return a message list.

    ``sample`` must expose ``sequences`` (LongTensor[1, T]) and ``action_mask``
    (LongTensor[1, T-1]) — i.e. an ``Experience`` or ``Samples`` instance.
    """
    if sample.sequences is None:
        return []

    sequence = sample.sequences[0].tolist()
    action_mask = sample.action_mask[0] if sample.action_mask is not None else None
    action_ranges = _assistant_ranges_from_action_mask(action_mask)
    if not action_ranges:
        return []

    messages: list[dict] = []
    for turn_idx, (start, end) in enumerate(action_ranges, start=1):
        assistant_message, tool_call_ids = _parse_gpt_oss_assistant_message(
            tokenizer,
            sequence[start:end],
            call_id_prefix=f"call_s{sample_idx}_t{turn_idx}",
        )
        messages.append(assistant_message)

        next_start = action_ranges[turn_idx][0] if turn_idx < len(action_ranges) else len(sequence)
        if end >= next_start:
            continue

        tool_messages = _parse_gpt_oss_tool_messages(tokenizer, sequence[end:next_start], tool_call_ids)
        messages.extend(tool_messages)

    return messages
