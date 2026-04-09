"""Tool-calling turn for GRPO training — single-class agent.

Implements ``AgentInstanceBase`` for multi-turn tool-calling rollouts.

The initial prompt arrives **already formatted** by ``--apply_chat_template``
in the preprocessing step.  ``reset()`` is a passthrough; only ``step()``
does real work (parse tool calls, execute, produce bridge text via
``protocol.render_tool_feedback``).
"""

import asyncio
import json
import os
import re
import time
import datetime
import torch
from typing import Any, Callable, Dict, Optional

from transformers import AutoTokenizer

from openrlhf.utils.tool_versions import get_version
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from openrlhf.utils.chat_protocol import GLMFlashProtocol, GPTOSSProtocol, InternS1Protocol, Qwen3Protocol, Qwen3CoderProtocol


# ---------------------------------------------------------------------------
# RDKit log capture helpers
# ---------------------------------------------------------------------------

# Attempt to import RDKit log-capture utilities once at module load.
try:
    from rdkit.Chem import rdBase as _rdBase
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _exec_with_rdkit_log_capture(fn: Callable, arguments: dict, tool_name: str):
    """Call fn(**arguments), returning (error_msg_or_empty, result_json).

    Runs the tool call inside a RDKit BlockLogs context so C++ parse
    messages are silenced at the C++ layer (they were already reaching
    the terminal anyway).  Any exception is caught and returned as an
    empty error_str so the caller can still log the SMILES.

    Returns (error_str, result_json).
    error_str is empty when the call succeeded without RDKit errors.
    When non-empty it contains a description of what went wrong.
    """
    # Fix for gpt-oss "raw" keyword argument issue:
    # If arguments contains 'raw', it's usually a string that failed to parse as JSON.
    # Try to parse it and extract 'smiles' or 'query_smiles'.
    if "raw" in arguments and len(arguments) == 1:
        raw_val = arguments["raw"]
        if isinstance(raw_val, str):
            try:
                parsed = json.loads(raw_val)
                if isinstance(parsed, dict):
                    arguments = parsed
            except Exception:
                pass

    smiles_arg = arguments.get("smiles", arguments.get("query_smiles", ""))

    if _RDKIT_AVAILABLE:
        ctx = _rdBase.BlockLogs()
    else:
        ctx = None

    error_str = ""
    try:
        if ctx is not None:
            ctx.__enter__()
        
        # If the function doesn't take **kwargs, we might still hit issues if extra args are present.
        # But most of these tools take specific arguments.
        raw = fn(**arguments)
        result = json.dumps({"result": raw, "function_name": tool_name, "arguments": arguments})
        # Check if RDKit reported an invalid molecule (MolFromSmiles returned None).
        # Most wrappers raise ValueError for invalid SMILES, but some return 'invalid'.
        if isinstance(raw, str) and "invalid" in raw.lower() and smiles_arg:
            error_str = f"Tool returned indication of invalid SMILES: {raw!r}"
    except Exception as e:
        error_str = str(e)
        # If it's an "unexpected keyword argument" error, try to fall back to just the SMILES if possible
        if "got an unexpected keyword argument" in error_str and smiles_arg:
             try:
                 # Try calling with only the smiles/query_smiles arg
                 key = "smiles" if "smiles" in arguments else "query_smiles"
                 raw = fn(**{key: smiles_arg})
                 error_str = "" # suppress error since we recovered
                 result = json.dumps({"result": raw, "function_name": tool_name, "arguments": {key: smiles_arg}})
             except Exception as e2:
                 error_str = f"{error_str} | Fallback failed: {e2}"
                 result = json.dumps({"error": error_str, "function_name": tool_name, "arguments": arguments})
        else:
            result = json.dumps({"error": error_str, "function_name": tool_name, "arguments": arguments})
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return error_str, result


_SMILES_ERROR_LOG_PATH: Optional[str] = None  # resolved once on first call


def _write_smiles_error_log(tool_name: str, arguments: dict, error_str: str) -> None:
    """Append a JSON record to the SMILES error log file (if configured).

    The log path is read from the ``OPENRLHF_SMILES_ERROR_LOG`` environment
    variable.  Nothing happens when the variable is unset.
    """
    global _SMILES_ERROR_LOG_PATH
    if _SMILES_ERROR_LOG_PATH is None:
        _SMILES_ERROR_LOG_PATH = os.environ.get("OPENRLHF_SMILES_ERROR_LOG", "")
    if not _SMILES_ERROR_LOG_PATH:
        return

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tool": tool_name,
        "smiles": arguments.get("smiles", arguments.get("query_smiles", None)),
        "extra_args": {k: v for k, v in arguments.items() if k not in ("smiles", "query_smiles")},
        "error": error_str,
    }
    try:
        os.makedirs(os.path.dirname(_SMILES_ERROR_LOG_PATH), exist_ok=True)
        with open(_SMILES_ERROR_LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never crash training over a logging failure


class ToolCallingTurn(AgentInstanceBase):
    """One trajectory of a tool-calling agent.

    Each instance owns its own conversation history and is created fresh by
    ``MultiTurnAgentExecutor.execute`` for every rollout.

    The prompt is expected to arrive already chat-templated (via
    ``--apply_chat_template`` in the data preprocessing).  The protocol is
    only used for two things:
      - ``parse_assistant_text``: extract tool calls from vLLM output
      - ``render_tool_feedback``: produce the bridge text between turns
    """

    def __init__(self, hf_tokenizer=None, reward_fn=None, enable_tool_calling_rewards=True):
        # ---- tokenizer (needed by protocol parsers) ----
        if hf_tokenizer is not None:
            self.tokenizer = hf_tokenizer
        else:
            model_path = os.environ.get("OPENRLHF_MODEL_PATH")
            if not model_path:
                raise ValueError("OPENRLHF_MODEL_PATH environment variable must be set")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ---- optional injectable reward function ----
        # Signature: (generated_text: str, label: str) -> float
        # Falls back to _default_reward_fn (A/B letter extraction) when None.
        self._reward_fn = reward_fn
        self._enable_tool_calling_rewards = enable_tool_calling_rewards

        # ---- protocol (parse + feedback only, not initial rendering) ----
        protocol_name = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "glm_flash")
        if protocol_name == "intern_s1":
            self.protocol = InternS1Protocol(self.tokenizer)
        elif protocol_name == "gpt_oss":
            self.protocol = GPTOSSProtocol(self.tokenizer)
        elif protocol_name == "qwen3":
            self.protocol = Qwen3Protocol(self.tokenizer)
        elif protocol_name == "qwen3_5":
            self.protocol = Qwen3CoderProtocol(self.tokenizer)
        elif protocol_name == "glm_flash":
            self.protocol = GLMFlashProtocol(self.tokenizer)
        else:
            raise ValueError(f"Unsupported chat protocol: {protocol_name}")

        # ---- tool callables (from version registry) ----
        tool_version = os.environ.get("OPENRLHF_TOOL_VERSION")
        if not tool_version:
            raise ValueError(
                "OPENRLHF_TOOL_VERSION environment variable must be set "
                "(e.g. v1, v2, v3, v4). Pass --tool_version to train_ppo_ray.py."
            )
        ver_cfg = get_version(tool_version)
        self.tools: Dict[str, Callable] = dict(ver_cfg["callables"])

    # ------------------------------------------------------------------
    # AgentInstanceBase interface
    # ------------------------------------------------------------------

    async def reset(self, states: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Passthrough — prompt is already chat-templated by preprocessing."""
        return {"observation": states.get("observation", "")}

    async def step(self, state_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Parse tool calls, execute, return upstream-contract dict."""
        action_text = state_dict["action_text"]
        action_token_ids = state_dict.get("action_token_ids")
        label = state_dict.get("label", "")
        action = self.protocol.parse_assistant_text(action_text, token_ids=action_token_ids)
        tool_calls = action.get("tool_calls", [])
        base_logs = {
            "tool_call_count": len(tool_calls),
        }
        if "parse_method" in action:
            parse_method = action["parse_method"]
            parse_failed = action.get("parse_failed", False)
            base_logs[f"parse_method__{parse_method}"] = 1
            base_logs["parse_failed"] = 1 if parse_failed else 0
            base_logs["tool_call_attempted"] = 1 if (len(tool_calls) > 0 or parse_failed) else 0

        if tool_calls:
            # Execute all tool calls in parallel, preserving order
            results = await asyncio.gather(*[self._execute_tool(tc) for tc in tool_calls])
            tool_msgs = []
            extra_logs = base_logs.copy()
            # Timing stats: each result carries its own duration
            durations = [r[1] for r in results]
            extra_logs["tool_time_total"] = sum(durations)
            extra_logs["tool_time_max_call"] = max(durations)
            for tc, (result, dur) in zip(tool_calls, results):
                tool_name = tc.get("name", "")
                tool_msgs.append({"name": tool_name, "content": result})
                if tool_name:
                    key = f"tool_count__{tool_name}"
                    extra_logs[key] = extra_logs.get(key, 0) + 1

            # Bridge text: close assistant turn + tool responses + open next turn
            feedback = self.protocol.render_tool_feedback(tool_msgs)
            feedback_token_ids = self.protocol.render_tool_feedback_token_ids(tool_msgs)

            #### small reward for well-formatted tool calls (harmony > regex > unparsed) ####
            # parse_method is only set by GPTOSSProtocol; None means a
            # non-GPT-OSS protocol parsed successfully — no bonus/penalty.
            parse_method = action.get("parse_method")
            if not self._enable_tool_calling_rewards:
                format_reward = 0
            elif parse_method is None:
                # Non-GPT-OSS protocol: no format shaping
                format_reward = 0
            elif parse_method == "primary":
                # Best case: harmony token-ID parser succeeded on first try
                format_reward = 0.1
            elif parse_method == "fallback":
                # Harmony succeeded after prepending assistant header — neutral
                format_reward = 0.1
            elif parse_method == "regex":
                # Had to fall back to regex — mild penalty
                format_reward = 0.1
            else:
                raise ValueError(f"Unknown parse method: {parse_method}")
            extra_logs["format_reward"] = format_reward
            #### end small reward for well-formatted tool calls ####

            return {
                "environment_feedback": feedback,
                "environment_feedback_token_ids": feedback_token_ids,
                "rewards": torch.tensor(format_reward),
                "done": False,
                "scores": 0.0,
                "extra_logs": extra_logs,
            }

        # Parse failed: model attempted a tool call but mangled the format.
        # Apply an explicit penalty to discourage malformed harmony headers.
        parse_failed = action.get("parse_failed", False)
        if parse_failed:
            parse_penalty = 0 if self._enable_tool_calling_rewards else 0
            base_logs["format_reward"] = parse_penalty
            return {
                "environment_feedback": "",
                "rewards": torch.tensor(parse_penalty),
                "done": True,
                "scores": 0.0,
                "extra_logs": base_logs,
            }

        # No tool calls → final answer
        generated_text = action.get("content", "")
        if self._reward_fn is not None:
            reward = self._reward_fn(generated_text, label)
        else:
            reward = self._default_reward_fn(generated_text, label)
        return {
            "environment_feedback": "",
            "rewards": torch.tensor(reward),
            "done": True,
            "scores": reward,
            "extra_logs": base_logs,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> tuple:
        """Execute a tool call and return (result_json, duration_seconds)."""
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        t0 = time.monotonic()
        error_str = ""
        if tool_name not in self.tools:
            result = json.dumps(
                {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": list(self.tools.keys()),
                }
            )
        else:
            try:
                error_str, result = _exec_with_rdkit_log_capture(
                    self.tools[tool_name], arguments, tool_name
                )
            except Exception as e:
                result = json.dumps(
                    {
                        "error": str(e),
                        "function_name": tool_name,
                        "arguments": arguments,
                    }
                )
        duration = time.monotonic() - t0
        # Log any invalid SMILES errors to the SMILES error log file.
        if error_str:
            _write_smiles_error_log(tool_name, arguments, error_str)
        return result, duration

    _ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-Za-z])\s*\)?")
    _PAREN_ANSWER_RE = re.compile(r"\(\s*([A-Za-z])\s*\)")

    def _default_reward_fn(self, generated_text: str, label: Optional[str]) -> float:
        """Default reward: 1.0 iff the model's Answer: (X) matches the label."""
        if not label:
            return 0.0

        # Prefer the post-think region when present; fall back to full text
        # if the answer is inside the <think> block (common with Qwen3.5).
        think_end = generated_text.find("</think>")
        if think_end != -1:
            answer_region = generated_text[think_end:]
        else:
            answer_region = generated_text

        match = self._ANSWER_RE.search(answer_region)
        # If nothing found in post-think region, search the full text
        # (models like Qwen3.5 may place Answer: inside the <think> block).
        if not match and think_end != -1:
            match = self._ANSWER_RE.search(generated_text)
        if match:
            pred = match.group(1).upper()
        else:
            # GPT-OSS frequently emits bare "(A)"/"(B)" without "Answer:" prefix.
            # Search post-think first, then fall back to full text.
            search_regions = [answer_region] if think_end == -1 else [answer_region, generated_text]
            pred = None
            for region in search_regions:
                paren_matches = self._PAREN_ANSWER_RE.findall(region)
                if paren_matches:
                    pred = paren_matches[-1].upper()
                    break
            if pred is None:
                for region in search_regions:
                    stripped = region.strip()
                    if len(stripped) == 1 and stripped.isalpha():
                        pred = stripped.upper()
                        break
            if pred is None:
                return 0.0

        # Extract letter from label too (handles "A", "(A)", "Answer: (A)", etc.)
        label_match = self._ANSWER_RE.search(label)
        if label_match:
            gold = label_match.group(1).upper()
        else:
            # Bare letter like "A" or "(A)"
            gold = label.strip().strip("()").upper()

        return 1.0 if pred == gold else 0.0


# ---------------------------------------------------------------------------
# Executor (required name for vllm_engine._load_agent_executor)
# ---------------------------------------------------------------------------
class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self, reward_fn=None, length_penalty_start: int = 0, enable_tool_calling_rewards: bool = True, **kwargs):
        super().__init__(ToolCallingTurn, reward_fn=reward_fn, length_penalty_start=length_penalty_start, enable_tool_calling_rewards=enable_tool_calling_rewards, **kwargs)


__all__ = ["ToolCallingTurn", "AgentExecutor"]
