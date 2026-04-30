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
import inspect
import torch
from typing import Any, Callable, Dict, Optional

from transformers import AutoTokenizer

from openrlhf.utils.tool_versions import get_version, resolve_tool_metric_metadata
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from openrlhf.utils.chat_protocol import (
    GLM51Protocol,
    GLMFlashProtocol,
    GPTOSSProtocol,
    InternS1Protocol,
    KimiK2Protocol,
    Qwen3CoderProtocol,
    Qwen3Protocol,
)
from openrlhf.utils.tdc_reward_model import extract_final_answer


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


def _is_known_task_smiles_lookup_error(error_str: str) -> bool:
    if not error_str:
        return False
    return "is not part of task" in error_str and "requires a known task molecule" in error_str


def _infer_task_from_observation_text(observation_text: str) -> Optional[str]:
    """Infer the active TDC task for task-bound tools from the rendered prompt."""
    if not observation_text:
        return None

    trim_match = re.search(
        r"Retrieve text-form local analog evidence for task ([A-Za-z0-9_]+)\.",
        observation_text,
    )
    if trim_match:
        return trim_match.group(1)

    if '"name": "compare_similar_mols"' in observation_text or "'name': 'compare_similar_mols'" in observation_text:
        from openrlhf.tools.therapeutic_tools.similarity import TASKS

        hits = [task for task in TASKS if task in observation_text]
        if len(hits) == 1:
            return hits[0]
    return None


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

    def __init__(
        self,
        hf_tokenizer=None,
        reward_fn=None,
        discard_failed_tool_traces: bool = True,
        enable_tool_calling_rewards=True,
        tool_calling_reward_until_step: int = -1,
        tool_calling_reward_mode: str = "auto",
        tool_calling_reward_naive_per_call: float = 0.1,
        tool_calling_reward_feature_single: float = 0.1,
        tool_calling_reward_feature_full: float = 0.065,
        tool_calling_reward_feature_max_count: int = 21,
        tool_calling_reward_max_rewarded_calls: int = -1,
        current_global_step: int = -1,
        total_training_steps: int = -1,
    ):
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
        self._discard_failed_tool_traces = bool(discard_failed_tool_traces)
        self._enable_tool_calling_rewards = enable_tool_calling_rewards
        self._tool_calling_reward_until_step = int(tool_calling_reward_until_step)
        self._tool_calling_reward_mode = tool_calling_reward_mode
        self._tool_calling_reward_naive_per_call = tool_calling_reward_naive_per_call
        self._tool_calling_reward_feature_single = tool_calling_reward_feature_single
        self._tool_calling_reward_feature_full = tool_calling_reward_feature_full
        self._tool_calling_reward_feature_max_count = max(1, int(tool_calling_reward_feature_max_count))
        self._tool_calling_reward_max_rewarded_calls = int(tool_calling_reward_max_rewarded_calls)
        self._current_global_step = int(current_global_step)
        self._total_training_steps = int(total_training_steps)
        self._rewarded_tool_calls_so_far = 0

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
        elif protocol_name == "kimi_k2":
            self.protocol = KimiK2Protocol(self.tokenizer)
        elif protocol_name == "glm51":
            self.protocol = GLM51Protocol(self.tokenizer)
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
        self.tool_version = tool_version
        self._task: Optional[str] = None

    # ------------------------------------------------------------------
    # AgentInstanceBase interface
    # ------------------------------------------------------------------

    async def reset(self, states: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Passthrough — prompt is already chat-templated by preprocessing."""
        self._task = states.get("task")
        return {"observation": states.get("observation", "")}

    async def step(self, state_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Parse tool calls, execute, return upstream-contract dict."""
        self._current_observation_text = state_dict.get("observation", "")
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
            failed_tool_turn = False
            smiles_lookup_failed = False
            for tc, (result, dur, error_str) in zip(tool_calls, results):
                tool_name = tc.get("name", "")
                tool_msgs.append(
                    {
                        "name": tool_name,
                        "content": result,
                        "tool_call_id": tc.get("id", ""),
                    }
                )
                if tool_name:
                    key = f"tool_count__{tool_name}"
                    extra_logs[key] = extra_logs.get(key, 0) + 1
                if error_str:
                    failed_tool_turn = True
                    smiles_lookup_failed = smiles_lookup_failed or _is_known_task_smiles_lookup_error(error_str)

            feature_request_metrics = self._accumulate_feature_request_metrics(tool_calls)
            extra_logs.update(feature_request_metrics)
            extra_logs["requested_get_features"] = 1.0 if feature_request_metrics["get_features_request_count"] > 0 else 0.0
            extra_logs["tool_execution_failed"] = 1.0 if failed_tool_turn else 0.0
            extra_logs["smiles_lookup_failed"] = 1.0 if smiles_lookup_failed else 0.0
            extra_logs["discard_from_training"] = 1.0 if (failed_tool_turn and self._discard_failed_tool_traces) else 0.0

            # Bridge text: close assistant turn + tool responses + open next turn
            feedback = self.protocol.render_tool_feedback(tool_msgs)
            feedback_token_ids = self.protocol.render_tool_feedback_token_ids(tool_msgs)

            #### small tool-calling reward (parsed tool call > unparsed) ####
            # parse_method is only set by GPTOSSProtocol; None means a
            # non-GPT-OSS protocol parsed successfully — no bonus/penalty.
            parse_method = action.get("parse_method")
            reward_disabled_by_step = (
                self._tool_calling_reward_until_step >= 0
                and self._current_global_step >= self._tool_calling_reward_until_step
            )

            if not self._enable_tool_calling_rewards or reward_disabled_by_step:
                tool_calling_reward = 0
            elif failed_tool_turn:
                tool_calling_reward = 0
            elif parse_method is None:
                # Non-GPT-OSS protocol: no tool-calling shaping
                tool_calling_reward = 0
            elif parse_method in ("primary", "fallback", "regex"):
                resolved_mode = self._resolve_tool_calling_reward_mode()
                tool_calling_reward, rewarded_calls, suppressed_calls = self._compute_step_tool_calling_reward(
                    tool_calls, resolved_mode
                )
                extra_logs["tool_call_rewarded_count"] = rewarded_calls
                extra_logs["tool_call_reward_suppressed_count"] = suppressed_calls
            else:
                raise ValueError(f"Unknown parse method: {parse_method}")
            extra_logs["tool_calling_reward"] = tool_calling_reward
            #### end small tool-calling reward ####

            return {
                "environment_feedback": feedback,
                "environment_feedback_token_ids": feedback_token_ids,
                "rewards": torch.tensor(tool_calling_reward),
                "done": bool(failed_tool_turn and self._discard_failed_tool_traces),
                "scores": 0.0,
                "extra_logs": extra_logs,
            }

        # Parse failed: model attempted a tool call but mangled the format.
        # Apply an explicit penalty to discourage malformed harmony headers.
        parse_failed = action.get("parse_failed", False)
        if parse_failed:
            parse_penalty = 0 if self._enable_tool_calling_rewards else 0
            base_logs["tool_calling_reward"] = parse_penalty
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
        """Execute a tool call and return (result_json, duration_seconds, error_str)."""
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        t0 = time.monotonic()
        error_str = ""
        if tool_name not in self.tools:
            error_str = f"Unknown tool: {tool_name}"
            result = json.dumps(
                {
                    "error": error_str,
                    "available_tools": list(self.tools.keys()),
                }
            )
        else:
            try:
                if isinstance(arguments, dict):
                    try:
                        params = inspect.signature(self.tools[tool_name]).parameters
                    except (TypeError, ValueError):
                        params = {}
                    resolved_task = getattr(self, "_task", None) or _infer_task_from_observation_text(
                        getattr(self, "_current_observation_text", "")
                    )
                    if resolved_task:
                        stripped_task = re.sub(r"_(train|valid|validation|val|test|eval)$", "", resolved_task)
                        try:
                            from openrlhf.tools.therapeutic_tools.v11 import _resolve_task_name
                            canonical_task = (
                                _resolve_task_name(stripped_task)
                                or _resolve_task_name(resolved_task)
                                or stripped_task
                            )
                        except Exception:
                            canonical_task = stripped_task
                        for inject_name in ("task", "task_name"):
                            if inject_name in params:
                                arguments = dict(arguments)
                                arguments[inject_name] = canonical_task
                                break
                error_str, result = _exec_with_rdkit_log_capture(
                    self.tools[tool_name], arguments, tool_name
                )
            except Exception as e:
                error_str = str(e)
                result = json.dumps(
                    {
                        "error": error_str,
                        "function_name": tool_name,
                        "arguments": arguments,
                    }
                )
        duration = time.monotonic() - t0
        # Log any invalid SMILES errors to the SMILES error log file.
        if error_str:
            _write_smiles_error_log(tool_name, arguments, error_str)
        return result, duration, error_str

    def _resolve_tool_calling_reward_mode(self) -> str:
        mode = (self._tool_calling_reward_mode or "auto").strip().lower()
        if mode != "auto":
            return mode
        # Backwards compatibility: v10 used naive per-tool rewards.
        return "naive" if self.tool_version == "v10" else "feature_aware"

    def _tool_signature_parameters(self, tool_name: str) -> Dict[str, inspect.Parameter]:
        fn = self.tools.get(tool_name)
        if fn is None:
            return {}
        try:
            return dict(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            return {}

    def _tool_reward_family(self, tool_name: str) -> str:
        params = self._tool_signature_parameters(tool_name)
        feature_param = params.get("feature_names")
        if feature_param is None:
            return "naive"

        # Distinguish feature-query tools from neighbor lookup tools using
        # callable signatures rather than exact names so renamed variants keep
        # the same reward behavior.
        if "task_name" in params or "include_labels" in params or tool_name.startswith("get_neighbors"):
            return "neighbor_optional_features"

        if feature_param.default is inspect.Signature.empty:
            return "feature_count_scaled"

        return "naive"

    def _tool_feature_vocab_size(self, tool_name: str) -> Optional[int]:
        fn = self.tools.get(tool_name)
        if fn is None:
            return None
        module = inspect.getmodule(fn)
        feature_names = getattr(module, "FEATURE_NAMES", None)
        if isinstance(feature_names, (list, tuple, set)):
            try:
                size = len(feature_names)
            except TypeError:
                return None
            return size if size > 0 else None
        return None

    def _effective_feature_max_count(self, tool_name: str) -> int:
        configured = self._tool_calling_reward_feature_max_count
        available = self._tool_feature_vocab_size(tool_name)
        if available is None:
            return configured
        return max(1, min(configured, available))

    @staticmethod
    def _requested_feature_count(arguments: Any) -> Optional[int]:
        if not isinstance(arguments, dict):
            return None
        fn_list = arguments.get("feature_names")
        if not isinstance(fn_list, list):
            return None
        return len(fn_list)

    def _accumulate_feature_request_metrics(self, tool_calls: list[Dict[str, Any]]) -> Dict[str, float]:
        metrics = {
            "get_features_request_count": 0.0,
            "get_features_requested_feature_total": 0.0,
            "get_features_requested_feature_count_count": 0.0,
            "get_neighbors_request_count": 0.0,
            "get_neighbors_requested_feature_total": 0.0,
            "get_neighbors_requested_feature_count_count": 0.0,
        }
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name", ""))
            arguments = tool_call.get("arguments", {})
            metric_metadata = resolve_tool_metric_metadata(self.tool_version, tool_name)
            if metric_metadata is None or not metric_metadata.get("count_request_metric", True):
                continue

            endpoint = metric_metadata.get("endpoint")
            requested_feature_count = self._requested_feature_count(arguments) or 0
            supports_feature_selection = "feature_names" in self._tool_signature_parameters(tool_name)

            if endpoint == "features":
                metrics["get_features_request_count"] += 1.0
                if supports_feature_selection:
                    metrics["get_features_requested_feature_total"] += float(requested_feature_count)
                    metrics["get_features_requested_feature_count_count"] += 1.0
            elif endpoint == "neighbors":
                metrics["get_neighbors_request_count"] += 1.0
                if supports_feature_selection:
                    metrics["get_neighbors_requested_feature_total"] += float(requested_feature_count)
                    metrics["get_neighbors_requested_feature_count_count"] += 1.0

        return metrics

    def _compute_tool_call_reward(self, tool_call: Dict[str, Any], mode: str) -> float:
        if mode == "naive":
            return float(self._tool_calling_reward_naive_per_call)
        if mode == "feature_aware":
            tool_name = str(tool_call.get("name", ""))
            args = tool_call.get("arguments", {})
            family = self._tool_reward_family(tool_name)
            requested_feature_count = self._requested_feature_count(args)

            if family == "neighbor_optional_features":
                return float(
                    self._tool_calling_reward_feature_single
                    if requested_feature_count and requested_feature_count > 0
                    else self._tool_calling_reward_feature_full
                )

            if family == "feature_count_scaled" and requested_feature_count is not None:
                max_count = self._effective_feature_max_count(tool_name)
                n = max(1, min(requested_feature_count, max_count))
                if max_count == 1:
                    return float(self._tool_calling_reward_feature_single)
                span = self._tool_calling_reward_feature_single - self._tool_calling_reward_feature_full
                return float(
                    self._tool_calling_reward_feature_single
                    - span * (n - 1) / (max_count - 1)
                )
            return float(self._tool_calling_reward_naive_per_call)
        raise ValueError(f"Unknown tool calling reward mode: {mode}")

    def _compute_step_tool_calling_reward(self, tool_calls: list[Dict[str, Any]], mode: str) -> tuple[float, int, int]:
        if self._tool_calling_reward_max_rewarded_calls >= 0:
            remaining_rewarded_calls = max(
                0, self._tool_calling_reward_max_rewarded_calls - self._rewarded_tool_calls_so_far
            )
        else:
            remaining_rewarded_calls = None

        tool_calling_reward = 0.0
        rewarded_calls = 0
        suppressed_calls = 0
        for tc in tool_calls:
            if remaining_rewarded_calls is not None and rewarded_calls >= remaining_rewarded_calls:
                suppressed_calls += 1
                continue
            tool_calling_reward += self._compute_tool_call_reward(tc, mode)
            rewarded_calls += 1

        self._rewarded_tool_calls_so_far += rewarded_calls
        return tool_calling_reward, rewarded_calls, suppressed_calls

    _ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-Za-z])\s*\)?")

    def _default_reward_fn(self, generated_text: str, label: Optional[str]) -> float:
        """Default reward: 1.0 iff the model's Answer: (X) matches the label."""
        if not label:
            return 0.0

        predicted = extract_final_answer(generated_text)
        if not predicted:
            return 0.0
        pred = predicted.strip().strip("()").upper()

        # Extract letter from label too (handles "A", "(A)", "Answer: (A)", etc.)
        label_match = self._ANSWER_RE.search(label)
        if label_match:
            gold = label_match.group(1).upper()
        else:
            # Bare letter like "A" or "(A)"
            gold = label.strip().strip("()").upper()

        if pred != gold:
            return 0.0

        return 1.0


# ---------------------------------------------------------------------------
# Executor (required name for vllm_engine._load_agent_executor)
# ---------------------------------------------------------------------------
class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(
        self,
        reward_fn=None,
        length_penalty_start: int = 0,
        discard_failed_tool_traces: bool = True,
        enable_tool_calling_rewards: bool = True,
        tool_calling_reward_until_step: int = -1,
        tool_calling_reward_mode: str = "auto",
        tool_calling_reward_naive_per_call: float = 0.1,
        tool_calling_reward_feature_single: float = 0.1,
        tool_calling_reward_feature_full: float = 0.065,
        tool_calling_reward_feature_max_count: int = 21,
        tool_calling_reward_max_rewarded_calls: int = -1,
        tool_calling_reward_cap: float = 0.25,
        **kwargs,
    ):
        super().__init__(
            ToolCallingTurn,
            reward_fn=reward_fn,
            length_penalty_start=length_penalty_start,
            tool_calling_reward_cap=tool_calling_reward_cap,
            discard_failed_tool_traces=discard_failed_tool_traces,
            enable_tool_calling_rewards=enable_tool_calling_rewards,
            tool_calling_reward_until_step=tool_calling_reward_until_step,
            tool_calling_reward_mode=tool_calling_reward_mode,
            tool_calling_reward_naive_per_call=tool_calling_reward_naive_per_call,
            tool_calling_reward_feature_single=tool_calling_reward_feature_single,
            tool_calling_reward_feature_full=tool_calling_reward_feature_full,
            tool_calling_reward_feature_max_count=tool_calling_reward_feature_max_count,
            tool_calling_reward_max_rewarded_calls=tool_calling_reward_max_rewarded_calls,
            **kwargs,
        )


__all__ = ["ToolCallingTurn", "AgentExecutor"]
