import heapq
import math
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import timedelta
from typing import Any, List, Optional, Tuple, Union

import ray
import torch
from tqdm import tqdm
from vllm import SamplingParams

from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.ppo_utils.length_penalty import apply_length_penalties
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.tool_versions import resolve_tool_metric_endpoint
from openrlhf.utils.utils import zero_pad_sequences

logger = init_logger(__name__)

_PREPENDED_NEIGHBOR_CONTEXT_RE = re.compile(
    r"(Nearest Neighbors from Training Set:|Nearest Neighbors for task\s+'[^']+'\s+\(k=\d+\):|KNN Predicted Label:\s*\([AB]\)|pseudo label from naive Morgan fingerprint KNN prediction is \([AB]\))",
    re.IGNORECASE,
)


def _prompt_has_neighbor_context(prompt: str) -> bool:
    return bool(prompt and _PREPENDED_NEIGHBOR_CONTEXT_RE.search(prompt))


def _tool_name_from_metric_key(metric_key: str) -> Optional[str]:
    if not metric_key.startswith("tool_count__"):
        return None
    return metric_key[len("tool_count__") :]


def _tool_metric_endpoint_for_key(tool_version: Optional[str], metric_key: str) -> Optional[str]:
    tool_name = _tool_name_from_metric_key(metric_key)
    if not tool_name or not tool_version:
        return None
    return resolve_tool_metric_endpoint(tool_version, tool_name)


def _response_requested_neighbors(response: dict, tool_version: Optional[str]) -> bool:
    extra_logs = response.get("extra_logs", {}) or {}
    for key, value in extra_logs.items():
        if _tool_metric_endpoint_for_key(tool_version, key) != "neighbors":
            continue
        if isinstance(value, torch.Tensor):
            value = value.flatten()[0].item()
        try:
            if float(value) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _apply_knn_reward_shaping(
    responses: list[dict],
    knn_pl: str | None,
    requested_neighbors: bool,
    correct_reversal_bonus: float,
    correct_stick_delta: float,
) -> None:
    """Apply reward shaping based on correctness relative to KNN pseudo-label.

    Applies configurable shaping when a sample is correct and either:
    - reverses the KNN pseudo-label, or
    - correctly sticks with the KNN pseudo-label.

    No shaping is applied when no KNN pseudo-label is available or neighbors
    were not requested/available.
    """
    if knn_pl is None or not requested_neighbors:
        return

    for response in responses:
        score_val = response.get("scores", None)
        if score_val is None:
            continue

        is_correct = float(score_val) > 0
        true_answer = response.get("label", "")
        knn_agrees_with_truth = knn_pl in str(true_answer)
        if not is_correct:
            continue

        if knn_agrees_with_truth:
            delta = correct_stick_delta
            delta_key = "knn_correct_stick_delta"
        else:
            delta = correct_reversal_bonus
            delta_key = "knn_correct_reversal_bonus"

        if delta == 0:
            continue

        response["reward"] = float(response.get("reward", 0.0)) + delta

        extra_logs = response.setdefault("extra_logs", {})
        extra_logs["knn_reward_delta"] = extra_logs.get("knn_reward_delta", 0.0) + delta
        extra_logs[delta_key] = extra_logs.get(delta_key, 0.0) + delta


def _coerce_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.flatten()[0].item())
    return float(value)


def _maybe_numeric_extra_log(value: Any) -> float | None:
    """Return a scalar float for numeric extra_logs values, else None.

    Experience.info is consumed as tensor-valued metrics downstream, so
    string/categorical metadata from extra_logs must be skipped here.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.flatten()[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_trace_tool_usage(response: dict, tool_version: Optional[str]) -> tuple[int, int, int, int]:
    extra_logs = response.get("extra_logs", {}) or {}
    total_tool_calls = 0
    unique_tool_calls = 0
    molecular_info_calls = 0
    neighbor_calls = 0

    for key, value in extra_logs.items():
        if not key.startswith("tool_count__"):
            continue
        try:
            count = _coerce_float(value)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue

        total_tool_calls += int(count)
        unique_tool_calls += 1
        endpoint = _tool_metric_endpoint_for_key(tool_version, key)
        if endpoint == "features":
            molecular_info_calls += int(count)
        if endpoint == "neighbors":
            neighbor_calls += int(count)

    return total_tool_calls, unique_tool_calls, molecular_info_calls, neighbor_calls


def _update_trace_diagnostic_bucket(bucket: dict[str, float], *, matched: bool, correct: bool) -> None:
    if not matched:
        return
    bucket["count"] += 1
    if correct:
        bucket["correct"] += 1


def _finalize_trace_diagnostic_bucket(bucket: dict[str, float], total_traces: int) -> tuple[float | None, float | None]:
    if total_traces <= 0:
        return None, None
    pct = bucket["count"] / total_traces * 100
    correctness = (bucket["correct"] / bucket["count"] * 100) if bucket["count"] > 0 else None
    return pct, correctness


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    index: (B,)
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, A)
    action_log_probs: (B, S)
    base_action_log_probs: (B, S)
    values: (B, S)
    returns: (B, S)
    advantages: (B, S)
    kl: (B, S)
    info: dict[str, list]
    """

    index: list[int] = None
    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None

    action_log_probs: torch.Tensor = None
    base_action_log_probs: torch.Tensor = None
    rollout_log_probs: torch.Tensor = None
    values: torch.Tensor = None
    returns: torch.Tensor = None
    advantages: torch.Tensor = None
    kl: torch.Tensor = None

    prompts: list[str] = None
    labels: list[str] = None
    rewards: torch.Tensor = None  # used for advantage calculation
    scores: torch.Tensor = None  # 0-1 reward used for dynamic sampling

    # the info field is used to store additional information
    # all the fields in the info will be logged to the tensorboard/wandb
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        index=None,
        sequences=None,
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_log_probs=None,
        values=None,
        returns=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        kl=None,
        prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.index = index
        self.sequences = sequences
        self.action_log_probs = action_log_probs
        self.base_action_log_probs = base_action_log_probs
        self.rollout_log_probs = rollout_log_probs
        self.values = values
        self.returns = returns
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.kl = kl
        self.prompts = prompts or []
        self.labels = labels or []
        self.rewards = rewards
        self.scores = scores
        self.info = info or []

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            # Collect ALL keys across every dict so that sparse keys
            # (e.g. tool_count__X present in only some samples) are
            # filled with a zero-tensor default for the missing samples.
            all_keys: set = set()
            for d in items:
                all_keys.update(d.keys())

            sorted_keys = sorted(all_keys)
            result = {key: [] for key in sorted_keys}
            for d in items:
                for key in sorted_keys:
                    if key in d:
                        result[key].append(d[key])
                    else:
                        # Infer a zero-valued placeholder that matches the
                        # type/shape of a real entry for this key.
                        _exemplar = next(dd[key] for dd in items if key in dd)
                        if isinstance(_exemplar, torch.Tensor):
                            result[key].append(torch.zeros_like(_exemplar))
                        elif isinstance(_exemplar, (int, float)):
                            result[key].append(type(_exemplar)(0))
                        else:
                            result[key].append(_exemplar)  # fallback: repeat as-is
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


def _collect_prompt_batch(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader."""
    indices, datasources, prompts, labels, knn_pseudo_labels, late_phase_prompt_flags, prompt_refs = [], [], [], [], [], [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            batch = next(dataloader_iter)
            # Support legacy tuple layouts as well as late-phase prompt flags.
            if len(batch) == 7:
                (
                    batch_indices,
                    batch_datasources,
                    batch_prompts,
                    batch_labels,
                    batch_knn,
                    batch_late_phase,
                    batch_prompt_refs,
                ) = batch
            elif len(batch) == 6:
                batch_indices, batch_datasources, batch_prompts, batch_labels, batch_knn, batch_late_phase = batch
                batch_prompt_refs = [None] * len(batch_prompts)
            elif len(batch) == 5:
                batch_indices, batch_datasources, batch_prompts, batch_labels, batch_knn = batch
                batch_late_phase = [False] * len(batch_prompts)
                batch_prompt_refs = [None] * len(batch_prompts)
            else:
                batch_indices, batch_datasources, batch_prompts, batch_labels = batch
                batch_knn = [None] * len(batch_prompts)
                batch_late_phase = [False] * len(batch_prompts)
                batch_prompt_refs = [None] * len(batch_prompts)
            remaining = num_prompts - len(prompts)
            indices.extend(batch_indices[:remaining])
            datasources.extend(batch_datasources[:remaining])
            prompts.extend(batch_prompts[:remaining])
            labels.extend(batch_labels[:remaining])
            knn_pseudo_labels.extend(batch_knn[:remaining])
            late_phase_prompt_flags.extend(batch_late_phase[:remaining])
            prompt_refs.extend(batch_prompt_refs[:remaining])
        except StopIteration:
            exhausted = True
            break

    return indices, datasources, prompts, labels, knn_pseudo_labels, late_phase_prompt_flags, prompt_refs, exhausted


class SamplesGenerator:
    """Stateless sample generator: pulls prompts and dispatches to rollout workers."""

    def __init__(
        self,
        strategy,
        prompts_dataloader,
        eval_dataloader,
        tokenizer,
        vllm_engines: List,
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []

        self.prompts_dataloader = None
        self.eval_dataloader = eval_dataloader
        from openrlhf.utils.run_paths import resolve_run_dir

        run_name = getattr(self.args, "wandb_run_name", "run")
        run_name = run_name.replace("/", "_")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.runs_dir = resolve_run_dir(project_root, run_name)
        self.rollout_trace_run_dir = os.path.join(self.runs_dir, "traces")
        os.makedirs(self.rollout_trace_run_dir, exist_ok=True)
        logger.info(f"Rollout traces enabled at: {self.rollout_trace_run_dir}")

        # vLLM stats persistence directory.
        self.vllm_stats_dir = os.path.join(self.runs_dir, "vllm_stats")
        os.makedirs(self.vllm_stats_dir, exist_ok=True)
        self.hidden_prompt_audit_dir = os.path.join(self.runs_dir, "hidden_prompt_audit")
        os.makedirs(self.hidden_prompt_audit_dir, exist_ok=True)

        # Last collected vLLM stats (for W&B logging from the trainer).
        self.last_vllm_stats: dict = {}

        # Smart replay: accumulate dataset indices by filter outcome across the episode.
        self._replay_hard_indices: set = set()
        self._replay_kept_indices: set = set()

        # Track discarded items for telemetry, regardless of smart replay.
        self._discarded_easy_indices: set = set()
        self._discarded_hard_indices: set = set()

        # Per-step filtering stats (reset each generate_samples call).
        self._step_too_easy_count = 0
        self._step_too_hard_count = 0
        self._step_scored_prompt_groups = 0
        self._step_kept_scored_prompt_groups = 0
        self._step_prompts_consumed = 0
        # Per-episode filtering stats (reset each episode).
        self._episode_easy_count = 0
        self._episode_hard_count = 0

        #### Oversampling: missed indices tracking ####
        self._missed_indices: set = set()
        self._step_missed_count = 0
        self._episode_missed_count = 0
        self._step_oversample_ratio = float(getattr(self.args, "oversample_ratio", 1.0))
        self._prompt_ref_registry: dict = {}
        self._late_phase_hidden_instruction = None
        #### end oversampling ####

        #### Easy/hard prompt tracking (Phase 12) ####
        self._easy_hard_collection = {"easy": [], "hard": []}
        self.eval_traces_dir = os.path.join(self.runs_dir, "eval_traces")
        os.makedirs(self.eval_traces_dir, exist_ok=True)
        #### end easy/hard tracking ####

        if prompts_dataloader is not None:
            self.set_prompts_dataloader(prompts_dataloader)

    def set_prompts_dataloader(self, prompts_dataloader) -> None:
        """Switch the active prompt dataloader and merge its samples into the ref registry."""
        self.prompts_dataloader = prompts_dataloader
        self._dataloader_iter = None
        if prompts_dataloader is None:
            return
        dataset = getattr(prompts_dataloader, "dataset", None)
        if dataset is None:
            return
        self._late_phase_hidden_instruction = getattr(dataset, "late_phase_hidden_instruction", None)
        if hasattr(dataset, "iter_prompt_samples"):
            for sample in dataset.iter_prompt_samples():
                prompt_ref = sample.get("prompt_ref")
                if prompt_ref is not None:
                    self._prompt_ref_registry[prompt_ref] = sample

    def _get_current_oversample_ratio(self, requested_ratio: float, global_step: Optional[int]) -> float:
        """Return the active oversample ratio for this step.

        When a start/end ramp is configured, interpolate linearly over
        ``oversample_ratio_ramp_steps`` (or ``max_steps`` by default).
        Otherwise return the requested static ratio unchanged.
        """
        start = getattr(self.args, "oversample_ratio_start", None)
        end = getattr(self.args, "oversample_ratio_end", None)
        if start is None and end is None:
            return requested_ratio

        if start is None:
            start = requested_ratio
        if end is None:
            end = requested_ratio

        ramp_steps = getattr(self.args, "oversample_ratio_ramp_steps", None)
        if ramp_steps is None:
            ramp_steps = getattr(self.args, "max_steps", None)
        if ramp_steps is None or ramp_steps <= 1:
            return float(end)

        step = max(0, int(global_step or 0))
        progress = min(step / max(ramp_steps - 1, 1), 1.0)
        return float(start + (end - start) * progress)

    def _to_jsonable(self, value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        return value

    def _write_hidden_prompt_stripping_audit(
        self,
        *,
        stripped_prompt_text: str,
        stripped_observation_text: str,
        hi_count: int,
        gen_marker_count: int,
        hi_start: int,
        hi_end: int,
        first_action: int,
    ) -> None:
        """Persist one representative stripped-sequence audit sample per run."""
        audit_dir = getattr(self, "hidden_prompt_audit_dir", "")
        if not audit_dir:
            return

        marker_path = os.path.join(audit_dir, ".sample0_stripped_written")
        try:
            fd = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return
        except OSError as exc:
            logger.warning("[hidden_instruction] Failed to initialize stripping audit in %s: %s", audit_dir, exc)
            return

        meta_path = os.path.join(audit_dir, "sample0_stripping_meta.json")
        meta = {
            "hidden_instruction_token_count": hi_count,
            "hidden_instruction_gen_marker_count": gen_marker_count,
            "hidden_instruction_token_start": hi_start,
            "hidden_instruction_token_end": hi_end,
            "first_action_token_start": first_action,
        }
        try:
            with open(os.path.join(audit_dir, "sample0_stripped_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(stripped_prompt_text)
            with open(os.path.join(audit_dir, "sample0_stripped_observation.txt"), "w", encoding="utf-8") as f:
                f.write(stripped_observation_text)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            logger.info("[hidden_instruction] Wrote stripped prompt audit files to %s", audit_dir)
        except OSError as exc:
            logger.warning("[hidden_instruction] Failed to write stripping audit files in %s: %s", audit_dir, exc)

    def _decode_trace(self, trace: dict) -> dict:
        """Decode observation_tokens into human-readable text sections."""
        obs_tokens = trace.get("observation_tokens", [])
        action_ranges = trace.get("action_ranges", [])
        if not obs_tokens:
            return {}

        decoded = {}
        decoded["full_text"] = self.tokenizer.decode(obs_tokens, skip_special_tokens=False)

        sections = []
        if action_ranges:
            first_start = action_ranges[0][0]
            if first_start > 0:
                sections.append(
                    {
                        "type": "prompt",
                        "token_range": [0, first_start],
                        "text": self.tokenizer.decode(obs_tokens[:first_start], skip_special_tokens=False),
                    }
                )

            for i, (start, end) in enumerate(action_ranges):
                sections.append(
                    {
                        "type": "action",
                        "index": i + 1,
                        "token_range": [start, end],
                        "text": self.tokenizer.decode(obs_tokens[start:end], skip_special_tokens=False),
                    }
                )
                if i + 1 < len(action_ranges):
                    next_start = action_ranges[i + 1][0]
                    if end < next_start:
                        sections.append(
                            {
                                "type": "observation",
                                "index": i + 1,
                                "token_range": [end, next_start],
                                "text": self.tokenizer.decode(obs_tokens[end:next_start], skip_special_tokens=False),
                            }
                        )
                else:
                    remaining = obs_tokens[end:]
                    if remaining:
                        sections.append(
                            {
                                "type": "trailing",
                                "token_range": [end, len(obs_tokens)],
                                "text": self.tokenizer.decode(remaining, skip_special_tokens=False),
                            }
                        )

        decoded["sections"] = sections

        # Extract response-only text (excluding prompt) using action_ranges.
        if action_ranges:
            first_action_start = action_ranges[0][0]
            decoded["response_text"] = self.tokenizer.decode(
                obs_tokens[first_action_start:], skip_special_tokens=True
            )
        else:
            decoded["response_text"] = decoded["full_text"]

        return decoded

    def _strip_token_ids(self, trace: dict) -> dict:
        trace_no_ids = dict(trace)
        for key in ("observation_tokens", "token_ids", "prompt_token_ids"):
            trace_no_ids.pop(key, None)
        return trace_no_ids


    #### Full-trace rollout helpers ####
    def _full_trace_enabled(self) -> bool:
        if not getattr(self.args, "save_all_traces", True):
            return False
        if not self.strategy.is_rank_0():
            return False
        from openrlhf.utils.full_trace import is_enabled as _bulk_enabled
        return _bulk_enabled()

    def _collect_rollout_full_trace_records(
        self,
        *,
        responses,
        processed_experiences,
        ds_idx,
        datasource,
        prompt_text,
        ref_label,
        global_step,
    ) -> None:
        """Append per-sample rollout records to ``self._full_trace_buffer``.

        Rollout schema is intentionally a strict subset of the eval schema:
        ``trace_messages`` (the structured per-turn reconstruction used for SFT
        distillation) is **never** emitted on rollout records, and the always-null
        eval-only fields (``task``/``smiles``/``source_messages``/...) are simply
        omitted.
        """
        for sample_idx, response in enumerate(responses):
            try:
                exp = processed_experiences[sample_idx] if sample_idx < len(processed_experiences) else None
                decoded = self._decode_trace(response)
                response_text = decoded.get("response_text", decoded.get("full_text", ""))

                rec = {
                    "phase": "rollout",
                    "global_step": int(global_step),
                    "datasource": datasource,
                    "ds_idx": ds_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt_text,
                    "label": ref_label,
                    "response": response_text,
                    "score": float(exp.scores[0].item()) if exp is not None and exp.scores is not None else None,
                    "reward": float(exp.rewards[0].item()) if exp is not None and exp.rewards is not None else None,
                    "response_length": int(exp.info.get("response_length", torch.tensor([0])).flatten()[0].item()) if exp is not None else None,
                    "completion_length": int(
                        exp.info.get("completion_length", torch.tensor([0])).flatten()[0].item()
                    ) if exp is not None else None,
                    "total_length": int(exp.info.get("total_length", torch.tensor([0])).flatten()[0].item()) if exp is not None else None,
                    "truncated": bool(exp.info.get("truncated", torch.tensor([0])).flatten()[0].item()) if exp is not None else None,
                    "extra_logs": response.get("extra_logs"),
                }
                self._full_trace_buffer.append(rec)
            except Exception as exc:  # pragma: no cover — logging-only path
                logger.warning("[full_trace] skipped a rollout record: %s", exc)
    #### end full-trace rollout helpers ####

    #### Prompt group trace methods (Phase 12) ####
    def _build_prompt_group_record(self, responses, ds_idx, datasource, global_step):
        """Build a serializable record of a prompt group with decoded outputs."""
        prompt = responses[0].get("prompt", "")
        label = responses[0].get("label", "")
        samples = []
        for r in responses:
            decoded = self._decode_trace(r)
            samples.append({
                "reward": r.get("reward"),
                "score": r.get("scores"),
                "response_text": decoded.get("response_text", decoded.get("full_text", "")),
            })
        return {
            "dataset_idx": ds_idx,
            "datasource": datasource,
            "global_step": global_step,
            "prompt": prompt,
            "label": label,
            "samples": samples,
        }

    def save_easy_hard_collection(self):
        """Write accumulated easy/hard examples to disk (Phase 12)."""
        if not self._easy_hard_collection["easy"] and not self._easy_hard_collection["hard"]:
            return

        # JSON format
        json_path = os.path.join(self.runs_dir, "easy_hard_prompts.json")
        try:
            with open(json_path, "w") as f:
                json.dump(self._easy_hard_collection, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Failed to write easy/hard JSON: {e}")

        # TXT format
        txt_path = os.path.join(self.runs_dir, "easy_hard_prompts.txt")
        try:
            with open(txt_path, "w") as f:
                for category in ("easy", "hard"):
                    for group in self._easy_hard_collection[category]:
                        step = group.get("global_step", "?")
                        ds_idx = group.get("dataset_idx", "?")
                        ds_name = group.get("datasource", "?")
                        f.write(f"[{category.upper()} @ step {step}] (dataset_idx={ds_idx}, datasource={ds_name})\n")
                        f.write(f"PROMPT: {group.get('prompt', '')}\n")
                        f.write(f"LABEL: {group.get('label', '')}\n\n")
                        for si, sample in enumerate(group.get("samples", []), 1):
                            reward = sample.get("reward", "?")
                            f.write(f"--- Sample {si} (reward={reward}) ---\n")
                            f.write(f"{sample.get('response_text', '')}\n\n")
                        f.write("=====================================\n\n")
        except Exception as e:
            logger.warning(f"Failed to write easy/hard TXT: {e}")
    #### end prompt group trace methods ####

    # ── vLLM stats collection ──────────────────────────────────────────

    def _collect_vllm_engine_stats(self) -> dict:
        """Collect and aggregate SchedulerStats from all vLLM engines."""
        if not self.vllm_engines:
            return {}
        try:
            refs = [engine.get_vllm_stats.remote() for engine in self.vllm_engines]
            per_engine = ray.get(refs)
        except Exception as e:
            logger.warning(f"Failed to collect vLLM stats: {e}")
            return {}

        # Aggregate raw samples from all engines.
        all_raw_samples = []
        all_kv, all_running, all_waiting = [], [], []
        all_pc_hit_rates = []
        total_poll_samples = 0

        for stats in per_engine:
            n = stats.get("num_samples", 0)
            total_poll_samples += n
            all_raw_samples.extend(stats.get("raw_samples", []))
            if n > 0:
                kv = stats["kv_cache_usage_pct"]
                all_kv.append(kv["mean"])
                all_running.append(stats["num_running_reqs"]["mean"])
                all_waiting.append(stats["num_waiting_reqs"]["mean"])
                all_pc_hit_rates.append(stats.get("prefix_cache_hit_rate", 0.0))

        if not all_kv:
            return {"num_engines": len(self.vllm_engines), "num_poll_samples": 0, "raw_samples": all_raw_samples}

        ne = len(all_kv)
        avg_pc_hit_rate = sum(all_pc_hit_rates) / len(all_pc_hit_rates) if all_pc_hit_rates else 0.0
        return {
            "num_engines": len(self.vllm_engines),
            "num_poll_samples": total_poll_samples,
            "kv_cache_usage_pct": {
                "mean": round(sum(all_kv) / ne, 4),
                "max": round(
                    max(s["kv_cache_usage_pct"]["max"] for s in per_engine if s.get("num_samples", 0) > 0), 4
                ),
            },
            "num_running_reqs": {
                "mean": round(sum(all_running) / ne, 2),
                "max": max(s["num_running_reqs"]["max"] for s in per_engine if s.get("num_samples", 0) > 0),
            },
            "num_waiting_reqs": {
                "mean": round(sum(all_waiting) / ne, 2),
                "max": max(s["num_waiting_reqs"]["max"] for s in per_engine if s.get("num_samples", 0) > 0),
            },
            "prefix_cache_hit_rate": round(avg_pc_hit_rate, 4),
            "raw_samples": all_raw_samples,
        }

    def _compute_token_throughput(self, experiences: List[Experience], wall_time: float) -> dict:
        """Compute decode/prefill token counts and throughput from experiences."""
        total_decode = 0
        total_prefill = 0
        for exp in experiences:
            if exp.action_mask is not None:
                total_decode += exp.action_mask.sum().item()
            if exp.attention_mask is not None and exp.action_mask is not None:
                total_prefill += exp.attention_mask.sum().item() - exp.action_mask.sum().item()

        total_rollout = total_decode + total_prefill

        result = {
            "total_decode_tokens": int(total_decode),
            "total_prefill_tokens": int(total_prefill),
            "total_rollout_tokens": int(total_rollout),
            "generation_wall_time_sec": round(wall_time, 2),
        }
        if wall_time > 0:
            result["decode_tokens_per_sec"] = round(total_decode / wall_time, 1)
            result["prefill_tokens_per_sec"] = round(total_prefill / wall_time, 1)
            result["total_rollout_tokens_per_sec"] = round(total_rollout / wall_time, 1)
        return result

    def _collect_and_write_vllm_stats(
        self,
        global_step: int,
        experiences: List[Experience],
        generation_wall_time: float,
        total_prompts: int,
        stats_type: str = "rollout",
    ):
        """Collect stats from engines, compute throughput, write JSONL and timeseries."""
        engine_stats = self._collect_vllm_engine_stats()
        throughput = self._compute_token_throughput(experiences, generation_wall_time)

        # Build the summary record.
        record = {
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "total_prompts": total_prompts,
            **throughput,
        }
        # Merge engine stats (excluding raw_samples, which go to timeseries).
        raw_samples = engine_stats.pop("raw_samples", [])
        record.update(engine_stats)

        # Write to the appropriate JSONL file.
        stats_file = "rollout_stats.jsonl" if stats_type == "rollout" else "eval_stats.jsonl"
        stats_path = os.path.join(self.vllm_stats_dir, stats_file)
        try:
            with open(stats_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write vLLM stats to {stats_path}: {e}")

        # Append raw scheduler time-series samples.
        if raw_samples:
            timeseries_path = os.path.join(self.vllm_stats_dir, "scheduler_timeseries.jsonl")
            try:
                with open(timeseries_path, "a") as f:
                    for sample in raw_samples:
                        f.write(json.dumps(sample) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write scheduler timeseries: {e}")

        # Store for W&B logging (flat keys for the trainer to prefix with vllm_).
        flat = {
            "vllm_generation_wall_time_sec": throughput.get("generation_wall_time_sec", 0),
            "vllm_decode_tokens_per_sec": throughput.get("decode_tokens_per_sec", 0),
            "vllm_prefill_tokens_per_sec": throughput.get("prefill_tokens_per_sec", 0),
            "vllm_total_decode_tokens": throughput.get("total_decode_tokens", 0),
            "vllm_total_prefill_tokens": throughput.get("total_prefill_tokens", 0),
            "vllm_total_rollout_tokens": throughput.get("total_rollout_tokens", 0),
            "vllm_total_rollout_tokens_per_sec": throughput.get("total_rollout_tokens_per_sec", 0),
        }
        if "kv_cache_usage_pct" in engine_stats:
            flat["vllm_kv_cache_usage_pct_mean"] = engine_stats["kv_cache_usage_pct"]["mean"]
            flat["vllm_kv_cache_usage_pct_max"] = engine_stats["kv_cache_usage_pct"]["max"]
        if "num_running_reqs" in engine_stats:
            flat["vllm_num_running_reqs_mean"] = engine_stats["num_running_reqs"]["mean"]
            flat["vllm_num_running_reqs_max"] = engine_stats["num_running_reqs"]["max"]
        if "num_waiting_reqs" in engine_stats:
            flat["vllm_num_waiting_reqs_mean"] = engine_stats["num_waiting_reqs"]["mean"]
            flat["vllm_num_waiting_reqs_max"] = engine_stats["num_waiting_reqs"]["max"]
        pc_hit_rate = engine_stats.get("prefix_cache_hit_rate", 0.0)
        flat["vllm_prefix_cache_hit_rate"] = pc_hit_rate

        self.last_vllm_stats = flat

        pc_info = f", prefix_cache_hit_rate={pc_hit_rate:.1%}"

        logger.info(
            f"vLLM stats (step {global_step}, {stats_type}): "
            f"decode={throughput.get('decode_tokens_per_sec', 0):.0f} tok/s, "
            f"prefill={throughput.get('prefill_tokens_per_sec', 0):.0f} tok/s, "
            f"kv_cache={engine_stats.get('kv_cache_usage_pct', {}).get('mean', 0):.1%}, "
            f"running={engine_stats.get('num_running_reqs', {}).get('mean', 0):.1f}, "
            f"waiting={engine_stats.get('num_waiting_reqs', {}).get('mean', 0):.1f}"
            f"{pc_info}"
        )

    def flush_timeseries_to_disk(self, global_step: int = -1) -> None:
        """Drain raw scheduler samples from all engine actors and append to disk.

        Unlike _collect_and_write_vllm_stats (called at end of generation), this
        only flushes the raw timeseries samples without touching the aggregated
        per-step summary that collect_and_reset() produces.  Call this:
          - after every eval
          - at the end of every global_step
          - before skip_training exits
        to keep Ray actor RAM bounded (samples accumulate every 30s during
        generation and would otherwise pile up until end-of-generation flush).
        """
        if not self.vllm_engines:
            return
        try:
            refs = [engine.get_and_flush_raw_samples.remote() for engine in self.vllm_engines]
            per_engine_samples = ray.get(refs)
        except Exception as e:
            logger.warning(f"Failed to flush timeseries from engines: {e}")
            return

        all_samples = []
        for samples in per_engine_samples:
            all_samples.extend(samples)

        if not all_samples:
            return

        timeseries_path = os.path.join(self.vllm_stats_dir, "scheduler_timeseries.jsonl")
        try:
            with open(timeseries_path, "a") as f:
                for sample in all_samples:
                    f.write(json.dumps(sample) + "\n")
        except Exception as e:
            logger.warning(f"Failed to flush timeseries to {timeseries_path}: {e}")
            return

        logger.debug(f"Flushed {len(all_samples)} timeseries samples to disk (step {global_step})")

    # ── eval ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        if getattr(self, "_eval_dataloader_iter", None) is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        # Wake both weights and KV cache — weights may still be asleep for
        # step-0 eval (before any broadcast_to_vllm has run).  Waking
        # already-awake weights is a no-op, so this is always safe.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_consumed, exhausted, _, _, _ = self._generate_vllm(
            dataloader_iter=self._eval_dataloader_iter,
            num_prompts=len(self.eval_dataloader),
            dynamic_filtering=False,
            allow_train_missed_fallback=False,
            discard_failed_tool_traces=False,
            **generate_kwargs,
        )
        # Collect vLLM stats for eval.
        global_step = int(generate_kwargs.get("global_step", 0))
        self._collect_and_write_vllm_stats(
            global_step=global_step,
            experiences=experiences,
            generation_wall_time=getattr(self, "_last_generation_wall_time", 0.0),
            total_prompts=prompts_consumed,
            stats_type="eval",
        )

        # Reclaim host RAM in vLLM engine workers accumulated during generation.
        batch_vllm_engine_call(self.vllm_engines, "gc_collect")

        # NOTE: We intentionally do NOT sleep vLLM after eval.  Eval always
        # runs between weight-sync and the next generate_samples() call, so
        # sleeping here would just cause a pointless sleep→wake round-trip.
        # The sleep that matters (freeing GPU for actor training) happens at
        # the end of generate_samples() instead.

        self._eval_dataloader_iter = None

        return experiences

    def get_replay_indices(self) -> Tuple[set, set]:
        """Return (hard_indices, kept_indices) accumulated during the episode.

        Note: kept_indices are tracked but excluded from the actual replay pool.
        They are returned for logging/diagnostics only.
        """
        return self._replay_hard_indices, self._replay_kept_indices

    #### Oversampling: missed indices getter ####
    def get_missed_indices(self) -> set:
        """Return missed/cancelled indices from oversampling."""
        return self._missed_indices
    #### end oversampling ####

    def get_easy_prompt_refs(self) -> set:
        return self._discarded_easy_indices

    def clear_replay_indices(self, reset_episode_counters: bool = True):
        """Reset replay tracking for a new episode."""
        self._replay_hard_indices = set()
        self._replay_kept_indices = set()
        self._discarded_easy_indices = set()
        self._discarded_hard_indices = set()
        #### Oversampling: reset missed indices per episode ####
        self._missed_indices = set()
        if reset_episode_counters:
            self._episode_easy_count = 0
            self._episode_hard_count = 0
            self._episode_missed_count = 0
        #### end oversampling ####

    def save_discarded_indices(self, episode: int):
        """Write the discarded indices of this episode to the runs_dir."""
        if not self.args.dynamic_filtering:
            return
        out_path = os.path.join(self.runs_dir, f"discarded_indices_ep{episode}.json")
        data = {
            "episode": episode,
            "too_easy": sorted(list(self._discarded_easy_indices)),
            "too_hard": sorted(list(self._discarded_hard_indices)),
            #### Oversampling: include missed indices ####
            "missed": sorted(list(self._missed_indices)),
            #### end oversampling ####
        }
        with open(out_path, "w") as f:
            json.dump(data, f)
        logger.info(
            f"Saved {len(self._discarded_easy_indices)} too_easy and {len(self._discarded_hard_indices)} too_hard indices to {out_path}"
        )

    @property
    def step_too_easy_pct(self) -> float:
        """Percentage of scored prompt groups this step that were too easy."""
        if self._step_scored_prompt_groups == 0:
            return 0.0
        return self._step_too_easy_count / self._step_scored_prompt_groups * 100

    @property
    def step_too_hard_pct(self) -> float:
        """Percentage of scored prompt groups this step that were too hard."""
        if self._step_scored_prompt_groups == 0:
            return 0.0
        return self._step_too_hard_count / self._step_scored_prompt_groups * 100

    #### Oversampling: missed percentage property ####
    @property
    def step_missed_pct(self) -> float:
        """Percentage of prompts consumed this step that were missed/cancelled."""
        if self._step_prompts_consumed == 0:
            return 0.0
        return self._step_missed_count / self._step_prompts_consumed * 100
    #### end oversampling ####

    @property
    def step_effective_prompts_consumed(self) -> int:
        """Prompts that actually completed filtering, excluding oversample cancellations."""
        return max(self._step_prompts_consumed - self._step_missed_count, 0)

    @property
    def step_filter_pass_rate(self) -> float:
        """Pass rate among prompt groups that completed and have scores."""
        if self._step_scored_prompt_groups == 0:
            return 0.0
        return self._step_kept_scored_prompt_groups / self._step_scored_prompt_groups * 100

    @property
    def step_oversample_ratio(self) -> float:
        return self._step_oversample_ratio

    @property
    def episode_filter_stats(self) -> dict:
        """Per-episode filtering stats for W&B logging."""
        return {
            "easy_discarded": self._episode_easy_count,
            "hard_kept": self._episode_hard_count,
            #### Oversampling: include missed count ####
            "missed_cancelled": self._episode_missed_count,
            #### end oversampling ####
        }

    @property
    def step_knn_stats(self) -> dict:
        """Per-step KNN reversal stats for W&B logging (knn/ section)."""
        return getattr(self, "_step_knn_stats", {})

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Produce one batch and indicate if the dataloader is exhausted."""
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)
            #### Oversampling: skip replay index reset if caller opts out ####
            if not generate_kwargs.pop("_skip_clear_replay", False):
                self.clear_replay_indices()
            #### end oversampling ####
        trace_step_idx = getattr(self, "_trace_step_idx", 0)
        self._trace_step_idx = trace_step_idx + 1

        # Reset per-step counters.
        self._step_too_easy_count = 0
        self._step_too_hard_count = 0
        self._step_scored_prompt_groups = 0
        self._step_kept_scored_prompt_groups = 0
        self._step_prompts_consumed = 0
        #### Oversampling: reset per-step missed count ####
        self._step_missed_count = 0
        #### end oversampling ####

        #### Full-trace rollout buffer (one JSONL per global_step) ####
        self._full_trace_buffer = [] if self._full_trace_enabled() else None
        #### end full-trace buffer ####

        # Wake sleeping vLLM engines before dispatching.
        # Wake both weights and KV cache — weights may still be asleep for
        # the first generation (before any broadcast_to_vllm has run).
        # Waking already-awake weights is a no-op, so this is always safe.
        _wake_start = time.time()
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
        self._last_vllm_wake_sec = time.time() - _wake_start

        #### Oversampling: extract and forward oversample_ratio ####
        oversample_ratio = generate_kwargs.pop(
            "oversample_ratio", getattr(self.args, "oversample_ratio", 1.0)
        )
        if generate_kwargs.pop("_apply_oversample_ramp", True):
            oversample_ratio = self._get_current_oversample_ratio(
                oversample_ratio,
                generate_kwargs.get("global_step", trace_step_idx),
            )
        self._step_oversample_ratio = float(oversample_ratio)
        #### end oversampling ####
        experiences, prompts_consumed, exhausted, _unused_prompt_groups, easy_ex, hard_ex = self._generate_vllm(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            dynamic_filtering=self.args.dynamic_filtering,
            trace_step_idx=trace_step_idx,
            oversample_ratio=oversample_ratio,
            **generate_kwargs,
        )
        self._step_prompts_consumed = prompts_consumed

        #### Collect easy/hard examples (Phase 12) ####
        if easy_ex:
            self._easy_hard_collection["easy"].append(easy_ex)
        if hard_ex:
            self._easy_hard_collection["hard"].append(hard_ex)
        #### end prompt group traces ####

        # Collect vLLM stats and write JSONL.
        global_step = int(generate_kwargs.get("global_step", trace_step_idx))
        self._collect_and_write_vllm_stats(
            global_step=global_step,
            experiences=experiences,
            generation_wall_time=getattr(self, "_last_generation_wall_time", 0.0),
            total_prompts=prompts_consumed,
            stats_type="rollout",
        )

        # Reclaim host RAM in vLLM engine workers accumulated during generation.
        _gc_start = time.time()
        batch_vllm_engine_call(self.vllm_engines, "gc_collect")
        self._last_vllm_gc_collect_sec = time.time() - _gc_start

        # Put engines back to sleep when enabled.
        _sleep_start = time.time()
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep", level=getattr(self.args, "vllm_sleep_level", 1))
        self._last_vllm_sleep_sec = time.time() - _sleep_start

        filter_pass_rate = None
        if self.args.dynamic_filtering and prompts_consumed:
            filter_pass_rate = self.step_filter_pass_rate

        if exhausted:
            self._dataloader_iter = None
            logger.info("Prompt dataloader is exhausted.")

        #### Full-trace rollout buffer flush ####
        if self._full_trace_buffer:
            from openrlhf.utils.full_trace import write_jsonl as _write_full_trace_jsonl

            run_name = getattr(self.args, "wandb_run_name", "run").replace("/", "_")
            _write_full_trace_jsonl(
                self._full_trace_buffer,
                phase="rollout",
                global_step=global_step,
                run_name=run_name,
            )
        self._full_trace_buffer = None
        #### end full-trace buffer flush ####

        return experiences, filter_pass_rate, prompts_consumed, exhausted

    def _generate_vllm(
        self, dataloader_iter, num_prompts: int, dynamic_filtering, **generate_kwargs
    ) -> Tuple[List[Experience], int, bool, list, Optional[dict], Optional[dict]]:
        """Generate a batch of Experiences with optional reward filtering."""
        allow_train_missed_fallback = generate_kwargs.pop("allow_train_missed_fallback", True)
        #### Oversampling: compute oversampled dispatch count ####
        oversample_ratio = generate_kwargs.pop("oversample_ratio", getattr(self.args, "oversample_ratio", 1.0))
        oversampled_count = math.ceil(num_prompts * oversample_ratio) if dynamic_filtering else num_prompts
        #### end oversampling ####

        step_idx = int(generate_kwargs.get("trace_step_idx", generate_kwargs.get("global_step", 0)))
        self._current_step_group_sizes = []  # Reset per-prompt group sizes for ERL

        # Set global step on all engines for time-series labeling.
        for engine in self.vllm_engines:
            engine.set_current_global_step.remote(step_idx)

        generation_start_time = time.time()

        prompts_consumed = 0
        #### Oversampling: collect oversampled_count prompts, fill from missed_indices ####
        (
            dataset_indices,
            ds_datasources,
            prompts,
            labels,
            ds_knn_pseudo_labels,
            ds_late_phase_prompts,
            prompt_refs,
            exhausted,
        ) = _collect_prompt_batch(dataloader_iter, oversampled_count)

        # Fill-in: when dataloader exhausts, supplement from missed_indices
        if allow_train_missed_fallback and exhausted and len(prompts) < oversampled_count and self._missed_indices:
            remaining_needed = oversampled_count - len(prompts)
            fill_indices = list(self._missed_indices)[:remaining_needed]
            for prompt_ref in fill_indices:
                sample = self._prompt_ref_registry.get(prompt_ref)
                if sample is None:
                    continue
                dataset_indices.append(sample.get("idx", -1))
                prompt_refs.append(prompt_ref)
                ds_datasources.append(sample.get("datasource", "unknown"))
                ds_knn_pseudo_labels.append(sample.get("knn_pseudo_label"))
                ds_late_phase_prompts.append(bool(sample.get("late_phase_prompt", False)))
                prompts.append(sample["prompt"])
                labels.append(sample["label"])
            self._missed_indices -= set(fill_indices)
            if len(prompts) < oversampled_count:
                logger.warning(
                    f"[Oversample] Could only fill {len(prompts)}/{oversampled_count} "
                    f"(dataloader exhausted, {len(self._missed_indices)} missed remaining)"
                )

        # If can't even fill num_prompts (base, not oversampled): mark as missed and skip
        if len(prompts) < num_prompts:
            for prompt_ref in prompt_refs:
                if prompt_ref is None:
                    continue
                self._missed_indices.add(prompt_ref)
                self._step_missed_count += 1
                self._episode_missed_count += 1
            self._last_generation_wall_time = 0.0
            return [], len(prompts), True, [], None, None
        #### end oversampling ####

        # Stop early if the prompt source is fully consumed and nothing collected.
        if exhausted and not prompts:
            self._last_generation_wall_time = 0.0
            return [], prompts_consumed, exhausted, [], None, None

        smart_replay = getattr(self.args, "smart_replay", False)
        prompt_tasks = []
        for prompt_ref in prompt_refs:
            task = None
            if prompt_ref is not None:
                sample = self._prompt_ref_registry.get(prompt_ref) or {}
                raw_record = sample.get("raw_record") or {}
                task = raw_record.get("task")
            prompt_tasks.append(task)

        dispatches = self._dispatch_prompts_to_vllm(
            prompts,
            labels,
            late_phase_prompt_flags=ds_late_phase_prompts,
            datasources=ds_datasources,
            tasks=prompt_tasks,
            **generate_kwargs,
        )

        pending_refs = [ref for ref, _ in dispatches]
        ref_to_engine = {ref: engine_idx for ref, engine_idx in dispatches}
        # Map each ref → its dataset index for smart replay tracking.
        ref_to_dataset_idx = {ref: prompt_refs[i] for i, (ref, _) in enumerate(dispatches)}
        ref_to_datasource = {ref: ds_datasources[i] for i, (ref, _) in enumerate(dispatches)} if ds_datasources else {}
        ref_to_knn_pl = {ref: ds_knn_pseudo_labels[i] for i, (ref, _) in enumerate(dispatches)} if ds_knn_pseudo_labels else {}
        ref_to_label = {ref: labels[i] for i, (ref, _) in enumerate(dispatches)}
        ref_to_prompt = {ref: prompts[i] for i, (ref, _) in enumerate(dispatches)}
        prompts_consumed += len(prompts)

        # Track how many outstanding requests each engine has.
        engine_pending = defaultdict(int)
        for _, engine_idx in dispatches:
            engine_pending[engine_idx] += 1

        accepted_experiences: List[Experience] = []
        accepted_prompt_groups = 0  #### Oversampling: track accepted groups for early termination ####
        pbar = tqdm(range(num_prompts), desc="Generate samples")
        filtered_count = 0
        episode_traces: list = []
        total_episodes = 0
        exhausted_during_refill = False

        #### Prompt group tracking (Phase 12) — easy/hard only; the per-step
        # ``step{N}_groups.json`` artifact was retired in favor of the off-path
        # JSONL writer (`openrlhf.utils.trace_writer`) and the curated
        # ``group_trace_step{N}_{mixed,needle}.json`` showcase files.
        step_prompt_groups: list = []  # kept for return-tuple compatibility; always empty
        step_easy_example = None
        step_hard_example = None
        #### end prompt group tracking ####

        #### Oversample label bias tracking ####
        bias_accepted_labels = []
        bias_skipped_labels = []  # early-termination cancelled
        bias_easy_labels = []     # filtered too-easy
        bias_hard_labels = []     # filtered too-hard
        #### end oversample label bias tracking ####

        #### KNN reversal tracking ####
        tool_version = getattr(self.args, "tool_version", None)
        requested_neighbors_total = 0
        requested_get_features_total = 0
        get_features_requested_feature_total = 0
        get_features_request_count = 0
        get_features_requested_feature_count_count = 0
        get_neighbors_requested_feature_total = 0
        get_neighbors_request_count = 0
        get_neighbors_requested_feature_count_count = 0
        processed_prompt_groups = 0  # prompt groups that actually returned (excludes cancelled)
        knn_total = 0       # prompts with KNN pseudo-label
        knn_reversed = 0    # model prediction != KNN pseudo-label
        knn_correct_reversal = 0   # reversed AND model got the right answer
        knn_incorrect_reversal = 0  # reversed AND model got the wrong answer
        knn_correct_stick = 0      # agreed with KNN AND model got the right answer
        knn_incorrect_stick = 0    # agreed with KNN AND model got the wrong answer
        total_traces = 0
        trace_diag_two_unique = {"count": 0, "correct": 0}
        trace_diag_three_plus = {"count": 0, "correct": 0}
        trace_diag_two_molinfo_one_neighbor = {"count": 0, "correct": 0}
        #### end KNN tracking ####

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                engine_idx = ref_to_engine.pop(ref)
                ds_idx = ref_to_dataset_idx.pop(ref, None)
                datasource = ref_to_datasource.pop(ref, None)
                ref_label = ref_to_label.pop(ref, None)
                prompt_text = ref_to_prompt.pop(ref, "")
                engine_pending[engine_idx] -= 1



                # Build Experience objects for each vLLM response returned from this worker.
                responses = ray.get(ref)
                processed_prompt_groups += 1
                total_episodes += len(responses)
                total_traces += len(responses)

                for response in responses:
                    total_tool_calls, unique_tool_calls, molecular_info_calls, neighbor_calls = _count_trace_tool_usage(
                        response, tool_version
                    )
                    is_correct = float(response.get("scores", 0) or 0) > 0
                    _update_trace_diagnostic_bucket(
                        trace_diag_two_unique,
                        matched=unique_tool_calls >= 2,
                        correct=is_correct,
                    )
                    _update_trace_diagnostic_bucket(
                        trace_diag_three_plus,
                        matched=total_tool_calls >= 3,
                        correct=is_correct,
                    )
                    _update_trace_diagnostic_bucket(
                        trace_diag_two_molinfo_one_neighbor,
                        matched=(molecular_info_calls >= 2 and neighbor_calls >= 1),
                        correct=is_correct,
                    )

                if getattr(self, "rollout_trace_run_dir", None) and not getattr(
                    self, "_has_saved_first_ever_trace", False
                ):
                    try:
                        self._has_saved_first_ever_trace = True
                        trace_path = os.path.join(self.rollout_trace_run_dir, "first_ever_trace.json")
                        trace = responses[0]
                        record = {
                            "engine_idx": engine_idx,
                            "trace": self._strip_token_ids(trace),
                            "decoded": self._decode_trace(trace),
                        }
                        with open(trace_path, "w") as f:
                            f.write(json.dumps(self._to_jsonable(record), ensure_ascii=True))
                    except Exception as e:
                        logger.error(f"Failed to save first ever trace: {e}")

                # Only keep the first trace per step.
                # Holding ALL responses in episode_traces leaks hundreds of MB in
                # multi-turn mode (each resp contains full observation_tokens + log_probs).
                if not episode_traces:
                    episode_traces.append((engine_idx, responses[0]))
                #### KNN reversal tracking per prompt group ####
                knn_pl = ref_to_knn_pl.pop(ref, None)
                requested_neighbors = _prompt_has_neighbor_context(prompt_text) or any(
                    _response_requested_neighbors(response, tool_version) for response in responses
                )
                if requested_neighbors:
                    requested_neighbors_total += 1

                prompt_group_requested_get_features = False
                for response in responses:
                    extra_logs = response.get("extra_logs", {}) or {}
                    feature_call_count = int(extra_logs.get("get_features_request_count", 0) or 0)
                    neighbor_call_count = int(extra_logs.get("get_neighbors_request_count", 0) or 0)
                    feature_requested_total = int(extra_logs.get("get_features_requested_feature_total", 0) or 0)
                    feature_requested_count_count = int(
                        extra_logs.get("get_features_requested_feature_count_count", 0) or 0
                    )
                    neighbor_requested_total = int(extra_logs.get("get_neighbors_requested_feature_total", 0) or 0)
                    neighbor_requested_count_count = int(
                        extra_logs.get("get_neighbors_requested_feature_count_count", 0) or 0
                    )

                    if feature_call_count > 0:
                        prompt_group_requested_get_features = True
                        get_features_request_count += feature_call_count
                        get_features_requested_feature_total += feature_requested_total
                        get_features_requested_feature_count_count += feature_requested_count_count
                    if neighbor_call_count > 0:
                        get_neighbors_request_count += neighbor_call_count
                        get_neighbors_requested_feature_total += neighbor_requested_total
                        get_neighbors_requested_feature_count_count += neighbor_requested_count_count

                if prompt_group_requested_get_features:
                    requested_get_features_total += 1
                if knn_pl is not None and responses and requested_neighbors:
                    knn_total += 1
                    # Use pure correctness scores, not shaped rewards.
                    scores = [float(r.get("scores", 0) or 0) for r in responses]
                    model_correct = sum(1 for s in scores if s > 0) > len(scores) / 2
                    true_answer = responses[0].get("label", "")
                    knn_agrees_with_truth = knn_pl in str(true_answer)
                    reversed_knn = model_correct != knn_agrees_with_truth
                    if reversed_knn:
                        knn_reversed += 1
                        if model_correct:
                            knn_correct_reversal += 1
                        else:
                            knn_incorrect_reversal += 1
                    else:
                        if model_correct:
                            knn_correct_stick += 1
                        else:
                            knn_incorrect_stick += 1
                _apply_knn_reward_shaping(
                    responses,
                    knn_pl,
                    requested_neighbors,
                    correct_reversal_bonus=getattr(self.args, "knn_correct_reversal_bonus", 0.25),
                    correct_stick_delta=getattr(self.args, "knn_correct_stick_delta", -0.15),
                )
                #### end KNN tracking ####

                processed_experiences = [
                    self._process_response_into_experience(response, **generate_kwargs) for response in responses
                ]
                # Filter out None entries from failed/empty generations.
                for experience in processed_experiences:
                    if experience is None:
                        continue
                    if prompt_text:
                        experience.prompts = [prompt_text]
                    if ref_label is not None:
                        experience.labels = [ref_label]

                experiences = [e for e in processed_experiences if e is not None]
                if not experiences:
                    logger.warning(
                        f"All responses from engine {engine_idx} were dropped before PPO "
                        f"(prompt group datasource={datasource}, ds_idx={ds_idx}); requesting replacements"
                    )

                #### Full-trace rollout buffer ####
                if getattr(self, "_full_trace_buffer", None) is not None:
                    self._collect_rollout_full_trace_records(
                        responses=responses,
                        processed_experiences=processed_experiences,
                        ds_idx=ds_idx,
                        datasource=datasource,
                        prompt_text=prompt_text,
                        ref_label=ref_label,
                        global_step=int(generate_kwargs.get("global_step", step_idx)),
                    )
                #### end full-trace ####

                # Drop experiences if the average score falls outside the allowed range.
                if dynamic_filtering and experiences and all(e.scores is not None for e in experiences):
                    self._step_scored_prompt_groups += 1
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.dynamic_filtering_reward_range
                    if avg_reward >= max_r:
                        # Too easy — drop; do NOT add to replay
                        filtered_count += 1
                        self._step_too_easy_count += 1
                        if ref_label is not None:
                            bias_easy_labels.append(ref_label)
                        self._episode_easy_count += 1
                        if ds_idx is not None:
                            self._discarded_easy_indices.add(ds_idx)
                        #### Capture easy example (Phase 12) ####
                        if step_easy_example is None and responses:
                            step_easy_example = self._build_prompt_group_record(
                                responses, ds_idx, datasource, step_idx
                            )
                        #### end capture easy ####
                        if filtered_count % 10 == 0:
                            logger.info(
                                "Dynamic filtering rejected group (too easy) "
                                f"(rejected={filtered_count}, accepted={len(accepted_experiences)}/{num_prompts}, "
                                f"avg_reward={avg_reward:.2f}, threshold=({min_r:.2f}, {max_r:.2f}))"
                            )
                        experiences = []
                    elif avg_reward <= min_r:
                        # Too hard — queue index for replay
                        filtered_count += 1
                        self._step_too_hard_count += 1
                        if ref_label is not None:
                            bias_hard_labels.append(ref_label)
                        self._episode_hard_count += 1
                        if ds_idx is not None:
                            self._discarded_hard_indices.add(ds_idx)
                        if smart_replay and ds_idx is not None:
                            self._replay_hard_indices.add(ds_idx)
                        #### Capture hard example (Phase 12) ####
                        if step_hard_example is None and responses:
                            step_hard_example = self._build_prompt_group_record(
                                responses, ds_idx, datasource, step_idx
                            )
                        #### end capture hard ####
                        if filtered_count % 10 == 0:
                            logger.info(
                                "Dynamic filtering rejected group (too hard) "
                                f"(rejected={filtered_count}, accepted={len(accepted_experiences)}/{num_prompts}, "
                                f"avg_reward={avg_reward:.2f}, threshold=({min_r:.2f}, {max_r:.2f}))"
                            )
                        experiences = []
                    else:
                        # In range — kept; queue index for replay
                        self._step_kept_scored_prompt_groups += 1
                        if smart_replay and ds_idx is not None:
                            self._replay_kept_indices.add(ds_idx)

                # Accept experiences and stop once enough have been gathered.
                if experiences:
                    retained_group_size = len(experiences)
                    for experience in experiences:
                        experience.info["prompt_group_size"] = torch.tensor([retained_group_size])
                    accepted_experiences.extend(experiences)
                    if ref_label is not None:
                        bias_accepted_labels.append(ref_label)
                    # Track per-prompt group size for variable-size ERL groups.
                    if not hasattr(self, "_current_step_group_sizes"):
                        self._current_step_group_sizes = []
                    self._current_step_group_sizes.append(retained_group_size)
                    accepted_prompt_groups += 1
                    pbar.set_postfix({"prompts_consumed": prompts_consumed})
                    pbar.update()

                    # (per-step ``step{N}_groups.json`` retired — off-path JSONL writer
                    # + showcase ``group_trace_step{N}_*.json`` cover the same use case)

                    #### Oversampling: early termination once enough accepted ####
                    if accepted_prompt_groups >= num_prompts:
                        cancelled_refs = list(pending_refs)
                        for cancel_ref in cancelled_refs:
                            missed_idx = ref_to_dataset_idx.get(cancel_ref)
                            if missed_idx is not None:
                                self._missed_indices.add(missed_idx)
                                self._step_missed_count += 1
                                self._episode_missed_count += 1
                            cancel_label = ref_to_label.get(cancel_ref)
                            if cancel_label is not None:
                                bias_skipped_labels.append(cancel_label)
                        logger.info(
                            f"[Oversample] Early termination: {accepted_prompt_groups} accepted, "
                            f"cancelled {len(cancelled_refs)} in-flight, "
                            f"{self._step_missed_count} missed this step"
                        )
                        # Cancel abandoned refs so vLLM engines are idle before sleep.
                        # ray.cancel(force=False) raises CancelledError inside the
                        # remote coroutine, which propagates through the await chain
                        # and triggers vLLM request abort. ray.get() then returns as
                        # soon as cancellation cleanup finishes — not when the original
                        # generation would have completed.
                        for cancel_ref in cancelled_refs:
                            ray.cancel(cancel_ref, force=False)
                        if cancelled_refs:
                            try:
                                ray.get(cancelled_refs)
                            except ray.exceptions.TaskCancelledError:
                                pass
                            except Exception:
                                pass  # best-effort
                        pending_refs = []
                        break
                    #### end oversampling ####

                # If rejected, request replacement prompts to keep filling the batch.
                else:
                    replace_ratio = getattr(self.args, "replace_discarded_prompts_ratio", 1.0)
                    num_replacements = max(1, math.ceil(replace_ratio))
                    (
                        new_ds_indices,
                        new_ds_datasources,
                        new_prompts,
                        new_labels,
                        new_knn_pls,
                        new_late_phase_prompts,
                        new_prompt_refs,
                        exhausted,
                    ) = _collect_prompt_batch(dataloader_iter, num_replacements)

                    #### Oversampling: fall back to missed_indices when dataloader exhausted ####
                    if allow_train_missed_fallback and exhausted and len(new_prompts) < num_replacements and self._missed_indices:
                        remaining = num_replacements - len(new_prompts)
                        fill_indices = list(self._missed_indices)[:remaining]
                        for prompt_ref in fill_indices:
                            sample = self._prompt_ref_registry.get(prompt_ref)
                            if sample is None:
                                continue
                            new_ds_indices.append(sample.get("idx", -1))
                            new_prompt_refs.append(prompt_ref)
                            new_ds_datasources.append(sample.get("datasource", "unknown"))
                            new_knn_pls.append(sample.get("knn_pseudo_label"))
                            new_late_phase_prompts.append(bool(sample.get("late_phase_prompt", False)))
                            new_prompts.append(sample["prompt"])
                            new_labels.append(sample["label"])
                        self._missed_indices -= set(fill_indices)
                    elif allow_train_missed_fallback and exhausted and not new_prompts and self._missed_indices:
                        fallback_idx = self._missed_indices.pop()
                        sample = self._prompt_ref_registry.get(fallback_idx)
                        if sample is not None:
                            new_ds_indices = [sample.get("idx", -1)]
                            new_prompt_refs = [fallback_idx]
                            new_ds_datasources = [sample.get("datasource", "unknown")]
                            new_knn_pls = [sample.get("knn_pseudo_label")]
                            new_late_phase_prompts = [bool(sample.get("late_phase_prompt", False))]
                            new_prompts = [sample["prompt"]]
                            new_labels = [sample["label"]]
                        else:
                            new_ds_indices = []
                            new_ds_datasources = []
                            new_knn_pls = []
                            new_late_phase_prompts = []
                            new_prompt_refs = []
                            new_prompts = []
                            new_labels = []
                    #### end oversampling ####

                    # Count every replacement prompt attempted this step,
                    # including those sourced from missed_indices after the
                    # dataloader has been exhausted.
                    prompts_consumed += len(new_prompts)

                    # Dataloader drained (and no missed fallback): drain in-flight refs.
                    # This avoids racing vLLM sleep/wake against active decode kernels.
                    if exhausted and not new_prompts:
                        logger.info(
                            "Prompt dataloader exhausted during refill; "
                            f"draining {len(pending_refs)} in-flight vLLM refs before sleep."
                        )
                        exhausted_during_refill = True
                    # Otherwise dispatch the new prompt to keep filling the queue.
                    elif new_prompts:
                        new_dispatches = self._dispatch_prompts_to_vllm(
                            new_prompts,
                            new_labels,
                            late_phase_prompt_flags=new_late_phase_prompts,
                            datasources=new_ds_datasources,
                            **generate_kwargs,
                        )
                        for j, (new_ref, new_engine_idx) in enumerate(new_dispatches):
                            pending_refs.append(new_ref)
                            ref_to_engine[new_ref] = new_engine_idx
                            ref_to_dataset_idx[new_ref] = new_prompt_refs[j] if j < len(new_prompt_refs) else None
                            ref_to_label[new_ref] = new_labels[j]
                            if new_ds_datasources:
                                ref_to_datasource[new_ref] = new_ds_datasources[j] if j < len(new_ds_datasources) else "unknown"
                            if new_knn_pls:
                                ref_to_knn_pl[new_ref] = new_knn_pls[j] if j < len(new_knn_pls) else None
                            engine_pending[new_engine_idx] += 1

                del responses  # free raw vLLM response dicts

        self._last_generation_wall_time = time.time() - generation_start_time

        two_unique_pct, two_unique_correctness = _finalize_trace_diagnostic_bucket(
            trace_diag_two_unique, total_traces
        )
        three_plus_pct, three_plus_correctness = _finalize_trace_diagnostic_bucket(
            trace_diag_three_plus, total_traces
        )
        two_molinfo_neighbor_pct, two_molinfo_neighbor_correctness = _finalize_trace_diagnostic_bucket(
            trace_diag_two_molinfo_one_neighbor, total_traces
        )

        #### Store KNN stats for W&B logging ####
        self._step_knn_stats = {
            "requested_get_neighbors_pct": (requested_neighbors_total / processed_prompt_groups * 100)
            if processed_prompt_groups > 0
            else None,
            "requested_get_features_pct": (requested_get_features_total / processed_prompt_groups * 100)
            if processed_prompt_groups > 0
            else None,
            "avg_get_features_requested_feature_count": (
                get_features_requested_feature_total / get_features_requested_feature_count_count
            )
            if get_features_requested_feature_count_count > 0
            else None,
            "avg_get_neighbors_requested_feature_count": (
                get_neighbors_requested_feature_total / get_neighbors_requested_feature_count_count
            )
            if get_neighbors_requested_feature_count_count > 0
            else None,
            "trace_total": total_traces,
            "trace_pct_at_least_2_unique_tools": two_unique_pct,
            "trace_correctness_at_least_2_unique_tools": two_unique_correctness,
            "trace_pct_at_least_3_tool_calls": three_plus_pct,
            "trace_correctness_at_least_3_tool_calls": three_plus_correctness,
            "trace_pct_at_least_2_molinfo_and_1_neighbor": two_molinfo_neighbor_pct,
            "trace_correctness_at_least_2_molinfo_and_1_neighbor": two_molinfo_neighbor_correctness,
            "knn_total": knn_total,
            "knn_reversal_pct": (knn_reversed / knn_total * 100) if knn_total > 0 else None,
            "knn_correct_reversal_pct": (knn_correct_reversal / knn_total * 100) if knn_total > 0 else None,
            "knn_incorrect_reversal_pct": (knn_incorrect_reversal / knn_total * 100) if knn_total > 0 else None,
            "knn_correct_stick_pct": (knn_correct_stick / knn_total * 100) if knn_total > 0 else None,
            "knn_stick_pct": ((knn_correct_stick + knn_incorrect_stick) / knn_total * 100) if knn_total > 0 else None,
            "knn_incorrect_stick_pct": (knn_incorrect_stick / knn_total * 100) if knn_total > 0 else None,
        }
        #### end KNN stats ####

        #### Oversample label bias: write per-step JSONL record ####
        if getattr(self, "runs_dir", None):
            try:
                self._write_oversample_label_bias(
                    step_idx, bias_accepted_labels, bias_skipped_labels,
                    bias_easy_labels, bias_hard_labels,
                )
            except Exception as e:
                logger.warning(f"[OversampleBias] Failed to write label bias record: {e}")
        #### end oversample label bias ####

        if smart_replay and not exhausted_during_refill:
            logger.info(
                f"[SmartReplay] Step done: replay buffer now has "
                f"{len(self._replay_hard_indices)} hard + {len(self._replay_kept_indices)} kept "
                f"+ {len(self._missed_indices)} missed "
                f"= {len(self._replay_hard_indices) + len(self._replay_kept_indices) + len(self._missed_indices)} total prompts"
            )

        if exhausted_during_refill:
            if smart_replay:
                logger.info(
                    f"[SmartReplay] Step done (exhausted): replay buffer now has "
                    f"{len(self._replay_hard_indices)} hard + {len(self._replay_kept_indices)} kept "
                    f"= {len(self._replay_hard_indices) + len(self._replay_kept_indices)} total prompts"
                )
            #### Oversampling: return partial results instead of [] ####
            return accepted_experiences, prompts_consumed, True, step_prompt_groups, step_easy_example, step_hard_example
            #### end oversampling ####

        return accepted_experiences, prompts_consumed, exhausted, step_prompt_groups, step_easy_example, step_hard_example

    @staticmethod
    def _label_bias_stats(labels: list) -> dict:
        """Compute majority-label percentage for a list of label strings."""
        if not labels:
            return {"count": 0, "majority_label": None, "majority_pct": None, "distribution": {}}
        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        return {
            "count": len(labels),
            "majority_label": majority_label,
            "majority_pct": round(majority_count / len(labels) * 100, 2),
            "distribution": dict(counts),
        }

    def _write_oversample_label_bias(
        self, step_idx: int,
        accepted: list, skipped: list, easy: list, hard: list,
    ):
        """Append one JSONL record per step to oversample_label_bias.jsonl."""
        bias_path = os.path.join(self.runs_dir, "oversample_label_bias.jsonl")
        record = {
            "global_step": step_idx,
            "accepted": self._label_bias_stats(accepted),
            "skipped_early_term": self._label_bias_stats(skipped),
            "filtered_easy": self._label_bias_stats(easy),
            "filtered_hard": self._label_bias_stats(hard),
            "all_dispatched": self._label_bias_stats(accepted + skipped + easy + hard),
        }
        with open(bias_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _dispatch_prompts_to_vllm(
        self,
        prompts: List[str],
        labels: List[str],
        late_phase_prompt_flags: Optional[List[bool]] = None,
        datasources: Optional[List[str]] = None,
        tasks: Optional[List[Optional[str]]] = None,
        **generate_kwargs,
    ) -> List:
        """Send prompts to rollout executors and return Ray object refs."""
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            **(
                {
                    "spaces_between_special_tokens": False,
                    "stop": self.args.vllm_stop_strings,
                    "include_stop_str_in_output": True,
                }
                if self.args.agent_func_path
                else {}
            ),
            logprobs=1 if self.args.enable_vllm_is_correction else None,
        )
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)
        n_samples_per_prompt = generate_kwargs.get("n_samples_per_prompt", self.args.n_samples_per_prompt)

        # Snapshot current pending rollout counts to balance upcoming work.
        pending_counts = ray.get([engine.get_num_unfinished_requests.remote() for engine in self.vllm_engines])
        engine_heap = [(count, idx) for idx, count in enumerate(pending_counts)]
        heapq.heapify(engine_heap)

        # Pre-compute engine assignment to keep loads even.
        engine_indices = []
        for _ in prompts:
            current_load, engine_idx = heapq.heappop(engine_heap)
            engine_indices.append(engine_idx)
            heapq.heappush(engine_heap, (current_load + n_samples_per_prompt, engine_idx))

        refs = []
        extra_hidden_instruction = self._late_phase_hidden_instruction
        for idx, (prompt, label) in enumerate(zip(prompts, labels)):
            # Spread work across engines/workers in load-aware order.
            engine_idx = engine_indices[idx]
            llm_engine = self.vllm_engines[engine_idx]
            prompt_hidden_instruction = None
            if late_phase_prompt_flags and idx < len(late_phase_prompt_flags) and late_phase_prompt_flags[idx]:
                prompt_hidden_instruction = extra_hidden_instruction
            prompt_datasource = datasources[idx] if datasources and idx < len(datasources) else None
            prompt_task = tasks[idx] if tasks and idx < len(tasks) else None
            ref = llm_engine.generate_responses.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=truncate_length,
                num_samples=n_samples_per_prompt,
                log_trajectory=(idx == 0),
                extra_hidden_instruction=prompt_hidden_instruction,
                datasource=prompt_datasource,
                task=prompt_task,
            )
            refs.append((ref, engine_idx))

        return refs

    def _process_response_into_experience(self, response, **generate_kwargs) -> Experience:
        """Turn a single vLLM response into an Experience."""
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)
        extra_logs = response.get("extra_logs", {}) or {}
        discard_failed_tool_traces = generate_kwargs.get("discard_failed_tool_traces", True)

        if discard_failed_tool_traces and float(extra_logs.get("discard_from_training", 0) or 0) > 0:
            logger.warning(
                "Dropping rollout trace before PPO due to failed tool execution "
                f"(prompt={response.get('prompt', '')[:120]!r}, label={response.get('label', '')!r})"
            )
            return None

        # Base rollout fields from the output.
        tokenized_observation = response["observation_tokens"].copy()
        tokenized_ranges = response["action_ranges"]
        reward_val = response.get("reward", None)
        score_val = response.get("scores", None)

        # Strip hidden instruction tokens before building training tensors.
        # These tokens guided generation but should not be seen during training.
        # Layout in observation: [...user | hi_tokens(H) | gen_marker(G) | action...]
        # hi_start = action_ranges[0][0] - G - H (robust to left-truncation)
        hi_count = response.get("hidden_instruction_token_count", 0)
        if hi_count > 0 and tokenized_ranges:
            gen_marker_count = response.get("hidden_instruction_gen_marker_count", 0)
            first_action = tokenized_ranges[0][0]
            hi_start = first_action - gen_marker_count - hi_count
            hi_end = hi_start + hi_count
            if hi_start >= 0 and hi_end <= len(tokenized_observation):
                tokenized_observation = tokenized_observation[:hi_start] + tokenized_observation[hi_end:]
                tokenized_ranges = [(s - hi_count, e - hi_count) for s, e in tokenized_ranges]
                if response.get("rollout_log_probs") is not None:
                    lp = response["rollout_log_probs"]
                    response["rollout_log_probs"] = lp[:hi_start] + lp[hi_end:]
                if tokenized_ranges:
                    stripped_first_action = tokenized_ranges[0][0]
                    stripped_prompt_ids = tokenized_observation[:stripped_first_action]
                else:
                    stripped_first_action = len(tokenized_observation)
                    stripped_prompt_ids = tokenized_observation
                self._write_hidden_prompt_stripping_audit(
                    stripped_prompt_text=self.tokenizer.decode(stripped_prompt_ids, skip_special_tokens=False),
                    stripped_observation_text=self.tokenizer.decode(tokenized_observation, skip_special_tokens=False),
                    hi_count=hi_count,
                    gen_marker_count=gen_marker_count,
                    hi_start=hi_start,
                    hi_end=hi_end,
                    first_action=first_action,
                )
                logger.debug(
                    f"[hidden_instruction] Stripped {hi_count} tokens at [{hi_start}:{hi_end}), "
                    f"new sequence length: {len(tokenized_observation)}"
                )
            else:
                logger.warning(
                    f"[hidden_instruction] Invalid strip range [{hi_start}:{hi_end}) for "
                    f"sequence of length {len(tokenized_observation)} "
                    f"(first_action={first_action}, gen_marker={gen_marker_count}), skipping."
                )

        sequences = torch.tensor(tokenized_observation, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokenized_observation))
        # Mark the action span within the concatenated tokens.
        action_mask = torch.zeros_like(attention_mask)
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        # Truncate everything to the configured context window.
        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")
        action_tokens = int(action_mask.sum().item())
        if action_tokens == 0:
            logger.warning(
                "Skipping rollout with zero action tokens after truncation (would produce NaNs). "
                f"prompt={response['prompt'][:200]!r}, label={response['label']!r}, "
                f"observation_tokens={len(tokenized_observation)}, action_ranges={tokenized_ranges}, truncate_length={truncate_length}"
            )
            return None

        # Align rollout logprobs with the truncated action span.
        if response["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(response["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Collect simple stats about lengths and clipping.
        # For multi-turn tool use, action_mask can contain multiple disjoint
        # assistant spans separated by tool-feedback tokens. Measuring from the
        # first action token to the last would incorrectly count those tool
        # outputs as part of the model response length. Use the mask sum so
        # response_length tracks only generated assistant tokens, and keep the
        # legacy span metric separately as completion_length.
        response_length = action_tokens
        ones_indices = torch.where(action_mask)[0]
        completion_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
        total_length = attention_mask.float().sum()
        is_clipped = total_length >= truncate_length

        # Check if response was truncated (hit max_tokens limit, finish_reason == "length")
        is_truncated = response.get("truncated", False)

        info = {
            "response_length": torch.tensor([response_length]),
            "completion_length": torch.tensor([completion_length]),
            "total_length": torch.tensor([total_length]),
            "response_clip_ratio": torch.tensor([is_clipped]),
            "truncated": torch.tensor([is_truncated]),
        }
        if reward_val is not None:
            info["reward"] = torch.tensor([reward_val])
        if score_val is not None:
            info["score"] = torch.tensor([score_val])

        # Convert extra logs to tensors for downstream consumers.
        for key, value in extra_logs.items():
            numeric_value = _maybe_numeric_extra_log(value)
            if numeric_value is None:
                continue
            info[key] = torch.tensor([numeric_value])

        # Generic distillation mask: any executor can tag extra_logs["distill"] = 1
        # to request SFT loss on this experience's action tokens.
        info["distill_mask"] = torch.tensor([float(extra_logs.get("distill", 0))])

        return Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            prompts=[response["prompt"]],
            labels=[response["label"]],
            rewards=torch.tensor([reward_val]) if reward_val is not None else None,
            scores=torch.tensor([score_val]) if score_val is not None else None,
            info=info,
        )


class RemoteExperienceMaker:
    def __init__(
        self,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy,
        tokenizer,
        **kwargs,
    ):
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args
        self.advantage_estimator = strategy.args.advantage_estimator

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.initial_model_group = initial_model_group
        self.tokenizer = tokenizer
        self.kl_ctl = kl_controller

    @staticmethod
    def _tensor_to_int(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.flatten()[0].item()
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _infer_prompt_group_sizes(self, rollout_samples: List[Experience]) -> List[int]:
        """Recover retained prompt group sizes from per-sample metadata.

        SamplesGenerator emits accepted samples in prompt-group order. When a
        broken tool trace is dropped, some groups shrink below
        ``n_samples_per_prompt``. We attach the retained size to each sample so
        the training-side experience maker can reconstruct the variable groups
        after the batch crosses process / actor boundaries.
        """
        if not rollout_samples:
            return []

        group_sizes = []
        sample_idx = 0
        while sample_idx < len(rollout_samples):
            info = getattr(rollout_samples[sample_idx], "info", None) or {}
            group_size = self._tensor_to_int(info.get("prompt_group_size"))
            if group_size is None:
                return []
            if group_size <= 0 or sample_idx + group_size > len(rollout_samples):
                logger.warning(
                    "Invalid prompt_group_size metadata at sample %d: group_size=%r, total_samples=%d",
                    sample_idx,
                    group_size,
                    len(rollout_samples),
                )
                return []

            for member_idx in range(sample_idx, sample_idx + group_size):
                member_info = getattr(rollout_samples[member_idx], "info", None) or {}
                member_group_size = self._tensor_to_int(member_info.get("prompt_group_size"))
                if member_group_size != group_size:
                    logger.warning(
                        "Inconsistent prompt_group_size metadata in group starting at sample %d: "
                        "expected=%d, found=%r at member %d",
                        sample_idx,
                        group_size,
                        member_group_size,
                        member_idx,
                    )
                    return []

            group_sizes.append(group_size)
            sample_idx += group_size

        return group_sizes

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        import math

        if getattr(self.args, "use_adaptive_batch", False):
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            samples_with_idx = [(i, l) for i, l in enumerate(total_lengths)]
            samples_with_idx.sort(key=lambda x: x[1], reverse=True)
            
            batch_indexes = []
            current_partition = []
            default_max_len = 0
            
            for idx, length in samples_with_idx:
                new_size = len(current_partition) + 1
                new_max_len = max(default_max_len, length) if current_partition else length
                if current_partition and (new_size * new_max_len > self.args.rollout_max_tokens_per_gpu):
                    batch_indexes.append([i for i, _ in current_partition])
                    current_partition = [(idx, length)]
                    default_max_len = length
                else:
                    current_partition.append((idx, length))
                    default_max_len = new_max_len
            if current_partition:
                batch_indexes.append([i for i, _ in current_partition])
                
            # Ensure num partitions is multiple of effective_actor_num
            while len(batch_indexes) % effective_actor_num != 0 or len(batch_indexes) < effective_actor_num:
                biggest_idx = max(range(len(batch_indexes)), key=lambda k: len(batch_indexes[k]))
                target = batch_indexes.pop(biggest_idx)
                mid = len(target) // 2
                if mid > 0:
                    batch_indexes.append(target[:mid])
                    batch_indexes.append(target[mid:])
                else:
                    batch_indexes.append([]) # Safety fallback

            # Sort partitions descending by sum of lengths for the interval distribution
            batch_indexes.sort(key=lambda p: sum([total_lengths[i] for i in p]) if p else 0, reverse=True)
            
            for micro_index in batch_indexes:
                if not micro_index: continue
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)

        elif self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = math.ceil(minimum_batch_num / effective_actor_num) * effective_actor_num
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)

            # Interleave samples_list so each contiguous chunk assigned to an actor has
            # an identical distribution of heavy and light microbatches.
            split_items = [
                samples_list[i : i + effective_actor_num] for i in range(0, len(samples_list), effective_actor_num)
            ]
            half = len(split_items) // 2
            first_half = split_items[:half]
            last_half = [item[::-1] for item in split_items[half:]]

            interval_items = []
            for i in range(half):
                interval_items.append(first_half[i])
                interval_items.append(last_half[-(i + 1)])
            if len(last_half) > len(first_half):
                interval_items.append(last_half[0])

            interval_merged = list(zip(*interval_items))
            flattened_samples_list = []
            for actor_chunks in interval_merged:
                flattened_samples_list.extend(actor_chunks)
            samples_list = flattened_samples_list
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size], self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        self._current_step_group_sizes = self._infer_prompt_group_sizes(rollout_samples)

        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples_list: List[Experience]) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        use_reward_model = samples_list[0].rewards is None
        if use_reward_model:
            if self.reward_model_group is None:
                raise ValueError("reward_model_group is required when rewards are not precomputed")
            # Batch call reward model
            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples_list),
            )
        else:
            r_refs = None

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models and r_refs is not None:
            ray.get(r_refs)
            ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

        # Batch call actor model
        action_log_probs_ref = self.actor_model_group.async_run_method_batch(
            method_name="forward",
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )

        # Sync to avoid GPU OOM when colocate models
        if args.colocate_all_models or args.colocate_actor_ref:
            ray.get(action_log_probs_ref)
            ray.get(self.actor_model_group.async_run_method(method_name="empty_cache"))

        # Batch call critic model
        if self.critic_model_group is not None:
            if args.colocate_critic_reward and r_refs is not None:
                ray.get(r_refs)
                ray.get(self.reward_model_group.async_run_method(method_name="empty_cache"))

            value_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )
            if args.colocate_all_models or args.colocate_critic_reward:
                ray.get(value_ref)
                ray.get(self.critic_model_group.async_run_method(method_name="empty_cache"))
        else:
            value_ref = ray.put([[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size))

        # Batch call initial model
        if self.initial_model_group is not None:
            base_action_log_probs_ref = self.initial_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                action_mask=action_mask_list,
                attention_mask=attention_mask_list,
            )

            if args.colocate_all_models or args.colocate_actor_ref:
                ray.get(base_action_log_probs_ref)
                ray.get(self.initial_model_group.async_run_method(method_name="empty_cache"))
        else:
            base_action_log_probs_ref = ray.put(
                [[None]] * (len(samples_list) * args.ring_attn_size * args.ds_tensor_parallel_size)
            )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        del action_log_probs_ref
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        del base_action_log_probs_ref
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])
        del value_ref

        # Process rewards based on source
        if use_reward_model:
            # Reward Model
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            del r_refs
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list), (
            f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"
        )

        # Process results for each sample
        for i, (samples, action_log_probs, base_action_log_probs, value) in enumerate(
            zip(samples_list, action_log_probs_list, base_action_log_probs_list, value_list)
        ):
            if (self.initial_model_group is not None) and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.strategy.args.kl_estimator,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
                logprobs_diff = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)
            logprobs_diff_mean = masked_mean(logprobs_diff, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean
            samples.info["logprobs_diff"] = logprobs_diff_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args

        # Apply length penalties (DAPO overlong / ProRL stop properly) - BEFORE dynamic indices processing
        apply_length_penalties(experiences, args)

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearrange samples
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        #### Compute prompt_tokens for prompt-level loss aggregation ####
        # prompt_tokens[i] = total action tokens across all completions for sequence i's prompt.
        # Computed in sorted order (prompt groups are contiguous), then remapped to original order.
        if getattr(args, "loss_aggregation", "sample") == "prompt":
            # Get per-sequence action token counts in original order, then sort
            raw_action_counts = torch.cat(
                [exp.action_mask.sum(dim=-1).float() for exp in experiences], dim=0
            )
            sorted_action_counts = torch.empty_like(raw_action_counts)
            sorted_action_counts[indices] = raw_action_counts  # to sorted order

        # Check if we have variable group sizes (ERL mode)
        prompt_group_sizes = getattr(self, "_current_step_group_sizes", None)
        use_variable_groups = (
            prompt_group_sizes
            and len(prompt_group_sizes) > 0
            and any(gs != args.n_samples_per_prompt for gs in prompt_group_sizes)
        )

        if not use_variable_groups and rewards.numel() % args.n_samples_per_prompt != 0:
            raise RuntimeError(
                "Reward batch size is not divisible by n_samples_per_prompt. "
                f"Got {rewards.numel()} rewards for n_samples_per_prompt={args.n_samples_per_prompt}. "
                "This usually means some traces were dropped but prompt-group metadata was not preserved."
            )

        if use_variable_groups:
            # Variable group sizes (ERL mode): split rewards by per-prompt group sizes
            reward_groups = list(torch.split(rewards, prompt_group_sizes))

            # Log group reward std per experience
            group_std_flat = []
            for group in reward_groups:
                std_val = group.std().item() if len(group) > 1 else 0.0
                group_std_flat.extend([std_val] * len(group))
            group_reward_stds = torch.tensor(group_std_flat)[indices].split(exp_len)
            for experience, group_reward_std in zip(experiences, group_reward_stds):
                experience.info["group_reward_std"] = group_reward_std

            # Reward shaping per group
            shaped_groups = []
            for group in reward_groups:
                n = len(group)
                if args.advantage_estimator == "rloo":
                    baseline = (group.sum() - group) / max(n - 1, 1)
                    shaped_groups.append(group - baseline)
                elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
                    shaped_groups.append(group - group.mean())
                elif args.advantage_estimator == "group_norm":
                    std = group.std() + 1e-9 if n > 1 else torch.tensor(1.0)
                    shaped_groups.append((group - group.mean()) / std)
                else:
                    shaped_groups.append(group)
            rewards = torch.cat(shaped_groups)[indices].split(exp_len)
        else:
            # Fixed group sizes (standard mode)
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)

            # log group reward std
            if args.n_samples_per_prompt > 1:
                group_reward_stds = (
                    rewards.std(-1, keepdim=True)
                    .repeat(1, args.n_samples_per_prompt)
                    .reshape(-1)[indices]
                    .split(exp_len)
                )
                for experience, group_reward_std in zip(experiences, group_reward_stds):
                    experience.info["group_reward_std"] = group_reward_std

            # reward shaping
            if args.advantage_estimator == "rloo":
                baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
                rewards = rewards - baseline
            elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
                rewards = rewards - rewards.mean(-1, keepdim=True)
            elif args.advantage_estimator == "group_norm":
                rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

            rewards = rewards.reshape(-1)[indices].split(exp_len)

        #### Store prompt_tokens in experiences ####
        if getattr(args, "loss_aggregation", "sample") == "prompt":
            if use_variable_groups:
                # Variable group sizes (ERL): group by prompt_group_sizes
                pt_groups = torch.split(sorted_action_counts, prompt_group_sizes)
                prompt_tokens_sorted = torch.cat(
                    [g.sum().expand(len(g)) for g in pt_groups]
                )
            else:
                # Fixed group sizes: reshape to (P, G)
                G = args.n_samples_per_prompt
                pt_grouped = sorted_action_counts.reshape(-1, G)
                # Sum within each prompt group, broadcast back to (P, G)
                prompt_tokens_sorted = pt_grouped.sum(dim=-1, keepdim=True).expand_as(pt_grouped).reshape(-1)

            # Remap from sorted order back to original, then split per experience
            prompt_tokens_orig = prompt_tokens_sorted[indices].split(exp_len)
            for experience, pt in zip(experiences, prompt_tokens_orig):
                experience.info["prompt_tokens"] = pt
        #### end prompt_tokens ####

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    args.gamma,
                    args.lambd,
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning("gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm")
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None

        # Normalize advantages across all experiences for GAE, REINFORCE, and REINFORCE-baseline
        if self.args.advantage_estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (advantages_vector * action_masks_vector).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = ((advantages_vector - mean).pow(2) * action_masks_vector).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
