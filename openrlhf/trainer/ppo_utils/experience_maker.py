import heapq
import json
import math
import os
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

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
from openrlhf.utils.utils import zero_pad_sequences

logger = init_logger(__name__)


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
        """Select specific fields from a list of Experience instances to create new Experience instances."""
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
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            #### Sparse key handling: fill missing keys with zero-valued placeholders ####
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
                        _exemplar = next(dd[key] for dd in items if key in dd)
                        if isinstance(_exemplar, torch.Tensor):
                            result[key].append(torch.zeros_like(_exemplar))
                        elif isinstance(_exemplar, (int, float)):
                            result[key].append(type(_exemplar)(0))
                        else:
                            result[key].append(_exemplar)
            #### end sparse key handling ####
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience."""
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


#### Updated _collect_prompt_batch: returns dataset indices for replay tracking ####
def _collect_prompt_batch(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader."""
    indices, prompts, labels = [], [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            batch_indices, _, batch_prompts, batch_labels = next(dataloader_iter)
            remaining = num_prompts - len(prompts)
            indices.extend(batch_indices[:remaining])
            prompts.extend(batch_prompts[:remaining])
            labels.extend(batch_labels[:remaining])
        except StopIteration:
            exhausted = True
            break

    return indices, prompts, labels, exhausted
#### end updated _collect_prompt_batch ####


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

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader

        #### Runs directory and trace setup (L6, L7, L8) ####
        run_name = getattr(self.args, "wandb_run_name", "run")
        run_name = run_name.replace("/", "_")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.runs_dir = os.path.join(project_root, "runs", run_name)
        self.rollout_trace_run_dir = os.path.join(self.runs_dir, "traces")
        os.makedirs(self.rollout_trace_run_dir, exist_ok=True)
        logger.info(f"Rollout traces enabled at: {self.rollout_trace_run_dir}")

        self.vllm_stats_dir = os.path.join(self.runs_dir, "vllm_stats")
        os.makedirs(self.vllm_stats_dir, exist_ok=True)
        self.last_vllm_stats: dict = {}
        #### end runs directory and trace setup ####

        #### Smart replay index tracking (for Phase 8) ####
        self._replay_hard_indices: set = set()
        self._replay_kept_indices: set = set()
        self._discarded_easy_indices: set = set()
        self._discarded_hard_indices: set = set()
        #### end smart replay index tracking ####

        #### Per-step filtering stats (L9, L10) ####
        self._step_too_easy_count = 0
        self._step_too_hard_count = 0
        self._step_prompts_consumed = 0
        self._episode_easy_count = 0
        self._episode_hard_count = 0
        #### end per-step filtering stats ####

        #### Oversampling: missed indices tracking (L10) ####
        self._missed_indices: set = set()
        self._step_missed_count = 0
        self._episode_missed_count = 0
        self._original_dataset = prompts_dataloader.dataset if prompts_dataloader is not None else None
        #### end oversampling ####

    #### Trace helper methods (L6) ####
    def _to_jsonable(self, value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        return value

    def _decode_trace(self, response: dict) -> dict:
        """Decode observation_tokens into human-readable text sections."""
        tokens = response.get("observation_tokens", [])
        ranges = response.get("action_ranges", [])
        if not tokens:
            return {"raw_prompt": response.get("prompt", ""), "sections": []}

        try:
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        except Exception:
            text = "<decode error>"

        sections = []
        prev_end = 0
        for start, end in ranges:
            if start > prev_end:
                obs_text = self.tokenizer.decode(tokens[prev_end:start], skip_special_tokens=False)
                label = "prompt" if prev_end == 0 else "observation"
                sections.append({"type": label, "start": prev_end, "end": start, "text": obs_text})
            action_text = self.tokenizer.decode(tokens[start:end], skip_special_tokens=False)
            sections.append({"type": "action", "start": start, "end": end, "text": action_text})
            prev_end = end
        if prev_end < len(tokens):
            trailing_text = self.tokenizer.decode(tokens[prev_end:], skip_special_tokens=False)
            sections.append({"type": "trailing", "start": prev_end, "end": len(tokens), "text": trailing_text})

        return {
            "full_text": text,
            "sections": sections,
            "reward": response.get("reward"),
            "scores": response.get("scores"),
            "prompt": response.get("prompt", ""),
            "label": response.get("label", ""),
        }

    def _strip_token_ids(self, trace: dict) -> dict:
        """Remove token ID arrays from trace to save disk space."""
        for key in ("observation_tokens", "token_ids", "prompt_token_ids"):
            trace.pop(key, None)
        return trace

    def _write_step_trace(self, step_idx: int, traces: list):
        """Write one rollout trace per step to disk (L6)."""
        if not traces:
            return
        path = os.path.join(self.rollout_trace_run_dir, f"step{step_idx}.jsonl")
        try:
            with open(path, "w") as f:
                for trace in traces[:1]:  # Only first trace per step
                    decoded = self._decode_trace(trace)
                    stripped = self._strip_token_ids(trace)
                    record = {**stripped, "decoded": decoded}
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write step trace: {e}")

    def _write_eval_trace(self, step_idx: int, traces: list):
        """Write eval trace to disk (L6)."""
        if not traces:
            return
        path = os.path.join(self.rollout_trace_run_dir, f"eval_{step_idx}.json")
        try:
            decoded_traces = []
            for trace in traces[:5]:  # Keep up to 5 eval traces
                decoded = self._decode_trace(trace)
                stripped = self._strip_token_ids(trace)
                decoded_traces.append({**stripped, "decoded": decoded})
            with open(path, "w") as f:
                json.dump(decoded_traces, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write eval trace: {e}")
    #### end trace helper methods ####

    #### vLLM stats collection methods (L5, L7, L8) ####
    def _collect_vllm_engine_stats(self) -> dict:
        """Collect scheduler stats from all vLLM engines."""
        if not self.vllm_engines:
            return {}

        try:
            refs = [engine.get_vllm_stats.remote() for engine in self.vllm_engines]
            all_stats = ray.get(refs)
        except Exception as e:
            logger.warning(f"Failed to collect vLLM stats: {e}")
            return {}

        # Aggregate across engines
        total_samples = sum(s.get("num_samples", 0) for s in all_stats)
        if total_samples == 0:
            return {"num_samples": 0}

        kv_means = [s["kv_cache_usage_pct"]["mean"] for s in all_stats if s.get("num_samples", 0) > 0]
        kv_maxes = [s["kv_cache_usage_pct"]["max"] for s in all_stats if s.get("num_samples", 0) > 0]
        running_means = [s["num_running_reqs"]["mean"] for s in all_stats if s.get("num_samples", 0) > 0]
        running_maxes = [s["num_running_reqs"]["max"] for s in all_stats if s.get("num_samples", 0) > 0]
        waiting_means = [s["num_waiting_reqs"]["mean"] for s in all_stats if s.get("num_samples", 0) > 0]
        waiting_maxes = [s["num_waiting_reqs"]["max"] for s in all_stats if s.get("num_samples", 0) > 0]
        pc_rates = [s.get("prefix_cache_hit_rate", 0.0) for s in all_stats if s.get("num_samples", 0) > 0]

        raw_samples = []
        for s in all_stats:
            raw_samples.extend(s.get("raw_samples", []))

        return {
            "num_samples": total_samples,
            "kv_cache_usage_pct": {
                "mean": round(sum(kv_means) / len(kv_means), 4) if kv_means else 0,
                "max": round(max(kv_maxes), 4) if kv_maxes else 0,
            },
            "num_running_reqs": {
                "mean": round(sum(running_means) / len(running_means), 2) if running_means else 0,
                "max": max(running_maxes) if running_maxes else 0,
            },
            "num_waiting_reqs": {
                "mean": round(sum(waiting_means) / len(waiting_means), 2) if waiting_means else 0,
                "max": max(waiting_maxes) if waiting_maxes else 0,
            },
            "prefix_cache_hit_rate": round(sum(pc_rates) / len(pc_rates), 4) if pc_rates else 0,
            "raw_samples": raw_samples,
        }

    def _compute_token_throughput(self, experiences: list, wall_time: float) -> dict:
        """Compute token throughput from experiences."""
        if not experiences or wall_time <= 0:
            return {}
        total_tokens = sum(int(e.info.get("total_length", torch.tensor([0])).sum().item()) for e in experiences)
        response_tokens = sum(int(e.info.get("response_length", torch.tensor([0])).sum().item()) for e in experiences)
        return {
            "total_tokens": total_tokens,
            "response_tokens": response_tokens,
            "tokens_per_sec": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
            "decode_tokens_per_sec": round(response_tokens / wall_time, 1) if wall_time > 0 else 0,
        }

    def _collect_and_write_vllm_stats(self, step_idx: int, experiences: list, wall_time: float, mode: str = "rollout"):
        """Collect vLLM stats, write to disk, and store flat metrics for W&B (L7, L8)."""
        engine_stats = self._collect_vllm_engine_stats()
        throughput = self._compute_token_throughput(experiences, wall_time)

        record = {
            "step": step_idx,
            "mode": mode,
            "wall_time_sec": round(wall_time, 2),
            "timestamp": datetime.now().isoformat(),
            **engine_stats,
            **throughput,
        }
        # Remove raw_samples from the summary record
        raw_samples = record.pop("raw_samples", [])

        # Write per-step summary
        fname = "rollout_stats.jsonl" if mode == "rollout" else "eval_stats.jsonl"
        try:
            with open(os.path.join(self.vllm_stats_dir, fname), "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write vLLM stats: {e}")

        # Append raw scheduler timeseries
        if raw_samples:
            try:
                with open(os.path.join(self.vllm_stats_dir, "scheduler_timeseries.jsonl"), "a") as f:
                    for sample in raw_samples:
                        f.write(json.dumps(sample, default=str) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write scheduler timeseries: {e}")

        # Store flat metrics for W&B logging
        flat = {}
        if engine_stats.get("num_samples", 0) > 0:
            flat["vllm/kv_cache_usage_mean"] = engine_stats["kv_cache_usage_pct"]["mean"]
            flat["vllm/kv_cache_usage_max"] = engine_stats["kv_cache_usage_pct"]["max"]
            flat["vllm/num_running_reqs_mean"] = engine_stats["num_running_reqs"]["mean"]
            flat["vllm/num_running_reqs_max"] = engine_stats["num_running_reqs"]["max"]
            flat["vllm/num_waiting_reqs_mean"] = engine_stats["num_waiting_reqs"]["mean"]
            flat["vllm/num_waiting_reqs_max"] = engine_stats["num_waiting_reqs"]["max"]
            flat["vllm/prefix_cache_hit_rate"] = engine_stats["prefix_cache_hit_rate"]
        if throughput:
            flat["vllm/tokens_per_sec"] = throughput["tokens_per_sec"]
            flat["vllm/decode_tokens_per_sec"] = throughput["decode_tokens_per_sec"]
        self.last_vllm_stats = flat

    def flush_timeseries_to_disk(self):
        """Drain raw scheduler samples from engine actors to disk (L8)."""
        if not self.vllm_engines:
            return
        try:
            refs = [engine.get_and_flush_raw_samples.remote() for engine in self.vllm_engines]
            all_samples = ray.get(refs)
            samples = []
            for engine_samples in all_samples:
                samples.extend(engine_samples)
            if samples:
                with open(os.path.join(self.vllm_stats_dir, "scheduler_timeseries.jsonl"), "a") as f:
                    for sample in samples:
                        f.write(json.dumps(sample, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to flush timeseries: {e}")
    #### end vLLM stats collection methods ####

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> List[Experience]:
        if getattr(self, "_eval_dataloader_iter", None) is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, traces, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._eval_dataloader_iter,
            num_prompts=len(self.eval_dataloader),
            dynamic_filtering=False,
            **generate_kwargs,
        )

        #### Write eval trace and collect stats (L6, L7) ####
        trace_step = getattr(self, "_trace_step_idx", 0)
        self._write_eval_trace(trace_step, traces)
        gen_time = getattr(self, "_last_generation_wall_time", 0)
        self._collect_and_write_vllm_stats(trace_step, experiences, gen_time, mode="eval")
        #### end eval trace and stats ####

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        self._eval_dataloader_iter = None

        return experiences

    #### Smart replay index management ####
    def get_replay_indices(self):
        return self._replay_hard_indices, self._replay_kept_indices

    def get_missed_indices(self):
        return self._missed_indices

    def clear_replay_indices(self):
        self._replay_hard_indices.clear()
        self._replay_kept_indices.clear()
        self._discarded_easy_indices.clear()
        self._discarded_hard_indices.clear()
        self._missed_indices.clear()
        self._episode_easy_count = 0
        self._episode_hard_count = 0
        self._episode_missed_count = 0

    def save_discarded_indices(self, episode: int):
        """Write filtering decisions per episode to disk (L12)."""
        record = {
            "episode": episode,
            "too_easy": sorted(self._discarded_easy_indices),
            "too_hard": sorted(self._discarded_hard_indices),
            "missed": sorted(self._missed_indices),
            "kept": sorted(self._replay_kept_indices),
        }
        path = os.path.join(self.runs_dir, f"discarded_indices_ep{episode}.json")
        try:
            with open(path, "w") as f:
                json.dump(record, f)
        except Exception as e:
            logger.warning(f"Failed to write discarded indices: {e}")

    @property
    def step_too_easy_pct(self):
        if self._step_prompts_consumed == 0:
            return 0.0
        return self._step_too_easy_count / self._step_prompts_consumed * 100

    @property
    def step_too_hard_pct(self):
        if self._step_prompts_consumed == 0:
            return 0.0
        return self._step_too_hard_count / self._step_prompts_consumed * 100

    @property
    def step_missed_pct(self):
        if self._step_prompts_consumed == 0:
            return 0.0
        return self._step_missed_count / self._step_prompts_consumed * 100

    @property
    def episode_filter_stats(self):
        return {
            "episode_too_easy": self._episode_easy_count,
            "episode_too_hard": self._episode_hard_count,
            "episode_missed": self._episode_missed_count,
        }
    #### end smart replay index management ####

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Produce one batch and indicate if the dataloader is exhausted."""
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)
            #### Clear replay indices at start of new episode ####
            if not generate_kwargs.pop("_skip_clear_replay", False):
                self.clear_replay_indices()
            #### end clear replay indices ####
            self._trace_step_idx = 0

        #### Reset per-step counters (L9, L10) ####
        self._step_too_easy_count = 0
        self._step_too_hard_count = 0
        self._step_prompts_consumed = 0
        self._step_missed_count = 0
        #### end reset per-step counters ####

        # Wake sleeping vLLM engines before dispatching.
        _wake_start = time.time()
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
        self._last_vllm_wake_sec = time.time() - _wake_start

        #### Oversampling ratio (L10) ####
        oversample_ratio = generate_kwargs.pop("oversample_ratio", getattr(self.args, "oversample_ratio", 1.0))
        #### end oversampling ratio ####

        experiences, traces, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            dynamic_filtering=self.args.dynamic_filtering,
            trace_step_idx=self._trace_step_idx,
            oversample_ratio=oversample_ratio,
            **generate_kwargs,
        )
        self._step_prompts_consumed = prompts_consumed

        #### Collect vLLM stats (L7) ####
        gen_time = getattr(self, "_last_generation_wall_time", 0)
        self._collect_and_write_vllm_stats(self._trace_step_idx, experiences, gen_time, mode="rollout")
        #### end collect stats ####

        #### GC collect on engines ####
        _gc_start = time.time()
        if self.vllm_engines:
            batch_vllm_engine_call(self.vllm_engines, "gc_collect")
        self._last_vllm_gc_collect_sec = time.time() - _gc_start
        #### end GC collect ####

        # Put engines back to sleep when enabled.
        _sleep_start = time.time()
        if self.args.vllm_enable_sleep:
            sleep_level = getattr(self.args, "vllm_sleep_level", 1)
            batch_vllm_engine_call(self.vllm_engines, "sleep", level=sleep_level)
        self._last_vllm_sleep_sec = time.time() - _sleep_start

        filter_pass_rate = None
        if self.args.dynamic_filtering and prompts_consumed:
            filter_pass_rate = self.args.rollout_batch_size / prompts_consumed * 100

        if exhausted:
            self._dataloader_iter = None
            logger.info("Prompt dataloader is exhausted.")

        self._trace_step_idx += 1

        return experiences, filter_pass_rate, prompts_consumed, exhausted

    def _generate_vllm(
        self, dataloader_iter, num_prompts: int, dynamic_filtering, **generate_kwargs
    ) -> Tuple[List[Experience], list, int, bool]:
        """Generate a batch of Experiences with optional reward filtering and oversampling."""
        prompts_consumed = 0

        #### Oversampling dispatch (L10) ####
        oversample_ratio = generate_kwargs.pop("oversample_ratio", 1.0)
        oversampled_count = math.ceil(num_prompts * oversample_ratio) if dynamic_filtering else num_prompts
        #### end oversampling dispatch ####

        trace_step_idx = generate_kwargs.pop("trace_step_idx", 0)

        #### Set global step on engines for time-series labeling ####
        if self.vllm_engines:
            for engine in self.vllm_engines:
                engine.set_current_global_step.remote(trace_step_idx)
        #### end set global step ####

        generation_start_time = time.time()

        ds_indices, prompts, labels, exhausted = _collect_prompt_batch(dataloader_iter, oversampled_count)
        # Stop early if the prompt source is fully consumed.
        if exhausted and len(prompts) < num_prompts:
            return [], [], prompts_consumed, exhausted

        pending_result = self._dispatch_prompts_to_vllm(prompts, labels, **generate_kwargs)
        prompts_consumed += len(prompts)

        # Build ref→engine and ref→dataset_index mappings
        ref_to_engine = {}
        ref_to_dataset_idx = {}
        pending_refs = []
        for ref, engine_idx in pending_result:
            pending_refs.append(ref)
            ref_to_engine[ref] = engine_idx
            if ds_indices:
                ref_to_dataset_idx[ref] = ds_indices[len(pending_refs) - 1] if len(pending_refs) - 1 < len(ds_indices) else None

        engine_pending = defaultdict(int)
        for ref in pending_refs:
            engine_pending[ref_to_engine[ref]] += 1

        smart_replay = getattr(self.args, "smart_replay", False)

        accepted_experiences: List[Experience] = []
        accepted_prompt_groups = 0
        episode_traces: list = []
        pbar = tqdm(range(num_prompts), desc="Generate samples")

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                engine_idx = ref_to_engine.get(ref, 0)
                engine_pending[engine_idx] = max(0, engine_pending[engine_idx] - 1)
                ds_idx = ref_to_dataset_idx.get(ref)

                try:
                    responses = ray.get(ref)
                except Exception as e:
                    logger.warning(f"Failed to get response: {e}")
                    continue

                # Save first trace for this step
                if responses and not episode_traces:
                    episode_traces.append(responses[0])

                # Build Experience objects for each vLLM response
                experiences = []
                for response in responses:
                    exp = self._process_response_into_experience(response, **generate_kwargs)
                    if exp is not None:
                        experiences.append(exp)

                # Drop experiences if the average score falls outside the allowed range.
                if dynamic_filtering and all(e.scores is not None for e in experiences) and experiences:
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.dynamic_filtering_reward_range

                    #### Split too_easy vs too_hard with telemetry (L9) ####
                    if avg_reward >= max_r:
                        self._step_too_easy_count += 1
                        self._episode_easy_count += 1
                        if smart_replay and ds_idx is not None:
                            self._discarded_easy_indices.add(ds_idx)
                        experiences = []
                    elif avg_reward <= min_r:
                        self._step_too_hard_count += 1
                        self._episode_hard_count += 1
                        if smart_replay and ds_idx is not None:
                            self._replay_hard_indices.add(ds_idx)
                            self._discarded_hard_indices.add(ds_idx)
                        experiences = []
                    else:
                        if smart_replay and ds_idx is not None:
                            self._replay_kept_indices.add(ds_idx)
                    #### end split too_easy vs too_hard ####

                # Accept experiences and stop once enough have been gathered.
                if experiences:
                    accepted_experiences.extend(experiences)
                    accepted_prompt_groups += 1
                    pbar.set_postfix({"prompts_consumed": prompts_consumed})
                    pbar.update()

                    #### Early termination when oversampled (L10) ####
                    if accepted_prompt_groups >= num_prompts and oversample_ratio > 1.0:
                        for remaining_ref in pending_refs:
                            try:
                                ray.cancel(remaining_ref, force=False)
                            except Exception:
                                pass
                            rem_ds_idx = ref_to_dataset_idx.get(remaining_ref)
                            if rem_ds_idx is not None:
                                self._missed_indices.add(rem_ds_idx)
                                self._step_missed_count += 1
                                self._episode_missed_count += 1
                        pending_refs = []
                        break
                    #### end early termination ####

                # If rejected, request a new prompt to keep filling the batch.
                elif not exhausted:
                    replace_ratio = getattr(self.args, "replace_discarded_prompts_ratio", 1.0)
                    if replace_ratio > 0:
                        new_ds_indices, new_prompts, new_labels, exhausted = _collect_prompt_batch(dataloader_iter, 1)
                        prompts_consumed += len(new_prompts)
                        if exhausted and not new_prompts:
                            continue
                        if new_prompts:
                            new_result = self._dispatch_prompts_to_vllm(new_prompts, new_labels, **generate_kwargs)
                            for new_ref, new_engine_idx in new_result:
                                pending_refs.append(new_ref)
                                ref_to_engine[new_ref] = new_engine_idx
                                if new_ds_indices:
                                    ref_to_dataset_idx[new_ref] = new_ds_indices[0]
                                engine_pending[new_engine_idx] += 1

        pbar.close()

        #### Write step trace (L6) ####
        self._write_step_trace(trace_step_idx, episode_traces)
        #### end write step trace ####

        self._last_generation_wall_time = time.time() - generation_start_time

        #### Smart replay logging ####
        if smart_replay and self.strategy.is_rank_0():
            logger.info(
                f"[Step {trace_step_idx}] Smart replay: "
                f"accepted={accepted_prompt_groups}, "
                f"too_easy={self._step_too_easy_count}, "
                f"too_hard={self._step_too_hard_count}, "
                f"missed={self._step_missed_count}"
            )
        #### end smart replay logging ####

        return accepted_experiences, episode_traces, prompts_consumed, exhausted

    def _dispatch_prompts_to_vllm(self, prompts: List[str], labels: List[str], **generate_kwargs) -> List[Tuple]:
        """Send prompts to rollout executors and return (ref, engine_idx) tuples."""
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
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
        for idx, (prompt, label) in enumerate(zip(prompts, labels)):
            llm_engine = self.vllm_engines[engine_indices[idx]]
            ref = llm_engine.generate_responses.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                num_samples=n_samples_per_prompt,
            )
            refs.append((ref, engine_indices[idx]))

        return refs

    def _process_response_into_experience(self, response, **generate_kwargs) -> Optional[Experience]:
        """Turn a single vLLM response into an Experience."""
        truncate_length = generate_kwargs.get("prompt_max_len", 1024) + generate_kwargs.get("max_new_tokens", 1024)

        # Base rollout fields from the output.
        tokenized_observation = response["observation_tokens"].copy()
        tokenized_ranges = response["action_ranges"]
        reward_val = response.get("reward", None)
        score_val = response.get("scores", None)

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

        #### Zero action token guard ####
        action_tokens = action_mask.sum().item()
        if action_tokens == 0:
            prompt_preview = response.get("prompt", "")[:100]
            logger.warning(
                f"Skipping experience with 0 action tokens (would cause NaN). "
                f"seq_len={len(tokenized_observation)}, ranges={tokenized_ranges}, "
                f"prompt={prompt_preview!r}"
            )
            return None
        #### end zero action token guard ####

        # Align rollout logprobs with the truncated action span.
        if response["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(response["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Collect simple stats about lengths and clipping.
        ones_indices = torch.where(action_mask)[0]
        response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
        total_length = attention_mask.float().sum()
        is_clipped = total_length >= truncate_length

        # Check if response was truncated (hit max_tokens limit, finish_reason == "length")
        is_truncated = response.get("truncated", False)

        info = {
            "response_length": torch.tensor([response_length]),
            "total_length": torch.tensor([total_length]),
            "response_clip_ratio": torch.tensor([is_clipped]),
            "truncated": torch.tensor([is_truncated]),
        }
        if reward_val is not None:
            info["reward"] = torch.tensor([reward_val])
        if score_val is not None:
            info["score"] = torch.tensor([score_val])

        # Convert extra logs to tensors for downstream consumers.
        extra_logs = response.get("extra_logs", {})
        for key, value in extra_logs.items():
            if isinstance(value, torch.Tensor):
                value = value.flatten()[0].item()
            info[key] = torch.tensor([value])

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

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [int(s.info["total_length"].item()) for s in rollout_samples]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )

            #### Adaptive batch: greedy bin-packing by descending length ####
            if getattr(self.args, "use_adaptive_batch", False):
                max_tokens = self.args.train_max_tokens_per_gpu
                sorted_indices = sorted(range(len(total_lengths)), key=lambda i: total_lengths[i], reverse=True)
                partitions = []
                partition_sums = []

                for idx in sorted_indices:
                    length = total_lengths[idx]
                    placed = False
                    for p_idx in range(len(partitions)):
                        if partition_sums[p_idx] + length <= max_tokens:
                            partitions[p_idx].append(idx)
                            partition_sums[p_idx] += length
                            placed = True
                            break
                    if not placed:
                        partitions.append([idx])
                        partition_sums.append(length)

                # Ensure partition count is a multiple of effective_actor_num
                while len(partitions) % effective_actor_num != 0:
                    biggest = max(range(len(partitions)), key=lambda i: len(partitions[i]))
                    if len(partitions[biggest]) < 2:
                        break
                    mid = len(partitions[biggest]) // 2
                    left = partitions[biggest][:mid]
                    right = partitions[biggest][mid:]
                    partitions[biggest] = left
                    partition_sums[biggest] = sum(total_lengths[i] for i in left)
                    partitions.append(right)
                    partition_sums.append(sum(total_lengths[i] for i in right))

                batch_indexes = partitions
            #### end adaptive batch ####
            else:
                minimum_batch_num = get_minimum_num_micro_batch_size(
                    total_lengths,
                    self.args.rollout_max_tokens_per_gpu,
                    self.args.ring_attn_size,
                    self.args.ds_tensor_parallel_size,
                )
                #### Ceiling fix: use math.ceil to prevent 0-batch partitions ####
                minimum_batch_num = math.ceil(minimum_batch_num / effective_actor_num) * effective_actor_num
                #### end ceiling fix ####
                num_batch = max(minimum_batch_num, effective_actor_num)
                batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)

            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(micro_batch, self.tokenizer.pad_token_id)
                samples_list.append(concat_samples)
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
        logger.info(f"Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples")

        args = self.strategy.args
        device = "cpu"

        # Extract all information from samples in one pass
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        use_reward_model = samples_list[0].rewards is None
        if use_reward_model:
            if self.reward_model_group is None:
                raise ValueError("reward_model_group is required when rewards are not precomputed")
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
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(ray.get(action_log_probs_ref)[::duplicate_factor], [])
        del action_log_probs_ref
        base_action_log_probs_list = sum(ray.get(base_action_log_probs_ref)[::duplicate_factor], [])
        del base_action_log_probs_ref
        value_list = sum(ray.get(value_ref)[::duplicate_factor], [])
        del value_ref

        # Process rewards based on source
        if use_reward_model:
            rewards_list = sum(ray.get(r_refs)[::duplicate_factor], [])
            del r_refs
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list) == len(action_log_probs_list) == len(base_action_log_probs_list) == len(value_list)
        ), f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"

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
        logger.info(f"Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        """
        args = self.strategy.args

        # Apply length penalties (DAPO overlong / ProRL stop properly) - BEFORE dynamic indices processing
        apply_length_penalties(experiences, args)

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        indices = torch.tensor(sum([experience.index for experience in experiences], []))
        raw_rewards = torch.cat([experience.rewards for experience in experiences], dim=0)
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True).repeat(1, args.n_samples_per_prompt).reshape(-1)[indices].split(exp_len)
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
