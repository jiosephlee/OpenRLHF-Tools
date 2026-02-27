import heapq
from collections import defaultdict
import json
import os
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
        run_name = getattr(self.args, "wandb_run_name", "run")
        run_name = run_name.replace("/", "_")

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.runs_dir = os.path.join(project_root, "runs", run_name)
        self.rollout_trace_run_dir = os.path.join(self.runs_dir, "traces")
        os.makedirs(self.rollout_trace_run_dir, exist_ok=True)
        logger.info(f"Rollout traces enabled at: {self.rollout_trace_run_dir}")

        # Smart replay: accumulate dataset indices by filter outcome across the episode.
        self._replay_hard_indices: set = set()
        self._replay_kept_indices: set = set()

        # Track discarded items for telemetry, regardless of smart replay.
        self._discarded_easy_indices: set = set()
        self._discarded_hard_indices: set = set()

        # Per-step filtering stats (reset each generate_samples call).
        self._step_too_easy_count = 0
        self._step_too_hard_count = 0
        self._step_prompts_consumed = 0
        # Per-episode filtering stats (reset each episode).
        self._episode_easy_count = 0
        self._episode_hard_count = 0

    def _to_jsonable(self, value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        return value

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
                sections.append({
                    "type": "prompt",
                    "token_range": [0, first_start],
                    "text": self.tokenizer.decode(obs_tokens[:first_start], skip_special_tokens=False),
                })

            for i, (start, end) in enumerate(action_ranges):
                sections.append({
                    "type": "action",
                    "index": i + 1,
                    "token_range": [start, end],
                    "text": self.tokenizer.decode(obs_tokens[start:end], skip_special_tokens=False),
                })
                if i + 1 < len(action_ranges):
                    next_start = action_ranges[i + 1][0]
                    if end < next_start:
                        sections.append({
                            "type": "observation",
                            "index": i + 1,
                            "token_range": [end, next_start],
                            "text": self.tokenizer.decode(obs_tokens[end:next_start], skip_special_tokens=False),
                        })
                else:
                    remaining = obs_tokens[end:]
                    if remaining:
                        sections.append({
                            "type": "trailing",
                            "token_range": [end, len(obs_tokens)],
                            "text": self.tokenizer.decode(remaining, skip_special_tokens=False),
                        })

        decoded["sections"] = sections
        return decoded

    def _strip_token_ids(self, trace: dict) -> dict:
        trace_no_ids = dict(trace)
        for key in ("observation_tokens", "token_ids", "prompt_token_ids"):
            trace_no_ids.pop(key, None)
        return trace_no_ids

    def _write_step_trace(self, step_idx: int, episode_traces: list, prompts_consumed: int, filtered_count: int, total_episodes: int = 0):
        if not self.rollout_trace_run_dir or not episode_traces:
            return
        step_id = step_idx + 1
        trace_path = os.path.join(self.rollout_trace_run_dir, f"step{step_id}.jsonl")
        # Only save the first trace per step to avoid excessive disk usage.
        engine_idx, trace = episode_traces[0]
        record = {
            "step": step_id,
            "episode": 0,
            "engine_idx": engine_idx,
            "prompts_consumed": prompts_consumed,
            "filtered_count": filtered_count,
            "total_episodes": total_episodes,
            "trace": self._strip_token_ids(trace),
            "decoded": self._decode_trace(trace),
        }
        with open(trace_path, "w") as f:
            f.write(json.dumps(self._to_jsonable(record), ensure_ascii=True) + "\n")

    def _write_eval_trace(self, eval_idx: int, episode_trace: tuple, total_episodes: int):
        if not self.rollout_trace_run_dir or episode_trace is None:
            return
        engine_idx, trace = episode_trace
        trace_path = os.path.join(self.rollout_trace_run_dir, f"eval_{eval_idx}.json")
        record = {
            "eval": eval_idx,
            "engine_idx": engine_idx,
            "total_episodes": total_episodes,
            "trace": self._strip_token_ids(trace),
            "decoded": self._decode_trace(trace),
        }
        with open(trace_path, "w") as f:
            f.write(json.dumps(self._to_jsonable(record), ensure_ascii=True))

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        if getattr(self, "_eval_dataloader_iter", None) is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._eval_dataloader_iter,
            num_prompts=len(self.eval_dataloader),
            dynamic_filtering=False,
            log_step_trace=False,
            **generate_kwargs,
        )
        self._write_eval_trace(
            int(generate_kwargs.get("global_step", 0)),
            getattr(self, "_last_episode_trace", None),
            getattr(self, "_last_total_episodes", 0),
        )

        # Reclaim host RAM in vLLM engine workers accumulated during generation.
        batch_vllm_engine_call(self.vllm_engines, "gc_collect")

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        self._eval_dataloader_iter = None

        return experiences

    def get_replay_indices(self) -> Tuple[set, set]:
        """Return (hard_indices, kept_indices) accumulated during the episode."""
        return self._replay_hard_indices, self._replay_kept_indices

    def clear_replay_indices(self):
        """Reset replay tracking for a new episode."""
        self._replay_hard_indices = set()
        self._replay_kept_indices = set()
        self._discarded_easy_indices = set()
        self._discarded_hard_indices = set()
        self._episode_easy_count = 0
        self._episode_hard_count = 0

    def save_discarded_indices(self, episode: int):
        """Write the discarded indices of this episode to the runs_dir."""
        if not self.args.dynamic_filtering:
            return
        out_path = os.path.join(self.runs_dir, f"discarded_indices_ep{episode}.json")
        data = {
            "episode": episode,
            "too_easy": sorted(list(self._discarded_easy_indices)),
            "too_hard": sorted(list(self._discarded_hard_indices))
        }
        with open(out_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved {len(self._discarded_easy_indices)} too_easy and {len(self._discarded_hard_indices)} too_hard indices to {out_path}")

    @property
    def step_too_easy_pct(self) -> float:
        """Percentage of prompts consumed this step that were too easy."""
        if self._step_prompts_consumed == 0:
            return 0.0
        return self._step_too_easy_count / self._step_prompts_consumed * 100

    @property
    def step_too_hard_pct(self) -> float:
        """Percentage of prompts consumed this step that were too hard."""
        if self._step_prompts_consumed == 0:
            return 0.0
        return self._step_too_hard_count / self._step_prompts_consumed * 100

    @property
    def episode_filter_stats(self) -> dict:
        """Per-episode filtering stats for W&B logging."""
        return {
            "easy_discarded": self._episode_easy_count,
            "hard_kept": self._episode_hard_count,
        }

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Produce one batch and indicate if the dataloader is exhausted."""
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)
            self.clear_replay_indices()
        trace_step_idx = getattr(self, "_trace_step_idx", 0)
        self._trace_step_idx = trace_step_idx + 1

        # Reset per-step counters.
        self._step_too_easy_count = 0
        self._step_too_hard_count = 0
        self._step_prompts_consumed = 0

        # Wake sleeping vLLM engines before dispatching.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        experiences, prompts_consumed, exhausted = self._generate_vllm(
            dataloader_iter=self._dataloader_iter,
            num_prompts=self.args.rollout_batch_size,
            dynamic_filtering=self.args.dynamic_filtering,
            trace_step_idx=trace_step_idx,
            **generate_kwargs,
        )
        self._step_prompts_consumed = prompts_consumed

        # Reclaim host RAM in vLLM engine workers accumulated during generation.
        batch_vllm_engine_call(self.vllm_engines, "gc_collect")

        # Put engines back to sleep when enabled.
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        filter_pass_rate = None
        if self.args.dynamic_filtering and prompts_consumed:
            filter_pass_rate = self.args.rollout_batch_size / prompts_consumed * 100

        if exhausted:
            self._dataloader_iter = None
            logger.info("Prompt dataloader is exhausted.")

        return experiences, filter_pass_rate, prompts_consumed, exhausted

    def _generate_vllm(
        self, dataloader_iter, num_prompts: int, dynamic_filtering, **generate_kwargs
    ) -> Tuple[List[Experience], int, bool]:
        """Generate a batch of Experiences with optional reward filtering."""
        step_idx = int(generate_kwargs.get("trace_step_idx", generate_kwargs.get("global_step", 0)))
        prompts_consumed = 0
        dataset_indices, prompts, labels, exhausted = _collect_prompt_batch(dataloader_iter, num_prompts)
        # Stop early if the prompt source is fully consumed.
        if exhausted and not prompts:
            return [], prompts_consumed, exhausted

        smart_replay = getattr(self.args, "smart_replay", False)

        multi_stage = getattr(self.args, "multi_stage_dispatch", False)
        n = len(prompts)

        if multi_stage:
            # Single-stage deferred dispatch: send 75% upfront, hold 25% as
            # reserve.  The reserve is dispatched as one bulk batch when any
            # engine drops to ≤4 pending requests.
            RESERVE_THRESHOLD = 4
            initial_count = max(1, int(n * 0.75))
            reserve_prompts = prompts[initial_count:]
            reserve_labels = labels[initial_count:]
            reserve_dataset_indices = dataset_indices[initial_count:]
            dispatches = self._dispatch_prompts_to_vllm(prompts[:initial_count], labels[:initial_count], **generate_kwargs)
        else:
            # Default: dispatch everything upfront.  The heap balancer in
            # _dispatch_prompts_to_vllm already spreads load evenly, and
            # vLLM's internal scheduler handles queuing efficiently.
            reserve_prompts = []
            reserve_labels = []
            reserve_dataset_indices = []
            dispatches = self._dispatch_prompts_to_vllm(prompts, labels, **generate_kwargs)

        pending_refs = [ref for ref, _ in dispatches]
        ref_to_engine = {ref: engine_idx for ref, engine_idx in dispatches}
        # Map each ref → its dataset index for smart replay tracking.
        ref_to_dataset_idx = {ref: dataset_indices[i] for i, (ref, _) in enumerate(dispatches)}
        prompts_consumed += len(prompts)
        reserve_dispatched = False

        # Track how many outstanding requests each engine has.
        engine_pending = defaultdict(int)
        for _, engine_idx in dispatches:
            engine_pending[engine_idx] += 1

        accepted_experiences: List[Experience] = []
        pbar = tqdm(range(num_prompts), desc="Generate samples")
        filtered_count = 0
        episode_traces: list = []
        total_episodes = 0
        exhausted_during_refill = False

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                engine_idx = ref_to_engine.pop(ref)
                ds_idx = ref_to_dataset_idx.pop(ref, None)
                engine_pending[engine_idx] -= 1

                # Single-stage reserve dispatch: when any engine drops to
                # ≤RESERVE_THRESHOLD pending, dispatch all reserve prompts at once.
                if multi_stage and reserve_prompts and not reserve_dispatched and engine_pending[engine_idx] <= RESERVE_THRESHOLD:
                    new_dispatches = self._dispatch_prompts_to_vllm(reserve_prompts, reserve_labels, **generate_kwargs)
                    for j, (new_ref, new_engine_idx) in enumerate(new_dispatches):
                        pending_refs.append(new_ref)
                        ref_to_engine[new_ref] = new_engine_idx
                        ref_to_dataset_idx[new_ref] = reserve_dataset_indices[j]
                        engine_pending[new_engine_idx] += 1
                    reserve_dispatched = True

                # Build Experience objects for each vLLM response returned from this worker.
                responses = ray.get(ref)
                total_episodes += len(responses)
                # Only keep the first trace per step — _write_step_trace only uses [0].
                # Holding ALL responses in episode_traces leaks hundreds of MB in
                # multi-turn mode (each resp contains full observation_tokens + log_probs).
                if not episode_traces:
                    episode_traces.append((engine_idx, responses[0]))
                experiences = [self._process_response_into_experience(response, **generate_kwargs) for response in responses]
                del responses  # free raw vLLM response dicts before processing next batch

                # Drop experiences if the average score falls outside the allowed range.
                if dynamic_filtering and all(e.scores is not None for e in experiences):
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.dynamic_filtering_reward_range
                    if avg_reward >= max_r:
                        # Too easy — drop; do NOT add to replay
                        filtered_count += 1
                        self._step_too_easy_count += 1
                        self._episode_easy_count += 1
                        if ds_idx is not None:
                            self._discarded_easy_indices.add(ds_idx)
                        if filtered_count <= 3 or filtered_count % 25 == 0:
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
                        self._episode_hard_count += 1
                        if ds_idx is not None:
                            self._discarded_hard_indices.add(ds_idx)
                        if smart_replay and ds_idx is not None:
                            self._replay_hard_indices.add(ds_idx)
                        if filtered_count <= 3 or filtered_count % 25 == 0:
                            logger.info(
                                "Dynamic filtering rejected group (too hard) "
                                f"(rejected={filtered_count}, accepted={len(accepted_experiences)}/{num_prompts}, "
                                f"avg_reward={avg_reward:.2f}, threshold=({min_r:.2f}, {max_r:.2f}))"
                            )
                        experiences = []
                    else:
                        # In range — kept; queue index for replay
                        if smart_replay and ds_idx is not None:
                            self._replay_kept_indices.add(ds_idx)

                # Accept experiences and stop once enough have been gathered.
                if experiences:
                    accepted_experiences.extend(experiences)
                    pbar.set_postfix({"prompts_consumed": prompts_consumed})
                    pbar.update()

                # If rejected, request a new prompt to keep filling the batch.
                else:
                    # Pull another prompt when the current one fails filtering.
                    new_ds_indices, new_prompts, new_labels, exhausted = _collect_prompt_batch(dataloader_iter, 1)
                    prompts_consumed += len(new_prompts)
                    # Dataloader drained: stop adding work and drain in-flight refs.
                    # This avoids racing vLLM sleep/wake against active decode kernels.
                    if exhausted:
                        logger.info(
                            "Prompt dataloader exhausted during refill; "
                            f"draining {len(pending_refs)} in-flight vLLM refs before sleep."
                        )
                        exhausted_during_refill = True
                    # Otherwise dispatch the new prompt to keep filling the queue.
                    else:
                        new_dispatches = self._dispatch_prompts_to_vllm(new_prompts, new_labels, **generate_kwargs)
                        for j, (new_ref, new_engine_idx) in enumerate(new_dispatches):
                            pending_refs.append(new_ref)
                            ref_to_engine[new_ref] = new_engine_idx
                            ref_to_dataset_idx[new_ref] = new_ds_indices[j]
                            engine_pending[new_engine_idx] += 1

        self._last_episode_trace = episode_traces[0] if episode_traces else None
        self._last_total_episodes = total_episodes
        if generate_kwargs.get("log_step_trace", True):
            self._write_step_trace(step_idx, episode_traces, prompts_consumed, filtered_count, total_episodes)

        if smart_replay and not exhausted_during_refill:
            logger.info(
                f"[SmartReplay] Step done: replay buffer now has "
                f"{len(self._replay_hard_indices)} hard + {len(self._replay_kept_indices)} kept "
                f"= {len(self._replay_hard_indices) + len(self._replay_kept_indices)} total prompts"
            )

        if exhausted_during_refill:
            if smart_replay:
                logger.info(
                    f"[SmartReplay] Step done (exhausted): replay buffer now has "
                    f"{len(self._replay_hard_indices)} hard + {len(self._replay_kept_indices)} kept "
                    f"= {len(self._replay_hard_indices) + len(self._replay_kept_indices)} total prompts"
                )
            return [], prompts_consumed, True

        return accepted_experiences, prompts_consumed, exhausted

    def _dispatch_prompts_to_vllm(self, prompts: List[str], labels: List[str], **generate_kwargs) -> List:
        """Send prompts to rollout executors and return Ray object refs."""
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens", 1024),
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            **({"spaces_between_special_tokens": False, "stop": self.args.vllm_stop_strings, "include_stop_str_in_output": True} if self.args.agent_func_path else {}),
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
            # Spread work across engines/workers in load-aware order.
            engine_idx = engine_indices[idx]
            llm_engine = self.vllm_engines[engine_idx]
            ref = llm_engine.generate_responses.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=truncate_length,
                num_samples=n_samples_per_prompt,
                log_trajectory=(idx == 0),
            )
            refs.append((ref, engine_idx))

        return refs

    def _process_response_into_experience(self, response, **generate_kwargs) -> Experience:
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
        action_tokens = int(action_mask.sum().item())
        if action_tokens == 0:
            raise ValueError(
                "Encountered rollout with zero action tokens after truncation; this will produce NaNs in PPO loss. "
                f"prompt={response['prompt'][:200]!r}, label={response['label']!r}, "
                f"observation_tokens={len(tokenized_observation)}, action_ranges={tokenized_ranges}, truncate_length={truncate_length}"
            )

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
        import math
        if self.args.use_dynamic_batch:
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
            split_items = [samples_list[i : i + effective_actor_num] for i in range(0, len(samples_list), effective_actor_num)]
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
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
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
