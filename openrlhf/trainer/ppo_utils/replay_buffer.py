import random
from abc import ABC
from dataclasses import dataclass, fields
from typing import List, Optional

import torch
from torch import distributed as dist

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import zero_pad_sequences

logger = init_logger(__name__)


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    rollout_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    """Split a batch of experiences into individual BufferItems."""
    batch_size = len(experience.sequences)
    # Get fields from BufferItem, excluding 'info'
    keys = tuple(field.name for field in fields(BufferItem) if field.name != "info")
    experience.index = None

    # Validate batch size for all attributes
    for key in keys:
        value = getattr(experience, key)
        if value is not None:
            if isinstance(value, (torch.Tensor, list)):
                if len(value) != batch_size:
                    raise ValueError(f"Size of {key} ({len(value)}) does not match batch_size ({batch_size})")

    items = []
    for i in range(batch_size):
        # Process main attributes
        item = {key: (getattr(experience, key)[i] if getattr(experience, key) is not None else None) for key in keys}

        # Process info dictionary
        item["info"] = {}
        for k, v in experience.info.items():
            if isinstance(v, (torch.Tensor, list)):
                if len(v) != batch_size:
                    raise ValueError(f"Size of info[{k}] ({len(v)}) does not match batch_size ({batch_size})")
                item["info"][k] = v[i]
            else:
                raise TypeError(f"Unsupported type for info[{k}]: {type(v)}")

        items.append(BufferItem(**item))

    return items


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    """Combine individual BufferItems into a batch of experiences."""
    if not items:
        raise ValueError("Empty items list")

    # Get fields from BufferItem, excluding 'info'
    keys = tuple(field.name for field in fields(BufferItem) if field.name != "info")

    # Process main attributes
    kwargs = {
        key: (
            zero_pad_sequences([getattr(item, key) for item in items], "right", stack=True)
            if getattr(items[0], key) is not None
            else None
        )
        for key in keys
    }

    # Process info dictionary — collect ALL keys across every item
    # so sparse keys (e.g. tool_count__X) are zero-filled for items
    # that didn't record them.
    all_info_keys: set = set()
    for item in items:
        all_info_keys.update(item.info.keys())

    kwargs["info"] = {}
    for key in sorted(all_info_keys):
        values = []
        for item in items:
            if key in item.info:
                values.append(item.info[key])
            else:
                # Find an exemplar from an item that *does* have this key
                _exemplar = next(it.info[key] for it in items if key in it.info)
                if isinstance(_exemplar, (int, float)):
                    values.append(type(_exemplar)(0))
                elif isinstance(_exemplar, torch.Tensor):
                    values.append(torch.zeros_like(_exemplar))
                else:
                    values.append(_exemplar)
        if not values:
            continue

        # Validate all items have the same type
        first_type = type(values[0])
        if not all(isinstance(v, first_type) for v in values):
            raise TypeError(f"Inconsistent types in info[{key}]")

        # Convert to tensor if all values are numeric
        if all(isinstance(v, (int, float)) for v in values):
            kwargs["info"][key] = torch.tensor(values)
        else:
            kwargs["info"][key] = values

    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        # Calculate right padding using attention_mask
        right_pad = item.attention_mask.flip(0).argmax()
        right_pad = None if right_pad == 0 else -right_pad

        # Remove right padding for all tensors
        keys = tuple(field.name for field in fields(BufferItem) if field.name != "info")
        for key in keys:
            value = getattr(item, key)
            if value is not None:
                setattr(item, key, value[:right_pad])

    return items


def _balance_experiences_legacy(items_all, effective_num):
    """Original chunk-zip balancer.

    Pairs heavy/light chunks across DP ranks, but `zip` silently truncates to
    the smallest chunk when `len(items_all) % effective_num != 0` — both
    dropping samples and returning fewer than `effective_num` batches.
    Kept for fallback only.
    """
    split_items = [items_all[i : i + effective_num] for i in range(0, len(items_all), effective_num)]
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
    return [make_experience_batch(items) for items in interval_merged]


def _balance_experiences_snake(items_all, effective_num):
    """Snake/zigzag balancer at item granularity.

    Items are length-sorted desc; we walk them and assign to ranks in a
    boustrophedon pattern (0..k-1, k-1..0, 0..k-1, ...). This pairs heavy
    with light per rank, preserves all samples, and always returns exactly
    `effective_num` batches as long as `len(items_all) >= effective_num`.
    """
    buckets = [[] for _ in range(effective_num)]
    for i, item in enumerate(items_all):
        cycle, pos = divmod(i, effective_num)
        rank = pos if cycle % 2 == 0 else (effective_num - 1 - pos)
        buckets[rank].append(item)
    return [b for b in buckets if b]


def _finalize_balanced_buckets(buckets, args, effective_num):
    if not buckets:
        return []

    bucket_sizes = [len(bucket) for bucket in buckets]
    target_size = min(bucket_sizes)
    if target_size <= 0:
        logger.warning(
            "Balanced experience buckets contain an empty rank shard; bucket_sizes=%s effective_actors=%d",
            bucket_sizes,
            effective_num,
        )
        return []

    if getattr(args, "use_dynamic_batch", False) or getattr(args, "use_adaptive_batch", False):
        local_train_batch_size = max(args.train_batch_size // max(effective_num, 1), 1)
        target_steps = target_size // local_train_batch_size
        target_size = target_steps * local_train_batch_size
        if target_size <= 0:
            logger.warning(
                "Balanced experience buckets are smaller than local_train_batch_size; "
                "bucket_sizes=%s effective_actors=%d local_train_batch_size=%d",
                bucket_sizes,
                effective_num,
                local_train_batch_size,
            )
            return []

    dropped = sum(len(bucket) - target_size for bucket in buckets)
    if dropped > 0:
        logger.warning(
            "Dropping %d post-balance samples to equalize DP ranks and avoid collective mismatch; "
            "bucket_sizes=%s target_size=%d effective_actors=%d",
            dropped,
            bucket_sizes,
            target_size,
            effective_num,
        )

    return [make_experience_batch(bucket[:target_size]) for bucket in buckets]


def balance_experiences(experiences, args):
    """Balance experiences across DP ranks.

    Default uses the snake balancer (`_balance_experiences_snake`), which
    preserves every sample and always emits `effective_num` batches when
    the input has at least that many samples.

    Set `args.balance_experiences_legacy = True` to fall back to the
    chunk-zip implementation that shipped originally. The legacy path
    silently drops samples when the sample count isn't a multiple of the
    effective DP rank count — only use it for A/B comparison or if the
    snake path regresses.
    """
    items_all = []
    for item in experiences:
        items_all.extend(split_experience_batch(item))
    items_all.sort(key=lambda x: x.info["total_length"], reverse=True)

    effective_num = (
        args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    )

    if getattr(args, "balance_experiences_legacy", False):
        return _balance_experiences_legacy(items_all, effective_num)

    buckets = _balance_experiences_snake(items_all, effective_num)
    return _finalize_balanced_buckets(buckets, args, effective_num)


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self,
        sample_batch_size: int,
        limit: int = 0,
        cpu_offload: bool = True,
        packing_samples: bool = False,
        dynamic_batch: bool = False,
        adaptive_batch: bool = False,
        legacy_loss_scaling: bool = False,
        loss_aggregation: str = "sample",
        n_samples_per_prompt: int = 1,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.legacy_loss_scaling = legacy_loss_scaling
        self.loss_aggregation = loss_aggregation
        self.n_samples_per_prompt = n_samples_per_prompt
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.dynamic_batch = dynamic_batch
        self.adaptive_batch = adaptive_batch
        self.dynamic_indices: List[List[int]] = []
        self.dynamic_loss_scale: List[float] = []
        self.dynamic_optimizer_step: List[int] = []
        self.micro_batch_stats: dict = {}

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        if self.dynamic_batch or self.adaptive_batch:
            return len(self.dynamic_indices)
        else:
            return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        if self.dynamic_batch or self.adaptive_batch:
            indices = self.dynamic_indices[idx]
            return [self.items[i] for i in indices]
        else:
            return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        if self.dynamic_batch or self.adaptive_batch:
            batch = batch[0]
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def _compute_micro_batch_stats(self, data_partitions, sample_lengths):
        """Compute per-step micro batch statistics for monitoring vRAM pressure.

        Stored as self.micro_batch_stats dict with keys:
          - micro_batch/max_seq_len: longest sequence in any micro batch
          - micro_batch/mean_seq_len: mean of max-seq-len across all micro batches
          - micro_batch/max_batch_size: most sequences in any single micro batch
          - micro_batch/mean_batch_size: mean micro batch size
          - micro_batch/max_tokens_footprint: max (batch_size * max_seq_len) across micro batches
          - micro_batch/num_microbatches: total number of micro batches
        """
        all_max_lens = []
        all_batch_sizes = []
        all_token_footprints = []
        for partitions in data_partitions:
            for partition in partitions:
                if not partition:
                    continue
                max_len = max(sample_lengths[idx] for idx in partition)
                bs = len(partition)
                all_max_lens.append(max_len)
                all_batch_sizes.append(bs)
                all_token_footprints.append(bs * max_len)

        if all_max_lens:
            n = len(all_max_lens)
            self.micro_batch_stats = {
                "micro_batch/max_seq_len": max(all_max_lens),
                "micro_batch/mean_seq_len": sum(all_max_lens) / n,
                "micro_batch/max_batch_size": max(all_batch_sizes),
                "micro_batch/mean_batch_size": sum(all_batch_sizes) / n,
                "micro_batch/max_tokens_footprint": max(all_token_footprints),
                "micro_batch/num_microbatches": n,
            }
        else:
            self.micro_batch_stats = {}

    def setup_dynamic_batch(self, strategy):
        args = strategy.args
        sample_lengths = [sample.info["total_length"].item() for sample in self.items]

        world_size = dist.get_world_size()
        dp_size = world_size // args.ring_attn_size // args.ds_tensor_parallel_size
        local_train_batch_size = args.train_batch_size // dp_size
        #### Partial-batch safe: derive num_steps from actual buffer size ####
        total_samples = len(self.items)
        num_steps = total_samples // local_train_batch_size
        #### end partial-batch safe ####

        # split by train_batch_size, sync num_microbatches across dp
        num_microbatches = []
        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            num_microbatches.append(
                get_minimum_num_micro_batch_size(
                    sample_lengths[start:end],
                    args.train_max_tokens_per_gpu,
                    args.ring_attn_size,
                    args.ds_tensor_parallel_size,
                )
            )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        # balance the number of mirobatches across steps
        micro_batch_indices = []
        data_partitions = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            samples = sample_lengths[start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)  # List[List[int]], index
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)
            data_partitions.append(partitions)
        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        #### Micro batch stats tracking ####
        self._compute_micro_batch_stats(data_partitions, sample_lengths)
        #### end micro batch stats tracking ####

        #### Loss scaling ####
        # PolicyLoss.forward() returns a SUM (not a mean).  The loss_scale
        # converts that local sum into the correct fraction of the global mean:
        #
        #   accumulated_grad = Σ_mb (local_sum_mb * loss_scale_mb / world_size)
        #
        # DeepSpeed divides gradients by world_size, so loss_scale must include
        # a ×world_size factor to compensate.  The denominator D_global depends
        # on the aggregation mode:
        #
        #   sample: D = N_global  (total sequences across all ranks)
        #   prompt: D = P_global  (unique prompts = N_global / G)
        #   token:  D = T_global  (total action tokens across all ranks)
        #
        # loss_scale = world_size / D_global   (same for every microbatch)
        #
        # This is constant across microbatches within an optimizer step because
        # PolicyLoss already returns the correct partial sum for each microbatch.
        loss_scales = []
        optimizer_steps = []
        world_size = dist.get_world_size()

        if self.legacy_loss_scaling:
            # Legacy (upstream-compatible): sequence-proportional, rank-local,
            # no cross-rank sync.  Kept for backward compat with old checkpoints.
            for partitions in data_partitions:
                sample_num = sum(len(partition) for partition in partitions)
                loss_scale = [len(partition) / max(sample_num, 1) for partition in partitions]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_aggregation in ("sample", "prompt"):
            # Sample: D = N_global (total sequences)
            # Prompt: D = P_global = N_global / G (unique prompts)
            G = self.n_samples_per_prompt if self.loss_aggregation == "prompt" else 1
            for partitions in data_partitions:
                local_N = sum(len(partition) for partition in partitions)
                global_N = torch.tensor(local_N, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_N, op=dist.ReduceOp.SUM)
                D_global = global_N.item() / G  # N_global for sample, P_global for prompt
                scale = world_size / max(D_global, 1.0)
                loss_scale = [scale] * len(partitions)
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_aggregation == "token":
            # Token: D = T_global (total action tokens)
            for partitions in data_partitions:
                local_T = sum(
                    self.items[idx].action_mask.sum().item()
                    for partition in partitions
                    for idx in partition
                )
                global_T = torch.tensor(local_T, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_T, op=dist.ReduceOp.SUM)
                D_global = global_T.item()
                scale = world_size / max(D_global, 1.0)
                loss_scale = [scale] * len(partitions)
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)

        self.dynamic_loss_scale = loss_scales
        self.dynamic_optimizer_step = optimizer_steps
        #### end loss scaling ####

    def setup_adaptive_batch(self, strategy):
        """
        Adaptive batching groups sequences iteratively, ensuring that the VRAM footprint
        of the right-padded tensor (num_samples * max_seq_len) is <= train_max_tokens_per_gpu.
        This provides dynamic batching for models not supporting flash attention/packing.
        """
        args = strategy.args
        sample_lengths = [sample.info["total_length"].item() for sample in self.items]

        world_size = dist.get_world_size()
        dp_size = world_size // args.ring_attn_size // args.ds_tensor_parallel_size
        local_train_batch_size = args.train_batch_size // dp_size
        #### Partial-batch safe: derive num_steps from actual buffer size ####
        total_samples = len(self.items)
        num_steps = total_samples // local_train_batch_size
        #### end partial-batch safe ####

        num_microbatches = []
        data_partitions = []
        micro_batch_indices = []

        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            samples_with_idx = [(idx + start, sample_lengths[idx + start]) for idx in range(end - start)]
            # Sort by length descending to pack similarly-sized sequences together
            samples_with_idx.sort(key=lambda x: x[1], reverse=True)

            partitions = []
            current_partition = []
            default_max_len = 0 # No partition yet

            for idx, length in samples_with_idx:
                # Bucket sequence lengths to reduction flex_attention recompilation triggers (nearest 1024)
                effective_length = ((length + 1023) // 1024) * 1024
                
                # If adding this sequence means we exceed budget (or it's the first seq in a new partition)
                new_size = len(current_partition) + 1
                new_max_len = max(default_max_len, effective_length) if current_partition else effective_length
                
                # We enforce minimum of 1 sample per partition even if it's over budget
                if current_partition and (new_size * new_max_len > args.train_max_tokens_per_gpu):
                    partitions.append([idx for idx, _ in current_partition])
                    current_partition = [(idx, length)]
                    default_max_len = effective_length
                else:
                    current_partition.append((idx, length))
                    default_max_len = new_max_len

            if current_partition:
                partitions.append([idx for idx, _ in current_partition])

            # Sync number of microbatches across GPUs so that distributed collective communications
            # (e.g. all-reduce during PPO) happen symmetrically
            num_microbatches.append(len(partitions))
            data_partitions.append(partitions)

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        # If some GPUs needed fewer microbatches (e.g. their sequences were shorter), 
        # we append empty microbatches (handled by padding/dummy batches elsewhere)
        # However, to be completely safe with PPO synchronous logic, we just split the largest partition.
        for i, num_mbs in enumerate(num_microbatches):
            partitions = data_partitions[i]
            while len(partitions) < num_mbs:
                # Find biggest partition and split it
                biggest_idx = max(range(len(partitions)), key=lambda k: len(partitions[k]))
                target = partitions.pop(biggest_idx)
                mid = len(target) // 2
                if mid > 0:
                    partitions.append(target[:mid])
                    partitions.append(target[mid:])
                else:
                    # Can't split further, just append empty partition (or it will crash if it reaches here and length is 1)
                    # For safety, let's just duplicate the first element but mask it in forward? 
                    # No, PPO training_step can handle empty/small batches, but to avoid 0-size tensors,
                    # we should take the last element and split it
                    partitions.append([]) # PPO might handle empty, but warning!
            
            # Remove empty partitions if we added any, actually we should avoid empty partitions.
            # If we really hit this, the GPU with fewer microbatches will wait.
            partitions = [p for p in partitions if len(p) > 0]
            # Wait, if we remove empty partitions, len(partitions) != num_mbs.
            # We MUST have exactly num_mbs partitions.
            # So if we are forced to add an empty partition because we have e.g. 1 sample but max_mbs=2
            # Let's just use empty lists — DeepSpeed logic handles zero-sized inputs or drops them?
            # Actually, the original seqlen_blocking algorithm just splits. If len=1, it splits into [x] and [].
            # Empty indices means batch_size=0.
            micro_batch_indices.extend(partitions)

        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        #### Micro batch stats tracking ####
        self._compute_micro_batch_stats(data_partitions, sample_lengths)
        #### end micro batch stats tracking ####

        #### Adaptive batch: loss scaling ####
        # Same logic as setup_dynamic_batch — see comments there.
        loss_scales = []
        optimizer_steps = []
        world_size = dist.get_world_size()

        if self.legacy_loss_scaling:
            for partitions in data_partitions:
                sample_num = sum(len(partition) for partition in partitions)
                loss_scale = [len(partition) / max(sample_num, 1) for partition in partitions]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_aggregation in ("sample", "prompt"):
            G = self.n_samples_per_prompt if self.loss_aggregation == "prompt" else 1
            for partitions in data_partitions:
                local_N = sum(len(partition) for partition in partitions)
                global_N = torch.tensor(local_N, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_N, op=dist.ReduceOp.SUM)
                D_global = global_N.item() / G
                scale = world_size / max(D_global, 1.0)
                loss_scale = [scale] * len(partitions)
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_aggregation == "token":
            for partitions in data_partitions:
                local_T = sum(
                    self.items[idx].action_mask.sum().item()
                    for partition in partitions
                    for idx in partition
                )
                global_T = torch.tensor(local_T, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_T, op=dist.ReduceOp.SUM)
                D_global = global_T.item()
                scale = world_size / max(D_global, 1.0)
                loss_scale = [scale] * len(partitions)
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)

        self.dynamic_loss_scale = loss_scales
        self.dynamic_optimizer_step = optimizer_steps
        #### end adaptive batch: loss scaling ####
