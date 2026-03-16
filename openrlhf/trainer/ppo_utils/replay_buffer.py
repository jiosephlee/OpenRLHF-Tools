import random
from abc import ABC
from dataclasses import dataclass, fields
from typing import List, Optional

import torch
from torch import distributed as dist

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import zero_pad_sequences


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

    # Process info dictionary
    kwargs["info"] = {}
    for key in items[0].info.keys():
        values = [item.info[key] for item in items]
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


def balance_experiences(experiences, args):
    """
    Balance experience accross dp
    Example:
        sorted lengths: [8,7,6,5,4,3,2,1], effective_num: 2
        first_half: [[8,7], [6,5]], last_half: [[3,4], [1,2]], interval_items: [[8,7], [1,2], [6,5], [3,4]]
        interval_merged: [[8,1,6,3], [7,2,5,4]]
    """
    # split experience, sort by total_length
    items_all = []
    for item in experiences:
        items_all.extend(split_experience_batch(item))
    items_all.sort(key=lambda x: x.info["total_length"], reverse=True)

    # split experience into chunks
    effective_num = (
        args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    )
    split_items = [items_all[i : i + effective_num] for i in range(0, len(items_all), effective_num)]
    half = len(split_items) // 2
    first_half = split_items[:half]
    last_half = [item[::-1] for item in split_items[half:]]

    # balance distribution by intervaling chunks
    interval_items = []
    for i in range(half):
        interval_items.append(first_half[i])
        interval_items.append(last_half[-(i + 1)])
    if len(last_half) > len(first_half):
        interval_items.append(last_half[0])

    interval_merged = list(zip(*interval_items))
    return [make_experience_batch(items) for items in interval_merged]


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
        #### Phase 3: adaptive batch, loss_type, legacy_loss_scaling ####
        adaptive_batch: bool = False,
        loss_type: str = "ppo",
        legacy_loss_scaling: bool = False,
        #### end Phase 3 ####
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.loss_type = loss_type
        self.legacy_loss_scaling = legacy_loss_scaling
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.dynamic_batch = dynamic_batch
        self.adaptive_batch = adaptive_batch
        self.dynamic_indices: List[List[int]] = []
        self.dynamic_loss_scale: List[float] = []
        self.dynamic_optimizer_step: List[int] = []

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
        if self.dynamic_batch:
            return len(self.dynamic_indices)
        else:
            return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        if self.dynamic_batch:
            indices = self.dynamic_indices[idx]
            return [self.items[i] for i in indices]
        else:
            return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        if self.dynamic_batch:
            batch = batch[0]
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    #### Micro batch stats tracking (L4) ####
    def _compute_micro_batch_stats(self, data_partitions, sample_lengths):
        """Compute per-step micro batch statistics for monitoring vRAM pressure."""
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
    #### end micro batch stats tracking ####

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

        # balance the number of microbatches across steps
        micro_batch_indices = []
        data_partitions = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            samples = sample_lengths[start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)
            data_partitions.append(partitions)
        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        self._compute_micro_batch_stats(data_partitions, sample_lengths)

        #### Loss scaling — fully separate legacy vs new paths ####
        loss_scales = []
        optimizer_steps = []

        if self.legacy_loss_scaling:
            # Legacy (upstream-compatible): sequence-proportional, rank-local,
            # no cross-rank sync, same formula for all loss types.
            for partitions in data_partitions:
                sample_num = sum(len(partition) for partition in partitions)
                loss_scale = [len(partition) / max(sample_num, 1) for partition in partitions]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_type in ("ppo", "gspo", "sapo", "dr_grpo"):
            # Sequence-level: loss_scale = B_mb * R / N_global_sequences
            world_size = dist.get_world_size()
            for partitions in data_partitions:
                local_N = sum(len(partition) for partition in partitions)
                global_N = torch.tensor(local_N, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_N, op=dist.ReduceOp.SUM)
                global_N = global_N.item()
                loss_scale = [len(partition) * world_size / max(global_N, 1.0) for partition in partitions]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_type in ("dapo", "cispo"):
            # Token-level with global normalization
            world_size = dist.get_world_size()
            for partitions in data_partitions:
                token_counts = []
                for partition in partitions:
                    tc = sum(self.items[idx].action_mask.sum().item() for idx in partition)
                    token_counts.append(tc)
                local_T = sum(token_counts)
                global_T = torch.tensor(local_T, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_T, op=dist.ReduceOp.SUM)
                global_T = global_T.item()
                loss_scale = [tc * world_size / max(global_T, 1.0) for tc in token_counts]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        else:
            # BNPO: token-level, rank-local only (no cross-rank normalization)
            for partitions in data_partitions:
                token_counts = []
                for partition in partitions:
                    tc = sum(self.items[idx].action_mask.sum().item() for idx in partition)
                    token_counts.append(tc)
                total_tokens = sum(token_counts)
                if total_tokens > 0:
                    loss_scale = [tc / total_tokens for tc in token_counts]
                else:
                    loss_scale = [1.0 / len(partitions)] * len(partitions)
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)

        self.dynamic_loss_scale = loss_scales
        self.dynamic_optimizer_step = optimizer_steps
        #### end loss scaling ####

    #### Adaptive batching (for models not supporting flash attention/packing) ####
    def setup_adaptive_batch(self, strategy):
        """Adaptive batching groups sequences iteratively, ensuring that the VRAM footprint
        of the right-padded tensor (num_samples * max_seq_len) is <= train_max_tokens_per_gpu."""
        args = strategy.args
        sample_lengths = [sample.info["total_length"].item() for sample in self.items]

        world_size = dist.get_world_size()
        dp_size = world_size // args.ring_attn_size // args.ds_tensor_parallel_size
        local_train_batch_size = args.train_batch_size // dp_size
        total_samples = len(self.items)
        num_steps = total_samples // local_train_batch_size

        num_microbatches = []
        data_partitions = []
        micro_batch_indices = []

        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            samples_with_idx = [(idx + start, sample_lengths[idx + start]) for idx in range(end - start)]
            samples_with_idx.sort(key=lambda x: x[1], reverse=True)

            partitions = []
            current_partition = []
            default_max_len = 0

            for idx, length in samples_with_idx:
                effective_length = ((length + 1023) // 1024) * 1024
                new_size = len(current_partition) + 1
                new_max_len = max(default_max_len, effective_length) if current_partition else effective_length

                if current_partition and (new_size * new_max_len > args.train_max_tokens_per_gpu):
                    partitions.append([idx for idx, _ in current_partition])
                    current_partition = [(idx, length)]
                    default_max_len = effective_length
                else:
                    current_partition.append((idx, length))
                    default_max_len = new_max_len

            if current_partition:
                partitions.append([idx for idx, _ in current_partition])

            num_microbatches.append(len(partitions))
            data_partitions.append(partitions)

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        for i, num_mbs in enumerate(num_microbatches):
            partitions = data_partitions[i]
            while len(partitions) < num_mbs:
                biggest_idx = max(range(len(partitions)), key=lambda k: len(partitions[k]))
                target = partitions.pop(biggest_idx)
                mid = len(target) // 2
                if mid > 0:
                    partitions.append(target[:mid])
                    partitions.append(target[mid:])
                else:
                    partitions.append([])
            partitions = [p for p in partitions if len(p) > 0]
            micro_batch_indices.extend(partitions)

        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        self._compute_micro_batch_stats(data_partitions, sample_lengths)

        # Loss scaling — same logic as setup_dynamic_batch
        loss_scales = []
        optimizer_steps = []

        if self.legacy_loss_scaling:
            for partitions in data_partitions:
                sample_num = sum(len(partition) for partition in partitions)
                loss_scale = [len(partition) / max(sample_num, 1) for partition in partitions]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_type in ("ppo", "gspo", "sapo", "dr_grpo"):
            world_size = dist.get_world_size()
            for partitions in data_partitions:
                local_N = sum(len(partition) for partition in partitions)
                global_N = torch.tensor(local_N, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_N, op=dist.ReduceOp.SUM)
                global_N = global_N.item()
                loss_scale = [len(partition) * world_size / max(global_N, 1.0) for partition in partitions]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        elif self.loss_type in ("dapo", "cispo"):
            world_size = dist.get_world_size()
            for partitions in data_partitions:
                token_counts = []
                for partition in partitions:
                    tc = sum(self.items[idx].action_mask.sum().item() for idx in partition)
                    token_counts.append(tc)
                local_T = sum(token_counts)
                global_T = torch.tensor(local_T, dtype=torch.float, device=torch.cuda.current_device())
                dist.all_reduce(global_T, op=dist.ReduceOp.SUM)
                global_T = global_T.item()
                loss_scale = [tc * world_size / max(global_T, 1.0) for tc in token_counts]
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)
        else:
            for partitions in data_partitions:
                token_counts = []
                for partition in partitions:
                    tc = sum(self.items[idx].action_mask.sum().item() for idx in partition)
                    token_counts.append(tc)
                total_tokens = sum(token_counts)
                if total_tokens > 0:
                    loss_scale = [tc / total_tokens for tc in token_counts]
                else:
                    loss_scale = [1.0 / len(partitions)] * len(partitions)
                optimizer_step = [0] * (len(partitions) - 1) + [1]
                loss_scales.extend(loss_scale)
                optimizer_steps.extend(optimizer_step)

        self.dynamic_loss_scale = loss_scales
        self.dynamic_optimizer_step = optimizer_steps
    #### end adaptive batching ####
