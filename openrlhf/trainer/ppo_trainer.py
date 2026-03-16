import ctypes
import gc
import json
import os
import time
from abc import ABC
from datetime import timedelta
from typing import Dict, Tuple

import ray
import torch
import transformers
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, SamplesGenerator
from openrlhf.trainer.ppo_utils.kl_controller import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


def prepare_datasets(strategy, tokenizer):
    args = strategy.args

    # prepare datasets
    train_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.prompt_split,
    )

    # Create train dataset
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, 1, True, True, prompts_dataset.collate_fn)

    # Create eval dataset if eval data exists
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval_split,
        )
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
        eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False, eval_dataset.collate_fn)
    else:
        eval_dataloader = None

    max_steps = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.num_episodes * args.max_epochs
    )
    return prompts_dataloader, eval_dataloader, max_steps


class BasePPOTrainer(ABC):
    """Training-side base class: model orchestration, logging/eval, PPO steps."""

    def __init__(
        self,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        tokenizer,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer

        if self.args.kl_target:
            self.kl_ctl = AdaptiveKLController(self.args.init_kl_coef, self.args.kl_target, self.args.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.args.init_kl_coef)

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            tokenizer,
        )

        # Tracking backends
        self.wandb_logger = WandbLogger(self.args) if self.args.use_wandb else None
        self.tensorboard_logger = TensorboardLogger(self.args) if self.args.use_tensorboard else None

    def fit(self, global_step: int = 0) -> None:
        raise NotImplementedError("fit method is not implemented")

    def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
        # Turn raw rollouts into PPO-ready trajectories with rewards.
        experiences = self.experience_maker.make_experience_batch(rollout_samples)

        # Peek at the first decoded sample for quick sanity check.
        _decode = self.tokenizer.decode if _TRANSFORMERS_V5 else self.tokenizer.batch_decode
        sample0 = [
            _decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[0],
            experiences[0].info["reward"][0].item(),
        ]
        print(sample0)

        # Balance experiences across DP ranks if needed.
        if self.args.use_dynamic_batch:
            experiences = balance_experiences(experiences, self.args)

        # Push experiences to actor (and critic) shards before PPO.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
        ray.get(refs)

        # Perform PPO optimization for actor/critic and gather metrics.
        status = self.ppo_train(global_step)

        # Sync weights to vLLM.
        if self.vllm_engines is not None:
            self.broadcast_to_vllm()

        # Refresh KL controller with the latest measurement.
        if "kl" in status:
            # TODO: KL controller must be FixedKLController; AdaptiveKLController is incompatible here.
            self.kl_ctl.update(status["kl"], self.args.rollout_batch_size * self.args.n_samples_per_prompt)

        status["generated_samples"] = sample0
        return status, global_step + 1

    def ppo_train(self, global_steps: int) -> Dict:
        """Run one PPO train step for critic + actor and return merged status dict."""
        status: dict = {}

        # Decide whether to train critic/actor this round (actor can be frozen initially).
        run_critic = self.critic_model_group is not None
        run_actor = global_steps > self.args.freezing_actor_steps and self.actor_model_group is not None

        def _run_sleep(group, **kwargs):
            # Sleep mode: reload -> fit -> offload (smaller GPU memory).
            ray.get(group.async_run_method(method_name="reload_states"))
            ref = group.async_run_method(method_name="fit", **kwargs)
            status.update(ray.get(ref)[0])
            ray.get(group.async_run_method(method_name="offload_states"))

        if self.args.deepspeed_enable_sleep:
            # Colocated/sleeping: run critic first, then actor.
            if run_critic:
                _run_sleep(self.critic_model_group)
            if run_actor:
                _run_sleep(self.actor_model_group, kl_ctl=self.kl_ctl.value)
        else:
            # Async: start jobs first, then wait and merge results.
            refs = []
            if run_critic:
                refs += self.critic_model_group.async_run_method(method_name="fit")
            if run_actor:
                refs += self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)

            for result in ray.get(refs):
                status.update(result)

        return status

    def broadcast_to_vllm(self) -> None:
        """Broadcast actor weights to vLLM engines.

        When vllm_enable_sleep is enabled, we use fine-grained control:
        1. Wake up only weights (not KV cache) to minimize GPU memory during weight sync
        2. Broadcast weights from actor model to vLLM
        3. Keep vLLM in weights-only state; KV cache will be woken up later before generation

        This approach reduces peak GPU memory during gradient sync by avoiding
        simultaneous allocation of both weights and KV cache.
        """
        if self.args.vllm_enable_sleep:
            # Wake up only weights for weight sync (not KV cache)
            # This avoids allocating KV cache memory during weight update
            batch_vllm_engine_call(self.vllm_engines, "wake_up", tags=["weights"])

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        # NOTE: We keep vLLM in weights-only state after weight sync.
        # KV cache will be woken up before generation in SamplesGenerator.

    def save_logs_and_checkpoints(self, global_step: int, logs_dict=None, client_states=None) -> None:
        logs_dict = logs_dict or {}
        if global_step % self.args.logging_steps == 0:
            if self.wandb_logger:
                self.wandb_logger.log_train(global_step, logs_dict)
            if self.tensorboard_logger:
                self.tensorboard_logger.log_train(global_step, logs_dict)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        client_states = client_states or {}
        if global_step % self.args.save_steps == 0:
            tag = f"global_step{global_step}"
            refs = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                refs.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(refs)

            #### Push HF checkpoint to Hub and optionally delete local copy ####
            if self.args.save_hf_ckpt and getattr(self.args, "push_to_hub", None):
                hf_ckpt_path = os.path.join(self.args.ckpt_path, f"{tag}_hf")
                if os.path.exists(hf_ckpt_path):
                    config_path = os.path.join(hf_ckpt_path, "training_config.json")
                    with open(config_path, "w") as f:
                        json.dump(vars(self.args), f, indent=2, default=str)

                    from huggingface_hub import HfApi

                    api = HfApi()
                    api.create_repo(self.args.push_to_hub, private=self.args.push_to_hub_private, exist_ok=True)
                    api.upload_folder(
                        folder_path=hf_ckpt_path,
                        repo_id=self.args.push_to_hub,
                        commit_message=f"Checkpoint {tag}",
                        revision=tag,
                    )
                    logger.info(f"Uploaded {tag} to {self.args.push_to_hub} (branch: {tag})")
                    if self.args.delete_local_after_push:
                        import shutil

                        shutil.rmtree(hf_ckpt_path, ignore_errors=True)
                        logger.info(f"Deleted local checkpoint {hf_ckpt_path}")
            #### end push HF checkpoint ####

    def init_checkpoint_states(self) -> Dict:
        ckpt_path = os.path.join(self.args.ckpt_path, "_actor")
        if self.args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            return checkpoint_states
        return {
            "episode": 0,
            "global_step": 0,
            "total_consumed_prompts": 0,
            "data_loader_state_dict": {},
        }


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines,
        **generate_kwargs,
    ) -> None:
        # get eval and save steps
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt

        # Tokenizer is shared across the sample generator and trainer to avoid duplicated loads.
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)
        self.generate_kwargs = generate_kwargs

        # sample generation
        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
        )

        # train
        super().__init__(
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            tokenizer,
        )

        #### Smart replay tracking (L11) ####
        self._round_counter = 0
        #### end smart replay tracking ####

    def get_max_steps(self):
        return self.max_steps

    #### Helper: empty model caches ####
    def _empty_all_model_caches(self):
        """Force garbage collection and clear GPU caches across all model groups."""
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    #### end helper ####

    #### Oversampling: LeftOverPrompts phase (L13) ####
    def _run_leftover_phase(self, episode: int, global_step: int, total_consumed_prompts: int) -> int:
        """Dispatch remaining missed_indices without oversampling."""
        missed_indices = self.samples_generator.get_missed_indices()
        if len(missed_indices) < self.args.rollout_batch_size:
            if missed_indices:
                logger.info(
                    f"[LeftOverPrompts] {len(missed_indices)} missed indices "
                    f"(< batch_size={self.args.rollout_batch_size}), deferring to smart replay"
                )
            return global_step

        logger.info(f"[LeftOverPrompts] Processing {len(missed_indices)} missed indices")

        # Create Subset dataloader from missed_indices.
        original_dataset = self.samples_generator._original_dataset
        subset = Subset(original_dataset, list(missed_indices))
        leftover_dataloader = DataLoader(
            subset, batch_size=1, shuffle=False, collate_fn=original_dataset.collate_fn
        )

        # Temporarily swap dataloader; clear consumed missed indices.
        saved_dataloader = self.samples_generator.prompts_dataloader
        self.samples_generator.prompts_dataloader = leftover_dataloader
        self.samples_generator._missed_indices = set()

        while True:
            log_step_trace = global_step % 2 == 0
            rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                self.samples_generator.generate_samples(
                    global_step=global_step, log_step_trace=log_step_trace,
                    oversample_ratio=1.0,  # NO oversampling in leftover phase
                    _skip_clear_replay=True,  # preserve replay indices
                    **self.generate_kwargs,
                )
            )
            total_consumed_prompts += prompts_consumed

            if is_exhausted:
                if rollout_samples:
                    status, global_step = self.train_step(rollout_samples, global_step)
                    log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                    logger.info(f"Global step {global_step} [leftover-partial]: {log_status}")
                    client_states = {
                        "episode": episode,
                        "global_step": global_step,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": {},
                    }
                    self.save_logs_and_checkpoints(global_step, status, client_states)
                    del rollout_samples, status
                    gc.collect()
                break

            status, global_step = self.train_step(rollout_samples, global_step)
            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = filter_pass_rate
            status["leftover/phase"] = 1

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"Global step {global_step} [leftover]: {log_status}")

            client_states = {
                "episode": episode,
                "global_step": global_step,
                "total_consumed_prompts": total_consumed_prompts,
                "data_loader_state_dict": {},
            }
            self.save_logs_and_checkpoints(global_step, status, client_states)

            if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                eval_generate_kwargs = self.generate_kwargs.copy()
                eval_generate_kwargs["temperature"] = self.args.eval_temperature
                eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                self.evaluate(global_step, **eval_generate_kwargs)

            del rollout_samples, status
            gc.collect()
            self._empty_all_model_caches()
            self.samples_generator.flush_timeseries_to_disk(global_step=global_step)

        # Restore original dataloader.
        self.samples_generator.prompts_dataloader = saved_dataloader
        # Any new missed_indices from this phase stay for smart replay.
        return global_step
    #### end oversampling ####

    #### Smart replay: replay filtered prompts (L11) ####
    def _run_replay_episodes(self, episode: int, global_step: int, total_consumed_prompts: int) -> int:
        """After the primary episode, replay filtered prompts for up to max_replay_rounds."""
        hard_indices, kept_indices = self.samples_generator.get_replay_indices()
        missed_indices = self.samples_generator.get_missed_indices()
        replay_indices = list(hard_indices | kept_indices | missed_indices)
        max_replay_rounds = getattr(self.args, "max_replay_rounds", 2)
        original_dataloader = self.samples_generator.prompts_dataloader

        for replay_round in range(max_replay_rounds):
            if len(replay_indices) < self.args.rollout_batch_size:
                logger.info(
                    f"[SmartReplay] Round {replay_round + 1}: only {len(replay_indices)} prompts "
                    f"(< rollout_batch_size={self.args.rollout_batch_size}), skipping."
                )
                break

            logger.info(
                f"[SmartReplay] Episode {episode + 1}, round {replay_round + 1}/{max_replay_rounds}: "
                f"replaying {len(replay_indices)} prompts (hard={len(hard_indices)}, kept={len(kept_indices)})"
            )

            subset = Subset(original_dataloader.dataset, replay_indices)
            replay_dataloader = DataLoader(
                subset,
                batch_size=1,
                shuffle=True,
                collate_fn=original_dataloader.dataset.collate_fn,
            )
            self.samples_generator.prompts_dataloader = replay_dataloader

            pbar = tqdm(
                range(len(replay_dataloader)),
                desc=f"Episode [{episode + 1}] Replay round {replay_round + 1}",
            )
            while True:
                log_step_trace = global_step % 2 == 0
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(
                        global_step=global_step, log_step_trace=log_step_trace, **self.generate_kwargs
                    )
                )
                total_consumed_prompts += prompts_consumed
                if is_exhausted:
                    break

                status, global_step = self.train_step(rollout_samples, global_step)
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = filter_pass_rate
                status["replay/round"] = replay_round + 1

                log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                logger.info(f"Global step {global_step} [replay]: {log_status}")

                client_states = {
                    "episode": episode,
                    "global_step": global_step,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": {},
                }
                self.save_logs_and_checkpoints(global_step, status, client_states)

                if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    eval_generate_kwargs = self.generate_kwargs.copy()
                    eval_generate_kwargs["temperature"] = self.args.eval_temperature
                    eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                    self.evaluate(global_step, **eval_generate_kwargs)

                pbar.update(prompts_consumed)
                del rollout_samples, status
                gc.collect()
                self._empty_all_model_caches()
                self.samples_generator.flush_timeseries_to_disk(global_step=global_step)

            # Log episode stats for this replay round.
            if self.wandb_logger:
                self.wandb_logger.log_episode(
                    self._round_counter,
                    episode,
                    replay_round + 1,
                    self.samples_generator.episode_filter_stats,
                )
                self._round_counter += 1

            # Run leftover phase after each replay round.
            global_step = self._run_leftover_phase(episode, global_step, total_consumed_prompts)

            # Eval at end of replay round (skip if last step already ran eval).
            if self.eval_dataloader and (global_step % self.args.eval_steps != 0):
                eval_generate_kwargs = self.generate_kwargs.copy()
                eval_generate_kwargs["temperature"] = self.args.eval_temperature
                eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                logger.info(f"Running end-of-replay-round evaluation at global_step {global_step}")
                self.evaluate(global_step, **eval_generate_kwargs)

            # Collect new replay indices from this round.
            hard_indices, kept_indices = self.samples_generator.get_replay_indices()
            missed_indices = self.samples_generator.get_missed_indices()
            replay_indices = list(hard_indices | kept_indices | missed_indices)
            logger.info(
                f"[SmartReplay] Round {replay_round + 1} done. "
                f"{len(replay_indices)} non-easy prompts remain "
                f"(hard={len(hard_indices)}, kept={len(kept_indices)}, missed={len(missed_indices)})."
            )

        # Restore original dataloader.
        self.samples_generator.prompts_dataloader = original_dataloader
        return global_step
    #### end smart replay ####

    def fit(self, global_step: int = 0) -> None:
        checkpoint_states = self.init_checkpoint_states()
        # Restore step and start_epoch
        start_episode = checkpoint_states["episode"]
        # Use checkpoint's global_step if resuming, otherwise use the parameter
        is_resuming = checkpoint_states["global_step"] > 0
        if is_resuming:
            global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states["total_consumed_prompts"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step:
            self.broadcast_to_vllm()
            state_dict = checkpoint_states["data_loader_state_dict"]
            if state_dict:
                self.prompts_dataloader.load_state_dict(state_dict)

        #### Run config saving (L26) ####
        runs_dir = getattr(self.samples_generator, "runs_dir", None)
        if runs_dir and self.strategy.is_rank_0():
            config_path = os.path.join(runs_dir, "openrlhf_config.json")
            try:
                with open(config_path, "w") as f:
                    json.dump(vars(self.args), f, indent=2, default=str)
                logger.info(f"Saved run config to {config_path}")
            except Exception as e:
                logger.warning(f"Failed to save run config: {e}")
        #### end run config saving ####

        #### Run timing tracking (L18) ####
        run_timing_records = []
        run_start_time = time.time()
        #### end run timing init ####

        #### Skip eval at step zero (L21) ####
        skip_eval_step_zero = getattr(self.args, "skip_eval_step_zero", True)
        if not skip_eval_step_zero and self.eval_dataloader and global_step == 0:
            eval_generate_kwargs = self.generate_kwargs.copy()
            eval_generate_kwargs["temperature"] = self.args.eval_temperature
            eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
            logger.info("Running step-zero evaluation")
            self.evaluate(global_step, **eval_generate_kwargs)
        #### end skip eval step zero ####

        for episode in range(start_episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                #### Phase timings (L23) ####
                t_rollout_start = time.time()
                #### end phase timing start ####

                # Draw one mini-batch of prompts; stop when loader is exhausted.
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(**self.generate_kwargs)
                )
                total_consumed_prompts += prompts_consumed
                if is_exhausted:
                    break

                #### Phase timings (L23) ####
                t_rollout_end = time.time()
                #### end phase timing rollout ####

                # Run PPO update on this batch and bump the global step counter.
                status, global_step = self.train_step(rollout_samples, global_step)

                #### Phase timings (L23) ####
                t_train_end = time.time()
                status["time/rollout"] = t_rollout_end - t_rollout_start
                status["time/train_step"] = t_train_end - t_rollout_end

                # Record run timing (L18)
                run_timing_records.append({
                    "global_step": global_step,
                    "episode": episode,
                    "t": t_train_end,
                    "rollout_time": t_rollout_end - t_rollout_start,
                    "train_time": t_train_end - t_rollout_end,
                    "total_time": t_train_end - t_rollout_start,
                    "prompts_consumed": prompts_consumed,
                })
                #### end phase timings ####

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = filter_pass_rate
                log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                logger.info(f"✨ Global step {global_step}: {log_status}")

                # logs/checkpoints
                client_states = {
                    "episode": episode,
                    "global_step": global_step,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(global_step, status, client_states)

                # TODO: Add evaluation mechanism for PPO
                if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    eval_generate_kwargs = self.generate_kwargs.copy()
                    eval_generate_kwargs["temperature"] = self.args.eval_temperature
                    eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                    self.evaluate(global_step, **eval_generate_kwargs)

                pbar.update(prompts_consumed)

            #### Smart replay & leftover integration (L11, L12, L13) ####
            # Save discarded prompts for offline analysis.
            self.samples_generator.save_discarded_indices(episode)

            # LeftOverPrompts phase after main episode, before smart replay.
            if getattr(self.args, "oversample_ratio", 1.0) > 1.0:
                global_step = self._run_leftover_phase(episode, global_step, total_consumed_prompts)

            # Eval at end of episode (skip if last step already ran eval).
            if self.eval_dataloader and (global_step % self.args.eval_steps != 0):
                eval_generate_kwargs = self.generate_kwargs.copy()
                eval_generate_kwargs["temperature"] = self.args.eval_temperature
                eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                logger.info(f"Running end-of-episode evaluation at global_step {global_step}")
                self.evaluate(global_step, **eval_generate_kwargs)

            # Smart replay: log initial-pass stats and run replay episodes.
            if getattr(self.args, "smart_replay", False):
                if self.wandb_logger:
                    self.wandb_logger.log_episode(
                        self._round_counter,
                        episode,
                        0,
                        self.samples_generator.episode_filter_stats,
                    )
                    self._round_counter += 1
                global_step = self._run_replay_episodes(episode, global_step, total_consumed_prompts)
            #### end smart replay & leftover integration ####

        #### Write run timing (L18) ####
        if runs_dir and self.strategy.is_rank_0() and run_timing_records:
            timing_path = os.path.join(runs_dir, "vllm_stats", "run_timing.jsonl")
            try:
                os.makedirs(os.path.dirname(timing_path), exist_ok=True)
                with open(timing_path, "w") as f:
                    for record in run_timing_records:
                        f.write(json.dumps(record) + "\n")
                logger.info(f"Wrote {len(run_timing_records)} timing records to {timing_path}")
            except Exception as e:
                logger.warning(f"Failed to write run timing: {e}")

            # Write run summary
            summary_path = os.path.join(runs_dir, "vllm_stats", "run_summary.json")
            try:
                total_elapsed = time.time() - run_start_time
                summary = {
                    "total_steps": global_step,
                    "total_elapsed_seconds": total_elapsed,
                    "total_consumed_prompts": total_consumed_prompts,
                    "avg_step_time": total_elapsed / max(global_step, 1),
                }
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Wrote run summary to {summary_path}")
            except Exception as e:
                logger.warning(f"Failed to write run summary: {e}")
        #### end write run timing ####

        # Close trackers
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    @torch.no_grad()
    def evaluate(self, global_step, **generate_kwargs):
        """Evaluate model performance on eval dataset."""
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # First collect all prompts and labels
        prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources
        for datasources, prompts, labels in self.eval_dataloader:
            # Create mapping for each prompt to its corresponding data source
            for prompt, datasource in zip(prompts, datasources):
                prompt_to_datasource[prompt] = datasource

        # Generate samples and calculate rewards
        samples_list = self.samples_generator.generate_eval_samples(**generate_kwargs)

        # duplicate prompts and labels for each sample
        all_prompts = sum([s.prompts for s in samples_list], [])

        n_samples_per_prompt = generate_kwargs["n_samples_per_prompt"]

        # Get rewards from samples, such as agent rewards or remote reward models
        rewards_list = []
        for samples in samples_list:
            rewards_list.append(samples.rewards)
        # Reshape rewards to (num_prompts, n_samples_per_prompt)
        rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

        # Collect local statistics for each data source
        global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

        # Process rewards in chunks of n_samples_per_prompt
        num_prompts = len(all_prompts) // n_samples_per_prompt
        for i in range(num_prompts):
            # Get the original prompt (first one in the chunk)
            original_prompt = all_prompts[i * n_samples_per_prompt]
            datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping
            if datasource not in global_metrics:
                global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

            # Get rewards for this chunk
            chunk_rewards = rewards[i]

            # Calculate pass@k and pass@1
            if n_samples_per_prompt > 1:
                global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
            global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
            global_metrics[datasource]["count"] += 1

        # Calculate global averages
        logs = {}
        for datasource, metrics in global_metrics.items():
            logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
            )
            logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

        # Log to wandb/tensorboard
        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")
