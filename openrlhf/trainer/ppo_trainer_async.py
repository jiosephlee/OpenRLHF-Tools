import asyncio
import gc

import ray
from ray.util.queue import Queue
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset

from openrlhf.trainer.ppo_trainer import BasePPOTrainer, prepare_datasets
from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


@ray.remote(num_cpus=0)
class VLLMLock:
    """Cross-actor mutex for vLLM critical section."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self._lock.acquire()

    async def release(self):
        self._lock.release()


@ray.remote
class GenerateSamplesActor:
    def __init__(
        self,
        pretrain,
        strategy,
        vllm_engines,
        *,
        vllm_lock,
        rollout_queue,
        rollout_slots,
        **generate_kwargs,
    ):
        self.args = strategy.args

        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)
        self.generate_kwargs = generate_kwargs

        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=self.prompts_dataloader,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
        )

        self.vllm_lock = vllm_lock
        self.rollout_queue = rollout_queue
        # Token bucket: size == rollout_queue capacity.
        # Generator MUST take a token BEFORE generating; trainer returns token AFTER consuming.
        self.rollout_slots = rollout_slots

    def get_max_steps(self):
        return self.max_steps

    def load_state_dict(self, state_dict):
        self.prompts_dataloader.load_state_dict(state_dict)

    def log_dataloader_order(self):
        """Log first/last 100 samples from the dataloader for validation.

        Mirrors PPOTrainer._log_dataloader_order() but runs on the generator actor
        which owns the prompts_dataloader.
        """
        import json as _json
        import os

        run_name = getattr(self.args, "wandb_run_name", "run").replace("/", "_")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, "runs", run_name, "dataloader_logs")
        os.makedirs(log_dir, exist_ok=True)

        all_samples = []
        for _indices, datasources, prompts, _labels in self.prompts_dataloader:
            for ds, prompt in zip(datasources, prompts):
                all_samples.append((ds, prompt))

        n = len(all_samples)
        head = 100
        tail = 100

        log_path = os.path.join(log_dir, "dataset_order.jsonl")
        with open(log_path, "w") as f:
            f.write(_json.dumps({"total_samples": n, "head": head, "tail": tail}) + "\n")
            for i in range(min(head, n)):
                ds, prompt = all_samples[i]
                f.write(_json.dumps({
                    "index": i,
                    "datasource": ds,
                    "prompt_tail": prompt[-100:] if len(prompt) > 100 else prompt,
                }) + "\n")
            if n > head + tail:
                f.write(_json.dumps({"gap": f"...skipped indices {head} to {n - tail - 1}..."}) + "\n")
            for i in range(max(head, n - tail), n):
                ds, prompt = all_samples[i]
                f.write(_json.dumps({
                    "index": i,
                    "datasource": ds,
                    "prompt_tail": prompt[-100:] if len(prompt) > 100 else prompt,
                }) + "\n")

        logger.info(f"[DataloaderLog] Wrote {n} sample summary to {log_path}")

        if n > 0:
            sample_prompt_path = os.path.join(log_dir, "sample0_prompt.txt")
            with open(sample_prompt_path, "w", encoding="utf-8") as f:
                f.write(all_samples[0][1])
            logger.info(f"[DataloaderLog] Wrote full system prompt of sample 0 to {sample_prompt_path}")

    def get_episode_filter_stats(self):
        """Return episode filter stats for W&B logging (called by PPOTrainerAsync after each episode)."""
        return self.samples_generator.episode_filter_stats

    def fit(self, episode: int, total_consumed_prompts: int):
        for episode in range(episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                # Backpressure: only generate if we have queue capacity (token available).
                self.rollout_slots.get(block=True)

                # vLLM critical section: generation must not overlap with broadcast_to_vllm().
                ray.get(self.vllm_lock.acquire.remote())
                try:
                    rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                        self.samples_generator.generate_samples(global_step=total_consumed_prompts, **self.generate_kwargs)
                    )
                    total_consumed_prompts += prompts_consumed
                    # Capture per-step filtering stats before they're reset by the next generate_samples call.
                    step_too_easy_pct = self.samples_generator.step_too_easy_pct
                    step_too_hard_pct = self.samples_generator.step_too_hard_pct
                finally:
                    ray.get(self.vllm_lock.release.remote())

                produced = bool(rollout_samples)
                if produced:
                    client_states = {
                        "episode": episode,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                    }
                    self.rollout_queue.put((rollout_samples, client_states, filter_pass_rate, step_too_easy_pct, step_too_hard_pct), block=True)
                    if prompts_consumed:
                        pbar.update(prompts_consumed)
                else:
                    # Nothing enqueued => trainer will never "consume" this slot,
                    # so we must return the token here (prevents token leak / deadlock).
                    self.rollout_slots.put(None, block=True)

                if is_exhausted:
                    break

            pbar.close()
            self.samples_generator.save_discarded_indices(episode)

            if getattr(self.args, "smart_replay", False):
                total_consumed_prompts = self._run_replay_episodes(episode, total_consumed_prompts)

        self.rollout_queue.put("done", block=True)

    def _run_replay_episodes(self, episode, total_consumed_prompts):
        """After primary episode, replay filtered prompts for up to max_replay_rounds."""
        hard_indices, kept_indices = self.samples_generator.get_replay_indices()
        replay_indices = list(hard_indices | kept_indices)
        max_replay_rounds = getattr(self.args, "max_replay_rounds", 2)
        original_dataloader = self.samples_generator.prompts_dataloader

        for replay_round in range(max_replay_rounds):
            if len(replay_indices) < self.args.rollout_batch_size:
                logger.info(
                    f"[AsyncSmartReplay] Round {replay_round + 1}: only {len(replay_indices)} prompts "
                    f"(< rollout_batch_size={self.args.rollout_batch_size}), skipping."
                )
                break

            logger.info(
                f"[AsyncSmartReplay] Episode {episode + 1}, round {replay_round + 1}/{max_replay_rounds}: "
                f"replaying {len(replay_indices)} prompts (hard={len(hard_indices)}, kept={len(kept_indices)})"
            )

            subset = Subset(original_dataloader.dataset, replay_indices)
            replay_dataloader = DataLoader(
                subset, batch_size=1, shuffle=True,
                collate_fn=original_dataloader.dataset.collate_fn,
            )
            self.samples_generator.prompts_dataloader = replay_dataloader

            while True:
                self.rollout_slots.get(block=True)
                ray.get(self.vllm_lock.acquire.remote())
                try:
                    rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                        self.samples_generator.generate_samples(
                            global_step=total_consumed_prompts, **self.generate_kwargs
                        )
                    )
                    total_consumed_prompts += prompts_consumed
                    step_too_easy_pct = self.samples_generator.step_too_easy_pct
                    step_too_hard_pct = self.samples_generator.step_too_hard_pct
                finally:
                    ray.get(self.vllm_lock.release.remote())

                if rollout_samples:
                    client_states = {
                        "episode": episode,
                        "total_consumed_prompts": total_consumed_prompts,
                        "data_loader_state_dict": {},  # ephemeral, not resumable
                    }
                    self.rollout_queue.put(
                        (rollout_samples, client_states, filter_pass_rate,
                         step_too_easy_pct, step_too_hard_pct),
                        block=True,
                    )
                else:
                    self.rollout_slots.put(None, block=True)

                if is_exhausted:
                    break

            hard_indices, kept_indices = self.samples_generator.get_replay_indices()
            replay_indices = list(hard_indices | kept_indices)
            logger.info(
                f"[AsyncSmartReplay] Round {replay_round + 1} done. "
                f"{len(replay_indices)} non-easy prompts remain (hard={len(hard_indices)}, kept={len(kept_indices)})."
            )

        self.samples_generator.prompts_dataloader = original_dataloader
        return total_consumed_prompts


@ray.remote
class TrainingActor(BasePPOTrainer):
    def __init__(
        self,
        pretrain,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_group,
        reference_model_group,
        vllm_engines,
        *,
        vllm_lock,
        rollout_queue,
        rollout_slots,
        **generate_kwargs,
    ):
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)

        super().__init__(
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            tokenizer,
        )

        # Evaluation support: load eval dataloader and create a samples generator.
        _, self.eval_dataloader, _ = prepare_datasets(strategy, tokenizer)
        self.samples_generator = SamplesGenerator(
            strategy=strategy,
            prompts_dataloader=None,
            eval_dataloader=self.eval_dataloader,
            tokenizer=tokenizer,
            vllm_engines=vllm_engines,
        )
        self.generate_kwargs = generate_kwargs

        self.vllm_lock = vllm_lock
        self.rollout_queue = rollout_queue
        self.rollout_slots = rollout_slots

    def _evaluate_with_lock(self, global_step):
        """Run evaluation while holding the vLLM lock to prevent generation overlap."""
        eval_generate_kwargs = self.generate_kwargs.copy()
        eval_generate_kwargs["temperature"] = self.args.eval_temperature
        eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
        ray.get(self.vllm_lock.acquire.remote())
        try:
            self.evaluate(global_step, **eval_generate_kwargs)
        finally:
            ray.get(self.vllm_lock.release.remote())

    def fit(self, global_step: int):
        while True:
            payload = self.rollout_queue.get(block=True)
            if payload == "done":
                break

            rollout_samples, client_states, filter_pass_rate, step_too_easy_pct, step_too_hard_pct = payload

            # Batch consumed => free one token to allow generator to produce next batch.
            self.rollout_slots.put(None, block=True)

            status, global_step = self.train_step(rollout_samples, global_step)

            if self.args.dynamic_filtering:
                status["dynamic_filtering_pass_rate"] = filter_pass_rate
                status["too_easy_pct"] = step_too_easy_pct
                status["too_hard_pct"] = step_too_hard_pct

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"✨ Global step {global_step}: {log_status}")

            client_states.update({"global_step": global_step})
            self.save_logs_and_checkpoints(global_step, status, client_states)

            # Periodic evaluation (mirrors sync trainer).
            if global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                self._evaluate_with_lock(global_step)

            # Free accumulated Ray object store refs and Python garbage.
            del rollout_samples, status
            gc.collect()

        # Final eval at end of training (skip if last step already ran eval).
        if self.eval_dataloader and (global_step % self.args.eval_steps != 0):
            logger.info(f"Running final evaluation at global_step {global_step}")
            self._evaluate_with_lock(global_step)

        self._write_final_tool_usage_plot()
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()

    def evaluate_step0(self):
        """Run evaluation at step 0 before async training starts."""
        if self.eval_dataloader and not self.args.skip_eval_step_zero:
            eval_generate_kwargs = self.generate_kwargs.copy()
            eval_generate_kwargs["temperature"] = self.args.eval_temperature
            eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
            self.evaluate(0, **eval_generate_kwargs)

    def broadcast_to_vllm(self):
        # vLLM critical section: must not overlap with generation.
        ray.get(self.vllm_lock.acquire.remote())
        try:
            super().broadcast_to_vllm()
        finally:
            ray.get(self.vllm_lock.release.remote())


@ray.remote
class PPOTrainerAsync:
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
        self.args = strategy.args

        # get eval and save steps
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt

        queue_size = getattr(strategy.args, "async_queue_size", 1)
        if queue_size <= 0:
            raise ValueError(f"async_queue_size must be positive, got {queue_size}")
        logger.info(f"queue_size={queue_size}")

        self.rollout_queue = Queue(maxsize=queue_size)

        # Token pool (counting semaphore) for queue capacity.
        self.rollout_slots = Queue(maxsize=queue_size)
        for _ in range(queue_size):
            self.rollout_slots.put(None, block=True)

        # Cross-actor mutex for vLLM critical section.
        self.vllm_lock = VLLMLock.remote()

        self.generator_actor = GenerateSamplesActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            vllm_engines=vllm_engines,
            vllm_lock=self.vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            **generate_kwargs,
        )

        self.trainer_actor = TrainingActor.remote(
            pretrain=pretrain,
            strategy=strategy,
            actor_model_group=actor_model_group,
            critic_model_group=critic_model_group,
            reward_model_group=reward_model_group,
            reference_model_group=reference_model_group,
            vllm_engines=vllm_engines,
            vllm_lock=self.vllm_lock,
            rollout_queue=self.rollout_queue,
            rollout_slots=self.rollout_slots,
            **generate_kwargs,
        )

    def fit(self) -> None:
        checkpoint_states = ray.get(self.trainer_actor.init_checkpoint_states.remote())

        # Restore step and epoch
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states.get("total_consumed_prompts", 0)
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step > 0:
            ray.get(
                [
                    self.generator_actor.load_state_dict.remote(checkpoint_states["data_loader_state_dict"]),
                    self.trainer_actor.broadcast_to_vllm.remote(),
                ]
            )

        # Log dataloader ordering for validation before training begins.
        ray.get(self.generator_actor.log_dataloader_order.remote())

        # Evaluate at step 0 (before any training) unless resuming from a checkpoint.
        if global_step == 0:
            ray.get(self.trainer_actor.evaluate_step0.remote())

        # Launch async training
        ray.get(
            [
                self.generator_actor.fit.remote(episode=start_episode, total_consumed_prompts=total_consumed_prompts),
                self.trainer_actor.fit.remote(global_step=global_step),
            ]
        )

    def get_max_steps(self):
        return ray.get(self.generator_actor.get_max_steps.remote())
