import ctypes
import gc
import json
import os
import time
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import Dict, Tuple

import ray
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.prompts_dataset import interleave_indices_by_datasource
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, SamplesGenerator
from openrlhf.trainer.ppo_utils.kl_controller import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import TensorboardLogger, WandbLogger, init_logger
from openrlhf.utils.tdc_reward_model import extract_final_answer
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


def _extract_tdc_binary_choice(text: str) -> str:
    answer = extract_final_answer(text)
    if not answer:
        raise ValueError(f"Failed to extract TDC binary choice from text: {text[:200]!r}")
    return answer.upper()


def _macro_f1(y_true, y_pred) -> float:
    classes = sorted(set(y_true) | set(y_pred))
    f1_sum = 0.0
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_sum += f1
    return f1_sum / len(classes)


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
    shuffle = not getattr(args, "curriculum_balanced", False)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, 1, True, shuffle, prompts_dataset.collate_fn)

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

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def _collect_eval_tool_usage(self, all_prompts, samples_list, prompt_to_datasource):
        per_dataset_counts = defaultdict(lambda: defaultdict(int))
        per_dataset_prompts_used = defaultdict(lambda: defaultdict(int))  # prompts that used tool at least once
        per_dataset_total_prompts = defaultdict(int)
        prompt_idx = 0
        for sample in samples_list:
            batch_size = len(sample.sequences)
            for i in range(batch_size):
                if prompt_idx >= len(all_prompts):
                    break
                datasource = prompt_to_datasource[all_prompts[prompt_idx]]
                per_dataset_total_prompts[datasource] += 1
                for key, value in sample.info.items():
                    if not key.startswith("tool_count__"):
                        continue
                    tool_name = key[len("tool_count__"):]
                    tool_count = int(value.flatten()[i].item())
                    per_dataset_counts[datasource][tool_name] += tool_count
                    if tool_count >= 1:
                        per_dataset_prompts_used[datasource][tool_name] += 1
                prompt_idx += 1

        per_dataset_counts = {
            ds: dict(sorted(tool_counts.items()))
            for ds, tool_counts in sorted(per_dataset_counts.items())
            if tool_counts
        }
        per_dataset_normalized = {}
        totals = {}
        for ds, tool_counts in per_dataset_counts.items():
            total = sum(tool_counts.values())
            totals[ds] = total
            per_dataset_normalized[ds] = {tool: count / total for tool, count in tool_counts.items()}

        # Fraction of prompts (per dataset) that used each tool at least once → 100% if every prompt used it
        per_dataset_usage_pct = {}
        for ds in per_dataset_total_prompts:
            n = per_dataset_total_prompts[ds]
            if n == 0:
                continue
            per_dataset_usage_pct[ds] = {
                tool: per_dataset_prompts_used[ds].get(tool, 0) / n
                for tool in per_dataset_counts.get(ds, {})
            }
            if not per_dataset_usage_pct[ds]:
                del per_dataset_usage_pct[ds]

        return per_dataset_counts, per_dataset_normalized, totals, per_dataset_usage_pct

    def _write_eval_tool_usage(self, global_step, per_dataset_counts, per_dataset_normalized, totals, per_dataset_usage_pct):
        if not per_dataset_counts:
            return
        run_dir = self.samples_generator.runs_dir
        output_dir = os.path.join(run_dir, "tool_usage_eval")
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "global_step": global_step,
            "per_dataset_counts": per_dataset_counts,
            "per_dataset_normalized": per_dataset_normalized,
            "totals": totals,
            "per_dataset_usage_pct": per_dataset_usage_pct,
        }
        with open(os.path.join(output_dir, f"eval_step_{global_step}.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=True)
        if not hasattr(self, "_eval_tool_usage_history"):
            self._eval_tool_usage_history = []
        self._eval_tool_usage_history.append(payload)

    def _write_eval_metrics(self, global_step, global_metrics, logs, n_samples_per_prompt):
        """Write per-task accuracy and macro-F1 to a local JSON in the run folder."""
        run_dir = self.samples_generator.runs_dir
        output_dir = os.path.join(run_dir, "eval_metrics")
        os.makedirs(output_dir, exist_ok=True)

        per_task = {}
        for datasource, metrics in global_metrics.items():
            count = metrics["count"]
            accuracy = metrics["pass1"] / count if count > 0 else 0.0
            per_task[datasource] = {
                "accuracy": accuracy,
                "macro_f1": logs.get(f"eval_{datasource}_macro_f1"),
                "count": count,
            }

        payload = {
            "global_step": global_step,
            "n_samples_per_prompt": n_samples_per_prompt,
            "per_task": per_task,
            "avg_accuracy": logs.get("eval_avg_pass1"),
            "avg_macro_f1": logs.get("eval_avg_macro_f1"),
        }
        out_path = os.path.join(output_dir, f"eval_step_{global_step}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        logger.info(f"[eval_metrics] Wrote per-task metrics to {out_path}")

    def _write_final_tool_usage_plot(self):
        history = getattr(self, "_eval_tool_usage_history", [])
        if not history:
            return

        import matplotlib.pyplot as plt

        output_dir = os.path.join(self.samples_generator.runs_dir, "tool_usage_eval")
        os.makedirs(output_dir, exist_ok=True)
        # Use % of prompts that used each tool (0–1); fall back to old normalized if missing
        key = "per_dataset_usage_pct" if history[0].get("per_dataset_usage_pct") else "per_dataset_normalized"
        datasets = sorted({ds for entry in history for ds in entry.get(key, {}).keys()})
        if not datasets:
            return

        nrows = len(datasets)
        fig, axes = plt.subplots(nrows, 1, figsize=(14, max(4, 3 * nrows)), squeeze=False)
        for row, ds in enumerate(datasets):
            ax = axes[row][0]
            tools = sorted({tool for entry in history for tool in entry.get(key, {}).get(ds, {}).keys()})
            if not tools:
                ax.set_axis_off()
                continue
            matrix = [[entry.get(key, {}).get(ds, {}).get(tool, 0.0) for tool in tools] for entry in history]
            heatmap = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(f"{ds} (% prompts used tool)")
            ax.set_xlabel("tool")
            ax.set_ylabel("eval_step")
            ax.set_xticks(range(len(tools)))
            ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=8)
            y_labels = [str(entry["global_step"]) for entry in history]
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=8)
            fig.colorbar(heatmap, ax=ax, fraction=0.025, pad=0.02)

        fig.tight_layout()
        output_path = os.path.join(output_dir, "tool_usage_phase_histograms.png")
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved tool usage phase plot to {output_path}")

    @torch.no_grad()
    def evaluate(self, global_step, **generate_kwargs):
        """Evaluate model performance on eval dataset."""
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        is_tdc_eval = bool(getattr(self.args, "tdc_tools", None)) or any(
            "tdc" in str(v).lower()
            for v in [getattr(self.args, "prompt_data", None), getattr(self.args, "eval_dataset", None)]
            if v is not None
        )

        # First collect all prompts and labels
        prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources
        for _indices, datasources, prompts, labels in self.eval_dataloader:
            # Create mapping for each prompt to its corresponding data source
            for prompt, datasource in zip(prompts, datasources):
                prompt_to_datasource[prompt] = datasource

        # Generate samples and calculate rewards
        samples_list = self.samples_generator.generate_eval_samples(global_step=global_step, **generate_kwargs)

        # duplicate prompts and labels for each sample
        all_prompts = sum([s.prompts for s in samples_list], [])
        all_labels = sum([s.labels for s in samples_list], [])

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

        # Average across all datasources
        if global_metrics:
            pass1_values = [logs[f"eval_{ds}_pass1"] for ds in global_metrics]
            logs["eval_avg_pass1"] = sum(pass1_values) / len(pass1_values)
            if n_samples_per_prompt > 1:
                passk_values = [logs[f"eval_{ds}_pass{n_samples_per_prompt}"] for ds in global_metrics]
                logs[f"eval_avg_pass{n_samples_per_prompt}"] = sum(passk_values) / len(passk_values)

        # TDC-only macro-F1 (per task/datasource + average)
        if is_tdc_eval:
            labels_by_datasource = defaultdict(list)
            preds_by_datasource = defaultdict(list)
            for prompt, label, sample in zip(all_prompts, all_labels, samples_list):
                datasource = prompt_to_datasource[prompt]
                text = self.tokenizer.decode(sample.sequences[0], skip_special_tokens=False)
                labels_by_datasource[datasource].append(_extract_tdc_binary_choice(label))
                preds_by_datasource[datasource].append(_extract_tdc_binary_choice(text))

            macro_f1_values = []
            for datasource in labels_by_datasource:
                macro_f1 = _macro_f1(labels_by_datasource[datasource], preds_by_datasource[datasource])
                logs[f"eval_{datasource}_macro_f1"] = macro_f1
                macro_f1_values.append(macro_f1)
            if macro_f1_values:
                logs["eval_avg_macro_f1"] = sum(macro_f1_values) / len(macro_f1_values)

        per_dataset_counts, per_dataset_normalized, totals, per_dataset_usage_pct = self._collect_eval_tool_usage(
            all_prompts, samples_list, prompt_to_datasource
        )
        self._write_eval_tool_usage(global_step, per_dataset_counts, per_dataset_normalized, totals, per_dataset_usage_pct)
        self._write_eval_metrics(global_step, global_metrics, logs, n_samples_per_prompt)

        # Log to wandb/tensorboard
        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

        # Eval generates the entire dataset at once — free the large result set.
        del samples_list, all_prompts, all_labels, rewards
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def train_step(self, rollout_samples, global_step: int) -> Tuple[Dict, int]:
        # Strip tool_count__* keys from rollout samples before they enter the
        # training pipeline.  These per-tool counters are sparse (each sample
        # only records the tools it actually called) and are only consumed
        # during eval in _collect_eval_tool_usage.  Letting them flow into
        # concat_experiences / balance_experiences / replay buffer / actor
        # training would require every downstream consumer to handle
        # mismatched key sets across ranks — so we remove them early.
        for sample in rollout_samples:
            sample.info = {k: v for k, v in sample.info.items() if not k.startswith("tool_count__")}

        # Turn raw rollouts into PPO-ready trajectories with rewards.
        experiences = self.experience_maker.make_experience_batch(rollout_samples)

        # Periodic lightweight trace for rollout quality without full text spam.
        sample0 = [
            self.tokenizer.decode(experiences[0].sequences[0], skip_special_tokens=True),
            experiences[0].info["reward"][0].item(),
        ]
        trace_interval = int(os.environ.get("OPENRLHF_TRACE_INTERVAL", "10"))
        if global_step >= 5 and (global_step - 5) % max(trace_interval, 1) == 0:
            sample_preview = sample0[0].replace("\n", "\\n")
            logger.info(
                f"[trace] step={global_step} reward={sample0[1]:.3f} "
                f"response_len={float(experiences[0].info['response_length'][0]):.0f} "
                f"total_len={float(experiences[0].info['total_length'][0]):.0f} "
                f"preview={sample_preview!r}"
            )

        # Balance experiences across DP ranks if needed.
        if self.args.use_dynamic_batch:
            experiences = balance_experiences(experiences, self.args)

        # Push experiences to actor (and critic) shards before PPO.
        refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
        if self.critic_model_group is not None:
            refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences))
        ray.get(refs)

        # Free local experience tensors — data now lives in actor replay buffers.
        del experiences, rollout_samples, refs

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

            # Push HF checkpoint to Hub and optionally delete local copy
            if self.args.save_hf_ckpt and self.args.push_to_hub:
                hf_ckpt_path = os.path.join(self.args.ckpt_path, f"{tag}_hf")
                if os.path.exists(hf_ckpt_path):
                    # Save training config alongside checkpoint
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
        # Tokenizer is shared across the sample generator and trainer to avoid duplicated loads.
        tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer)
        self.prompts_dataloader, self.eval_dataloader, self.max_steps = prepare_datasets(strategy, tokenizer)

        # get eval and save steps
        if strategy.args.eval_steps == -1:
            strategy.args.eval_steps = float("inf")  # do not evaluate
        if strategy.args.save_steps == -1:
            strategy.args.save_steps = float("inf")  # do not save ckpt
        logger.info(f"save_steps={strategy.args.save_steps} (max_steps={self.max_steps})")
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

    def get_max_steps(self):
        return self.max_steps

    def _run_replay_episodes(self, episode: int, global_step: int, total_consumed_prompts: int) -> int:
        """After the primary episode, replay filtered prompts for up to max_replay_rounds."""
        hard_indices, kept_indices = self.samples_generator.get_replay_indices()
        replay_indices = list(hard_indices | kept_indices)
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

            # Build a dataloader over the replay subset.
            if getattr(self.args, "curriculum_balanced", False):
                replay_indices = interleave_indices_by_datasource(
                    replay_indices, original_dataloader.dataset.datasources, self.args.seed
                )
                replay_shuffle = False
            else:
                replay_shuffle = True

            subset = Subset(original_dataloader.dataset, replay_indices)
            replay_dataloader = DataLoader(
                subset, batch_size=1, shuffle=replay_shuffle,
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
                logger.info(f"✨ Global step {global_step} [replay]: {log_status}")

                client_states = {
                    "episode": episode,
                    "global_step": global_step,
                    "total_consumed_prompts": total_consumed_prompts,
                    "data_loader_state_dict": {},  # replay dataloader state is ephemeral
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

            # Collect new replay indices from this round (everything that wasn't too easy).
            hard_indices, kept_indices = self.samples_generator.get_replay_indices()
            replay_indices = list(hard_indices | kept_indices)
            logger.info(
                f"[SmartReplay] Round {replay_round + 1} done. "
                f"{len(replay_indices)} non-easy prompts remain (hard={len(hard_indices)}, kept={len(kept_indices)})."
            )

        # Restore original dataloader.
        self.samples_generator.prompts_dataloader = original_dataloader
        return global_step

    def _log_dataloader_order(self):
        """Log first/last 100 samples from the dataloader for validation."""
        import json as _json

        run_name = getattr(self.args, "wandb_run_name", "run").replace("/", "_")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, "runs", run_name, "dataloader_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Collect all samples (iterate the full dataloader once)
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

    def fit(self) -> None:
        checkpoint_states = self.init_checkpoint_states()
        # Restore step and start_epoch
        start_episode = checkpoint_states["episode"]
        global_step = checkpoint_states["global_step"]
        total_consumed_prompts = checkpoint_states["total_consumed_prompts"]
        # Keep vLLM weights and dataloader states in sync when resuming.
        if global_step:
            self.broadcast_to_vllm()
            state_dict = checkpoint_states["data_loader_state_dict"]
            if state_dict:
                self.prompts_dataloader.load_state_dict(state_dict)

        # Log dataloader ordering for validation before training begins.
        self._log_dataloader_order()

        # Evaluate at step 0 (before any training) unless resuming from a checkpoint.
        if global_step == 0 and self.eval_dataloader and not self.args.skip_eval_step_zero:
            eval_generate_kwargs = self.generate_kwargs.copy()
            eval_generate_kwargs["temperature"] = self.args.eval_temperature
            eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
            self.evaluate(global_step, **eval_generate_kwargs)

        for episode in range(start_episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                # Draw one mini-batch of prompts; stop when loader is exhausted.
                log_step_trace = global_step % 2 == 0
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(
                        global_step=global_step, log_step_trace=log_step_trace, **self.generate_kwargs
                    )
                )
                total_consumed_prompts += prompts_consumed
                if is_exhausted:
                    break

                # Run PPO update on this batch and bump the global step counter.
                status, global_step = self.train_step(rollout_samples, global_step)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = filter_pass_rate
                    status["too_easy_pct"] = self.samples_generator.step_too_easy_pct
                    status["too_hard_pct"] = self.samples_generator.step_too_hard_pct
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

                # Free accumulated Ray object store refs and Python garbage to
                # prevent host-RAM growth across training steps.
                del rollout_samples, status
                gc.collect()

            # --- Save discarded prompts for offline analysis ---
            self.samples_generator.save_discarded_indices(episode)

            # --- Smart replay: log episode stats and run replay episodes ---
            if getattr(self.args, "smart_replay", False):
                if self.wandb_logger:
                    self.wandb_logger.log_episode(episode, self.samples_generator.episode_filter_stats)
                global_step = self._run_replay_episodes(episode, global_step, total_consumed_prompts)

        # Final eval at end of training (skip if last step already ran eval)
        if self.eval_dataloader and (global_step % self.args.eval_steps != 0):
            eval_generate_kwargs = self.generate_kwargs.copy()
            eval_generate_kwargs["temperature"] = self.args.eval_temperature
            eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
            logger.info(f"Running final evaluation at global_step {global_step}")
            self.evaluate(global_step, **eval_generate_kwargs)

        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        # Close trackers
        self._write_final_tool_usage_plot()
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
