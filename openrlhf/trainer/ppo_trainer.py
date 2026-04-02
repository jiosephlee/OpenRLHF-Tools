import ctypes
import gc
import json
import os
import time
from abc import ABC
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import ray
import torch
import transformers
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

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

        # Monotonic counter for episode/round wandb x-axis.
        # Each initial episode pass and each replay round gets its own tick.
        self._round_counter = 0

        # Tracking backends
        self.wandb_logger = WandbLogger(self.args) if self.args.use_wandb else None
        self.tensorboard_logger = TensorboardLogger(self.args) if self.args.use_tensorboard else None

    def _get_response_text(self, seq, action_mask=None) -> str:
        """Helper to decode a sequence and strip the prompt and special/pad tokens.

        If *action_mask* is provided (shape ``(S-1,)`` aligned with ``seq[1:]``),
        the first action token position is used to split prompt from response —
        much more reliable than hunting for protocol-specific string markers.
        """
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        end_idx = len(seq)
        while end_idx > 0 and seq[end_idx - 1].item() in (pad_id, eos_id, 0):
            end_idx -= 1

        # ── Fast path: use action_mask to locate response start ──
        if action_mask is not None:
            ones = torch.where(action_mask)[0]
            if len(ones) > 0:
                # action_mask[k] corresponds to seq[k+1]
                resp_start = ones[0].item() + 1
                return self.tokenizer.decode(seq[resp_start:end_idx], skip_special_tokens=True)

        # ── Fallback: string-marker splitting ──
        protocol = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "")
        if protocol == "gpt_oss":
            _split_marker = "<|start|>assistant"
        elif protocol in ("intern_s1", "qwen3"):
            _split_marker = "<|im_start|>assistant"
        else:
            _split_marker = None

        full_text = self.tokenizer.decode(seq[:end_idx], skip_special_tokens=False)
        if _split_marker and _split_marker in full_text:
            return full_text[full_text.index(_split_marker):]
        return self.tokenizer.decode(seq[:end_idx], skip_special_tokens=True)

    def fit(self, global_step: int = 0) -> None:
        raise NotImplementedError("fit method is not implemented")

    def _collect_eval_tool_usage(self, all_prompts, samples_list, prompt_to_datasource):
        per_dataset_counts = defaultdict(lambda: defaultdict(int))
        per_dataset_prompts_used = defaultdict(lambda: defaultdict(int))  # prompts that used tool at least once
        per_dataset_total_prompts = defaultdict(int)
        per_dataset_parse_stats = defaultdict(lambda: defaultdict(int))
        prompt_idx = 0
        for sample in samples_list:
            batch_size = len(sample.sequences)
            for i in range(batch_size):
                if prompt_idx >= len(all_prompts):
                    break
                datasource = prompt_to_datasource[all_prompts[prompt_idx]]
                per_dataset_total_prompts[datasource] += 1
                for key, value in sample.info.items():
                    if key.startswith("tool_count__"):
                        tool_name = key[len("tool_count__") :]
                        tool_count = int(value.flatten()[i].item())
                        per_dataset_counts[datasource][tool_name] += tool_count
                        if tool_count >= 1:
                            per_dataset_prompts_used[datasource][tool_name] += 1
                    elif key.startswith("parse_method__") or key in [
                        "parse_failed",
                        "tool_call_attempted",
                        "tool_call_count",
                    ]:
                        count = int(value.flatten()[i].item())
                        per_dataset_parse_stats[datasource][key] += count
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
                tool: per_dataset_prompts_used[ds].get(tool, 0) / n for tool in per_dataset_counts.get(ds, {})
            }
            if not per_dataset_usage_pct[ds]:
                del per_dataset_usage_pct[ds]

        # Convert parse stats from defaultdict to dict
        per_dataset_parse_stats = {k: dict(v) for k, v in per_dataset_parse_stats.items()}

        return per_dataset_counts, per_dataset_normalized, totals, per_dataset_usage_pct, per_dataset_parse_stats

    def _write_eval_tool_usage(
        self,
        global_step,
        per_dataset_counts,
        per_dataset_normalized,
        totals,
        per_dataset_usage_pct,
        per_dataset_parse_stats,
    ):
        if not per_dataset_counts and not per_dataset_parse_stats:
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
            "per_dataset_parse_stats": per_dataset_parse_stats,
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
        prompt_to_knn_pl = {}      # prompt -> KNN pseudo-label (or None)
        for batch in self.eval_dataloader:
            # Support both 4-tuple (legacy) and 5-tuple (with knn_pseudo_labels)
            if len(batch) == 5:
                _indices, datasources, prompts, labels, knn_pls = batch
            else:
                _indices, datasources, prompts, labels = batch
                knn_pls = [None] * len(prompts)
            for prompt, datasource, knn_pl in zip(prompts, datasources, knn_pls):
                prompt_to_datasource[prompt] = datasource
                prompt_to_knn_pl[prompt] = knn_pl

        # Generate samples and calculate rewards
        samples_list = self.samples_generator.generate_eval_samples(global_step=global_step, **generate_kwargs)

        # duplicate prompts and labels for each sample
        all_prompts = sum([s.prompts for s in samples_list], [])
        all_labels = sum([s.labels for s in samples_list], [])

        n_samples_per_prompt = generate_kwargs["n_samples_per_prompt"]

        # Get rewards from samples, such as agent rewards or remote reward models
        rewards_list = []
        scores_list = []
        for samples in samples_list:
            rewards_list.append(samples.rewards)
            # scores is the pure correctness signal (0/1), without format bonuses
            scores_list.append(samples.scores.item() if samples.scores is not None else samples.rewards)
        # Reshape to (num_prompts, n_samples_per_prompt)
        rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)
        scores = torch.tensor(scores_list).reshape(-1, n_samples_per_prompt)

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

            # Use scores (pure correctness 0/1) for accuracy, not rewards (which include format bonuses)
            chunk_scores = scores[i]

            # Calculate pass@k and pass@1
            if n_samples_per_prompt > 1:
                global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_scores.max().float().item()
            global_metrics[datasource]["pass1"] += chunk_scores.mean().float().item()
            global_metrics[datasource]["count"] += 1

        # Collect response lengths for correct/incorrect samples.
        correct_lengths = []
        incorrect_lengths = []
        for i in range(num_prompts):
            for j in range(n_samples_per_prompt):
                exp = samples_list[i * n_samples_per_prompt + j]
                resp_len = int(exp.action_mask.sum().item()) if exp.action_mask is not None else 0
                if scores[i][j].item() > 0:
                    correct_lengths.append(resp_len)
                else:
                    incorrect_lengths.append(resp_len)

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

        # Log response length metrics (correct vs incorrect).
        if correct_lengths:
            logs["eval_avg_length_correct"] = sum(correct_lengths) / len(correct_lengths)
        if incorrect_lengths:
            logs["eval_avg_length_incorrect"] = sum(incorrect_lengths) / len(incorrect_lengths)

        # TDC-only macro-F1 (per task/datasource + average)
        if is_tdc_eval:
            labels_by_datasource = defaultdict(list)
            preds_by_datasource = defaultdict(list)
            for prompt, label, sample in zip(all_prompts, all_labels, samples_list):
                datasource = prompt_to_datasource[prompt]
                text = self._get_response_text(sample.sequences[0])
                try:
                    pred = _extract_tdc_binary_choice(text)
                except ValueError:
                    pred = "UNPARSEABLE"
                    logger.warning(f"[eval] Unparseable prediction for {datasource}: {text[:200]!r}")
                labels_by_datasource[datasource].append(_extract_tdc_binary_choice(label))
                preds_by_datasource[datasource].append(pred)

            macro_f1_values = []
            for datasource in labels_by_datasource:
                macro_f1 = _macro_f1(labels_by_datasource[datasource], preds_by_datasource[datasource])
                logs[f"eval_{datasource}_macro_f1"] = macro_f1
                macro_f1_values.append(macro_f1)
            if macro_f1_values:
                logs["eval_avg_macro_f1"] = sum(macro_f1_values) / len(macro_f1_values)

        per_dataset_counts, per_dataset_normalized, totals, per_dataset_usage_pct, per_dataset_parse_stats = (
            self._collect_eval_tool_usage(all_prompts, samples_list, prompt_to_datasource)
        )
        self._write_eval_tool_usage(
            global_step,
            per_dataset_counts,
            per_dataset_normalized,
            totals,
            per_dataset_usage_pct,
            per_dataset_parse_stats,
        )
        self._write_eval_metrics(global_step, global_metrics, logs, n_samples_per_prompt)

        #### KNN eval metrics ####
        knn_eval_total = 0
        knn_eval_reversed = 0
        knn_eval_correct_reversal = 0
        knn_eval_incorrect_reversal = 0
        for i in range(num_prompts):
            original_prompt = all_prompts[i * n_samples_per_prompt]
            knn_pl = prompt_to_knn_pl.get(original_prompt)
            if knn_pl is None:
                continue
            knn_eval_total += 1
            true_answer = all_labels[i * n_samples_per_prompt]
            chunk_scores = scores[i]
            model_correct = chunk_scores.max().item() > 0
            knn_agrees_with_truth = knn_pl in str(true_answer)
            reversed_knn = model_correct != knn_agrees_with_truth
            if reversed_knn:
                knn_eval_reversed += 1
                if model_correct:
                    knn_eval_correct_reversal += 1
                else:
                    knn_eval_incorrect_reversal += 1
        if knn_eval_total > 0:
            logs["knn_eval_reversal_pct"] = knn_eval_reversed / knn_eval_total * 100
            logs["knn_eval_correct_reversal_pct"] = knn_eval_correct_reversal / knn_eval_total * 100
            logs["knn_eval_incorrect_reversal_pct"] = knn_eval_incorrect_reversal / knn_eval_total * 100
            logs["knn_eval_total"] = knn_eval_total
        #### end KNN eval metrics ####

        # Log to wandb/tensorboard
        if self.wandb_logger:
            self.wandb_logger.log_eval(global_step, logs)
        if self.tensorboard_logger:
            self.tensorboard_logger.log_eval(global_step, logs)

        #### Eval sample saving (Phase 12) ####
        try:
            eval_traces_dir = getattr(self.samples_generator, "eval_traces_dir", None)
            if eval_traces_dir and self.strategy.is_rank_0():
                correct_samples = {}  # datasource -> sample dict
                wrong_samples = {}    # datasource -> sample dict

                for i in range(num_prompts):
                    original_prompt = all_prompts[i * n_samples_per_prompt]
                    datasource = prompt_to_datasource.get(original_prompt, "unknown")
                    chunk_scores = scores[i]
                    chunk_rewards = rewards[i]

                    best_idx = chunk_scores.argmax().item()
                    worst_idx = chunk_scores.argmin().item()

                    # Correct sample: highest score > 0, one per datasource
                    if chunk_scores[best_idx].item() > 0 and datasource not in correct_samples:
                        exp = samples_list[i * n_samples_per_prompt + best_idx]
                        correct_samples[datasource] = {
                            "prompt": original_prompt,
                            "response": self._get_response_text(exp.sequences[0], exp.action_mask[0] if exp.action_mask is not None else None),
                            "reward": chunk_rewards[best_idx].item(),
                        }

                    # Wrong sample: prefer truly wrong (score <= 0), otherwise keep lowest-score
                    worst_score = chunk_scores[worst_idx].item()
                    if datasource not in wrong_samples or (
                        worst_score <= 0 and wrong_samples[datasource].get("score", 1) > 0
                    ):
                        exp = samples_list[i * n_samples_per_prompt + worst_idx]
                        wrong_samples[datasource] = {
                            "prompt": original_prompt,
                            "response": self._get_response_text(exp.sequences[0], exp.action_mask[0] if exp.action_mask is not None else None),
                            "reward": chunk_rewards[worst_idx].item(),
                            "score": worst_score,
                        }

                eval_trace = {"global_step": global_step, "datasources": {}}
                for ds in set(list(correct_samples) + list(wrong_samples)):
                    eval_trace["datasources"][ds] = {
                        "correct": correct_samples.get(ds),
                        "wrong": wrong_samples.get(ds),
                    }

                trace_path = os.path.join(eval_traces_dir, f"eval_step_{global_step}.json")
                with open(trace_path, "w") as f:
                    json.dump(eval_trace, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Failed to save eval traces: {e}")
        #### end eval sample saving ####

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

        self._empty_all_model_caches()

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


        ###### Forward Pass For Initial Log Probs ######
        forward_start_time = time.time()
    
        # Turn raw rollouts into PPO-ready trajectories with rewards.
        experiences = self.experience_maker.make_experience_batch(rollout_samples)

        time_forward_pass = time.time() - forward_start_time
        status = {}
        status["time/forward_pass"] = time_forward_pass
        ###### End of Forward Pass ######


        # Periodic lightweight trace for rollout quality without full text spam.
        _decode = self.tokenizer.decode if _TRANSFORMERS_V5 else self.tokenizer.batch_decode
        sample0 = [
            _decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)[0],
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

        # Log interesting reward group trajectories every 4 steps.
        if global_step % 4 == 0:
            self._log_interesting_groups(experiences, global_step)

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

        ###### Beginning of Backwards Pass ######

        # Perform PPO optimization for actor/critic and gather metrics.
        backward_start_time = time.time()
        ppo_status = self.ppo_train(global_step)
        time_backward_pass = time.time() - backward_start_time

        # Compute policy training throughput (mirrors nemo-rl's policy_training_tokens_per_sec_per_gpu).
        # total_trained_tokens is a raw sum (not averaged) injected by ppo_actor.ppo_train().
        # We pop it here so it doesn't flow into W&B as a raw float average.
        total_trained_tokens = ppo_status.pop("total_trained_tokens", None)
        train_time = time_backward_pass
        num_training_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        if total_trained_tokens and train_time > 0:
            status["policy/total_trained_tokens"] = total_trained_tokens
            status["policy/training_tokens_per_sec_per_gpu"] = round(
                total_trained_tokens / train_time / num_training_gpus, 1
            )

        status.update(ppo_status)
        status["time/backward_pass"] = time_backward_pass

        ###### End of Backwards Pass ######


        ###### Sync Weights ######
        # Flush PyTorch's caching allocator on Actor/Critic workers so that
        # freed GPU memory is returned to the driver.  In colocate mode the
        # Actor and vLLM share the same physical GPUs — without this flush
        # vLLM's cumem allocator cannot reclaim memory that PyTorch still
        # holds in its cache, causing OOM when waking KV cache later.
        if self.vllm_engines is not None and self.args.vllm_enable_sleep:
            self._empty_all_model_caches()

        # Sync weights to vLLM.
        sync_start_time = time.time()
        if self.vllm_engines is not None:
            self.broadcast_to_vllm()
        time_sync_weights = time.time() - sync_start_time
        status["time/sync_weights"] = time_sync_weights
        status["time/sync_wake_weights"] = getattr(self, "_last_wake_weights_sec", 0.0)

        # Surface vLLM sleep/wake/gc timings from the last rollout.
        if hasattr(self, "samples_generator"):
            status["time/vllm_wake"] = getattr(self.samples_generator, "_last_vllm_wake_sec", 0.0)
            status["time/vllm_sleep"] = getattr(self.samples_generator, "_last_vllm_sleep_sec", 0.0)
            status["time/vllm_gc_collect"] = getattr(self.samples_generator, "_last_vllm_gc_collect_sec", 0.0)

        ###### End of Sync Weights ######

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

        def _run_sleep(group, group_name, **kwargs):
            # Sleep mode: reload -> fit -> offload (smaller GPU memory).
            _reload_start = time.time()
            ray.get(group.async_run_method(method_name="reload_states"))
            _reload_sec = time.time() - _reload_start

            ref = group.async_run_method(method_name="fit", **kwargs)
            status.update(ray.get(ref)[0])

            _offload_start = time.time()
            ray.get(group.async_run_method(method_name="offload_states"))
            _offload_sec = time.time() - _offload_start

            status[f"time/ds_{group_name}_reload_states"] = _reload_sec
            status[f"time/ds_{group_name}_offload_states"] = _offload_sec

        if self.args.deepspeed_enable_sleep:
            # Colocated/sleeping: run critic first, then actor.
            if run_critic:
                _run_sleep(self.critic_model_group, "critic")
            if run_actor:
                _run_sleep(self.actor_model_group, "actor", kl_ctl=self.kl_ctl.value)
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
        _wake_weights_start = time.time()
        if self.args.vllm_enable_sleep:
            # Wake up only weights for weight sync (not KV cache)
            # This avoids allocating KV cache memory during weight update
            batch_vllm_engine_call(self.vllm_engines, "wake_up", tags=["weights"])
        self._last_wake_weights_sec = time.time() - _wake_weights_start

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        # NOTE: We keep vLLM in weights-only state after weight sync.
        # KV cache will be woken up before generation in SamplesGenerator.

    def _log_interesting_groups(self, experiences, global_step: int) -> None:
        """Find and log decoded trajectories for 'needle' and 'mixed' reward groups."""
        try:
            trace_dir = getattr(self.samples_generator, "rollout_trace_run_dir", None)
            if not trace_dir:
                return

            n_samples = self.args.n_samples_per_prompt
            if n_samples < 2:
                return

            # Collect rewards and indices across all experience shards, sort into prompt order.
            indices = torch.tensor(sum([exp.index for exp in experiences], []))
            raw_rewards = torch.cat([exp.rewards for exp in experiences], dim=0)
            rewards = torch.empty_like(raw_rewards)
            rewards[indices] = raw_rewards

            # Also collect sequences, action_masks, prompts, labels in the same sorted order.
            # Pad sequences and action_masks to the same length before concatenating (shards may differ).
            max_seq_len = max(exp.sequences.size(1) for exp in experiences)
            padded = []
            padded_masks = []
            for exp in experiences:
                seq = exp.sequences
                if seq.size(1) < max_seq_len:
                    seq = torch.nn.functional.pad(seq, (0, max_seq_len - seq.size(1)), value=self.tokenizer.pad_token_id or 0)
                padded.append(seq)
                if exp.action_mask is not None:
                    am = exp.action_mask
                    # action_mask is (B, S-1); pad to max_seq_len - 1
                    target_len = max_seq_len - 1
                    if am.size(1) < target_len:
                        am = torch.nn.functional.pad(am, (0, target_len - am.size(1)), value=0)
                    padded_masks.append(am)
            all_sequences = torch.cat(padded, dim=0)
            sequences = torch.empty_like(all_sequences)
            sequences[indices] = all_sequences

            # Sort action_masks into prompt order (if available).
            action_masks = None
            if padded_masks and len(padded_masks) == len(padded):
                all_action_masks = torch.cat(padded_masks, dim=0)
                action_masks = torch.empty_like(all_action_masks)
                action_masks[indices] = all_action_masks

            all_prompts = sum([exp.prompts for exp in experiences], [])
            all_labels = sum([exp.labels for exp in experiences], [])
            sorted_prompts = [""] * len(all_prompts)
            sorted_labels = [""] * len(all_labels)
            for i, idx in enumerate(indices.tolist()):
                sorted_prompts[idx] = all_prompts[i]
                sorted_labels[idx] = all_labels[i]

            num_prompts = len(rewards) // n_samples
            if num_prompts == 0:
                return

            reward_groups = rewards[: num_prompts * n_samples].view(num_prompts, n_samples)

            found = {}  # type -> group_index
            for gi in range(num_prompts):
                group = reward_groups[gi]
                high_mask = group > 0.5
                n_high = high_mask.sum().item()
                n_total = n_samples

                if "needle" not in found and n_high == 1 and (n_total - n_high) >= 1:
                    found["needle"] = gi
                if "mixed" not in found:
                    frac = n_high / n_total
                    if 0.4 <= frac <= 0.6:
                        found["mixed"] = gi

                if len(found) == 2:
                    break

            if not found:
                return

            for gtype, gi in found.items():
                start = gi * n_samples
                end = start + n_samples
                group_rewards = rewards[start:end].tolist()
                group_prompt = sorted_prompts[start] if start < len(sorted_prompts) else ""
                group_label = sorted_labels[start] if start < len(sorted_labels) else ""
                samples = []
                for si in range(start, end):
                    seq = sequences[si]
                    am = action_masks[si] if action_masks is not None else None
                    response_text = self._get_response_text(seq, am)
                    samples.append({
                        "reward": group_rewards[si - start],
                        "response_text": response_text,
                    })

                record = {
                    "step": global_step,
                    "type": gtype,
                    "group_rewards": group_rewards,
                    "prompt": group_prompt,
                    "label": group_label,
                    "samples": samples,
                }
                trace_path = os.path.join(trace_dir, f"group_trace_step{global_step}_{gtype}.json")
                with open(trace_path, "w") as f:
                    json.dump(record, f, ensure_ascii=True, indent=2)
                logger.info(f"[group_trace] step={global_step} type={gtype} rewards={group_rewards} -> {trace_path}")

        except Exception as e:
            logger.warning(f"[group_trace] Failed to log interesting groups at step {global_step}: {e}")

    def _empty_all_model_caches(self) -> None:
        """Force PyTorch caching allocator to release memory back to CUDA/OS
        before the next vLLM wake_up cycle. gc.collect() on the controller
        triggers destruction of tensors on PyTorch workers, putting memory
        into the PyTorch cache but not the OS natively.
        """
        refs = self.actor_model_group.async_run_method(method_name="empty_cache")
        if getattr(self, "critic_model_group", None) is not None:
            refs.extend(self.critic_model_group.async_run_method(method_name="empty_cache"))
        if getattr(self, "reference_model_group", None) is not None:
            refs.extend(self.reference_model_group.async_run_method(method_name="empty_cache"))
        if getattr(self, "reward_model_group", None) is not None:
            refs.extend(self.reward_model_group.async_run_method(method_name="empty_cache"))
        ray.get(refs)

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

    #### Oversampling: LeftOverPrompts phase ####
    def _run_leftover_phase(self, episode: int, global_step: int, total_consumed_prompts: int) -> int:
        """Dispatch remaining missed_indices in two passes.

        Phase 1: mild oversampling (default 1.5×) with dynamic filtering still active.
        Phase 2: no oversampling (1.0×) to sweep up whatever Phase 1 left behind.
        """
        missed_indices = self.samples_generator.get_missed_indices()
        if len(missed_indices) < self.args.rollout_batch_size:
            if missed_indices:
                logger.info(
                    f"[LeftOverPrompts] {len(missed_indices)} missed indices "
                    f"(< batch_size={self.args.rollout_batch_size}), deferring to smart replay"
                )
            return global_step

        saved_dataloader = self.samples_generator.prompts_dataloader

        # --- Phase 1: mild oversampling ---
        leftover_oversample = getattr(self.args, "leftover_oversample_ratio", 1.5)
        logger.info(
            f"[LeftOverPrompts] Phase 1: processing {len(missed_indices)} missed indices "
            f"(oversample_ratio={leftover_oversample})"
        )
        global_step = self._run_leftover_pass(
            episode, global_step, total_consumed_prompts,
            missed_indices, oversample_ratio=leftover_oversample, phase_tag="leftover-p1",
        )

        # --- Phase 2: no oversampling, sweep remaining ---
        missed_indices = self.samples_generator.get_missed_indices()
        if len(missed_indices) >= self.args.rollout_batch_size:
            logger.info(
                f"[LeftOverPrompts] Phase 2: processing {len(missed_indices)} remaining missed indices "
                f"(oversample_ratio=1.0)"
            )
            global_step = self._run_leftover_pass(
                episode, global_step, total_consumed_prompts,
                missed_indices, oversample_ratio=1.0, phase_tag="leftover-p2",
            )
        elif missed_indices:
            logger.info(
                f"[LeftOverPrompts] Phase 2: {len(missed_indices)} remaining "
                f"(< batch_size={self.args.rollout_batch_size}), deferring to smart replay"
            )

        # Restore original dataloader.
        self.samples_generator.prompts_dataloader = saved_dataloader
        # Any new missed_indices from this phase stay for smart replay.
        return global_step

    def _run_leftover_pass(
        self, episode: int, global_step: int, total_consumed_prompts: int,
        missed_indices: set, oversample_ratio: float, phase_tag: str,
    ) -> int:
        """Run a single leftover pass over the given missed indices."""
        original_dataset = self.samples_generator._original_dataset
        subset = Subset(original_dataset, list(missed_indices))
        leftover_dataloader = DataLoader(
            subset, batch_size=1, shuffle=False, collate_fn=original_dataset.collate_fn
        )

        # Temporarily swap dataloader; clear consumed missed indices.
        self.samples_generator.prompts_dataloader = leftover_dataloader
        self.samples_generator._missed_indices = set()

        while True:
            log_step_trace = global_step % 2 == 0
            rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                self.samples_generator.generate_samples(
                    global_step=global_step, log_step_trace=log_step_trace,
                    oversample_ratio=oversample_ratio,
                    _skip_clear_replay=True,  # preserve replay indices
                    **self.generate_kwargs,
                )
            )
            total_consumed_prompts += prompts_consumed

            if is_exhausted:
                if rollout_samples:
                    status, global_step = self.train_step(rollout_samples, global_step)
                    log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                    logger.info(f"✨ Global step {global_step} [{phase_tag}-partial]: {log_status}")
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
            status[f"leftover/phase"] = 1 if "p1" in phase_tag else 2

            log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
            logger.info(f"✨ Global step {global_step} [{phase_tag}]: {log_status}")

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
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            self._empty_all_model_caches()
            self.samples_generator.flush_timeseries_to_disk(global_step=global_step)

        return global_step
    #### end oversampling ####

    def _run_replay_episodes(self, episode: int, global_step: int, total_consumed_prompts: int) -> int:
        """After the primary episode, replay filtered prompts for up to max_replay_rounds."""
        hard_indices, kept_indices = self.samples_generator.get_replay_indices()
        #### Oversampling: include missed indices in replay pool ####
        missed_indices = self.samples_generator.get_missed_indices()
        replay_indices = list(hard_indices | missed_indices)
        #### end oversampling ####
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
                f"replaying {len(replay_indices)} prompts (hard={len(hard_indices)}, kept={len(kept_indices)} tracked)"
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
                subset,
                batch_size=1,
                shuffle=replay_shuffle,
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
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

                self._empty_all_model_caches()

                # Flush accumulated timeseries samples from Ray actor memory to disk.
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

            #### Oversampling: run leftover phase after each replay round ####
            if getattr(self.args, "oversample_ratio", 1.0) > 1.0:
                global_step = self._run_leftover_phase(episode, global_step, total_consumed_prompts)
            #### end oversampling ####

            # Eval at end of replay round (skip if last step already ran eval).
            if self.eval_dataloader and (global_step % self.args.eval_steps != 0):
                eval_generate_kwargs = self.generate_kwargs.copy()
                eval_generate_kwargs["temperature"] = self.args.eval_temperature
                eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                logger.info(f"Running end-of-replay-round evaluation at global_step {global_step}")
                self.evaluate(global_step, **eval_generate_kwargs)

            # Collect new replay indices from this round (everything that wasn't too easy).
            hard_indices, kept_indices = self.samples_generator.get_replay_indices()
            #### Oversampling: include missed indices in next replay round ####
            missed_indices = self.samples_generator.get_missed_indices()
            replay_indices = list(hard_indices | missed_indices)
            #### end oversampling ####
            logger.info(
                f"[SmartReplay] Round {replay_round + 1} done. "
                f"{len(replay_indices)} replay prompts remain "
                f"(hard={len(hard_indices)}, kept={len(kept_indices)} tracked, missed={len(missed_indices)})."
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
        for batch in self.prompts_dataloader:
            _indices, datasources, prompts = batch[0], batch[1], batch[2]
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
                f.write(
                    _json.dumps(
                        {
                            "index": i,
                            "datasource": ds,
                            "prompt_tail": prompt[-100:] if len(prompt) > 100 else prompt,
                        }
                    )
                    + "\n"
                )
            if n > head + tail:
                f.write(_json.dumps({"gap": f"...skipped indices {head} to {n - tail - 1}..."}) + "\n")
            for i in range(max(head, n - tail), n):
                ds, prompt = all_samples[i]
                f.write(
                    _json.dumps(
                        {
                            "index": i,
                            "datasource": ds,
                            "prompt_tail": prompt[-100:] if len(prompt) > 100 else prompt,
                        }
                    )
                    + "\n"
                )

        logger.info(f"[DataloaderLog] Wrote {n} sample summary to {log_path}")

        if n > 0:
            sample_prompt_path = os.path.join(log_dir, "sample0_prompt.txt")
            with open(sample_prompt_path, "w", encoding="utf-8") as f:
                f.write(all_samples[0][1])
            logger.info(f"[DataloaderLog] Wrote full system prompt of sample 0 to {sample_prompt_path}")

    def _get_vllm_stats_dir(self):
        """Return the vllm_stats directory path (lazily created)."""
        stats_dir = getattr(self.samples_generator, "vllm_stats_dir", None)
        if stats_dir is None:
            run_name = getattr(self.args, "wandb_run_name", "run").replace("/", "_")
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            stats_dir = os.path.join(project_root, "runs", run_name, "vllm_stats")
            os.makedirs(stats_dir, exist_ok=True)
        return stats_dir

    def _write_run_timing(
        self,
        global_step: int,
        rollout_wall_sec: float,
        gap_rollout_to_train: float,
        train_wall_sec: float,
        time_forward_pass: float,
        time_backward_pass: float,
        time_sync_weights: float,
        gap_train_to_rollout: float,
        eval_wall_sec: Optional[float] = None,
        extra_timings: Optional[Dict[str, float]] = None,
    ):
        """Append one line to run_timing.jsonl."""
        record = {
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "rollout_wall_sec": round(rollout_wall_sec, 2),
            "gap_rollout_to_train": round(gap_rollout_to_train, 2),
            "train_wall_sec": round(train_wall_sec, 2),
            "time_forward_pass": round(time_forward_pass, 2),
            "time_backward_pass": round(time_backward_pass, 2),
            "time_sync_weights": round(time_sync_weights, 2),
            "gap_train_to_rollout": round(gap_train_to_rollout, 2),
        }
        if eval_wall_sec is not None:
             record["eval_wall_sec"] = round(eval_wall_sec, 2)
        if extra_timings:
            for k, v in extra_timings.items():
                record[k] = round(v, 2)
             
        timing_path = os.path.join(self._get_vllm_stats_dir(), "run_timing.jsonl")
        try:
            with open(timing_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write run timing: {e}")

    def _write_run_summary(self, total_steps: int):
        """Read run_timing.jsonl and write run_summary.json with averages."""
        timing_path = os.path.join(self._get_vllm_stats_dir(), "run_timing.jsonl")
        if not os.path.exists(timing_path):
            return

        try:
            records = []
            with open(timing_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            if not records:
                return

            n = len(records)
            evals = [r["eval_wall_sec"] for r in records if "eval_wall_sec" in r]
            summary = {
                "total_steps": n,
                "avg_rollout_wall_sec": round(sum(r["rollout_wall_sec"] for r in records) / n, 2),
                "avg_gap_rollout_to_train": round(sum(r["gap_rollout_to_train"] for r in records) / n, 2),
                "avg_train_wall_sec": round(sum(r["train_wall_sec"] for r in records) / n, 2),
                "avg_time_forward_pass": round(sum(r["time_forward_pass"] for r in records) / n, 2),
                "avg_time_backward_pass": round(sum(r["time_backward_pass"] for r in records) / n, 2),
                "avg_time_sync_weights": round(sum(r["time_sync_weights"] for r in records) / n, 2),
                "avg_gap_train_to_rollout": round(sum(r["gap_train_to_rollout"] for r in records) / n, 2),
            }
            if evals:
                 summary["avg_eval_wall_sec"] = round(sum(evals) / len(evals), 2)

            # Read rollout_stats.jsonl for KV cache averages if available.
            rollout_path = os.path.join(self._get_vllm_stats_dir(), "rollout_stats.jsonl")
            if os.path.exists(rollout_path):
                rollout_records = []
                with open(rollout_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rollout_records.append(json.loads(line))
                if rollout_records:
                    kv_means = [r["kv_cache_usage_pct"]["mean"] for r in rollout_records if "kv_cache_usage_pct" in r]
                    running_means = [r["num_running_reqs"]["mean"] for r in rollout_records if "num_running_reqs" in r]
                    if kv_means:
                        summary["vllm_avg_kv_cache_usage_pct"] = round(sum(kv_means) / len(kv_means), 4)
                    if running_means:
                        summary["vllm_avg_num_running_reqs"] = round(sum(running_means) / len(running_means), 2)

            summary_path = os.path.join(self._get_vllm_stats_dir(), "run_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Wrote run summary to {summary_path}")

        except Exception as e:
            logger.warning(f"Failed to write run summary: {e}")

    def _write_scheduler_timeseries_plot(self):
        """Generate a 4-panel matplotlib figure from scheduler_timeseries.jsonl.

        X-axis: equally-spaced global steps.  Within each step, samples are
        spread across the step's x-range proportionally by their relative
        timestamp.  Per-engine lines are overlaid, and each step gets a
        distinct alternating background shade for visual grouping.
        """
        timeseries_path = os.path.join(self._get_vllm_stats_dir(), "scheduler_timeseries.jsonl")
        if not os.path.exists(timeseries_path):
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            records = []
            with open(timeseries_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            if not records:
                return

            # Group by global_step, sorted.
            from collections import OrderedDict

            step_groups = OrderedDict()
            for r in records:
                gs = r["global_step"]
                step_groups.setdefault(gs, []).append(r)

            sorted_steps = sorted(step_groups.keys())
            engines = sorted(set(r["engine"] for r in records))

            # Assign each step an equal-width x-range: [i, i+1).
            # Within each step, samples are placed proportionally by their
            # timestamp offset within the step's time span.
            for r in records:
                gs = r["global_step"]
                step_idx = sorted_steps.index(gs)
                group = step_groups[gs]
                t_min = min(s["t"] for s in group)
                t_max = max(s["t"] for s in group)
                t_span = t_max - t_min if t_max > t_min else 1.0
                r["x"] = step_idx + (r["t"] - t_min) / t_span * 0.9  # leave small gap

            fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
            titles = ["KV Cache Usage %", "Running Requests", "Waiting Requests", "Prefix Cache Hit Rate"]
            keys = ["kv_cache_usage", "num_running", "num_waiting", "prefix_cache_hit_rate"]

            # Alternating background shading per step.
            for ax in axes:
                for i, gs in enumerate(sorted_steps):
                    color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
                    ax.axvspan(i, i + 1, facecolor=color, alpha=0.5)

            for ax, title, key in zip(axes, titles, keys):
                for eng in engines:
                    eng_records = sorted([r for r in records if r["engine"] == eng], key=lambda r: r["x"])
                    xs = [r["x"] for r in eng_records]
                    vals = [r[key] for r in eng_records]
                    ax.plot(xs, vals, label=f"Engine {eng}", alpha=0.7, linewidth=0.8, marker=".", markersize=2)
                ax.set_title(title, fontsize=11)
                ax.set_ylabel(title)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")

            # X-axis: show step labels at the center of each step range.
            tick_positions = [i + 0.45 for i in range(len(sorted_steps))]
            tick_labels = [str(gs) for gs in sorted_steps]
            # Show at most 30 labels to avoid crowding.
            if len(tick_labels) > 30:
                step_size = max(1, len(tick_labels) // 30)
                tick_positions = tick_positions[::step_size]
                tick_labels = tick_labels[::step_size]
            axes[-1].set_xticks(tick_positions)
            axes[-1].set_xticklabels(tick_labels, fontsize=8, rotation=45)
            axes[-1].set_xlabel("Global Step")

            fig.suptitle("vLLM Scheduler Stats per Step", fontsize=13)
            fig.tight_layout()

            plot_path = os.path.join(self._get_vllm_stats_dir(), "scheduler_timeseries.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"Wrote scheduler timeseries plot to {plot_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping timeseries plot")
        except Exception as e:
            logger.warning(f"Failed to write scheduler timeseries plot: {e}")

    def fit(self, global_step: int = 0) -> None:
        init_start_time = time.time()
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

        # Log dataloader ordering for validation before training begins.
        self._log_dataloader_order()

        # Evaluate at step 0 (before any training) unless resuming from a checkpoint.
        if global_step == 0 and self.eval_dataloader and not self.args.skip_eval_step_zero:
            eval_generate_kwargs = self.generate_kwargs.copy()
            eval_generate_kwargs["temperature"] = self.args.eval_temperature
            eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
            
            init_wall_sec = time.time() - init_start_time
            logger.info(f"[Timing] Initialization took {init_wall_sec:.2f}s")
            
            self.evaluate(global_step, **eval_generate_kwargs)

        last_train_end_time = time.time()
        last_eval_wall_sec = 0.0

        # --skip_training: exit after step-0 eval without entering the training loop.
        if getattr(self.args, "skip_training", False):
            logger.info("--skip_training is set: skipping training loop and exiting after step-0 eval.")
            # Flush any accumulated timeseries samples before exit.
            self.samples_generator.flush_timeseries_to_disk(global_step=global_step)
            self._write_run_summary(global_step)
            self._write_scheduler_timeseries_plot()
            self._write_final_tool_usage_plot()
            if self.wandb_logger:
                self.wandb_logger.close()
            if self.tensorboard_logger:
                self.tensorboard_logger.close()
            return

        for episode in range(start_episode, self.args.num_episodes):
            dataset_length = len(self.prompts_dataloader)
            pbar = tqdm(
                range(dataset_length),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                initial=total_consumed_prompts % max(dataset_length, 1),
            )
            while True:
                iteration_start_time = time.time()

                # Calculate gap from the end of the previous training loop (or initialization)
                # to the start of this generation step, excluding any time spent evaluating.
                rollout_start_time = time.time()
                gap_train_to_rollout = rollout_start_time - last_train_end_time - last_eval_wall_sec

                # Draw one mini-batch of prompts; stop when loader is exhausted.
                log_step_trace = global_step % 2 == 0
                rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
                    self.samples_generator.generate_samples(
                        global_step=global_step, log_step_trace=log_step_trace, **self.generate_kwargs
                    )
                )
                rollout_wall_sec = time.time() - rollout_start_time
                rollout_end_time = time.time()

                total_consumed_prompts += prompts_consumed
                #### Oversampling: train on partial batch before breaking ####
                if is_exhausted:
                    if rollout_samples:
                        train_start_time = time.time()
                        status, global_step = self.train_step(rollout_samples, global_step)
                        train_wall_sec = time.time() - train_start_time
                        last_train_end_time = time.time()
                        log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                        logger.info(f"✨ Global step {global_step} [partial-batch]: {log_status}")
                        client_states = {
                            "episode": episode,
                            "global_step": global_step,
                            "total_consumed_prompts": total_consumed_prompts,
                            "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                        }
                        self.save_logs_and_checkpoints(global_step, status, client_states)
                        del rollout_samples, status
                        gc.collect()
                    break
                #### end oversampling ####

                # Run PPO update on this batch and bump the global step counter.
                train_start_time = time.time()
                gap_rollout_to_train = train_start_time - rollout_end_time
                status, global_step = self.train_step(rollout_samples, global_step)
                train_wall_sec = time.time() - train_start_time
                last_train_end_time = time.time()

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = filter_pass_rate
                    status["too_easy_pct"] = self.samples_generator.step_too_easy_pct
                    status["too_hard_pct"] = self.samples_generator.step_too_hard_pct

                #### Oversampling: telemetry ####
                if getattr(self.args, "oversample_ratio", 1.0) > 1.0:
                    status["oversample/missed_count"] = self.samples_generator._step_missed_count
                    status["oversample/missed_pct"] = self.samples_generator.step_missed_pct
                #### end oversampling ####

                #### KNN reversal metrics ####
                knn_stats = self.samples_generator.step_knn_stats
                if knn_stats.get("knn_total", 0) > 0:
                    for k, v in knn_stats.items():
                        if v is not None:
                            status[k] = v
                #### end KNN metrics ####

                # Merge vLLM stats into status for W&B logging.
                vllm_stats = getattr(self.samples_generator, "last_vllm_stats", {})
                
                # Remove duplicated wall time from vllm_stats to rely purely on the trainer's measurement
                vllm_stats.pop("vllm_generation_wall_time_sec", None)
                
                status.update(vllm_stats)
                status["vllm_rollout_wall_sec"] = round(rollout_wall_sec, 2)
                status["vllm_train_wall_sec"] = round(train_wall_sec, 2)

                log_status = {k: v for k, v in status.items() if k not in ["generated_samples"]}
                logger.info(f"✨ Global step {global_step}: {log_status}")
                
                current_eval_wall_sec = None

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
                    eval_start_time = time.time()
                    eval_generate_kwargs = self.generate_kwargs.copy()
                    eval_generate_kwargs["temperature"] = self.args.eval_temperature
                    eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                    self.evaluate(global_step, **eval_generate_kwargs)
                    current_eval_wall_sec = time.time() - eval_start_time

                last_eval_wall_sec = current_eval_wall_sec if current_eval_wall_sec else 0.0

                # Write training timing JSONL.
                # Compute total iteration time and misc overhead.
                iteration_wall_sec = time.time() - iteration_start_time
                _eval_sec = current_eval_wall_sec if current_eval_wall_sec else 0.0
                accounted_sec = rollout_wall_sec + train_wall_sec + _eval_sec
                misc_gap_sec = max(0.0, iteration_wall_sec - accounted_sec)

                # Collect sub-phase timings for sleep/wake/reload/offload.
                _extra = {
                    k.replace("time/", ""): v
                    for k, v in status.items()
                    if k.startswith("time/") and k not in {
                        "time/forward_pass", "time/backward_pass", "time/sync_weights",
                    }
                }
                _extra["iteration_wall_sec"] = iteration_wall_sec
                _extra["misc_gap_sec"] = misc_gap_sec
                # Include micro batch partition stats for local tracking.
                _extra.update({k: v for k, v in status.items() if k.startswith("micro_batch/")})
                self._write_run_timing(
                    global_step=global_step,
                    rollout_wall_sec=rollout_wall_sec,
                    gap_rollout_to_train=gap_rollout_to_train,
                    train_wall_sec=train_wall_sec,
                    time_forward_pass=status.get("time/forward_pass", 0.0),
                    time_backward_pass=status.get("time/backward_pass", 0.0),
                    time_sync_weights=status.get("time/sync_weights", 0.0),
                    gap_train_to_rollout=gap_train_to_rollout,
                    eval_wall_sec=current_eval_wall_sec,
                    extra_timings=_extra,
                )

                pbar.update(prompts_consumed)

                # Free accumulated Ray object store refs and Python garbage to
                # prevent host-RAM growth across training steps.
                del rollout_samples, status
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

                self._empty_all_model_caches()

                # Flush accumulated timeseries samples from Ray actor memory to disk.
                self.samples_generator.flush_timeseries_to_disk(global_step=global_step)

            # --- Save discarded prompts for offline analysis ---
            self.samples_generator.save_discarded_indices(episode)

            #### Flush easy/hard collection (Phase 12) ####
            self.samples_generator.save_easy_hard_collection()
            #### end flush easy/hard ####

            #### Oversampling: LeftOverPrompts phase after main episode, before smart replay ####
            if getattr(self.args, "oversample_ratio", 1.0) > 1.0:
                global_step = self._run_leftover_phase(episode, global_step, total_consumed_prompts)
            #### end oversampling ####

            # Eval at end of episode (skip if last step already ran eval).
            if self.eval_dataloader and (global_step % self.args.eval_steps != 0):
                eval_generate_kwargs = self.generate_kwargs.copy()
                eval_generate_kwargs["temperature"] = self.args.eval_temperature
                eval_generate_kwargs["n_samples_per_prompt"] = self.args.eval_n_samples_per_prompt
                logger.info(f"Running end-of-episode evaluation at global_step {global_step}")
                self.evaluate(global_step, **eval_generate_kwargs)

            # --- Smart replay: log initial-pass stats and run replay episodes ---
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

        # Write run summary and timeseries plot.
        self._write_run_summary(global_step)
        self._write_scheduler_timeseries_plot()

        #### Final flush easy/hard collection (Phase 12) ####
        self.samples_generator.save_easy_hard_collection()
        #### end final flush easy/hard ####

        # Close trackers
        self._write_final_tool_usage_plot()
        if self.wandb_logger:
            self.wandb_logger.close()
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
