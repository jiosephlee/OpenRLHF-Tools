# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for vLLM."""

import logging
import os
import sys
from typing import Any, Dict

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("openrlhf")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger


class WandbLogger:
    """Handle wandb setup and training-time logging."""

    def __init__(self, args) -> None:
        import wandb

        if not wandb.api.api_key:
            wandb.login(key=args.use_wandb)
        wandb.init(
            entity=args.wandb_org,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=args.__dict__,
            reinit=True,
        )

        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("parse/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("system/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("vllm/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("batch/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("knn/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("eval/global_step")
        wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
        wandb.define_metric("episode/round")
        wandb.define_metric("episode/*", step_metric="episode/round")
        self.handle = wandb
        self.samples_table = wandb.Table(columns=["global_step", "text", "reward"])

    # Keys routed to the "parse/" wandb panel instead of "train/".
    _PARSE_KEYS = {"parse_failed", "tool_call_attempted"}

    @staticmethod
    def _route_key(k: str) -> str:
        """Return the wandb section prefix for a given metric key."""
        if k.startswith("knn_"):
            return "knn"
        if k.startswith("micro_batch/"):
            return "batch"
        if k.startswith("vllm_"):
            return "vllm"
        if k.startswith("parse_method__") or k in WandbLogger._PARSE_KEYS:
            return "parse"
        if k in ("cpu_percent", "cpu_cores_affinity"):
            return "system"
        return "train"

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        import psutil

        logs_dict = dict(logs_dict)
        logs_dict["cpu_percent"] = psutil.cpu_percent()
        try:
            logs_dict["cpu_cores_affinity"] = len(os.sched_getaffinity(0))
        except AttributeError:
            pass

        generated_samples = logs_dict.pop("generated_samples", None)

        logs = {"train/global_step": global_step}
        for k, v in logs_dict.items():
            if k.startswith("time/"):
                continue
            if v is not None:
                section = self._route_key(k)
                logs[f"{section}/{k}"] = v

        if generated_samples:
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
            new_table = self.handle.Table(columns=self.samples_table.columns, data=self.samples_table.data)
            new_table.add_data(global_step, *generated_samples)
            self.samples_table = new_table
            logs["train/generated_samples"] = new_table

        # Single wandb.log() call to avoid step_sync issues — two separate
        # calls would create an extra W&B internal step without
        # train/global_step, causing metric corruption with step_sync=True.
        self.handle.log(logs)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        import psutil

        logs_dict = dict(logs_dict)
        logs_dict["cpu_percent"] = psutil.cpu_percent()
        try:
            logs_dict["cpu_cores_affinity"] = len(os.sched_getaffinity(0))
        except AttributeError:
            pass

        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"eval/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
        self.handle.log(logs)

    def log_episode(self, round_idx: int, episode: int, replay_round: int, logs_dict: Dict[str, Any]) -> None:
        """Log per-round episode metrics.

        Args:
            round_idx: Monotonically increasing counter (x-axis tick).
            episode: Which training episode this round belongs to.
            replay_round: 0 for the initial pass, 1+ for replay rounds.
            logs_dict: Filter stats (easy_discarded, hard_kept, etc.).
        """
        logs_dict = dict(logs_dict)
        logs_dict["episode"] = episode
        logs_dict["replay_round"] = replay_round

        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"episode/%s" % k: v for k, v in {**metrics, "round": round_idx}.items()}
        self.handle.log(logs)

    def close(self) -> None:
        self.handle.finish()


class TensorboardLogger:
    """Handle tensorboard setup and training-time logging."""

    def __init__(self, args) -> None:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(args.use_tensorboard, exist_ok=True)
        log_dir = os.path.join(args.use_tensorboard, args.wandb_run_name)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        generated_samples = logs_dict.get("generated_samples")
        for k, v in logs_dict.items():
            if k.startswith("time/"):
                continue
            if k == "generated_samples" and v is not None:
                text, reward = generated_samples
                formatted_text = f"Sample:\\n{text}\\n\\nReward: {reward:.4f}"
                self.writer.add_text("train/generated_samples", formatted_text, global_step)
            elif v is not None:
                section = WandbLogger._route_key(k)
                self.writer.add_scalar(f"{section}/{k}", v, global_step)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        for k, v in logs_dict.items():
            self.writer.add_scalar(f"eval/{k}", v, global_step)

    def close(self) -> None:
        self.writer.close()
