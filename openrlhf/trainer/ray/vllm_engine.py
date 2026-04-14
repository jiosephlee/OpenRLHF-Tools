import asyncio
import os
import threading
import time as time_mod
from copy import deepcopy
from typing import Any, Dict, List, Optional

import ray
import vllm
from packaging import version
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.utils import random_uuid

from openrlhf.utils.agent import AgentExecutorBase, SingleTurnAgentExecutor
from openrlhf.utils.logging_utils import init_logger

from .utils import get_bundle_indices, ray_noset_visible_devices

logger = init_logger(__name__)


def _load_agent_executor(agent_func_path: str, **kwargs) -> AgentExecutorBase:
    assert agent_func_path.endswith(".py"), "Agent path must be a Python file"
    import importlib.util

    spec = importlib.util.spec_from_file_location("agent_module", agent_func_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)

    assert hasattr(agent_module, "AgentExecutor"), "Agent module must contain AgentExecutor class"
    agent_executor_cls = agent_module.AgentExecutor
    assert issubclass(agent_executor_cls, AgentExecutorBase), "AgentExecutor must inherit from AgentExecutorBase"
    return agent_executor_cls(**kwargs)


class _VLLMStatsPoller:
    """Background thread that reads vLLM V1 SchedulerStats every ~3 seconds.

    Accumulates min/max/mean for per-step summaries and stores every raw
    sample for time-series export.  Thread-safe via a simple lock.
    """

    def __init__(self, llm_engine, engine_id: int = 0, poll_interval: float = 3.0):
        self._llm = llm_engine
        self._engine_id = engine_id
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._current_global_step: int = -1

        # Accumulated stats since last collection.
        self._samples: List[Dict] = []
        self._kv_cache_vals: List[float] = []
        self._running_vals: List[int] = []
        self._waiting_vals: List[int] = []
        self._pc_hit_rates: List[float] = []

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._stop = threading.Event()
        self._paused = threading.Event()  # When set, polling is paused.
        self._thread.start()

    # -- polling loop --------------------------------------------------

    def _read_prefix_cache_hit_rate(self) -> float:
        """Read the prefix cache hit rate from vLLM's CachingMetrics.

        CachingMetrics accumulates across scheduler steps (sliding window over
        last 1000 requests), so a single poll gives an accurate rolling rate.
        """
        try:
            lm = getattr(self._llm, "logger_manager", None)
            if lm is None:
                return 0.0
            for sl in getattr(lm, "stat_loggers", []):
                pcm = getattr(sl, "prefix_caching_metrics", None)
                if pcm is not None and not pcm.empty:
                    return pcm.hit_rate
                # Nested: PerEngineStatLoggerAdapter
                for pe_logger in getattr(sl, "per_engine_stat_loggers", {}).values():
                    pcm = getattr(pe_logger, "prefix_caching_metrics", None)
                    if pcm is not None and not pcm.empty:
                        return pcm.hit_rate
        except Exception:
            pass
        return 0.0

    def _read_scheduler_stats(self):
        """Read the latest SchedulerStats from vLLM V1's AsyncLLM.

        Stats flow: EngineCore → output_handler → logger_manager.record()
        which sets ``LoggingStatLogger.last_scheduler_stats``.

        The logger_manager.stat_loggers list contains AggregateStatLoggerBase
        instances. A plain LoggingStatLogger is wrapped inside a
        PerEngineStatLoggerAdapter.  We probe both direct and nested paths.

        Returns None when no real stats have been recorded yet.
        """
        try:
            llm = self._llm  # vLLM V1 AsyncLLM instance

            lm = getattr(llm, "logger_manager", None)
            if lm is None:
                if not getattr(self, "_warned_no_logger_manager", False):
                    self._warned_no_logger_manager = True
                    logger.warning(
                        "vLLM engine %d: logger_manager is None — "
                        "stats polling will not work (log_stats disabled?)",
                        self._engine_id,
                    )
                return None

            def _is_active(s):
                """Check if SchedulerStats represents actual scheduler activity."""
                if s is None:
                    return False
                # Accept if any of these indicate real activity:
                # - step_counter > 0 means scheduler has run at least once
                # - running+waiting > 0 means there are active requests
                # - kv_cache_usage > 0 means KV cache has been allocated
                return (
                    getattr(s, "step_counter", 0) > 0
                    or getattr(s, "num_running_reqs", 0) > 0
                    or getattr(s, "num_waiting_reqs", 0) > 0
                    or getattr(s, "kv_cache_usage", 0.0) > 0
                )

            for sl in getattr(lm, "stat_loggers", []):
                # Direct: LoggingStatLogger or AggregatedLoggingStatLogger
                stats = getattr(sl, "last_scheduler_stats", None)
                if _is_active(stats):
                    if not getattr(self, "_logged_first_stats", False):
                        self._logged_first_stats = True
                        logger.info(
                            "vLLM engine %d: first scheduler stats read — "
                            "kv_cache=%.2f%%, running=%d, waiting=%d (via %s)",
                            self._engine_id,
                            getattr(stats, "kv_cache_usage", 0.0) * 100,
                            getattr(stats, "num_running_reqs", 0),
                            getattr(stats, "num_waiting_reqs", 0),
                            type(sl).__name__,
                        )
                    return stats

                # Nested: PerEngineStatLoggerAdapter wraps per-engine loggers
                pe_loggers = getattr(sl, "per_engine_stat_loggers", {})
                for pe_logger in pe_loggers.values():
                    pe_stats = getattr(pe_logger, "last_scheduler_stats", None)
                    if _is_active(pe_stats):
                        if not getattr(self, "_logged_first_stats", False):
                            self._logged_first_stats = True
                            logger.info(
                                "vLLM engine %d: first scheduler stats read — "
                                "kv_cache=%.2f%%, running=%d, waiting=%d "
                                "(via %s → %s)",
                                self._engine_id,
                                getattr(pe_stats, "kv_cache_usage", 0.0) * 100,
                                getattr(pe_stats, "num_running_reqs", 0),
                                getattr(pe_stats, "num_waiting_reqs", 0),
                                type(sl).__name__,
                                type(pe_logger).__name__,
                            )
                        return pe_stats

            return None
        except Exception as e:
            if not getattr(self, "_warned_read_error", False):
                self._warned_read_error = True
                logger.warning("vLLM engine %d: error reading scheduler stats: %s", self._engine_id, e)
            return None

    def _poll_loop(self):
        while not self._stop.wait(self._poll_interval):
            if self._paused.is_set():
                continue
            stats = self._read_scheduler_stats()
            if stats is None:
                continue

            kv = getattr(stats, "kv_cache_usage", 0.0)
            running = getattr(stats, "num_running_reqs", 0)
            waiting = getattr(stats, "num_waiting_reqs", 0)

            # Prefix cache hit rate from vLLM's CachingMetrics (sliding window
            # over last 1000 requests).  The per-step SchedulerStats only holds
            # a single-step delta that we'd miss between polls, but the
            # LoggingStatLogger accumulates them into prefix_caching_metrics.
            pc_hit_rate = self._read_prefix_cache_hit_rate()

            sample = {
                "t": time_mod.time(),
                "global_step": self._current_global_step,
                "engine": self._engine_id,
                "kv_cache_usage": round(kv, 4),
                "num_running": running,
                "num_waiting": waiting,
                "prefix_cache_hit_rate": round(pc_hit_rate, 4),
            }

            with self._lock:
                self._samples.append(sample)
                self._kv_cache_vals.append(kv)
                self._running_vals.append(running)
                self._waiting_vals.append(waiting)
                self._pc_hit_rates.append(pc_hit_rate)

    # -- public API ----------------------------------------------------

    def set_global_step(self, step: int):
        self._current_global_step = step

    def flush_raw_samples(self) -> List[Dict]:
        """Drain and return the raw sample list without touching aggregated stats.

        Called periodically to write timeseries to disk and free memory in the
        Ray actor.  Aggregated vals (_kv_cache_vals etc.) remain intact so that
        collect_and_reset() can still produce a correct per-step summary.
        """
        with self._lock:
            samples = self._samples
            self._samples = []
        return samples

    def collect_and_reset(self) -> Dict:
        """Return accumulated stats + raw samples, then reset."""
        with self._lock:
            samples = self._samples
            kv = self._kv_cache_vals
            running = self._running_vals
            waiting = self._waiting_vals
            pc_hit_rates = self._pc_hit_rates

            self._samples = []
            self._kv_cache_vals = []
            self._running_vals = []
            self._waiting_vals = []
            self._pc_hit_rates = []

        n = len(kv)
        if n == 0:
            return {"num_samples": 0, "raw_samples": []}

        # Use the last polled hit rate (it's a sliding window, so the most
        # recent reading is the most accurate).
        last_pc_hit_rate = pc_hit_rates[-1] if pc_hit_rates else 0.0

        return {
            "num_samples": n,
            "kv_cache_usage_pct": {
                "mean": round(sum(kv) / n, 4),
                "max": round(max(kv), 4),
                "min": round(min(kv), 4),
            },
            "num_running_reqs": {
                "mean": round(sum(running) / n, 2),
                "max": max(running),
            },
            "num_waiting_reqs": {
                "mean": round(sum(waiting) / n, 2),
                "max": max(waiting),
            },
            "prefix_cache_hit_rate": round(last_pc_hit_rate, 4),
            "raw_samples": samples,
        }

    def pause(self):
        """Pause polling (e.g. when engine is sleeping)."""
        self._paused.set()

    def resume(self):
        """Resume polling (e.g. when engine wakes up)."""
        self._paused.clear()

    def stop(self):
        self._stop.set()


@ray.remote
class LLMRayActor:
    """Async vLLM-backed actor that exposes generation utilities."""

    async def __init__(
        self,
        *args,
        bundle_indices: list = None,
        agent_func_path: Optional[str] = None,
        remote_rm_url: Optional[str] = None,
        agent_max_steps: int = 5,
        vllm_stop_strings: Optional[list] = None,
        chat_protocol: str = "glm_flash",
        tool_version: Optional[str] = None,
        length_penalty_start: int = 0,
        enable_tool_calling_rewards: bool = True,
        tool_calling_reward_mode: str = "auto",
        tool_calling_reward_naive_per_call: float = 0.1,
        tool_calling_reward_feature_single: float = 0.1,
        tool_calling_reward_feature_full: float = 0.05,
        tool_calling_reward_feature_max_count: int = 21,
        tool_calling_reward_until_step: int = -1,
        **kwargs,
    ):
        self._configure_device_env(
            backend=kwargs.get("distributed_executor_backend"),
            bundle_indices=bundle_indices,
            num_gpus=kwargs.pop("num_gpus"),
        )
        self._configure_vllm_env(version, vllm, kwargs.pop("full_determinism", False))

        model_path = kwargs.get("model", "")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Configure agent environment variables
        if agent_func_path:
            os.environ["OPENRLHF_MODEL_PATH"] = model_path
            os.environ["OPENRLHF_CHAT_PROTOCOL"] = chat_protocol
            os.environ["OPENRLHF_MAX_STEPS"] = str(agent_max_steps)
            if tool_version:
                os.environ["OPENRLHF_TOOL_VERSION"] = tool_version

        # Store agent config for generation
        self.agent_max_steps = agent_max_steps
        self.vllm_stop_strings = vllm_stop_strings

        # Hidden instruction: injected into prompts during generation,
        # stripped from token sequences before training.
        self.hidden_instruction = os.environ.get("OPENRLHF_HIDDEN_INSTRUCTION")
        self.chat_protocol_name = chat_protocol

        # Execution mode mapping:
        # - custom agent executor: user-provided AgentExecutorBase subclass
        # - single-turn with optional reward: default executor
        if agent_func_path:
            self.executor = _load_agent_executor(
                agent_func_path,
                length_penalty_start=length_penalty_start,
                enable_tool_calling_rewards=enable_tool_calling_rewards,
                tool_calling_reward_until_step=tool_calling_reward_until_step,
                tool_calling_reward_mode=tool_calling_reward_mode,
                tool_calling_reward_naive_per_call=tool_calling_reward_naive_per_call,
                tool_calling_reward_feature_single=tool_calling_reward_feature_single,
                tool_calling_reward_feature_full=tool_calling_reward_feature_full,
                tool_calling_reward_feature_max_count=tool_calling_reward_feature_max_count,
            )
        else:
            self.executor = SingleTurnAgentExecutor(
                remote_rm_url,
                length_penalty_start=length_penalty_start,
            )

        self.kwargs = kwargs

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

        # Background stats poller (reads SchedulerStats every ~5s).
        # engine_id is set later by create_vllm_engines via set_engine_id.
        self._stats_poller = _VLLMStatsPoller(self.llm, engine_id=0)

    def _configure_device_env(self, backend, bundle_indices, num_gpus):
        if backend == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif ray_noset_visible_devices():
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

    def _configure_vllm_env(self, version, vllm, full_determinism: bool):
        # Latest vLLM's CuMemAllocator is incompatible with expandable_segments.
        # Strip it from the allocator config so vLLM can init its memory pool.
        for alloc_var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            val = os.environ.get(alloc_var, "")
            if "expandable_segments" in val:
                parts = [p.strip() for p in val.split(",") if "expandable_segments" not in p]
                if parts:
                    os.environ[alloc_var] = ",".join(parts)
                else:
                    os.environ.pop(alloc_var, None)

        if version.parse(vllm.__version__) <= version.parse("0.8.5"):
            logger.warning(
                "vLLM version %s may be older than 0.8.5; proceeding anyway (custom build assumed)", vllm.__version__
            )

        # Prevent inheriting trainer process-group rendezvous env into vLLM workers.
        # vLLM V1 initializes its own distributed context and can collide on MASTER_PORT.
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        if not os.environ.get("RAY_ADDRESS"):
            from ray._private.worker import global_worker

            os.environ["RAY_ADDRESS"] = global_worker.gcs_client.address

        os.environ["VLLM_USE_V1"] = "1"

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False, fp4_quantize_format=None):
        return await self.llm.collective_rpc(
            "update_weight",
            args=(name, dtype, shape, empty_cache, fp4_quantize_format),
        )

    async def update_weight_cuda_ipc(
        self, name, dtype, shape, ipc_handles, empty_cache=False, fp4_quantize_format=None
    ):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache, fp4_quantize_format),
        )

    async def initialize_weight_reload(self):
        """Prepare model for layerwise weight reloading before weight sync."""
        return await self.llm.collective_rpc("initialize_weight_reload")

    async def post_weight_sync(self):
        """Run process_weights_after_loading() after all weights are synced.

        Handles MXFP4 swizzling/interleaving and any other post-load processing.
        Must be called once after all weights have been broadcast.
        """
        return await self.llm.collective_rpc("post_weight_sync")

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        logger.info(f"vLLM sleep requested (level={level})")
        self._stats_poller.pause()
        await self.llm.sleep(level=level)
        import torch

        torch.cuda.empty_cache()

    async def wake_up(self, tags=["weights", "kv_cache"]):
        """Wake up the engine from sleep mode.

        Args:
            tags: Optional list of tags to selectively wake up.
                  Use ["weights"] to wake up only model weights (for weight sync).
                  Use ["kv_cache"] to wake up only KV cache (after weight sync).
                  Use None to wake up everything.
        """
        logger.info(f"vLLM wake_up requested (tags={tags})")
        # Wake tags sequentially to avoid peak memory spike from
        # simultaneous weights + kv_cache allocation (can OOM after
        # a few rollouts due to GPU memory fragmentation).
        for tag in tags:
            await self.llm.wake_up(tags=[tag])
        self._stats_poller.resume()

    async def generate(self, prompt_token_ids, sampling_params):
        """Token-level generation for rollout executors."""
        #### ray.cancel() graceful abort ####
        request_id = random_uuid()
        params = deepcopy(sampling_params)
        params.output_kind = RequestOutputKind.FINAL_ONLY
        generator = self.llm.generate(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            params,
            request_id=request_id,
        )

        final_output = None
        try:
            async for request_output in generator:
                final_output = request_output
        except asyncio.CancelledError:
            await self.llm.abort(request_id)
            raise
        #### end ray.cancel() graceful abort ####

        return final_output

    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in vLLM engine."""
        return self.llm.output_processor.get_num_unfinished_requests()

    def set_engine_id(self, engine_id: int):
        """Set this engine's index (for time-series labeling)."""
        self._stats_poller._engine_id = engine_id

    def set_current_global_step(self, step: int):
        """Update the global step used in time-series samples."""
        self._stats_poller.set_global_step(step)
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.set_current_global_step(step)

    def get_vllm_stats(self) -> Dict:
        """Return accumulated scheduler stats + raw samples, then reset."""
        return self._stats_poller.collect_and_reset()

    def get_and_flush_raw_samples(self) -> List[Dict]:
        """Drain raw timeseries samples from memory without resetting summary stats.

        Called periodically (end of eval, end of global_step) to write samples
        to disk and free Ray actor RAM without disrupting the per-step summary
        that collect_and_reset() produces at the end of generation.
        """
        return self._stats_poller.flush_raw_samples()

    async def gc_collect(self):
        """Force garbage collection and return freed pages to the OS.

        Multi-turn agent execution creates many intermediate objects per request
        (RequestOutputs, token lists, deepcopied SamplingParams).  Python's GC
        frees them, but glibc's malloc doesn't return pages to the OS — causing
        RSS to grow indefinitely.  malloc_trim(0) forces that release.
        """
        import ctypes
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass  # non-Linux or musl libc

    # Map protocol names to their generation prompt markers (the string
    # that starts the assistant turn in a chat-templated prompt).
    _GENERATION_PROMPT_MARKERS = {
        "glm_flash": "<|assistant|>\n",
        "intern_s1": "<|im_start|>assistant\n<think>",
        "qwen3": "<|im_start|>assistant\n",
        "qwen3_coder": "<|im_start|>assistant\n",
        "gpt_oss": "<|start|>assistant",
    }

    def _inject_hidden_instruction(self, prompt: str) -> tuple:
        """Inject hidden instruction into prompt and compute token metadata.

        Returns:
            (augmented_prompt, hi_count, gen_marker_count) where hi_count is the
            number of hidden instruction tokens and gen_marker_count is the number
            of generation prompt marker tokens. These are used downstream to
            compute the exact strip position from the actual observation tokens
            (which may be truncated). Returns (prompt, 0, 0) if no injection.
        """
        hi = self.hidden_instruction
        if not hi:
            return prompt, 0, 0

        marker = self._GENERATION_PROMPT_MARKERS.get(self.chat_protocol_name)
        if marker is None:
            logger.warning(
                f"[hidden_instruction] Unknown chat_protocol '{self.chat_protocol_name}', "
                f"appending hidden instruction to end of prompt as fallback."
            )
            augmented = prompt + "\n" + hi
            marker = ""
        else:
            idx = prompt.rfind(marker)
            if idx == -1:
                logger.warning(
                    f"[hidden_instruction] Generation prompt marker {marker!r} not found "
                    f"in prompt, appending to end as fallback."
                )
                augmented = prompt + "\n" + hi
                marker = ""
            else:
                augmented = prompt[:idx] + "\n" + hi + "\n" + prompt[idx:]

        # Compute token counts
        orig_ids = self.hf_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        aug_ids = self.hf_tokenizer(augmented, add_special_tokens=False)["input_ids"]
        hi_count = len(aug_ids) - len(orig_ids)

        if hi_count <= 0:
            logger.warning(
                f"[hidden_instruction] Token count delta is {hi_count}, skipping injection."
            )
            return prompt, 0, 0

        gen_marker_count = len(self.hf_tokenizer(marker, add_special_tokens=False)["input_ids"]) if marker else 0

        return augmented, hi_count, gen_marker_count

    async def generate_responses(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        num_samples: int = 1,
        log_trajectory: bool = False,
    ):
        """Generate N samples for a single prompt. log_trajectory: log init + trajectory only for first prompt in episode.

        If the executor implements execute_batch(), delegates to it (supports
        variable-size output, e.g. ERL retries). Otherwise loops N times.
        """
        # Inject hidden instruction into the prompt for generation
        augmented_prompt, hi_count, gen_marker_count = self._inject_hidden_instruction(prompt)

        if hasattr(self.executor, "execute_batch"):
            results = await self.executor.execute_batch(
                prompt=augmented_prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=max_length,
                num_samples=num_samples,
                hf_tokenizer=self.hf_tokenizer,
                llm_engine=self,
                log_trajectory=log_trajectory,
            )
        else:
            tasks = [
                self.executor.execute(
                    prompt=augmented_prompt,
                    label=label,
                    sampling_params=sampling_params,
                    max_length=max_length,
                    hf_tokenizer=self.hf_tokenizer,
                    llm_engine=self,
                    log_trajectory=log_trajectory,
                )
                for _ in range(num_samples)
            ]
            results = await asyncio.gather(*tasks)

        # Tag each result with hidden instruction metadata for downstream stripping.
        # We pass hi_count and gen_marker_count so the experience maker can compute
        # the exact strip position from the actual observation tokens (robust to
        # left-truncation inside the executor).
        if hi_count > 0:
            for result in results:
                result["hidden_instruction_token_count"] = hi_count
                result["hidden_instruction_gen_marker_count"] = gen_marker_count
                # Store the original prompt (without hidden instruction) for logging
                result["prompt"] = prompt

        # Periodically return freed pages to OS.  Each prompt generates many
        # intermediate objects across N samples × T turns; without malloc_trim
        # glibc keeps the pages mapped and RSS grows indefinitely.
        self._gc_counter = getattr(self, "_gc_counter", 0) + 1
        if self._gc_counter % 32 == 0:
            import ctypes
            import gc

            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        return results


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    vllm_sleep_level=1,
    logprobs_mode=None,
    agent_func_path: Optional[str] = None,
    remote_rm_url: Optional[str] = None,
    agent_max_steps: int = 5,
    vllm_stop_strings: Optional[list] = None,
    chat_protocol: str = "glm_flash",
    tool_version: Optional[str] = None,
    length_penalty_start: int = 0,
    enable_tool_calling_rewards: bool = True,
    tool_calling_reward_until_step: int = -1,
    tool_calling_reward_mode: str = "auto",
    tool_calling_reward_naive_per_call: float = 0.1,
    tool_calling_reward_feature_single: float = 0.1,
    tool_calling_reward_feature_full: float = 0.05,
    tool_calling_reward_feature_max_count: int = 21,
    reduce_cuda_graph: bool = False,
    vllm_cudagraph_max_capture_size: Optional[int] = None,
    kv_cache_dtype: str = "auto",
    max_num_batched_tokens: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    # ERL config — propagated as env vars to Ray workers
    erl_hard_threshold: Optional[float] = None,
    erl_k: int = 4,
    erl_memory: bool = False,
    erl_max_memory: int = 5,
    erl_max_reflection_tokens: int = 512,
    language_model_only: bool = False,
    hidden_instruction: Optional[str] = None,
):
    """Spin up a set of vLLM Ray actors with consistent placement."""
    # Propagate hidden instruction via env var so LLMRayActor can read it.
    if hidden_instruction:
        os.environ["OPENRLHF_HIDDEN_INSTRUCTION"] = hidden_instruction

    # Propagate ERL config via env vars so ERLExecutor can read them
    # inside the Ray worker (set before LLMRayActor.__init__ calls
    # _load_agent_executor).
    if erl_hard_threshold is not None:
        os.environ["OPENRLHF_ERL_HARD_THRESHOLD"] = str(erl_hard_threshold)
        os.environ["OPENRLHF_ERL_K"] = str(erl_k)
        os.environ["OPENRLHF_ERL_MEMORY"] = "1" if erl_memory else "0"
        os.environ["OPENRLHF_ERL_MAX_MEMORY"] = str(erl_max_memory)
        os.environ["OPENRLHF_ERL_MAX_REFLECTION_TOKENS"] = str(erl_max_reflection_tokens)

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "mp"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # allow two engines to share one GPU in hybrid mode
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        actor_kwargs = {
            "model": pretrain,
            "enforce_eager": enforce_eager,
            "worker_extension_cls": "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed + i,
            "distributed_executor_backend": distributed_executor_backend,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "full_determinism": full_determinism,
            "gpu_memory_utilization": gpu_memory_utilization,
            "bundle_indices": bundle_indices,
            "num_gpus": 0.2 if use_hybrid_engine else 1,
            "enable_sleep_mode": vllm_enable_sleep,
            "kv_cache_dtype": kv_cache_dtype,
        }

        if language_model_only:
            actor_kwargs["limit_mm_per_prompt"] = {}
            # vLLM uses --limit-mm-per-prompt with empty dict to skip vision encoder loading
            # when the model has a vision component but we only need text.
            # This is the programmatic equivalent of --language-model-only.

        if max_num_batched_tokens is not None:
            actor_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

        if max_num_seqs is not None:
            actor_kwargs["max_num_seqs"] = max_num_seqs

        if reduce_cuda_graph and not enforce_eager:
            from vllm.config import CompilationConfig, CompilationMode

            max_capture = vllm_cudagraph_max_capture_size if vllm_cudagraph_max_capture_size is not None else 256
            actor_kwargs["compilation_config"] = CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                max_cudagraph_capture_size=max_capture,
                pass_config={"fuse_allreduce_rms": True, "eliminate_noops": True, "fuse_attn_quant": True},
            )
            actor_kwargs["async_scheduling"] = True
        elif vllm_cudagraph_max_capture_size is not None and not enforce_eager:
            from vllm.config import CompilationConfig, CompilationMode

            actor_kwargs["compilation_config"] = CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                max_cudagraph_capture_size=vllm_cudagraph_max_capture_size,
            )

        actor_kwargs.update(
            {
                "agent_func_path": agent_func_path,
                "remote_rm_url": remote_rm_url,
                "agent_max_steps": agent_max_steps,
                "vllm_stop_strings": vllm_stop_strings,
                "chat_protocol": chat_protocol,
                "tool_version": tool_version,
                "length_penalty_start": length_penalty_start,
                "enable_tool_calling_rewards": enable_tool_calling_rewards,
                "tool_calling_reward_until_step": tool_calling_reward_until_step,
                "tool_calling_reward_mode": tool_calling_reward_mode,
                "tool_calling_reward_naive_per_call": tool_calling_reward_naive_per_call,
                "tool_calling_reward_feature_single": tool_calling_reward_feature_single,
                "tool_calling_reward_feature_full": tool_calling_reward_feature_full,
                "tool_calling_reward_feature_max_count": tool_calling_reward_feature_max_count,
            }
        )

        if logprobs_mode:
            actor_kwargs["logprobs_mode"] = logprobs_mode
            actor_kwargs["max_logprobs"] = 1
            # assert version.parse(vllm.__version__) > version.parse("0.10.0"), (
            #     "vLLM > 0.10.0 is required for logprobs_mode"
            # )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                max_concurrency=1000,
            ).remote(**actor_kwargs)
        )

    # Assign engine IDs for time-series labeling.
    for i, engine in enumerate(vllm_engines):
        ray.get(engine.set_engine_id.remote(i))

    if vllm_enable_sleep:
        # Always use level 1 for the initial sleep so model weights are
        # backed up to CPU and can be restored on the first wake_up.
        # Level 2 would discard weights, leaving garbage after wake_up
        # (no weight sync has happened yet to reload them).
        batch_vllm_engine_call(vllm_engines, "sleep", level=1)

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """Call the same method on a list of engines and gather results."""
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
