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


#### _VLLMStatsPoller: background scheduler stats collection (L5) ####
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
        """Read the prefix cache hit rate from vLLM's CachingMetrics."""
        try:
            lm = getattr(self._llm, "logger_manager", None)
            if lm is None:
                return 0.0
            for sl in getattr(lm, "stat_loggers", []):
                pcm = getattr(sl, "prefix_caching_metrics", None)
                if pcm is not None and not pcm.empty:
                    return pcm.hit_rate
                for pe_logger in getattr(sl, "per_engine_stat_loggers", {}).values():
                    pcm = getattr(pe_logger, "prefix_caching_metrics", None)
                    if pcm is not None and not pcm.empty:
                        return pcm.hit_rate
        except Exception:
            pass
        return 0.0

    def _read_scheduler_stats(self):
        """Read the latest SchedulerStats from vLLM V1's AsyncLLM."""
        try:
            llm = self._llm

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
                if s is None:
                    return False
                return (
                    getattr(s, "step_counter", 0) > 0
                    or getattr(s, "num_running_reqs", 0) > 0
                    or getattr(s, "num_waiting_reqs", 0) > 0
                    or getattr(s, "kv_cache_usage", 0.0) > 0
                )

            for sl in getattr(lm, "stat_loggers", []):
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
        """Drain and return the raw sample list without touching aggregated stats."""
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


#### end _VLLMStatsPoller ####


@ray.remote
class LLMRayActor:
    """Async vLLM-backed actor that exposes generation utilities."""

    async def __init__(
        self,
        *args,
        bundle_indices: list = None,
        agent_func_path: Optional[str] = None,
        remote_rm_url: Optional[str] = None,
        #### Agent params for Phase 6 ####
        length_penalty_max_length: int = 0,
        enable_tool_calling_rewards: bool = True,
        #### end agent params ####
        **kwargs,
    ):
        self._configure_device_env(
            backend=kwargs.get("distributed_executor_backend"),
            bundle_indices=bundle_indices,
            num_gpus=kwargs.pop("num_gpus"),
        )
        self._configure_vllm_env(version, vllm, kwargs.pop("full_determinism", False))

        # Execution mode mapping:
        # - custom agent executor: user-provided AgentExecutorBase subclass
        # - single-turn with optional reward: default executor
        if agent_func_path:
            self.executor = _load_agent_executor(
                agent_func_path,
                length_penalty_max_length=length_penalty_max_length,
                enable_tool_calling_rewards=enable_tool_calling_rewards,
            )
        else:
            self.executor = SingleTurnAgentExecutor(
                remote_rm_url,
                length_penalty_max_length=length_penalty_max_length,
            )

        self.kwargs = kwargs

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

        #### Background stats poller (L5) ####
        self._stats_poller = _VLLMStatsPoller(self.llm, engine_id=0)
        #### end stats poller init ####

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

    #### Enhanced vLLM env configuration ####
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
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if full_determinism:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        if not os.environ.get("RAY_ADDRESS"):
            from ray._private.worker import global_worker

            os.environ["RAY_ADDRESS"] = global_worker.gcs_client.address

        os.environ["VLLM_USE_V1"] = "1"
    #### end enhanced vLLM env configuration ####

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight",
            args=(name, dtype, shape, empty_cache),
        )

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache),
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    #### Sleep with memory reclaim + poller pause ####
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
        for tag in tags:
            await self.llm.wake_up(tags=[tag])
        self._stats_poller.resume()
    #### end sleep with memory reclaim ####

    #### Generate with graceful ray.cancel() abort ####
    async def generate(self, prompt_token_ids, sampling_params):
        """Token-level generation for rollout executors."""
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

        return final_output
    #### end generate with graceful abort ####

    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in vLLM engine."""
        return self.llm.output_processor.get_num_unfinished_requests()

    #### Stats collection methods (L5) ####
    def set_engine_id(self, engine_id: int):
        """Set this engine's index (for time-series labeling)."""
        self._stats_poller._engine_id = engine_id

    def set_current_global_step(self, step: int):
        """Update the global step used in time-series samples."""
        self._stats_poller.set_global_step(step)

    def get_vllm_stats(self) -> Dict:
        """Return accumulated scheduler stats + raw samples, then reset."""
        return self._stats_poller.collect_and_reset()

    def get_and_flush_raw_samples(self) -> List[Dict]:
        """Drain raw timeseries samples from memory without resetting summary stats."""
        return self._stats_poller.flush_raw_samples()
    #### end stats collection methods ####

    #### GC collect for long-running agents ####
    async def gc_collect(self):
        """Force garbage collection and return freed pages to the OS."""
        import ctypes
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass  # non-Linux or musl libc
    #### end GC collect ####

    async def generate_responses(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        num_samples: int = 1,
    ):
        """Generate N samples for a single prompt."""
        tasks = [
            self.executor.execute(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=max_length,
                hf_tokenizer=hf_tokenizer,
                llm_engine=self,
            )
            for _ in range(num_samples)
        ]
        return await asyncio.gather(*tasks)


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
    #### vllm_sleep_level param ####
    vllm_sleep_level=1,
    #### end vllm_sleep_level ####
    logprobs_mode=None,
    agent_func_path: Optional[str] = None,
    remote_rm_url: Optional[str] = None,
    #### Agent params for Phase 6 ####
    length_penalty_max_length: int = 0,
    enable_tool_calling_rewards: bool = True,
    vllm_stop_strings: Optional[List[str]] = None,
    #### end agent params ####
):
    """Spin up a set of vLLM Ray actors with consistent placement."""
    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
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
        }

        actor_kwargs.update(
            {
                "agent_func_path": agent_func_path,
                "remote_rm_url": remote_rm_url,
                #### Agent params for Phase 6 ####
                "length_penalty_max_length": length_penalty_max_length,
                "enable_tool_calling_rewards": enable_tool_calling_rewards,
                #### end agent params ####
            }
        )


        if logprobs_mode:
            actor_kwargs["logprobs_mode"] = logprobs_mode
            actor_kwargs["max_logprobs"] = 1
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                #### Increase max concurrency for async agent workloads ####
                max_concurrency=1000,
                #### end max_concurrency ####
            ).remote(**actor_kwargs)
        )

    #### Assign engine IDs for time-series labeling (L5) ####
    for i, engine in enumerate(vllm_engines):
        ray.get(engine.set_engine_id.remote(i))
    #### end engine ID assignment ####

    if vllm_enable_sleep:
        #### Use explicit sleep level (default 1 to backup weights to CPU) ####
        batch_vllm_engine_call(vllm_engines, "sleep", level=vllm_sleep_level)
        #### end explicit sleep level ####

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
