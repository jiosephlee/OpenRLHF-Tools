"""Off-path JSONL trace writer.

Producer (the trainer thread) submits trace records via :meth:`TraceWriter.submit_many`,
which only does ``queue.put_nowait`` and never blocks on disk I/O. A single daemon
consumer thread per ``(phase, run_name)`` pair drains the queue, serializes records
to JSONL, and writes them to ``$OPENRLHF_FULL_TRACE_DIR/.../full_traces/{phase}_step_{N}.jsonl``.

Back-pressure semantics (failure mode = drop, never block training):

* Bounded queue (``OPENRLHF_TRACE_QUEUE_SIZE``, default 4096). Full → record dropped,
  ``stats.dropped_full`` incremented.
* Per-step wall-clock budget (``OPENRLHF_TRACE_BUDGET_SEC``, default 2.0s). Once
  cumulative serialize+write time for a given ``global_step`` exceeds the budget,
  further submissions for that step are dropped, ``stats.dropped_budget`` incremented,
  and ``stats.budget_exceeded_steps`` is bumped (once per step).
* No ``fsync``. Periodic ``flush()`` every 256 records or 5s, whichever comes first.
* Daemon thread: dies with the process. ``shutdown()`` joins with a 5s timeout.

Stats can be sampled via :meth:`TraceWriter.snapshot_stats` and logged to wandb.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_FLUSH_EVERY_N = 256
_FLUSH_EVERY_SEC = 5.0
_SENTINEL = object()


@dataclass
class _PhaseStats:
    submitted: int = 0
    written: int = 0
    dropped_full: int = 0
    dropped_budget: int = 0
    bytes_written: int = 0
    budget_exceeded_steps: int = 0


@dataclass
class _StreamState:
    """One queue + thread + file handle, keyed by ``(phase, run_name)``."""

    phase: str
    run_name: str
    q: queue.Queue
    thread: threading.Thread
    stats: _PhaseStats = field(default_factory=_PhaseStats)
    # Serialized time budget bookkeeping (consumer thread only).
    current_step: Optional[int] = None
    step_serialize_sec: float = 0.0
    step_budget_announced: bool = False
    # File handle bookkeeping (consumer thread only).
    fh = None
    fh_step: Optional[int] = None
    records_since_flush: int = 0
    last_flush_t: float = 0.0


class TraceWriter:
    """Process-wide singleton owning all background trace streams."""

    _instance_lock = threading.Lock()
    _instance: Optional["TraceWriter"] = None

    def __init__(self) -> None:
        self._streams_lock = threading.Lock()
        self._streams: dict[tuple[str, str], _StreamState] = {}
        self._shutdown = False
        self._queue_size = max(64, _env_int("OPENRLHF_TRACE_QUEUE_SIZE", 4096))
        self._budget_sec = max(0.0, _env_float("OPENRLHF_TRACE_BUDGET_SEC", 2.0))
        atexit.register(self.shutdown)

    # ---- public API ----------------------------------------------------

    @classmethod
    def instance(cls) -> "TraceWriter":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit_many(
        self,
        records: Iterable[dict],
        *,
        phase: str,
        global_step: int,
        run_name: str,
    ) -> Optional[str]:
        """Enqueue ``records`` for asynchronous append. Never blocks, never raises.

        Returns the destination path on success (best-effort: file may not exist
        yet on disk by the time this returns). Returns ``None`` when the bulk
        directory is unset/unwritable.
        """
        if phase not in ("rollout", "eval"):
            raise ValueError(f"phase must be 'rollout' or 'eval', got {phase!r}")
        if self._shutdown:
            return None

        from openrlhf.utils.run_paths import resolve_bulk_trace_dir

        bulk_dir = resolve_bulk_trace_dir(run_name)
        if bulk_dir is None:
            return None

        state = self._get_or_create_stream(phase, run_name)
        out_path = os.path.join(bulk_dir, f"{phase}_step_{int(global_step)}.jsonl")

        for rec in records:
            state.stats.submitted += 1
            try:
                state.q.put_nowait((int(global_step), out_path, rec))
            except queue.Full:
                state.stats.dropped_full += 1
        return out_path

    def snapshot_stats(self) -> dict:
        """Return per-phase counters. Safe to call from any thread."""
        out: dict = {}
        with self._streams_lock:
            for (phase, _), state in self._streams.items():
                bucket = out.setdefault(phase, _PhaseStats())
                bucket.submitted += state.stats.submitted
                bucket.written += state.stats.written
                bucket.dropped_full += state.stats.dropped_full
                bucket.dropped_budget += state.stats.dropped_budget
                bucket.bytes_written += state.stats.bytes_written
                bucket.budget_exceeded_steps += state.stats.budget_exceeded_steps
        return {phase: vars(stats) for phase, stats in out.items()}

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Signal all streams to drain and join. Idempotent."""
        if self._shutdown:
            return
        self._shutdown = True
        with self._streams_lock:
            states = list(self._streams.values())
        for state in states:
            try:
                state.q.put_nowait(_SENTINEL)
            except queue.Full:
                pass
        deadline = time.monotonic() + timeout
        for state in states:
            remaining = max(0.0, deadline - time.monotonic())
            state.thread.join(timeout=remaining)

    # ---- internals -----------------------------------------------------

    def _get_or_create_stream(self, phase: str, run_name: str) -> _StreamState:
        key = (phase, run_name)
        with self._streams_lock:
            state = self._streams.get(key)
            if state is not None:
                return state
            q: queue.Queue = queue.Queue(maxsize=self._queue_size)
            thread = threading.Thread(
                target=self._consumer_loop,
                args=(phase, run_name, q),
                name=f"trace-writer-{phase}-{run_name[:32]}",
                daemon=True,
            )
            state = _StreamState(phase=phase, run_name=run_name, q=q, thread=thread)
            self._streams[key] = state
            thread.start()
            return state

    def _consumer_loop(self, phase: str, run_name: str, q: queue.Queue) -> None:
        key = (phase, run_name)
        state = self._streams[key]
        budget_sec = self._budget_sec
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                step, out_path, rec = item
                self._handle_record(state, step, out_path, rec, budget_sec)
        except Exception as exc:  # pragma: no cover — never let writer thread die
            logger.warning("[trace_writer] %s consumer crashed: %s", phase, exc)
        finally:
            self._close_handle(state)

    def _handle_record(
        self,
        state: _StreamState,
        step: int,
        out_path: str,
        rec: dict,
        budget_sec: float,
    ) -> None:
        # Step rollover: reset budget bookkeeping and re-open file if path changed.
        if state.current_step != step:
            state.current_step = step
            state.step_serialize_sec = 0.0
            state.step_budget_announced = False
        if state.fh is None or state.fh_step != step:
            self._open_handle(state, out_path, step)

        # Per-step wall-clock budget.
        if budget_sec > 0.0 and state.step_serialize_sec >= budget_sec:
            state.stats.dropped_budget += 1
            if not state.step_budget_announced:
                state.stats.budget_exceeded_steps += 1
                state.step_budget_announced = True
                logger.warning(
                    "[trace_writer] %s step=%d exceeded %.2fs budget; dropping further records this step",
                    state.phase,
                    step,
                    budget_sec,
                )
            return

        t0 = time.monotonic()
        try:
            payload = json.dumps(rec, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("[trace_writer] %s skipped non-serializable record: %s", state.phase, exc)
            return
        try:
            assert state.fh is not None
            state.fh.write(payload)
            state.fh.write("\n")
            state.records_since_flush += 1
            state.stats.written += 1
            state.stats.bytes_written += len(payload) + 1
            now = time.monotonic()
            if (
                state.records_since_flush >= _FLUSH_EVERY_N
                or (now - state.last_flush_t) >= _FLUSH_EVERY_SEC
            ):
                state.fh.flush()
                state.records_since_flush = 0
                state.last_flush_t = now
        except OSError as exc:
            logger.warning("[trace_writer] %s write failed: %s", state.phase, exc)
            self._close_handle(state)
        finally:
            state.step_serialize_sec += time.monotonic() - t0

    def _open_handle(self, state: _StreamState, out_path: str, step: int) -> None:
        self._close_handle(state)
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            state.fh = open(out_path, "a", buffering=8192)
            state.fh_step = step
            state.records_since_flush = 0
            state.last_flush_t = time.monotonic()
            self._apply_retention(os.path.dirname(out_path), state.phase)
        except OSError as exc:
            logger.warning("[trace_writer] failed to open %s: %s", out_path, exc)
            state.fh = None
            state.fh_step = None

    def _close_handle(self, state: _StreamState) -> None:
        fh = state.fh
        if fh is None:
            return
        try:
            fh.flush()
            fh.close()
        except OSError:
            pass
        state.fh = None
        state.fh_step = None

    @staticmethod
    def _apply_retention(bulk_dir: str, phase: str) -> None:
        keep_raw = os.environ.get("OPENRLHF_FULL_TRACE_KEEP_LAST", "").strip()
        if not keep_raw:
            return
        try:
            keep = int(keep_raw)
        except ValueError:
            return
        if keep <= 0:
            return
        import glob
        import re

        step_re = re.compile(rf"^{phase}_step_(\d+)\.jsonl$")
        matches = []
        for path in glob.glob(os.path.join(bulk_dir, f"{phase}_step_*.jsonl")):
            m = step_re.match(os.path.basename(path))
            if m:
                matches.append((int(m.group(1)), path))
        if len(matches) <= keep:
            return
        matches.sort()
        for _, path in matches[: len(matches) - keep]:
            try:
                os.remove(path)
            except OSError as exc:
                logger.warning("[trace_writer] retention: failed to remove %s: %s", path, exc)
