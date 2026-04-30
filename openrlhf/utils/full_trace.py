"""Per-sample full-trace writer used by both rollout and eval paths.

Records land at::

    $OPENRLHF_FULL_TRACE_DIR/{eval|train}/{Monday}/<run_name>/full_traces/
        {rollout|eval}_step_{N}.jsonl

This module is now a thin shim over :class:`openrlhf.utils.trace_writer.TraceWriter`,
which runs all I/O on a daemon thread with a bounded queue and a per-step wall-clock
budget. The trainer thread never blocks on shared-FS writes; under back-pressure
records are dropped (counters exposed via ``TraceWriter.snapshot_stats``).

CLI / env knobs:
    --save_all_traces (default True; alias --save_sft_distill_traces)
    --save_distill_trace_messages (default False)
        When set, eval records include a structured ``trace_messages`` field
        suitable for SFT distillation. Rollout records never include this field.
    OPENRLHF_FULL_TRACE_DIR        unset = disabled, no writes
    OPENRLHF_FULL_TRACE_KEEP_LAST  N  (FIFO retention per phase)
    OPENRLHF_TRACE_QUEUE_SIZE      writer queue depth (default 4096)
    OPENRLHF_TRACE_BUDGET_SEC      per-step serialize budget (default 2.0)

Schemas (asymmetric — missing keys mean "absent"):

    Rollout records:
        phase, global_step, datasource, ds_idx, sample_idx,
        prompt, label, response, score, reward, response_length,
        completion_length, total_length, truncated, extra_logs

    Eval records:
        phase, global_step, datasource, task, smiles, sample_idx, ds_idx,
        prompt, label, response, score, reward, response_length,
        completion_length, total_length, truncated, extra_logs,
        source_messages, source_answer, source_label, [pred],
        [trace_messages]   # only when --save_distill_trace_messages is set
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    return bool(os.environ.get("OPENRLHF_FULL_TRACE_DIR", "").strip())


def write_jsonl(records: Iterable[dict], phase: str, global_step: int, run_name: str) -> Optional[str]:
    """Submit ``records`` to the off-path :class:`TraceWriter`. Never raises.

    Returns the destination path on success or ``None`` when full-trace logging
    is disabled. Submission is asynchronous; the actual file may not be flushed
    by the time this returns.
    """
    from openrlhf.utils.trace_writer import TraceWriter

    return TraceWriter.instance().submit_many(
        records, phase=phase, global_step=global_step, run_name=run_name
    )
