"""Filesystem layout helpers for the runs/ directory."""

import os
from datetime import date, datetime, timedelta
from typing import Optional


def categorize_run(run_name: str) -> str:
    return "eval" if run_name.startswith("eval-") else "train"


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_folder(when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    return monday_of(when.date()).isoformat()


def resolve_run_dir(project_root: str, run_name: str, *, when: Optional[datetime] = None) -> str:
    """Return runs/{eval|train}/{YYYY-MM-DD Monday}/<run_name> under project_root.

    Creates parent dirs on first use; idempotent across processes.
    """
    safe = run_name.replace("/", "_")
    path = os.path.join(project_root, "runs", categorize_run(safe), week_folder(when), safe)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_bulk_trace_dir(run_name: str, *, when: Optional[datetime] = None) -> Optional[str]:
    """Return <OPENRLHF_FULL_TRACE_DIR>/{eval|train}/{Monday}/<run_name>/full_traces.

    Returns ``None`` when the env var is unset (caller treats that as "disabled").
    Mirrors :func:`resolve_run_dir` so the bulk tree mechanically tracks the
    in-repo runs tree.
    """
    bulk_root = os.environ.get("OPENRLHF_FULL_TRACE_DIR", "").strip()
    if not bulk_root:
        return None
    safe = run_name.replace("/", "_")
    path = os.path.join(bulk_root, categorize_run(safe), week_folder(when), safe, "full_traces")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    return path
