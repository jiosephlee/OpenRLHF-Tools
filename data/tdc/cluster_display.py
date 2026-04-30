"""Helpers for resolving cluster::<id> column names to human-readable descriptions.

Used by the local-attribution / reasoning-trace narrators after `apply_cluster_dedup`
is applied to feature frames.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CLUSTER_PATH = (
    REPO_ROOT / "openrlhf" / "tools" / "therapeutic_tools" / "cache"
    / "sparse_feature_clusters_official_v15.json"
)


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_cluster_payload(path: str | Path = DEFAULT_CLUSTER_PATH) -> dict:
    return _load(str(path))


@lru_cache(maxsize=4)
def _by_id(path: str) -> dict[str, dict]:
    payload = _load(path)
    return {c["cluster_id"]: c for c in payload["clusters"]}


def cluster_display(cluster_col: str, path: str | Path = DEFAULT_CLUSTER_PATH) -> str:
    """`cluster::cluster_0123` -> e.g. 'nitro-aromatic'. Returns the raw column on miss."""
    if not cluster_col.startswith("cluster::"):
        return cluster_col
    cid = cluster_col[len("cluster::") :]
    cluster = _by_id(str(path)).get(cid)
    if cluster is None:
        return cluster_col
    return cluster["display_name"]


def cluster_namespaces(cluster_col: str, path: str | Path = DEFAULT_CLUSTER_PATH) -> list[str]:
    if not cluster_col.startswith("cluster::"):
        return []
    cid = cluster_col[len("cluster::") :]
    cluster = _by_id(str(path)).get(cid)
    if cluster is None:
        return []
    return list(cluster.get("namespaces", []))
