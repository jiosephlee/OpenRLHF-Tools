"""Build reusable local-attribution prompt artifacts for v17 ridge on deduplicated_no_salts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tdc.local_attribution_prompt_artifacts_common import (  # noqa: E402
    SPARSE_PREFIXES,
    TOP_K_TOTAL,
    TaskModelBundle,
    compute_model_decision_summary,
    dedupe_ranked,
    extract_local_contributions,
    render_evidence_block,
)
from data.tdc.ml_prompt_artifacts import local_attribution_dir, write_local_attribution_records  # noqa: E402
from data.tdc.v17_ridge_no_salts_common import (  # noqa: E402
    BACKEND,
    DEFAULT_FEATURE_CACHE,
    DEFAULT_LOCAL_ATTR_DEBUG_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SUMMARY_PATH,
    TOOL_VERSION,
)
from ml_experiments.feature_io import configure_single_thread_runtime, get_feature_frame  # noqa: E402
from ml_experiments.model_family import make_estimator, resolve_sparse_indices  # noqa: E402
from ml_experiments.preprocessing import fit_feature_preprocessor, transform_feature_frame  # noqa: E402

configure_single_thread_runtime()


def load_task_configs(summary_path: Path, tasks: Optional[list[str]] = None) -> dict[str, dict]:
    with summary_path.open() as f:
        summary = json.load(f)
    config_map = {
        entry["task"]: {
            "model_family": entry.get("model_family", summary.get("model_family", "linear")),
            "params": entry["best_params"],
        }
        for entry in summary.get("per_task", [])
    }
    if tasks is None:
        return config_map
    return {task: config_map[task] for task in tasks if task in config_map}


def load_raw_split(task: str, split: str, raw_dir: Path) -> pd.DataFrame:
    raw_path = raw_dir / task / f"{split}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    df = pd.read_csv(raw_path)
    if "Drug" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"{raw_path} missing Drug/Y columns")
    return df


def fit_task_model(task: str, config: dict, raw_dir: Path, feature_cache_dir: Path) -> TaskModelBundle:
    train_df = load_raw_split(task, "train", raw_dir)
    train_smiles = train_df["Drug"].astype(str).tolist()
    y_train = train_df["Y"].astype(int).to_numpy()

    train_features = get_feature_frame(task, "train", train_smiles, BACKEND, feature_cache_dir=feature_cache_dir, feature_jobs=8)
    preprocessor = fit_feature_preprocessor(train_features, scale_features=True)
    transformed_train = transform_feature_frame(train_features, preprocessor)
    surviving_columns = list(preprocessor["surviving_columns"])
    sparse_indices = resolve_sparse_indices(surviving_columns, SPARSE_PREFIXES)

    model = make_estimator(config["model_family"], model_jobs=-1, params=config["params"], sparse_indices=sparse_indices)
    model.fit(transformed_train.to_numpy(), y_train)

    return TaskModelBundle(
        task=task,
        model_family=config["model_family"],
        params=config["params"],
        preprocessor=preprocessor,
        surviving_columns=surviving_columns,
        sparse_indices=sparse_indices,
        model=model,
    )


def build_prompt_records_for_split(
    task: str,
    split: str,
    raw_dir: Path,
    feature_cache_dir: Path,
    bundle: TaskModelBundle,
) -> tuple[list[dict], dict]:
    raw_path = raw_dir / task / f"{split}.csv"
    if not raw_path.exists():
        return [], {}

    df = pd.read_csv(raw_path)
    smiles_list = df["Drug"].astype(str).tolist()
    feature_df = get_feature_frame(task, split, smiles_list, BACKEND, feature_cache_dir=feature_cache_dir, feature_jobs=8)
    transformed_df = transform_feature_frame(feature_df, bundle.preprocessor)

    records = []
    pos_counts = []
    neg_counts = []
    block_lengths = []
    missing_count = 0

    for idx, row in df.iterrows():
        smiles = str(row["Drug"])
        label = int(row["Y"])
        transformed_row = transformed_df.iloc[idx].to_numpy(dtype=float)
        contribution_items = extract_local_contributions(bundle, transformed_row, feature_df.iloc[idx])
        decision_summary = compute_model_decision_summary(bundle, transformed_row)
        evidence_block = render_evidence_block(contribution_items, decision_summary)
        selected_items = dedupe_ranked(contribution_items, top_k=TOP_K_TOTAL)
        pos_selected = sum(1 for item in selected_items if item["contribution"] > 0)
        neg_selected = sum(1 for item in selected_items if item["contribution"] < 0)
        pos_counts.append(pos_selected)
        neg_counts.append(neg_selected)
        block_lengths.append(len(evidence_block))
        if not contribution_items:
            missing_count += 1

        records.append(
            {
                "task": task,
                "split": split,
                "smiles": smiles,
                "label": label,
                "prompt_block": evidence_block,
            }
        )

    stats = {
        "records": len(records),
        "avg_block_chars": float(np.mean(block_lengths)) if block_lengths else 0.0,
        "avg_positive_items": float(np.mean(pos_counts)) if pos_counts else 0.0,
        "avg_negative_items": float(np.mean(neg_counts)) if neg_counts else 0.0,
        "missing_evidence_count": int(missing_count),
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local-attribution artifacts for v17 ridge on deduplicated_no_salts")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Input split directory")
    parser.add_argument("--feature-cache-dir", default=str(DEFAULT_FEATURE_CACHE), help="Feature cache directory")
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH), help="Path to saved model summary JSON")
    parser.add_argument("--debug-dir", default=str(DEFAULT_LOCAL_ATTR_DEBUG_DIR), help="Directory for summary stats")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to build")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    feature_cache_dir = Path(args.feature_cache_dir)
    summary_path = Path(args.summary_path)
    debug_dir = Path(args.debug_dir)

    task_configs = load_task_configs(summary_path, args.tasks)
    tasks = list(task_configs.keys())

    bundles = {}
    for task in tasks:
        bundles[task] = fit_task_model(task, task_configs[task], raw_dir, feature_cache_dir)
        print(f"  Fitted attribution model for {task} ({bundles[task].model_family})")

    summary: dict[str, dict] = {}
    root = local_attribution_dir(TOOL_VERSION)
    root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        summary[task] = {}
        for split in args.splits:
            records, stats = build_prompt_records_for_split(task, split, raw_dir, feature_cache_dir, bundles[task])
            if not records:
                continue
            out_path = write_local_attribution_records(TOOL_VERSION, task, split, records)
            summary[task][split] = stats
            print(f"  {task}/{split}: {len(records)} prompt blocks -> {out_path}")

    manifest = {
        "tool_version": TOOL_VERSION,
        "artifact_type": "local_attribution",
        "source_backend": BACKEND,
        "summary_path": str(summary_path),
        "tasks": tasks,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPrompt artifacts written to {root}")
    print(f"Debug summary written to {debug_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
