"""Shared helpers for building reusable per-sample local-attribution prompt artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tdc.ml_prompt_artifacts import RESULTS_BASE, local_attribution_dir, write_local_attribution_records  # noqa: E402
from ml_experiments.feature_io import (  # noqa: E402
    DEFAULT_FEATURE_CACHE,
    configure_single_thread_runtime,
    get_feature_frame,
)
from ml_experiments.model_family import make_estimator, resolve_sparse_indices  # noqa: E402
from ml_experiments.preprocessing import fit_feature_preprocessor, transform_feature_frame  # noqa: E402
from openrlhf.tools.therapeutic_tools.display_names import get_semantic_display_name  # noqa: E402

configure_single_thread_runtime()

TOOL_VERSION = "v16_no_neighbor"
BACKEND = "v14_tools"
SPARSE_PREFIXES = ("accfg::", "rdalert::", "toxalert::", "cluster::")
SUMMARY_PATH = RESULTS_BASE / "v14_coefficients" / "summary.json"
DEFAULT_RAW_DIR = _SCRIPT_DIR / "raw_deduplicated"
DEFAULT_DEBUG_DIR = _SCRIPT_DIR / "debug_v16_no_neighbor_local_attribution_prompts"
TOP_K_TOTAL = 32
TASK_CONFIG_OVERRIDES = {
    "SARSCoV2_3CLPro_Diamond": RESULTS_BASE / "v14_lr_sparse_fs" / "hp" / "v14_tools_SARSCoV2_3CLPro_Diamond_best_hp.json",
    "SARSCoV2_Vitro_Touret": RESULTS_BASE / "v14_lr_sparse_fs" / "hp" / "v14_tools_SARSCoV2_Vitro_Touret_best_hp.json",
}
V16_CONSOLIDATED_FEATURES = [
    "molecular_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
]

DENSE_FEATURE_LABELS = {
    "logp": "logP",
    "logd_7_4": "logD at pH 7.4",
    "esol_log_s": "aqueous solubility",
    "minimol_log_s": "aqueous solubility",
    "mol_weight": "molecular weight",
    "heavy_atom_count": "heavy atom count",
    "rotatable_bonds": "rotatable bond count",
    "fsp3": "fraction of sp3 carbons",
    "labute_asa": "Labute surface area",
    "npr1": "PMI ratio 1",
    "npr2": "PMI ratio 2",
    "molar_refractivity": "molar refractivity",
    "tpsa": "topological polar surface area",
    "hba": "hydrogen bond acceptor count",
    "hbd": "hydrogen bond donor count",
    "charge_polarization": "charge polarization",
    "max_gasteiger_charge": "maximum Gasteiger partial charge",
    "min_gasteiger_charge": "minimum Gasteiger partial charge",
    "charge_at_7_4": "net charge at pH 7.4",
    "neutral_fraction_7_4": "neutral fraction at pH 7.4",
    "ion_acid": "acidic form at pH 7.4",
    "ion_base": "basic form at pH 7.4",
    "ion_neutral": "neutral form at pH 7.4",
    "ion_zwitterion": "zwitterionic form at pH 7.4",
    "ion_anion": "anionic form at pH 7.4",
    "ion_cation": "cationic form at pH 7.4",
    "no_count": "nitrogen and oxygen atom count",
    "heteroatom_count": "heteroatom count",
    "most_acidic_pka": "most acidic pKa",
    "most_basic_pka": "most basic pKa",
    "num_acidic_sites": "number of acidic ionizable sites",
    "num_basic_sites": "number of basic ionizable sites",
    "aromatic_rings": "aromatic ring count",
    "aliphatic_rings": "aliphatic ring count",
    "saturated_rings": "saturated ring count",
    "total_rings": "total ring count",
    "heterocycles": "heterocycle count",
    "largest_aromatic_system": "largest aromatic system size",
    "num_aliphatic_carbocycles": "aliphatic carbocycle count",
    "num_aromatic_carbocycles": "aromatic carbocycle count",
    "num_saturated_carbocycles": "saturated carbocycle count",
    "bertz_ct": "Bertz complexity",
    "balaban_j": "Balaban J index",
    "hall_kier_alpha": "Hall-Kier alpha",
    "kappa1": "Kappa 1",
    "kappa2": "Kappa 2",
    "kappa3": "Kappa 3",
    "max_estate_index": "maximum E-state index",
    "min_estate_index": "minimum E-state index",
    "amide_bonds": "amide bond count",
    "bridgehead_atoms": "bridgehead atom count",
    "macrocycle_count": "macrocycle count",
    "spiro_atoms": "spiro atom count",
    "stereocenters": "stereocenter count",
    "num_R_stereo": "R stereocenter count",
    "num_S_stereo": "S stereocenter count",
    "num_unspecified_atom_stereo": "unspecified stereocenter count",
}


@dataclass
class TaskModelBundle:
    task: str
    model_family: str
    params: dict
    preprocessor: dict
    surviving_columns: list[str]
    sparse_indices: list[int]
    model: object


def load_task_configs(tasks: Optional[list[str]] = None) -> dict[str, dict]:
    with SUMMARY_PATH.open() as f:
        summary = json.load(f)
    config_map = {
        entry["task"]: {
            "model_family": entry["model_family"],
            "params": entry["params"],
        }
        for entry in summary
    }
    for task, hp_path in TASK_CONFIG_OVERRIDES.items():
        with hp_path.open() as f:
            hp_summary = json.load(f)
        config_map[task] = {
            "model_family": hp_summary["model_family"],
            "params": hp_summary["best_params"],
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


def format_numeric(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{float(value):.2f}"


def canonicalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def humanize_sparse_name(name: str) -> str:
    return " ".join(name.replace("_", " ").split())


def strength_word(abs_value: float, max_abs: float) -> str:
    ratio = abs_value / max_abs if max_abs > 0 else 0.0
    if ratio >= 0.6:
        return "Strong"
    if ratio >= 0.25:
        return "Moderate"
    return "Weak"


def sigmoid(logit: float) -> float:
    if logit >= 0:
        z = np.exp(-logit)
        return float(1.0 / (1.0 + z))
    z = np.exp(logit)
    return float(z / (1.0 + z))


def label_from_prob(prob_pos: float) -> str:
    return "B" if prob_pos >= 0.5 else "A"


def decision_strength_word(prob_pos: float) -> str:
    margin = abs(prob_pos - 0.5)
    if margin >= 0.35:
        return "strongly"
    if margin >= 0.15:
        return "moderately"
    return "weakly"


DISPLAY_NAME_KEY_OVERRIDES = {
    "logp": "MolLogP",
    "logd_7_4": "logD_74",
    "mol_weight": "MolWt",
    "heavy_atom_count": "HeavyAtomCount",
    "rotatable_bonds": "NumRotatableBonds",
    "fsp3": "FractionCSP3",
    "labute_asa": "LabuteASA",
    "molar_refractivity": "MolMR",
    "tpsa": "TPSA",
    "hba": "NumHAcceptors",
    "hbd": "NumHDonors",
    "neutral_fraction_7_4": "f_neutral_7_4",
    "heteroatom_count": "NumHeteroatoms",
    "bertz_ct": "BertzCT",
    "hall_kier_alpha": "HallKierAlpha",
    "max_estate_index": "MaxEStateIndex",
    "min_estate_index": "MinEStateIndex",
    "amide_bonds": "NumAmideBonds",
}


def feature_display_name(name: str) -> str:
    if name.startswith("cluster::"):
        from data.tdc.cluster_display import cluster_display
        return cluster_display(name)
    if name.startswith(SPARSE_PREFIXES):
        return get_semantic_display_name(name.split("::", 1)[1])
    if name in DENSE_FEATURE_LABELS:
        return DENSE_FEATURE_LABELS[name]
    return get_semantic_display_name(DISPLAY_NAME_KEY_OVERRIDES.get(name, name))


IONIZATION_INDICATOR_PHRASES = {
    "ion_acid": "an acidic form at pH 7.4",
    "ion_base": "a basic form at pH 7.4",
    "ion_neutral": "a neutral form at pH 7.4",
    "ion_zwitterion": "a zwitterionic form at pH 7.4",
    "ion_anion": "an anionic form at pH 7.4",
    "ion_cation": "a cationic form at pH 7.4",
}


IONIZATION_AND_SOLUBILITY_FEATURES = {
    "logd_7_4",
    "esol_log_s",
    "minimol_log_s",
    "neutral_fraction_7_4",
    "most_acidic_pka",
    "most_basic_pka",
    "num_acidic_sites",
    "num_basic_sites",
    "ion_acid",
    "ion_base",
    "ion_neutral",
    "ion_zwitterion",
    "ion_anion",
    "ion_cation",
}

STRUCTURE_AND_TOPOLOGY_FEATURES = {
    "aromatic_rings",
    "aliphatic_rings",
    "saturated_rings",
    "total_rings",
    "heterocycles",
    "largest_aromatic_system",
    "num_aliphatic_carbocycles",
    "num_aromatic_carbocycles",
    "num_saturated_carbocycles",
    "bridgehead_atoms",
    "macrocycle_count",
    "spiro_atoms",
}

MOLECULAR_PROFILE_FEATURES = {
    "logp",
    "mol_weight",
    "heavy_atom_count",
    "rotatable_bonds",
    "fsp3",
    "labute_asa",
    "npr1",
    "npr2",
    "molar_refractivity",
    "tpsa",
    "hba",
    "hbd",
    "charge_polarization",
    "max_gasteiger_charge",
    "min_gasteiger_charge",
    "no_count",
    "heteroatom_count",
    "bertz_ct",
    "balaban_j",
    "hall_kier_alpha",
    "kappa1",
    "kappa2",
    "kappa3",
    "max_estate_index",
    "min_estate_index",
    "amide_bonds",
    "stereocenters",
}


def map_feature_to_v16_group(name: str) -> str:
    if name.startswith("rdalert::") or name.startswith("toxalert::"):
        return "alert_screening"
    if name.startswith("accfg::"):
        return "structure_and_topology"
    if name.startswith("cluster::"):
        from data.tdc.cluster_display import cluster_namespaces
        ns = set(cluster_namespaces(name))
        if ns & {"rdalert", "toxalert"}:
            return "alert_screening"
        if "accfg" in ns:
            return "structure_and_topology"
        return "alert_screening"
    if name in IONIZATION_AND_SOLUBILITY_FEATURES:
        return "ionization_and_solubility"
    if name in STRUCTURE_AND_TOPOLOGY_FEATURES:
        return "structure_and_topology"
    if name in MOLECULAR_PROFILE_FEATURES:
        return "molecular_profile"
    return "molecular_profile"


def describe_dense_feature(name: str, raw_value: float, transformed_value: float) -> str:
    if name in IONIZATION_INDICATOR_PHRASES:
        phrase = IONIZATION_INDICATOR_PHRASES[name]
        if raw_value >= 0.5:
            return f"the molecule having {phrase}"
        return f"the molecule lacking {phrase}"

    label = feature_display_name(name)
    if transformed_value >= 0.75:
        return f"a higher-than-typical {label}"
    if transformed_value <= -0.75:
        return f"a lower-than-typical {label}"
    return f"a typical {label}"


def describe_sparse_feature(name: str) -> str:
    if name.startswith("cluster::"):
        from data.tdc.cluster_display import cluster_display, cluster_namespaces
        label = cluster_display(name)
        ns = set(cluster_namespaces(name))
        if ns == {"accfg"}:
            return f"contains functional group {label}"
        return f"matches alert {label}"
    if name.startswith("accfg::"):
        return f"contains functional group {humanize_sparse_name(name[7:])}"
    if name.startswith("rdalert::") or name.startswith("toxalert::"):
        return f"matches alert {humanize_sparse_name(name.split('::', 1)[1])}"
    return humanize_sparse_name(name)


def reconstruct_selected_names(model_family: str, model, feature_names: list[str], sparse_indices: list[int]) -> list[str]:
    if model_family in {"linear", "linear_lasso", "linear_lasso_aggressive", "linear_elasticnet"}:
        return list(feature_names)
    if model_family == "linear_fs":
        selector = model.named_steps["selector"]
        mask = selector.get_support()
        return [feature_names[i] for i, keep in enumerate(mask) if keep]
    if model_family in {"linear_sparse_fs", "linear_sparse_fs_mi"}:
        ct = model.named_steps["feature_split"]
        dense_indices = list(ct.transformers_[0][2])
        sparse_selector = ct.named_transformers_["sparse"]
        sparse_mask = sparse_selector.get_support()
        sparse_kept_indices = [sparse_indices[i] for i, keep in enumerate(sparse_mask) if keep]
        return [feature_names[i] for i in dense_indices] + [feature_names[i] for i in sparse_kept_indices]
    raise ValueError(f"Unsupported local-attribution family: {model_family}")


def extract_local_contributions(bundle: TaskModelBundle, transformed_row: np.ndarray, raw_row: pd.Series) -> list[dict]:
    model_family = bundle.model_family
    model = bundle.model

    if model_family in {"linear", "linear_lasso", "linear_lasso_aggressive", "linear_elasticnet"}:
        selected_names = bundle.surviving_columns
        selected_values = transformed_row
        coefs = model.coef_.ravel()
    elif model_family == "linear_fs":
        selector = model.named_steps["selector"]
        clf = model.named_steps["classifier"]
        mask = selector.get_support()
        selected_names = [bundle.surviving_columns[i] for i, keep in enumerate(mask) if keep]
        selected_values = transformed_row[mask]
        coefs = clf.coef_.ravel()
    elif model_family in {"linear_sparse_fs", "linear_sparse_fs_mi"}:
        ct = model.named_steps["feature_split"]
        clf = model.named_steps["classifier"]
        selected_names = reconstruct_selected_names(model_family, model, bundle.surviving_columns, bundle.sparse_indices)
        selected_values = np.asarray(ct.transform(transformed_row.reshape(1, -1))).ravel()
        coefs = clf.coef_.ravel()
    else:
        raise ValueError(f"Unsupported local-attribution family: {model_family}")

    items = []
    for name, feature_value, coef in zip(selected_names, selected_values, coefs):
        contribution = float(feature_value * coef)
        if abs(contribution) < 1e-10:
            continue
        raw_value = raw_row.get(name, np.nan)
        is_sparse = name.startswith(SPARSE_PREFIXES)
        if is_sparse:
            if pd.isna(raw_value) or float(raw_value) <= 0:
                continue
            description = describe_sparse_feature(name)
        else:
            if pd.isna(raw_value):
                continue
            description = describe_dense_feature(name, float(raw_value), float(feature_value))

        items.append(
            {
                "feature": name,
                "feature_display_name": feature_display_name(name),
                "description": description,
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "raw_value": None if pd.isna(raw_value) else float(raw_value),
            }
        )
    return items


def dedupe_ranked(items: list[dict], top_k: int) -> list[dict]:
    selected = []
    seen = set()
    for item in sorted(items, key=lambda x: x["abs_contribution"], reverse=True):
        key = canonicalize_label(item["description"])
        if not key or key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= top_k:
            break
    return selected


def recommended_v16_groups(items: list[dict]) -> list[str]:
    ranked = dedupe_ranked(items, top_k=TOP_K_TOTAL)
    groups = []
    seen = set()
    for item in ranked:
        group = map_feature_to_v16_group(item["feature"])
        if group in seen:
            continue
        groups.append(group)
        seen.add(group)
    return [g for g in V16_CONSOLIDATED_FEATURES if g in seen]


def compute_model_decision_summary(bundle: TaskModelBundle, transformed_row: np.ndarray) -> dict:
    model_family = bundle.model_family
    model = bundle.model

    if model_family in {"linear", "linear_lasso", "linear_lasso_aggressive", "linear_elasticnet"}:
        intercept = float(model.intercept_.ravel()[0])
        final_logit = float(np.dot(transformed_row, model.coef_.ravel()) + intercept)
    elif model_family == "linear_fs":
        selector = model.named_steps["selector"]
        clf = model.named_steps["classifier"]
        mask = selector.get_support()
        selected_values = transformed_row[mask]
        intercept = float(clf.intercept_.ravel()[0])
        final_logit = float(np.dot(selected_values, clf.coef_.ravel()) + intercept)
    elif model_family in {"linear_sparse_fs", "linear_sparse_fs_mi"}:
        ct = model.named_steps["feature_split"]
        clf = model.named_steps["classifier"]
        selected_values = np.asarray(ct.transform(transformed_row.reshape(1, -1))).ravel()
        intercept = float(clf.intercept_.ravel()[0])
        final_logit = float(np.dot(selected_values, clf.coef_.ravel()) + intercept)
    else:
        raise ValueError(f"Unsupported local-attribution family: {model_family}")

    base_prob_pos = sigmoid(intercept)
    final_prob_pos = sigmoid(final_logit)
    return {
        "base_prob_pos": base_prob_pos,
        "base_label": label_from_prob(base_prob_pos),
        "final_prob_pos": final_prob_pos,
        "final_label": label_from_prob(final_prob_pos),
        "final_strength": decision_strength_word(final_prob_pos),
    }


def tool_requirement_lines(suggested_groups: list[str]) -> list[str]:
    if len(suggested_groups) == 4:
        return [
            "Tool-use requirement:",
            "You must call the get_features tool once before giving your final answer. For this example, the top attributed features span all four feature groups, so I suggest requesting all of the features.",
        ]

    suggested_groups_text = ", ".join(suggested_groups) if suggested_groups else ", ".join(V16_CONSOLIDATED_FEATURES)
    return [
        "Tool-use requirement:",
        "You must call the get_features tool once before giving your final answer.",
        f"For this example, the top attributed features point most strongly to these feature groups, so I suggest requesting: {suggested_groups_text}.",
    ]


def render_evidence_block(items: list[dict], decision_summary: dict) -> str:
    base_label = decision_summary["base_label"]
    base_prob_pos = decision_summary["base_prob_pos"]
    final_label = decision_summary["final_label"]
    final_prob_pos = decision_summary["final_prob_pos"]
    final_strength = decision_summary["final_strength"]

    if not items:
        return (
            "Tool-use requirement:\n"
            "You must call the get_features tool once before giving your final answer. For this example, the top attributed features span all four feature groups, so I suggest requesting all of the features.\n\n"
            "Model-derived evidence for this molecule:\n"
            f"Auxiliary model base rate before molecule-specific evidence: leans toward ({base_label}) with P(B)={base_prob_pos:.3f}.\n"
            "The auxiliary linear model did not produce enough stable molecule-specific "
            "signal to summarize for this example.\n"
            f"Overall, after combining the base rate with the available molecule-specific evidence, the auxiliary linear model ultimately points toward ({final_label}) {final_strength} with P(B)={final_prob_pos:.3f}."
        )

    selected_items = dedupe_ranked(items, top_k=TOP_K_TOTAL)
    positives = [item for item in selected_items if item["contribution"] > 0]
    negatives = [item for item in selected_items if item["contribution"] < 0]
    max_abs = max(item["abs_contribution"] for item in selected_items) if selected_items else 1.0
    suggested_groups = recommended_v16_groups(selected_items)
    lines = [
        *tool_requirement_lines(suggested_groups),
        "",
        "Model-derived evidence for this molecule:",
        f"Auxiliary model base rate before molecule-specific evidence: leans toward ({base_label}) with P(B)={base_prob_pos:.3f}.",
    ]

    def add_section(title: str, section_items: list[dict], fallback: str) -> None:
        lines.append(title)
        if not section_items:
            lines.append(f"- {fallback}")
            return
        for item in section_items:
            strength = strength_word(item["abs_contribution"], max_abs)
            lines.append(
                f"- {strength}: feature `{item['feature']}` ({item['feature_display_name']}) -> "
                f"{item['description']}. contribution={item['contribution']:+.4f}"
            )

    add_section("Signals pushing toward (B):", positives, "No strong molecule-specific features were selected on this side.")
    add_section("Signals pushing toward (A):", negatives, "No strong molecule-specific features were selected on this side.")
    lines.append(
        f"Overall, after combining the base rate with this molecule-specific evidence, the auxiliary linear model ultimately points toward ({final_label}) {final_strength} with P(B)={final_prob_pos:.3f}."
    )
    return "\n".join(lines)


def fit_task_model(task: str, config: dict, raw_dir: Path) -> TaskModelBundle:
    train_df = load_raw_split(task, "train", raw_dir)
    train_smiles = train_df["Drug"].astype(str).tolist()
    y_train = train_df["Y"].astype(int).to_numpy()

    train_features = get_feature_frame(task, "train", train_smiles, BACKEND, feature_cache_dir=Path(DEFAULT_FEATURE_CACHE), feature_jobs=8)
    from ml_experiments.cluster_dedup import apply_cluster_dedup
    train_features = apply_cluster_dedup(train_features)
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


def build_prompt_records_for_split(task: str, split: str, raw_dir: Path, bundle: TaskModelBundle) -> tuple[list[dict], dict]:
    raw_path = raw_dir / task / f"{split}.csv"
    if not raw_path.exists():
        return [], {}

    df = pd.read_csv(raw_path)
    smiles_list = df["Drug"].astype(str).tolist()
    feature_df = get_feature_frame(task, split, smiles_list, BACKEND, feature_cache_dir=Path(DEFAULT_FEATURE_CACHE), feature_jobs=8)
    from ml_experiments.cluster_dedup import apply_cluster_dedup
    feature_df = apply_cluster_dedup(feature_df)
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
    parser = argparse.ArgumentParser(description="Build reusable local-attribution prompt artifacts for v16_no_neighbor")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Input split directory")
    parser.add_argument("--debug-dir", default=str(DEFAULT_DEBUG_DIR), help="Directory for summary stats")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks (default: all supported tasks)")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Splits to build")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    debug_dir = Path(args.debug_dir)
    task_configs = load_task_configs(args.tasks)
    tasks = list(task_configs.keys())

    bundles = {}
    for task in tasks:
        bundles[task] = fit_task_model(task, task_configs[task], raw_dir)
        print(f"  Fitted attribution model for {task} ({bundles[task].model_family})")

    summary: dict[str, dict] = {}
    root = local_attribution_dir(TOOL_VERSION)
    root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        summary[task] = {}
        for split in args.splits:
            records, stats = build_prompt_records_for_split(task, split, raw_dir, bundles[task])
            if not records:
                continue
            out_path = write_local_attribution_records(TOOL_VERSION, task, split, records)
            summary[task][split] = stats
            print(f"  {task}/{split}: {len(records)} prompt blocks -> {out_path}")

    manifest = {
        "tool_version": TOOL_VERSION,
        "artifact_type": "local_attribution",
        "source_backend": BACKEND,
        "summary_path": str(SUMMARY_PATH),
        "tasks": tasks,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPrompt artifacts written to {root}")
    print(f"Debug summary written to {debug_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
