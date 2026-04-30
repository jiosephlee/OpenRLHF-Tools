"""Shared helpers for building reusable per-sample local reasoning-trace artifacts."""

from __future__ import annotations

import argparse
import json
import math
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

from data.tdc.ml_prompt_artifacts import RESULTS_BASE, sample_prompt_variant_dir, write_sample_prompt_records  # noqa: E402
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
ARTIFACT_VARIANT = "local_reasoning_trace"
BACKEND = "v14_tools"
TRACE_SCHEMA_VERSION = "v3"  # bumped: controlled 4-way template variants for
                              #          opener/connectors/evidence/update/closer
SPARSE_PREFIXES = ("accfg::", "rdalert::", "toxalert::", "cluster::")
SUMMARY_PATH = RESULTS_BASE / "v14_coefficients" / "summary.json"
DEFAULT_RAW_DIR = _SCRIPT_DIR / "raw_deduplicated"
DEFAULT_DEBUG_DIR = _SCRIPT_DIR / "debug_v16_no_neighbor_local_reasoning_traces"
TOP_K_PER_SIDE = 7
TRACE_FEATURE_FAMILY_REGISTRY_PATH = _SCRIPT_DIR / "metadata" / "local_trace_feature_families.json"
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

TASK_LABEL_REFERENTS = {
    "AMES": {
        "A": {"precise": "being non-mutagenic", "short": "non-mutagenicity"},
        "B": {"precise": "being mutagenic", "short": "mutagenicity"},
    },
    "BBB_Martins": {
        "A": {"precise": "not crossing the BBB", "short": "not crossing the BBB"},
        "B": {"precise": "crossing the BBB", "short": "crossing the BBB"},
    },
    "Bioavailability_Ma": {
        "A": {"precise": "having oral bioavailability below 20%", "short": "low oral bioavailability"},
        "B": {"precise": "having oral bioavailability of at least 20%", "short": "higher oral bioavailability"},
    },
    "CYP2C9_Substrate_CarbonMangels": {
        "A": {"precise": "not being a CYP2C9 substrate", "short": "not being a CYP2C9 substrate"},
        "B": {"precise": "being a CYP2C9 substrate", "short": "being a CYP2C9 substrate"},
    },
    "CYP2D6_Substrate_CarbonMangels": {
        "A": {"precise": "not being a CYP2D6 substrate", "short": "not being a CYP2D6 substrate"},
        "B": {"precise": "being a CYP2D6 substrate", "short": "being a CYP2D6 substrate"},
    },
    "CYP3A4_Substrate_CarbonMangels": {
        "A": {"precise": "not being a CYP3A4 substrate", "short": "not being a CYP3A4 substrate"},
        "B": {"precise": "being a CYP3A4 substrate", "short": "being a CYP3A4 substrate"},
    },
    "Carcinogens_Lagunin": {
        "A": {"precise": "not being carcinogenic", "short": "not being carcinogenic"},
        "B": {"precise": "being carcinogenic", "short": "being carcinogenic"},
    },
    "ClinTox": {
        "A": {"precise": "not being toxic", "short": "not being toxic"},
        "B": {"precise": "being toxic", "short": "being toxic"},
    },
    "DILI": {
        "A": {"precise": "not causing DILI", "short": "not causing DILI"},
        "B": {"precise": "causing DILI", "short": "causing DILI"},
    },
    "HIA_Hou": {
        "A": {"precise": "not being absorbed", "short": "poor absorption"},
        "B": {"precise": "being absorbed", "short": "absorption"},
    },
    "PAMPA_NCATS": {
        "A": {"precise": "not being PAMPA-permeable", "short": "low PAMPA permeability"},
        "B": {"precise": "being PAMPA-permeable", "short": "PAMPA permeability"},
    },
    "Pgp_Broccatelli": {
        "A": {"precise": "not inhibiting P-gp", "short": "not inhibiting P-gp"},
        "B": {"precise": "inhibiting P-gp", "short": "inhibiting P-gp"},
    },
    "SARSCoV2_3CLPro_Diamond": {
        "A": {"precise": "not binding SARS-CoV-2 3CL protease", "short": "not binding 3CL protease"},
        "B": {"precise": "binding SARS-CoV-2 3CL protease", "short": "binding 3CL protease"},
    },
    "SARSCoV2_Vitro_Touret": {
        "A": {"precise": "not inhibiting SARS-CoV-2 replication", "short": "not inhibiting viral replication"},
        "B": {"precise": "inhibiting SARS-CoV-2 replication", "short": "inhibiting viral replication"},
    },
    "Skin_Reaction": {
        "A": {"precise": "not causing a skin reaction", "short": "not causing a skin reaction"},
        "B": {"precise": "causing a skin reaction", "short": "causing a skin reaction"},
    },
    "hERG": {
        "A": {"precise": "not blocking hERG", "short": "not blocking hERG"},
        "B": {"precise": "blocking hERG", "short": "blocking hERG"},
    },
}

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
    if ratio >= 0.7:
        return "Strong"
    if ratio >= 0.4:
        return "Moderate"
    return "Weak"


def downgrade_strength_label(label: str) -> str:
    if label == "strong":
        return "moderate"
    if label == "moderate":
        return "weak"
    return "weak"


def top_feature_probability_shift(base_prob_pos: float, selected_items: list[dict]) -> float:
    if not selected_items:
        return 0.0
    top_contribution = float(selected_items[0]["contribution"])
    return feature_probability_shift(base_prob_pos, top_contribution)


def feature_probability_shift(base_prob_pos: float, contribution: float) -> float:
    return abs(sigmoid(logit(base_prob_pos) + contribution) - base_prob_pos)


def load_trace_feature_families() -> list[dict]:
    if not TRACE_FEATURE_FAMILY_REGISTRY_PATH.exists():
        return []
    payload = json.loads(TRACE_FEATURE_FAMILY_REGISTRY_PATH.read_text())
    return payload.get("families", [])


TRACE_FEATURE_FAMILIES = load_trace_feature_families()


def mergeable_description_parts(description: str) -> Optional[tuple[str, str]]:
    if description.startswith("matches alert "):
        return ("alert", description[len("matches alert ") :])
    if description.startswith("contains functional group "):
        return ("functional_group", description[len("contains functional group ") :])
    return None


def item_merge_family(item: dict) -> Optional[dict]:
    haystack = " || ".join(
        [
            str(item.get("feature", "")).lower(),
            str(item.get("feature_display_name", "")).lower(),
            str(item.get("description", "")).lower(),
        ]
    )
    for family in TRACE_FEATURE_FAMILIES:
        needles = [needle.lower() for needle in family.get("match_any", [])]
        if needles and any(needle in haystack for needle in needles):
            return family

    parts = mergeable_description_parts(str(item.get("description", "")))
    if parts is None:
        return None
    return {
        "family_id": parts[0],
        "family_label": None,
        "fallback_render_kind": parts[0],
    }


def join_human_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        key = canonicalize_label(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def stable_choice(key: str, options: list[str]) -> str:
    if not options:
        raise ValueError("stable_choice requires at least one option")
    return options[sum(ord(ch) for ch in key) % len(options)]


def summarize_labels(labels: list[str], related_noun: str) -> str:
    labels = unique_in_order(labels)
    if len(labels) <= 4:
        return join_human_list(labels)
    preview = join_human_list(labels[:3])
    remainder = len(labels) - 3
    return f"{preview}, plus {remainder} related {related_noun}"


def sequential_connector(position: int, total_units: int, style: str) -> str:
    if style == "a":
        if position == 0:
            return "From the tool results, we first notice"
        if position == total_units - 1:
            return "Finally,"
        return "Next,"
    if style == "b":
        if position == 0:
            return "The first notable signal is"
        if position == total_units - 1:
            return "The last notable signal is"
        return "Another signal is"
    if style == "c":
        if position == 0:
            return "A first clue is"
        if position == total_units - 1:
            return "A final clue is"
        return "A further clue is"
    if position == 0:
        return "One early feature is"
    if position == total_units - 1:
        return "The remaining feature to note is"
    return "A related feature is"


def merged_description(items: list[dict]) -> str:
    family = item_merge_family(items[0]) if items else None
    parts = [mergeable_description_parts(item["description"]) for item in items]
    if family and family.get("family_label"):
        if parts and all(part is not None for part in parts):
            theme = parts[0][0]
            labels = [part[1] for part in parts if part is not None]
            if theme == "alert":
                return f"{family['family_label']}, reflected in alerts such as {summarize_labels(labels, 'alerts')}"
            if theme == "functional_group":
                return f"{family['family_label']}, reflected in functional groups such as {summarize_labels(labels, 'functional groups')}"
        descriptions = [item["description"] for item in items]
        return f"{family['family_label']}, reflected in {join_human_list(unique_in_order(descriptions))}"
    if not parts or any(part is None for part in parts):
        descriptions = [item["description"] for item in items]
        return join_human_list(unique_in_order(descriptions))

    theme = parts[0][0]
    labels = [part[1] for part in parts if part is not None]
    if theme == "alert":
        return f"alerts such as {summarize_labels(labels, 'alerts')}"
    if theme == "functional_group":
        return f"functional groups such as {summarize_labels(labels, 'functional groups')}"
    return items[0]["description"]


def grouped_signal_sentence(connector: str, merged: str, strength: str, direction: str, style: str) -> str:
    if style == "a":
        return f"{connector} {merged}, all {strength_adverb(strength)} pointing toward {direction}."
    if style == "b":
        return f"{connector} {merged}. Taken together, these are {strength} signals for {direction}."
    if style == "c":
        return f"{connector} {merged}, which collectively provide {strength} evidence for {direction}."
    return f"{connector} {merged}. As a group, these features point {strength_adverb(strength)} toward {direction}."


def _singleton_phrase(description: str) -> str:
    """Convert a structured `describe_sparse_feature` result into a noun phrase
    suitable for inline narration after connectors like "we first notice"."""
    if description.startswith("matches alert "):
        return description[len("matches alert ") :]
    if description.startswith("contains functional group "):
        return description[len("contains functional group ") :]
    return description


def singleton_signal_sentence(
    connector: str,
    description: str,
    strength: str,
    direction: str,
    style: str,
    effect: Optional[str] = None,
) -> str:
    description = _singleton_phrase(description)
    if effect is None:
        if style == "a":
            return f"{connector} {description}, which is {strength} evidence pointing toward {direction}."
        if style == "b":
            return f"{connector} {description}. This is {strength} evidence for {direction}."
        if style == "c":
            return f"{connector} {description}, adding a {strength} signal for {direction}."
        return f"{connector} {description}, a {strength} indication of {direction}."
    if style == "a":
        return f"{connector} {description}, which is {strength} evidence pointing toward {direction} and {effect}."
    if style == "b":
        return f"{connector} {description}. This is {strength} evidence for {direction} and {effect}."
    if style == "c":
        return f"{connector} {description}, a {strength} signal for {direction}, and it {effect}."
    return f"{connector} {description}, a {strength} indication of {direction}, and it {effect}."


def finite_probability_update_clause(prev_prob: float, next_prob: float, target_readable: str, style: str) -> str:
    if next_prob > prev_prob + 1e-4:
        if style == "a":
            return (
                f"pushes the running estimate for {target_readable} from "
                f"{format_percent(prev_prob)} up to about {format_percent(next_prob)}"
            )
        if style == "b":
            return (
                f"moves the running estimate for {target_readable} from "
                f"{format_percent(prev_prob)} to about {format_percent(next_prob)}"
            )
        if style == "c":
            return (
                f"shifts the running estimate for {target_readable} from "
                f"{format_percent(prev_prob)} to roughly {format_percent(next_prob)}"
            )
        return (
            f"changes the running estimate for {target_readable} from "
            f"{format_percent(prev_prob)} to approximately {format_percent(next_prob)}"
        )
    if next_prob < prev_prob - 1e-4:
        if style == "a":
            return (
                f"pulls the running estimate for {target_readable} from "
                f"{format_percent(prev_prob)} down to about {format_percent(next_prob)}"
            )
        if style == "b":
            return (
                f"moves the running estimate for {target_readable} from "
                f"{format_percent(prev_prob)} down to about {format_percent(next_prob)}"
            )
        if style == "c":
            return (
                f"shifts the running estimate for {target_readable} from "
                f"{format_percent(prev_prob)} down to roughly {format_percent(next_prob)}"
            )
        return (
            f"changes the running estimate for {target_readable} from "
            f"{format_percent(prev_prob)} down to approximately {format_percent(next_prob)}"
        )
    if style == "a":
        return (
            f"leaves the running estimate for {target_readable} essentially unchanged near "
            f"{format_percent(next_prob)}"
        )
    if style == "b":
        return (
            f"keeps the running estimate for {target_readable} about where it was, near "
            f"{format_percent(next_prob)}"
        )
    if style == "c":
        return (
            f"holds the running estimate for {target_readable} roughly steady at "
            f"{format_percent(next_prob)}"
        )
    return (
        f"changes the running estimate for {target_readable} very little, leaving it near "
        f"{format_percent(next_prob)}"
    )


def gerund_probability_update_clause(prev_prob: float, next_prob: float, target_readable: str, style: str) -> str:
    finite = finite_probability_update_clause(prev_prob, next_prob, target_readable, style)
    replacements = {
        "pushes": "pushing",
        "moves": "moving",
        "shifts": "shifting",
        "changes": "changing",
        "pulls": "pulling",
        "leaves": "leaving",
        "keeps": "keeping",
        "holds": "holding",
    }
    for old, new in replacements.items():
        if finite.startswith(old + " "):
            return new + finite[len(old) :]
    return finite


def legacy_strength_labels_for_selected_items(
    selected_items: list[dict],
    base_prob_pos: float,
    *,
    top_feature_shift_threshold: Optional[float] = None,
) -> list[str]:
    if not selected_items:
        return []

    max_abs = max(item["abs_contribution"] for item in selected_items)
    labels = [strength_bucket(item["abs_contribution"], max_abs) for item in selected_items]
    if top_feature_shift_threshold is not None and top_feature_probability_shift(base_prob_pos, selected_items) <= top_feature_shift_threshold:
        labels = [downgrade_strength_label(label) for label in labels]
    return labels


def strength_labels_for_selected_items(
    selected_items: list[dict],
    base_prob_pos: float,
    *,
    task_strength_thresholds: Optional[dict] = None,
    strength_method: str = "task_probability_shift_percentile",
    top_feature_shift_threshold: Optional[float] = None,
) -> list[str]:
    if not selected_items:
        return []

    if strength_method == "legacy_relative_top_feature_downgrade":
        return legacy_strength_labels_for_selected_items(
            selected_items,
            base_prob_pos,
            top_feature_shift_threshold=top_feature_shift_threshold,
        )

    if strength_method == "task_probability_shift_percentile" and task_strength_thresholds:
        moderate_min = float(task_strength_thresholds["moderate_min"])
        strong_min = float(task_strength_thresholds["strong_min"])
        moderate_abs_min = float(task_strength_thresholds.get("moderate_abs_min", 0.0))
        strong_abs_min = float(task_strength_thresholds.get("strong_abs_min", 0.0))
        labels = []
        for item in selected_items:
            shift = feature_probability_shift(base_prob_pos, float(item["contribution"]))
            if shift >= strong_min and shift >= strong_abs_min:
                labels.append("strong")
            elif shift >= moderate_min and shift >= moderate_abs_min:
                labels.append("moderate")
            else:
                labels.append("weak")
        return labels

    return legacy_strength_labels_for_selected_items(
        selected_items,
        base_prob_pos,
        top_feature_shift_threshold=top_feature_shift_threshold,
    )


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


def semantic_family_key(item: dict) -> str:
    feature = item["feature"]
    if feature.startswith(SPARSE_PREFIXES):
        return f"sparse::{canonicalize_label(item['feature_display_name'])}"
    return f"dense::{canonicalize_label(feature)}"


def dedupe_ranked(items: list[dict], top_k: Optional[int] = None) -> list[dict]:
    selected = []
    seen = set()
    for item in sorted(items, key=lambda x: x["abs_contribution"], reverse=True):
        key = semantic_family_key(item)
        if not key or key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if top_k is not None and len(selected) >= top_k:
            break
    return selected


def recommended_v16_groups(items: list[dict]) -> list[str]:
    ranked = dedupe_ranked(items, top_k=TOP_K_PER_SIDE * 2)
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


def format_percent(prob: float) -> str:
    pct = prob * 100.0
    if abs(pct - 50.0) < 2.0:
        return f"{pct:.1f}%"
    return f"{pct:.0f}%"


def probability_strength_word(prob: float) -> str:
    margin = abs(prob - 0.5)
    if margin >= 0.35:
        return "strong"
    if margin >= 0.20:
        return "moderate"
    if margin >= 0.10:
        return "mild"
    return "very slight"


def signed_score(value: float) -> str:
    return f"{value:+.2f}"


def oxford_join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def feature_narrative(item: dict) -> str:
    return item["description"]


def label_probability(prob_pos: float, label: str) -> float:
    return prob_pos if label == "B" else 1.0 - prob_pos


def label_reference_phrase(task: Optional[str], label: str, *, style: str = "short") -> str:
    if task is None:
        return f"class {label}"
    phrases = TASK_LABEL_REFERENTS.get(task)
    if phrases is None:
        return f"class {label}"
    label_info = phrases.get(label)
    if label_info is None:
        return f"class {label}"
    if isinstance(label_info, str):
        return label_info
    return label_info.get(style, label_info.get("short", label_info.get("precise", f"class {label}")))


def strip_gerund(phrase: str) -> str:
    for prefix in ("being ", "having "):
        if phrase.startswith(prefix):
            return phrase[len(prefix):]
    return phrase


def clipped_probability(prob: float) -> float:
    return float(min(max(prob, 1e-6), 1.0 - 1e-6))


def logit(prob: float) -> float:
    prob = clipped_probability(prob)
    return float(np.log(prob / (1.0 - prob)))


def boundary_characterization(prob: float) -> str:
    margin = abs(prob - 0.5)
    if margin < 0.10:
        return "close"
    if margin < 0.20:
        return "mixed"
    if margin < 0.35:
        return "moderate"
    return "clear"


def target_support_items(selected_items: list[dict], target_label: str) -> tuple[list[dict], list[dict]]:
    target_sign = 1 if target_label == "B" else -1
    supporting = [item for item in selected_items if contribution_sign(item) == target_sign]
    opposing = [item for item in selected_items if contribution_sign(item) == -target_sign]
    return supporting, opposing


def summary_feature_budget(prob_pos: float) -> int:
    margin = abs(prob_pos - 0.5)
    if margin >= 0.35:
        return 8
    if margin >= 0.20:
        return 12
    if margin >= 0.10:
        return 16
    return 20


def minimum_opposite_feature_count(prob_pos: float) -> int:
    if abs(prob_pos - 0.5) < 0.20:
        return 2
    return 1


def contribution_sign(item: dict) -> int:
    if item["contribution"] > 0:
        return 1
    if item["contribution"] < 0:
        return -1
    return 0


def checkpoint_count(num_items: int) -> int:
    return max(1, int(math.floor(math.sqrt(max(1, num_items)))))


def checkpoint_indices(num_items: int) -> list[int]:
    count = checkpoint_count(num_items)
    return sorted({
        max(1, min(num_items, int(math.ceil(((idx + 1) * num_items) / count))))
        for idx in range(count)
    })


def checkpoint_intro(chunk_size: int, checkpoint_idx: int, total_checkpoints: int) -> str:
    if checkpoint_idx == 0:
        return (
            "After this first piece of evidence"
            if chunk_size == 1
            else f"After these first {chunk_size} pieces of evidence"
        )
    if checkpoint_idx == total_checkpoints - 1:
        return (
            "After this final piece of evidence"
            if chunk_size == 1
            else f"After these final {chunk_size} pieces of evidence"
        )
    return (
        "After this next piece of evidence"
        if chunk_size == 1
        else f"After these next {chunk_size} pieces of evidence"
    )


def strength_bucket(abs_value: float, max_abs: float) -> str:
    return strength_word(abs_value, max_abs).lower()


def strength_adverb(strength: str) -> str:
    if strength == "strong":
        return "strongly"
    if strength == "moderate":
        return "moderately"
    return "weakly"


def select_summary_items(items: list[dict], decision_summary: dict) -> list[dict]:
    ranked = dedupe_ranked(items)
    if not ranked:
        return []

    budget = min(summary_feature_budget(decision_summary["final_prob_pos"]), len(ranked))
    return sorted(ranked[:budget], key=lambda x: x["abs_contribution"], reverse=True)


def render_prompt_block(
    items: list[dict],
    decision_summary: dict,
    selected_items: Optional[list[dict]] = None,
    *,
    task_strength_thresholds: Optional[dict] = None,
    strength_method: str = "task_probability_shift_percentile",
    top_feature_shift_threshold: Optional[float] = None,
    include_probability_updates: bool = True,
) -> str:
    base_label = decision_summary["base_label"]
    base_prob_pos = decision_summary["base_prob_pos"]
    final_label = decision_summary["final_label"]
    final_prob_pos = decision_summary["final_prob_pos"]

    if not items:
        if not include_probability_updates:
            return (
                "Evidence summary from a prior tool-informed pass:\n"
                "- Molecule-specific evidence: not enough stable local signal to summarize."
            )
        return (
            "Evidence summary from a prior tool-informed pass:\n"
            f"- Auxiliary prior: {probability_strength_word(base_prob_pos)} lean toward ({base_label}) at about {format_percent(base_prob_pos if base_label == 'B' else 1 - base_prob_pos)}.\n"
            "- Molecule-specific evidence: not enough stable local signal to summarize.\n"
            f"- Auxiliary readout after combining evidence: {probability_strength_word(final_prob_pos)} lean toward ({final_label}) at about {format_percent(final_prob_pos if final_label == 'B' else 1 - final_prob_pos)}."
        )

    selected_items = selected_items or select_summary_items(items, decision_summary)
    positives = [item for item in selected_items if item["contribution"] > 0]
    negatives = [item for item in selected_items if item["contribution"] < 0]

    # If a side is empty in the selected set, surface the strongest opposing
    # feature from the unfiltered items so the trace doesn't claim "no signal"
    # on that side when a small one exists.
    selected_ids = {id(item) for item in selected_items}
    if not positives:
        all_pos = [it for it in items if it["contribution"] > 0 and id(it) not in selected_ids]
        if all_pos:
            positives = [max(all_pos, key=lambda it: it["contribution"])]
            selected_items = list(selected_items) + positives
    if not negatives:
        all_neg = [it for it in items if it["contribution"] < 0 and id(it) not in selected_ids]
        if all_neg:
            negatives = [min(all_neg, key=lambda it: it["contribution"])]
            selected_items = list(selected_items) + negatives
    strength_labels = strength_labels_for_selected_items(
        selected_items,
        base_prob_pos,
        task_strength_thresholds=task_strength_thresholds,
        strength_method=strength_method,
        top_feature_shift_threshold=top_feature_shift_threshold,
    )
    strength_by_item = {id(item): label.title() for item, label in zip(selected_items, strength_labels)}
    suggested_groups = recommended_v16_groups(selected_items)
    suggested_groups_text = ", ".join(suggested_groups) if suggested_groups else ", ".join(V16_CONSOLIDATED_FEATURES)

    lines = [
        "Evidence summary from a prior tool-informed pass:",
        f"- Reviewed feature groups: {suggested_groups_text}.",
    ]
    if include_probability_updates:
        lines.append(
            f"- Auxiliary prior before molecule-specific evidence: {probability_strength_word(base_prob_pos)} lean toward ({base_label}) at about {format_percent(base_prob_pos if base_label == 'B' else 1 - base_prob_pos)}."
        )
    lines.append("- Features pushing toward (B):")

    def add_section(section_items: list[dict], fallback: str) -> None:
        if not section_items:
            lines.append(f"  - {fallback}")
            return
        for item in section_items:
            strength = strength_by_item[id(item)]
            lines.append(f"  - {strength}: {feature_narrative(item)}")

    add_section(positives, "No notable selected features on this side.")
    lines.append("- Features pushing toward (A):")
    add_section(negatives, "No notable selected features on this side.")
    if include_probability_updates:
        lines.append(
            f"- Auxiliary readout after combining evidence: {probability_strength_word(final_prob_pos)} lean toward ({final_label}) at about {format_percent(final_prob_pos if final_label == 'B' else 1 - final_prob_pos)}."
        )
    return "\n".join(lines)


NUMERIC_FIDELITY_MAX_GAP = 0.10
NUMERIC_FIDELITY_UNCERTAIN_BAND = (0.45, 0.55)


def render_reasoning_trace(
    selected_items: list[dict],
    decision_summary: dict,
    target_label: str,
    *,
    task: Optional[str] = None,
    task_strength_thresholds: Optional[dict] = None,
    strength_method: str = "task_probability_shift_percentile",
    top_feature_shift_threshold: Optional[float] = None,
    include_probability_updates: bool = True,
    numeric_fidelity_filter: bool = True,
    smiles: Optional[str] = None,
) -> Optional[str]:
    aux_label = decision_summary["final_label"]
    aux_prob_pos = decision_summary["final_prob_pos"]
    base_prob_pos = decision_summary["base_prob_pos"]
    target_base_prob = label_probability(base_prob_pos, target_label)
    target_final_prob = label_probability(aux_prob_pos, target_label)
    target_boundary = boundary_characterization(target_final_prob)
    aux_boundary = boundary_characterization(aux_prob_pos)
    target_phrase_precise = label_reference_phrase(task, target_label, style="precise")
    aux_phrase_precise = label_reference_phrase(task, aux_label, style="precise")
    target_phrase = label_reference_phrase(task, target_label, style="short")
    aux_phrase = label_reference_phrase(task, aux_label, style="short")
    target_readable = strip_gerund(target_phrase)
    aux_readable = strip_gerund(aux_phrase)
    target_precise_readable = strip_gerund(target_phrase_precise)

    template_key = f"{task or ''}::{smiles or ''}::{target_label}"
    opener_style = stable_choice(f"opener::{template_key}", ["a", "b", "c", "d"])
    connector_style = stable_choice(f"connector::{template_key}", ["a", "b", "c", "d"])
    singleton_style = stable_choice(f"singleton::{template_key}", ["a", "b", "c", "d"])
    grouped_style = stable_choice(f"grouped::{template_key}", ["a", "b", "c", "d"])
    update_style = stable_choice(f"update::{template_key}", ["a", "b", "c", "d"])
    closer_style = stable_choice(f"closer::{template_key}", ["a", "b", "c", "d"])
    final_style = stable_choice(f"final::{template_key}", ["a", "b", "c", "d"])

    lines: list[str]
    if include_probability_updates:
        prior_margin = abs(target_base_prob - 0.5)
        if prior_margin < 0.05:
            magnitude = "essentially balanced"
        elif prior_margin < 0.15:
            magnitude = "slight"
        elif prior_margin < 0.30:
            magnitude = "moderate"
        else:
            magnitude = "strong"

        if magnitude == "essentially balanced":
            prior_clause = "an essentially balanced starting point"
        elif target_base_prob > 0.5:
            prior_clause = f"a {magnitude} starting lean toward {target_readable}"
        else:
            prior_clause = f"a {magnitude} starting lean away from {target_readable}"

        if opener_style == "a":
            opener = (
                f"First, the base prior for {target_precise_readable} is about "
                f"{format_percent(target_base_prob)}, so before looking at molecule-specific "
                f"evidence we have {prior_clause}."
            )
        elif opener_style == "b":
            opener = (
                f"Base rate for {target_precise_readable} sits at about "
                f"{format_percent(target_base_prob)}, which gives {prior_clause} before any "
                f"molecule-specific evidence."
            )
        elif opener_style == "c":
            opener = (
                f"Starting from a prior of about {format_percent(target_base_prob)} for "
                f"{target_precise_readable}, we begin with {prior_clause}."
            )
        else:
            opener = (
                f"Before considering the specific molecular features, the background probability of "
                f"{target_precise_readable} is about {format_percent(target_base_prob)}, so the "
                f"starting position is {prior_clause}."
            )
        lines = [opener]
    else:
        lines = []

    if not selected_items:
        if include_probability_updates:
            lines.append("Analyzing the tool results, there is not enough stable local evidence to build a detailed molecule-specific case from the selected features.")
        else:
            lines.append("Looking at the tool results, there is not enough stable local evidence to build a detailed molecule-specific case from the selected features.")
        selected_target_prob = target_base_prob
    else:
        running_logit = logit(base_prob_pos)
        strength_labels = strength_labels_for_selected_items(
            selected_items,
            base_prob_pos,
            task_strength_thresholds=task_strength_thresholds,
            strength_method=strength_method,
            top_feature_shift_threshold=top_feature_shift_threshold,
        )
        weak_start = next((idx for idx, bucket in enumerate(strength_labels) if bucket == "weak"), len(selected_items))
        weak_tail_checkpoints: list[int] = []
        weak_tail_checkpoint_set: set[int] = set()
        last_weak_checkpoint_prob = None
        nonweak_groups: list[dict] = []
        nonweak_group_lookup: dict[tuple[str, str, str], int] = {}
        for idx in range(weak_start):
            item = selected_items[idx]
            strength = strength_labels[idx]
            if strength == "weak":
                continue
            direction = target_label if contribution_sign(item) == (1 if target_label == "B" else -1) else ("A" if target_label == "B" else "B")
            direction_phrase = label_reference_phrase(task, direction)
            direction_readable = strip_gerund(direction_phrase)
            family = item_merge_family(item)
            family_id = family["family_id"] if family is not None else f"singleton::{idx}"
            group_key = (direction, strength, family_id)
            group_idx = nonweak_group_lookup.get(group_key)
            if group_idx is None:
                nonweak_group_lookup[group_key] = len(nonweak_groups)
                nonweak_groups.append(
                    {
                        "items": [item],
                        "direction_readable": direction_readable,
                        "strength": strength,
                    }
                )
            else:
                nonweak_groups[group_idx]["items"].append(item)

        weak_groups: list[dict] = []
        weak_group_lookup: dict[tuple[str, str, str], int] = {}
        for idx in range(weak_start, len(selected_items)):
            item = selected_items[idx]
            strength = strength_labels[idx]
            direction = target_label if contribution_sign(item) == (1 if target_label == "B" else -1) else ("A" if target_label == "B" else "B")
            direction_phrase = label_reference_phrase(task, direction)
            direction_readable = strip_gerund(direction_phrase)
            family = item_merge_family(item)
            family_id = family["family_id"] if family is not None else f"weak-singleton::{idx}"
            group_key = (direction, strength, family_id)
            group_idx = weak_group_lookup.get(group_key)
            if group_idx is None:
                weak_group_lookup[group_key] = len(weak_groups)
                weak_groups.append(
                    {
                        "items": [item],
                        "direction": direction,
                        "direction_readable": direction_readable,
                        "strength": strength,
                    }
                )
            else:
                weak_groups[group_idx]["items"].append(item)

        weak_group_count = len(weak_groups)
        if weak_group_count > 0:
            weak_tail_checkpoints = checkpoint_indices(weak_group_count)
            weak_tail_checkpoint_set = set(weak_tail_checkpoints)
            last_weak_checkpoint_prob = target_base_prob if not nonweak_groups else None

        total_units = len(nonweak_groups) + weak_group_count
        unit_position = 0

        for group in nonweak_groups:
            prev_prob = label_probability(sigmoid(running_logit), target_label)
            group_items = group["items"]
            connector = sequential_connector(unit_position, total_units, connector_style)
            for grouped_item in group_items:
                running_logit += float(grouped_item["contribution"])
            next_prob = label_probability(sigmoid(running_logit), target_label)
            direction_readable = group["direction_readable"]
            strength = group["strength"]

            if len(group_items) > 1:
                base_sentence = grouped_signal_sentence(
                    connector, merged_description(group_items), strength, direction_readable, grouped_style
                )
                if include_probability_updates:
                    effect_clause = gerund_probability_update_clause(prev_prob, next_prob, target_readable, update_style)
                    if base_sentence.endswith("."):
                        base_sentence = base_sentence[:-1] + ", " + effect_clause + "."
                    else:
                        base_sentence = base_sentence + ", " + effect_clause + "."
                lines.append(base_sentence)
            else:
                item = group_items[0]
                effect = None
                if include_probability_updates:
                    effect = finite_probability_update_clause(prev_prob, next_prob, target_readable, update_style)
                lines.append(
                    singleton_signal_sentence(
                        connector, item["description"], strength, direction_readable, singleton_style, effect
                    )
                )
            unit_position += 1

        for weak_group_idx, group in enumerate(weak_groups, start=1):
            prev_prob = label_probability(sigmoid(running_logit), target_label)
            group_items = group["items"]
            for grouped_item in group_items:
                running_logit += float(grouped_item["contribution"])
            next_prob = label_probability(sigmoid(running_logit), target_label)
            direction_readable = group["direction_readable"]
            strength = group["strength"]
            connector = sequential_connector(unit_position, total_units, connector_style)
            if weak_group_idx == 1 and last_weak_checkpoint_prob is None:
                last_weak_checkpoint_prob = prev_prob
            if len(group_items) > 1:
                base_sentence = grouped_signal_sentence(
                    connector, merged_description(group_items), strength, direction_readable, grouped_style
                )
            else:
                item = group_items[0]
                base_sentence = singleton_signal_sentence(
                    connector, item["description"], strength, direction_readable, singleton_style
                )
            if include_probability_updates and weak_group_idx in weak_tail_checkpoint_set:
                effect_clause = (
                    "taken together with the prior weak signals, this "
                    + finite_probability_update_clause(last_weak_checkpoint_prob, next_prob, target_readable, update_style)
                )
                if base_sentence.endswith("."):
                    base_sentence = base_sentence[:-1] + "; " + effect_clause + "."
                else:
                    base_sentence = base_sentence + "; " + effect_clause + "."
                last_weak_checkpoint_prob = next_prob
            lines.append(base_sentence)
            unit_position += 1
        selected_target_prob = label_probability(sigmoid(running_logit), target_label)

    residual_matches_final = abs(selected_target_prob - target_final_prob) < 0.005

    if numeric_fidelity_filter:
        lo, hi = NUMERIC_FIDELITY_UNCERTAIN_BAND
        # Both probs must land on the gold side (target_label) with >=5pp margin,
        # AND the running cutoff must agree with the aux final within 10pp. This
        # is symmetric across A/B because *_target_prob == P(target_label).
        if (
            abs(selected_target_prob - target_final_prob) > NUMERIC_FIDELITY_MAX_GAP
            or selected_target_prob <= hi
            or target_final_prob <= hi
        ):
            return None

    selected_boundary = boundary_characterization(selected_target_prob)

    if not include_probability_updates:
        if aux_label == target_label:
            if target_boundary == "close":
                lines.append(
                    f"Taken together, the evidence only narrowly favors {target_readable}."
                )
            elif target_boundary == "mixed":
                lines.append(
                    f"Taken together, the evidence leans toward {target_readable}, but there is still meaningful pull from the other side."
                )
            elif target_boundary == "moderate":
                lines.append(
                    f"Taken together, the evidence supports {target_readable} with a reasonably solid lean."
                )
            else:
                lines.append(
                    f"Taken together, the evidence clearly supports {target_readable}."
                )
        else:
            if aux_boundary == "close":
                lines.append(
                    f"Taken together, the selected evidence points toward {target_readable}, but the full auxiliary model still lands on {aux_readable}, so this looks like a close and conflicted case."
                )
            elif aux_boundary == "mixed":
                lines.append(
                    f"Taken together, the selected evidence points toward {target_readable}, but the full auxiliary model still favors {aux_readable}, so the case remains mixed rather than cleanly resolved."
                )
            elif aux_boundary == "moderate":
                lines.append(
                    f"Taken together, the selected evidence points toward {target_readable}, but the full auxiliary model still favors {aux_readable}, making this a harder disagreement example."
                )
            else:
                lines.append(
                    f"Taken together, the selected evidence points toward {target_readable}, but the full auxiliary model clearly favors {aux_readable}, so this is a hard disagreement example."
                )
    else:
        # Trace closes at the running cutoff; that prob *is* the reported final.
        # Filter above guarantees |cutoff - aux_final| <= 0.10 and both outside
        # the uncertain [0.45, 0.55] band, so the conclusion is well-grounded.
        boundary_qualifier = {
            "close": f"only narrowly favoring {target_readable}",
            "mixed": f"leaning toward {target_readable} but with meaningful pull from the other side",
            "moderate": f"a reasonably solid lean toward {target_readable}",
            "clear": f"clearly favoring {target_readable}",
        }[selected_boundary]
        if closer_style == "a":
            lines.append(
                f"With these signals combined, the running estimate for {target_readable} "
                f"stands at about {format_percent(selected_target_prob)}, {boundary_qualifier}."
            )
        elif closer_style == "b":
            lines.append(
                f"Net running estimate for {target_readable}: about "
                f"{format_percent(selected_target_prob)} — {boundary_qualifier}."
            )
        elif closer_style == "c":
            lines.append(
                f"After combining the major signals, the running estimate for {target_readable} "
                f"settles near {format_percent(selected_target_prob)}, {boundary_qualifier}."
            )
        else:
            lines.append(
                f"Overall running estimate for {target_readable}: about "
                f"{format_percent(selected_target_prob)}, still {boundary_qualifier}."
            )

    if final_style == "a":
        lines.append(f"I would conclude that this molecule is more consistent with {target_readable}.")
    elif final_style == "b":
        lines.append(f"Overall, this molecule looks more consistent with {target_readable}.")
    elif final_style == "c":
        lines.append(f"Taken together, the profile is more consistent with {target_readable}.")
    else:
        lines.append(f"The overall evidence is more consistent with {target_readable}.")
    lines.append(f"Answer: ({target_label})")
    return "\n\n".join(lines)


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
    prompt_block_lengths = []
    trace_lengths = []
    missing_count = 0
    agreement_count = 0
    numeric_filter_dropped = 0

    for idx, row in df.iterrows():
        smiles = str(row["Drug"])
        label = int(row["Y"])
        target_label = "B" if label == 1 else "A"
        transformed_row = transformed_df.iloc[idx].to_numpy(dtype=float)
        contribution_items = extract_local_contributions(bundle, transformed_row, feature_df.iloc[idx])
        decision_summary = compute_model_decision_summary(bundle, transformed_row)
        selected_items = select_summary_items(contribution_items, decision_summary)
        prompt_block = render_prompt_block(contribution_items, decision_summary, selected_items=selected_items)
        reasoning_trace = render_reasoning_trace(selected_items, decision_summary, target_label, task=task, smiles=smiles)
        if reasoning_trace is None:
            numeric_filter_dropped += 1
            continue
        pos_selected = len([item for item in selected_items if item["contribution"] > 0])
        neg_selected = len([item for item in selected_items if item["contribution"] < 0])
        pos_counts.append(pos_selected)
        neg_counts.append(neg_selected)
        prompt_block_lengths.append(len(prompt_block))
        trace_lengths.append(len(reasoning_trace))
        if not contribution_items:
            missing_count += 1
        if decision_summary["final_label"] == target_label:
            agreement_count += 1

        records.append(
            {
                "task": task,
                "split": split,
                "smiles": smiles,
                "label": label,
                "answer": f"({target_label})",
                "prompt_block": prompt_block,
                "reasoning_trace": reasoning_trace,
                "selected_features": [
                    {
                        "feature": item["feature"],
                        "feature_display_name": item["feature_display_name"],
                        "description": item["description"],
                        "contribution": round(float(item["contribution"]), 4),
                    }
                    for item in selected_items
                ],
                "auxiliary_model": {
                    "base_prob_pos": round(float(decision_summary["base_prob_pos"]), 4),
                    "base_label": decision_summary["base_label"],
                    "final_prob_pos": round(float(decision_summary["final_prob_pos"]), 4),
                    "final_label": decision_summary["final_label"],
                    "agrees_with_target": decision_summary["final_label"] == target_label,
                },
                "trace_version": TRACE_SCHEMA_VERSION,
            }
        )

    stats = {
        "records": len(records),
        "avg_prompt_block_chars": float(np.mean(prompt_block_lengths)) if prompt_block_lengths else 0.0,
        "avg_trace_chars": float(np.mean(trace_lengths)) if trace_lengths else 0.0,
        "avg_positive_items": float(np.mean(pos_counts)) if pos_counts else 0.0,
        "avg_negative_items": float(np.mean(neg_counts)) if neg_counts else 0.0,
        "missing_evidence_count": int(missing_count),
        "agreement_rate": float(agreement_count / len(records)) if records else 0.0,
        "numeric_filter_dropped": int(numeric_filter_dropped),
    }
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable local reasoning-trace artifacts for v16_no_neighbor")
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
    root = sample_prompt_variant_dir(TOOL_VERSION, ARTIFACT_VARIANT)
    root.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        summary[task] = {}
        for split in args.splits:
            records, stats = build_prompt_records_for_split(task, split, raw_dir, bundles[task])
            if not records:
                continue
            out_path = write_sample_prompt_records(TOOL_VERSION, ARTIFACT_VARIANT, task, split, records)
            summary[task][split] = stats
            print(f"  {task}/{split}: {len(records)} reasoning-trace records -> {out_path}")

    manifest = {
        "tool_version": TOOL_VERSION,
        "artifact_type": ARTIFACT_VARIANT,
        "source_backend": BACKEND,
        "summary_path": str(SUMMARY_PATH),
        "tasks": tasks,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nReasoning-trace artifacts written to {root}")
    print(f"Debug summary written to {debug_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
