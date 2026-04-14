"""Versioned tool registry for GRPO training.

Versions (incremental):
  - v1: RDKit basic + AccFG
  - v2: v1 + remove_salts (standardize_tools)
  - v3: v2 + predict_pka + estimate_logd + get_3d_exposed_polar_surface
  - v4: v2 + predict_pka + estimate_logd + score_structural_alerts (no 3DEPSA)
  - v5: v4 + RAscore + predict_metabolic_sites (SyGMa)
         + predict_electronic_properties (GFN2-xTB)
  - v6: 7 consolidated tools (molecule_profile, functional_groups, ring_systems,
         adme, structural_alerts, remove_salts, predict_metabolites)
         + task-specific get_3d_properties for permeability/binding tasks
  - v7: v6 + find_similar_molecules (KNN neighbor lookup)
  - v8: v7 with predict_metabolites task-specific (CYP, DILI, Bioavailability,
         ClinTox, Carcinogens, AMES, hERG). Uses deduplicated_canonicalized
         dataset as base. predict_solubility is an internal subtool only.
  - v9: v8 + decision_tree_analysis (RF feature attributions from LLM4SD).
         Two dataset variants: v9 (no pseudo_label), v9_pseudo (with pseudo_label).
  - v10: consolidated v10 tools. Replaces the v7/v8/v9 fine-grained surface
         with get_molecular_properties plus task-specific get_similar_neighbors.

Usage::

    from openrlhf.utils.tool_versions import get_version

    ver = get_version("v3")
    schemas = ver["basic_schemas"]          # list of OpenAI tool dicts
    task_map = ver["task_specific_map"]     # {task: [extra tool dicts]}
    callables = ver["callables"]            # {name: callable}
"""

import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Global exclusion set — tool names listed here are stripped from ALL versions
# (both schemas and callables).  Edit this set to trim the tool surface.
# ---------------------------------------------------------------------------
_EXCLUDED_TOOLS: set = {
    "get_exact_molecular_weight",
}

# Tools that upstream RDKIT_BASIC_OPENAI_TOOLS bundles but we manage via
# Haydn wrappers only.  Strip them from the base import so they don't leak
# into every version; versions that want them add them explicitly from
# HAYDN_OPENAI_TOOLS.
_HAYDN_ONLY_TOOLS: set = {
    "analyze_ring_systems",
    "classify_ionization",
    "compute_similarity",
    "score_structural_alerts",
    "extract_pharmacophore_features",
    "match_substructure",
    "find_mcs",
}

_ALL_STRIPPED: set = _EXCLUDED_TOOLS | _HAYDN_ONLY_TOOLS


def _filter_schemas(schemas: List[Dict[str, Any]], *, haydn_passthrough: bool = False) -> List[Dict[str, Any]]:
    """Remove globally-excluded and (by default) Haydn-only tools.

    Set *haydn_passthrough=True* when adding Haydn tools explicitly so that
    only the global exclusion set applies.
    """
    blocked = _EXCLUDED_TOOLS if haydn_passthrough else _ALL_STRIPPED
    return [t for t in schemas if t["function"]["name"] not in blocked]


def _filter_callables(callables: Dict[str, Callable], *, haydn_passthrough: bool = False) -> Dict[str, Callable]:
    blocked = _EXCLUDED_TOOLS if haydn_passthrough else _ALL_STRIPPED
    return {k: v for k, v in callables.items() if k not in blocked}


# ---------------------------------------------------------------------------
# Lazy loader for v1-v5 (Intern-S1-recipe dependency)
# ---------------------------------------------------------------------------
_legacy_versions: Dict[str, Dict[str, Any]] | None = None


def _ensure_intern_s1_on_path() -> None:
    """Add Intern-S1-recipe/tools to sys.path so `from tools.X import ...` works."""
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _INTERN_S1_ROOT = _PROJECT_ROOT / "Intern-S1-recipe"
    assert (_INTERN_S1_ROOT / "tools").is_dir(), (
        f"Intern-S1-recipe/tools not found at {_INTERN_S1_ROOT}/tools. "
        f"Run: git submodule update --init Intern-S1-recipe"
    )
    if str(_INTERN_S1_ROOT) not in sys.path:
        sys.path.insert(0, str(_INTERN_S1_ROOT))
    _TOOLS_PATH = str(_INTERN_S1_ROOT / "tools")

    #### Python 3.11 compat: prefer tools_py311/ for patched files, fall through to tools/ ####
    _TOOLS_PY311_PATH = str(_INTERN_S1_ROOT / "tools_py311")
    _USE_PY311_COMPAT = sys.version_info < (3, 12) and os.path.isdir(_TOOLS_PY311_PATH)
    _tools_search_path = [_TOOLS_PY311_PATH, _TOOLS_PATH] if _USE_PY311_COMPAT else [_TOOLS_PATH]
    #### end Python 3.11 compat ####

    _existing_tools_pkg = sys.modules.get("tools")
    if _existing_tools_pkg is None:
        _pkg = types.ModuleType("tools")
        _pkg.__path__ = _tools_search_path
        _pkg.__package__ = "tools"
        sys.modules["tools"] = _pkg
    else:
        _existing_path = list(getattr(_existing_tools_pkg, "__path__", []))
        for _p in reversed(_tools_search_path):
            if _p not in _existing_path:
                _existing_path.insert(0, _p)
        _existing_tools_pkg.__path__ = _existing_path


def _build_legacy_versions() -> Dict[str, Dict[str, Any]]:
    """Import from Intern-S1-recipe and build v1-v5 version dicts."""
    global _legacy_versions
    if _legacy_versions is not None:
        return _legacy_versions

    _ensure_intern_s1_on_path()

    from tools.RDKit_tools import (
        RDKIT_BASIC_OPENAI_TOOLS,
        TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
        get_molecular_weight,
        get_exact_molecular_weight,
        get_heavy_atom_count,
        get_mol_logp,
        get_tpsa,
        get_hbd,
        get_hba,
        get_num_rotatable_bonds,
        get_fraction_csp3,
        get_mol_mr,
        get_ring_count,
        get_num_aromatic_rings,
        get_formal_charge,
        get_qed,
        get_num_heteroatoms,
        get_labute_asa,
        get_max_abs_partial_charge,
        get_min_abs_partial_charge,
        get_max_estate_index,
        get_min_estate_index,
        get_num_aromatic_atoms,
        get_fraction_aromatic_atoms,
        get_num_positive_charge_atoms,
        get_num_negative_charge_atoms,
        get_num_aliphatic_rings,
        get_num_saturated_rings,
        get_num_heterocycles,
        get_num_aromatic_heterocycles,
        get_num_aliphatic_heterocycles,
        get_num_saturated_heterocycles,
        get_num_amide_bonds,
        get_bertz_ct,
        get_balaban_j,
        get_ipc,
        get_hall_kier_alpha,
        get_kappa1,
        get_kappa2,
        get_kappa3,
        get_num_atom_stereo_centers,
        get_num_unspecified_atom_stereo_centers,
    )
    from tools.AccFG import AccFG_OPENAI_TOOLS, cached_describe_high_level_fg_fragments
    from tools.standardize_tools import STANDARDIZE_OPENAI_TOOLS, remove_salts

    # Optional imports
    try:
        from tools.ePSA_3D import get_3d_exposed_polar_surface, SASA_OPENAI_TOOLS
    except ImportError:
        get_3d_exposed_polar_surface = None
        SASA_OPENAI_TOOLS = []

    try:
        from tools.pka_related_tools import predict_pka, estimate_logd, PKA_TOOL, LOGD_TOOL
    except ImportError:
        predict_pka = None
        estimate_logd = None
        PKA_TOOL = None
        LOGD_TOOL = None

    try:
        from tools.rascore_tools import predict_synthesizability, RASCORE_TOOL
    except ImportError:
        predict_synthesizability = None
        RASCORE_TOOL = None

    try:
        from tools.metabolism_tools import predict_metabolic_sites, METABOLISM_TOOL
    except ImportError:
        predict_metabolic_sites = None
        METABOLISM_TOOL = None

    try:
        from tools.electronic_tools import predict_electronic_properties, ELECTRONIC_TOOL
    except ImportError:
        predict_electronic_properties = None
        ELECTRONIC_TOOL = None

    # Shared callables
    _RDKIT_ACCFG_CALLABLES: Dict[str, Callable] = {
        "describe_high_level_fg_fragments": cached_describe_high_level_fg_fragments,
        "get_molecular_weight": get_molecular_weight,
        "get_exact_molecular_weight": get_exact_molecular_weight,
        "get_heavy_atom_count": get_heavy_atom_count,
        "get_mol_logp": get_mol_logp,
        "get_tpsa": get_tpsa,
        "get_hbd": get_hbd,
        "get_hba": get_hba,
        "get_num_rotatable_bonds": get_num_rotatable_bonds,
        "get_fraction_csp3": get_fraction_csp3,
        "get_labute_asa": get_labute_asa,
        "get_mol_mr": get_mol_mr,
        "get_ring_count": get_ring_count,
        "get_num_aromatic_rings": get_num_aromatic_rings,
        "get_formal_charge": get_formal_charge,
        "get_qed": get_qed,
        "get_num_heteroatoms": get_num_heteroatoms,
        "get_max_abs_partial_charge": get_max_abs_partial_charge,
        "get_min_abs_partial_charge": get_min_abs_partial_charge,
        "get_max_estate_index": get_max_estate_index,
        "get_min_estate_index": get_min_estate_index,
        "get_num_aromatic_atoms": get_num_aromatic_atoms,
        "get_fraction_aromatic_atoms": get_fraction_aromatic_atoms,
        "get_num_positive_charge_atoms": get_num_positive_charge_atoms,
        "get_num_negative_charge_atoms": get_num_negative_charge_atoms,
        "get_num_aliphatic_rings": get_num_aliphatic_rings,
        "get_num_saturated_rings": get_num_saturated_rings,
        "get_num_heterocycles": get_num_heterocycles,
        "get_num_aromatic_heterocycles": get_num_aromatic_heterocycles,
        "get_num_aliphatic_heterocycles": get_num_aliphatic_heterocycles,
        "get_num_saturated_heterocycles": get_num_saturated_heterocycles,
        "get_num_amide_bonds": get_num_amide_bonds,
        "get_bertz_ct": get_bertz_ct,
        "get_balaban_j": get_balaban_j,
        "get_ipc": get_ipc,
        "get_hall_kier_alpha": get_hall_kier_alpha,
        "get_kappa1": get_kappa1,
        "get_kappa2": get_kappa2,
        "get_kappa3": get_kappa3,
        "get_num_atom_stereo_centers": get_num_atom_stereo_centers,
        "get_num_unspecified_atom_stereo_centers": get_num_unspecified_atom_stereo_centers,
    }

    # Version schemas
    _V1_SCHEMAS = _filter_schemas(RDKIT_BASIC_OPENAI_TOOLS + AccFG_OPENAI_TOOLS)
    _V2_SCHEMAS = _V1_SCHEMAS + _filter_schemas(STANDARDIZE_OPENAI_TOOLS)

    _V3_EXTRA_SCHEMAS: List[Dict[str, Any]] = []
    if PKA_TOOL is not None:
        _V3_EXTRA_SCHEMAS.append(PKA_TOOL)
    if LOGD_TOOL is not None:
        _V3_EXTRA_SCHEMAS.append(LOGD_TOOL)
    _V3_SCHEMAS = _V2_SCHEMAS + _filter_schemas(_V3_EXTRA_SCHEMAS + SASA_OPENAI_TOOLS)

    # Haydn wrappers
    try:
        from openrlhf.utils.haydn_wrappers import HAYDN_OPENAI_TOOLS, HAYDN_CALLABLES
    except ImportError:
        HAYDN_OPENAI_TOOLS = []
        HAYDN_CALLABLES = {}

    # v4
    if PKA_TOOL is None or LOGD_TOOL is None:
        raise ImportError(
            "v4 requires predict_pka and estimate_logd but molgpka is not installed. "
            "Install it with: pip install molgpka"
        )
    _V4_HAYDN_NAMES = {"score_structural_alerts"}
    _V4_EXTRA_SCHEMAS = [PKA_TOOL, LOGD_TOOL]
    _V4_SCHEMAS = _V2_SCHEMAS + _filter_schemas(
        _V4_EXTRA_SCHEMAS + [
            t for t in HAYDN_OPENAI_TOOLS if t["function"]["name"] in _V4_HAYDN_NAMES
        ],
        haydn_passthrough=True,
    )

    # Version callables
    _V1_CALLABLES = _filter_callables(_RDKIT_ACCFG_CALLABLES)
    _V2_CALLABLES = _filter_callables({**_V1_CALLABLES, "remove_salts": remove_salts})

    _V3_CALLABLES: Dict[str, Callable] = dict(_V2_CALLABLES)
    if predict_pka is not None:
        _V3_CALLABLES["predict_pka"] = predict_pka
    if estimate_logd is not None:
        _V3_CALLABLES["estimate_logd"] = estimate_logd
    if get_3d_exposed_polar_surface is not None:
        _V3_CALLABLES["get_3d_exposed_polar_surface"] = get_3d_exposed_polar_surface

    _V4_CALLABLES: Dict[str, Callable] = {
        **_V2_CALLABLES,
        "predict_pka": predict_pka,
        "estimate_logd": estimate_logd,
        **{k: v for k, v in HAYDN_CALLABLES.items() if k in _V4_HAYDN_NAMES},
    }

    # v5
    _V5_EXTRA_SCHEMAS: List[Dict[str, Any]] = []
    for _tool_schema in [RASCORE_TOOL, METABOLISM_TOOL, ELECTRONIC_TOOL]:
        if _tool_schema is not None:
            _V5_EXTRA_SCHEMAS.append(_tool_schema)
    _V5_SCHEMAS = _V4_SCHEMAS + _filter_schemas(_V5_EXTRA_SCHEMAS)

    _V5_CALLABLES: Dict[str, Callable] = dict(_V4_CALLABLES)
    for _name, _fn in [
        ("predict_synthesizability", predict_synthesizability),
        ("predict_metabolic_sites", predict_metabolic_sites),
        ("predict_electronic_properties", predict_electronic_properties),
    ]:
        if _fn is not None:
            _V5_CALLABLES[_name] = _fn

    _legacy_versions = {
        "v1": {
            "basic_schemas": _V1_SCHEMAS,
            "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
            "callables": _V1_CALLABLES,
        },
        "v2": {
            "basic_schemas": _V2_SCHEMAS,
            "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
            "callables": _V2_CALLABLES,
        },
        "v3": {
            "basic_schemas": _V3_SCHEMAS,
            "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
            "callables": _V3_CALLABLES,
        },
        "v4": {
            "basic_schemas": _V4_SCHEMAS,
            "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
            "callables": _V4_CALLABLES,
        },
        "v5": {
            "basic_schemas": _V5_SCHEMAS,
            "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
            "callables": _V5_CALLABLES,
        },
    }
    return _legacy_versions


# ---------------------------------------------------------------------------
# v6: 7 core tools + task-specific 3D (no Intern-S1 dep)
# ---------------------------------------------------------------------------
from openrlhf.tools.therapeutic_tools import (
    GET_MOLECULE_PROFILE_TOOL as _MOLECULE_PROFILE_SCHEMA,
    ANALYZE_FUNCTIONAL_GROUPS_TOOL as _FUNCTIONAL_GROUPS_SCHEMA,
    ANALYZE_RING_SYSTEMS_TOOL as _RING_SYSTEMS_SCHEMA,
    ASSESS_ADME_PROPERTIES_TOOL as _ADME_SCHEMA,
    SCREEN_STRUCTURAL_ALERTS_TOOL as _STRUCTURAL_ALERTS_SCHEMA,
    REMOVE_SALTS_TOOL as _REMOVE_SALTS_SCHEMA,
    PREDICT_METABOLITES_TOOL as _PREDICT_METABOLITES_SCHEMA,
    PREDICT_SOLUBILITY_TOOL as _PREDICT_SOLUBILITY_SCHEMA,
    DECISION_TREE_ANALYSIS_TOOL as _DECISION_TREE_SCHEMA,
    GET_3D_PROPERTIES_TOOL as _3D_PROPERTIES_SCHEMA,
    FIND_SIMILAR_MOLECULES_TOOL as _FIND_SIMILAR_SCHEMA,
    SIMILAR_MOLECULES_TASK_SCHEMAS as _SIMILAR_TASK_SCHEMAS,
    GET_MOLECULAR_PROPERTIES_TOOL as _GET_MOLECULAR_PROPERTIES_SCHEMA,
    V10_TASK_NEIGHBOR_TOOL_SCHEMAS as _V10_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V10_TASK_NEIGHBOR_CALLABLES as _V10_TASK_NEIGHBOR_CALLABLES,
    GET_FEATURES_TOOL as _GET_FEATURES_SCHEMA,
    GET_NEIGHBORS_TOOL as _GET_NEIGHBORS_SCHEMA,
    V11_TASK_NEIGHBOR_TOOL_SCHEMAS as _V11_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V11_TASK_NEIGHBOR_CALLABLES as _V11_TASK_NEIGHBOR_CALLABLES,
    V12_GET_FEATURES_TOOL as _V12_GET_FEATURES_SCHEMA,
    V12_GET_NEIGHBORS_TOOL as _V12_GET_NEIGHBORS_SCHEMA,
    V12_TASK_NEIGHBOR_TOOL_SCHEMAS as _V12_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V12_TASK_NEIGHBOR_CALLABLES as _V12_TASK_NEIGHBOR_CALLABLES,
    v12_get_features as _V12_GET_FEATURES_CALLABLE,
    v12_get_neighbors as _V12_GET_NEIGHBORS_CALLABLE,
    V13_GET_FEATURES_TOOL as _V13_GET_FEATURES_SCHEMA,
    V13_GET_NEIGHBORS_TOOL as _V13_GET_NEIGHBORS_SCHEMA,
    V13_TASK_NEIGHBOR_TOOL_SCHEMAS as _V13_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V13_TASK_NEIGHBOR_CALLABLES as _V13_TASK_NEIGHBOR_CALLABLES,
    v13_get_features as _V13_GET_FEATURES_CALLABLE,
    v13_get_neighbors as _V13_GET_NEIGHBORS_CALLABLE,
    _FUNCTION_MAP as _ALL_CALLABLES,
)

# v6 basic: 7 tools (no find_similar_molecules, no get_3d_properties)
_V6_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _MOLECULE_PROFILE_SCHEMA,
    _FUNCTIONAL_GROUPS_SCHEMA,
    _RING_SYSTEMS_SCHEMA,
    _ADME_SCHEMA,
    _STRUCTURAL_ALERTS_SCHEMA,
    _REMOVE_SALTS_SCHEMA,
    _PREDICT_METABOLITES_SCHEMA,
]

# Tasks where 3D conformational properties (ePSA, PMI shape) are informative.
# These involve membrane permeability, oral absorption, or 3D binding-site interactions.
_TASKS_WITH_3D = {
    "Bioavailability_Ma",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "BBB_Martins",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
}

_V6_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    task: [_3D_PROPERTIES_SCHEMA] for task in _TASKS_WITH_3D
}

# v6 callables: everything except find_similar_molecules
_V6_CALLABLES: Dict[str, Callable] = {
    k: v for k, v in _ALL_CALLABLES.items() if k != "find_similar_molecules"
}

_V6_VERSION = {
    "basic_schemas": _V6_BASIC_SCHEMAS,
    "task_specific_map": _V6_TASK_MAP,
    "callables": _V6_CALLABLES,
}

# ---------------------------------------------------------------------------
# v7: v6 + per-task find_similar_molecules_{task}
# ---------------------------------------------------------------------------
_V7_BASIC_SCHEMAS: List[Dict[str, Any]] = list(_V6_BASIC_SCHEMAS)  # no generic find_similar in basic

# Merge 3D and per-task similarity schemas into one task map
_V7_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}
for _task in set(list(_TASKS_WITH_3D) + list(_SIMILAR_TASK_SCHEMAS)):
    _extras: List[Dict[str, Any]] = []
    if _task in _TASKS_WITH_3D:
        _extras.append(_3D_PROPERTIES_SCHEMA)
    if _task in _SIMILAR_TASK_SCHEMAS:
        _extras.append(_SIMILAR_TASK_SCHEMAS[_task])
    _V7_TASK_MAP[_task] = _extras

_V7_VERSION = {
    "basic_schemas": _V7_BASIC_SCHEMAS,
    "task_specific_map": _V7_TASK_MAP,
    "callables": dict(_ALL_CALLABLES),  # includes all find_similar_molecules_{task} callables
}

# ---------------------------------------------------------------------------
# v8: v7 with predict_metabolites task-specific (predict_solubility is internal only)
# ---------------------------------------------------------------------------
_V8_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _MOLECULE_PROFILE_SCHEMA,
    _FUNCTIONAL_GROUPS_SCHEMA,
    _RING_SYSTEMS_SCHEMA,
    _ADME_SCHEMA,
    _STRUCTURAL_ALERTS_SCHEMA,
    _REMOVE_SALTS_SCHEMA,
]

# Tasks where metabolic fate is informative (CYP substrates, hepatotox,
# metabolic activation → mutagenicity/carcinogenicity, bioavailability).
_TASKS_WITH_METABOLISM = {
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "DILI",
    "Bioavailability_Ma",
    "ClinTox",
    "Carcinogens_Lagunin",
    "AMES",
    "hERG",
}

_V8_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}
for _task in set(list(_TASKS_WITH_3D) + list(_SIMILAR_TASK_SCHEMAS) + list(_TASKS_WITH_METABOLISM)):
    _extras: List[Dict[str, Any]] = []
    if _task in _TASKS_WITH_3D:
        _extras.append(_3D_PROPERTIES_SCHEMA)
    if _task in _TASKS_WITH_METABOLISM:
        _extras.append(_PREDICT_METABOLITES_SCHEMA)
    if _task in _SIMILAR_TASK_SCHEMAS:
        _extras.append(_SIMILAR_TASK_SCHEMAS[_task])
    _V8_TASK_MAP[_task] = _extras

_V8_CALLABLES: Dict[str, Callable] = dict(_ALL_CALLABLES)

_V8_VERSION = {
    "basic_schemas": _V8_BASIC_SCHEMAS,
    "task_specific_map": _V8_TASK_MAP,
    "callables": _V8_CALLABLES,
}

# ---------------------------------------------------------------------------
# v9: v8 + decision_tree_analysis (RF feature attributions)
# ---------------------------------------------------------------------------
_V9_BASIC_SCHEMAS: List[Dict[str, Any]] = _V8_BASIC_SCHEMAS + [_DECISION_TREE_SCHEMA]

_V9_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    _task: _extras + [_DECISION_TREE_SCHEMA]
    for _task, _extras in _V8_TASK_MAP.items()
}

_V9_CALLABLES: Dict[str, Callable] = dict(_ALL_CALLABLES)

_V9_VERSION = {
    "basic_schemas": _V9_BASIC_SCHEMAS,
    "task_specific_map": _V9_TASK_MAP,
    "callables": _V9_CALLABLES,
}

# ---------------------------------------------------------------------------
# v10: consolidated molecular summary + task-specific neighbor lookup
# ---------------------------------------------------------------------------
_V10_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _GET_MOLECULAR_PROPERTIES_SCHEMA,
]

_V10_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    _task: [_schema] for _task, _schema in _V10_TASK_NEIGHBOR_TOOL_SCHEMAS.items()
}

_V10_CALLABLES: Dict[str, Callable] = {
    "get_molecular_properties": _ALL_CALLABLES["get_molecular_properties"],
    **_V10_TASK_NEIGHBOR_CALLABLES,
}

_V10_VERSION = {
    "basic_schemas": _V10_BASIC_SCHEMAS,
    "task_specific_map": _V10_TASK_MAP,
    "callables": _V10_CALLABLES,
}

# ---------------------------------------------------------------------------
# v11: generalized get_features + get_neighbors (task as parameter)
# ---------------------------------------------------------------------------
_V11_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _GET_FEATURES_SCHEMA,
]

_V11_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    _task: [_schema] for _task, _schema in _V11_TASK_NEIGHBOR_TOOL_SCHEMAS.items()
}

_V11_CALLABLES: Dict[str, Callable] = {
    "get_features": _ALL_CALLABLES["get_features"],
    "get_neighbors": _ALL_CALLABLES["get_neighbors"],
    **_V11_TASK_NEIGHBOR_CALLABLES,
}

_V11_VERSION = {
    "basic_schemas": _V11_BASIC_SCHEMAS,
    "task_specific_map": _V11_TASK_MAP,
    "callables": _V11_CALLABLES,
}

# ---------------------------------------------------------------------------
# v12: v11 structure with granular physicochemical property features
# ---------------------------------------------------------------------------
_V12_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V12_GET_FEATURES_SCHEMA,
]

_V12_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    _task: [_schema] for _task, _schema in _V12_TASK_NEIGHBOR_TOOL_SCHEMAS.items()
}

_V12_CALLABLES: Dict[str, Callable] = {
    "get_features": _V12_GET_FEATURES_CALLABLE,
    "get_neighbors": _V12_GET_NEIGHBORS_CALLABLE,
    **_V12_TASK_NEIGHBOR_CALLABLES,
}

_V12_VERSION = {
    "basic_schemas": _V12_BASIC_SCHEMAS,
    "task_specific_map": _V12_TASK_MAP,
    "callables": _V12_CALLABLES,
}

# ---------------------------------------------------------------------------
# v13: v12 structure with top-20 SFT-backed RDKit descriptor additions
# ---------------------------------------------------------------------------
_V13_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V13_GET_FEATURES_SCHEMA,
]

_V13_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    _task: [_schema] for _task, _schema in _V13_TASK_NEIGHBOR_TOOL_SCHEMAS.items()
}

_V13_CALLABLES: Dict[str, Callable] = {
    "get_features": _V13_GET_FEATURES_CALLABLE,
    "get_neighbors": _V13_GET_NEIGHBORS_CALLABLE,
    **_V13_TASK_NEIGHBOR_CALLABLES,
}

_V13_VERSION = {
    "basic_schemas": _V13_BASIC_SCHEMAS,
    "task_specific_map": _V13_TASK_MAP,
    "callables": _V13_CALLABLES,
}

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
_ALL_VERSIONS = {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13"}

# Keep TOOL_VERSIONS for backwards compat but populate lazily
TOOL_VERSIONS: Dict[str, Dict[str, Any]] = {
    "v6": _V6_VERSION,
    "v7": _V7_VERSION,
    "v8": _V8_VERSION,
    "v9": _V9_VERSION,
    "v10": _V10_VERSION,
    "v11": _V11_VERSION,
    "v12": _V12_VERSION,
    "v13": _V13_VERSION,
}


def get_version(ver: str) -> Dict[str, Any]:
    """Return the tool version config dict, or raise ValueError."""
    if ver not in _ALL_VERSIONS:
        raise ValueError(
            f"Unknown tool version {ver!r}. "
            f"Available: {sorted(_ALL_VERSIONS)}"
        )
    if ver in ("v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13"):
        return TOOL_VERSIONS[ver]
    # Lazy-load legacy versions on first access
    legacy = _build_legacy_versions()
    TOOL_VERSIONS.update(legacy)
    return legacy[ver]


__all__ = ["TOOL_VERSIONS", "get_version"]
