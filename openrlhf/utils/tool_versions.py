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
  - v14: v12 feature vocabulary plus extra cache-backed scalars; ``metabolites``
         removed from get_features. Does NOT include v13 SFT descriptors.
  - v14_no_neighbor: same ``get_features`` as v14; no ``get_neighbors`` /
    ``get_neighbors_*`` tools.
  - v14_consolidated: v11-style grouped feature buckets, v14 scalar additions
    folded into the consolidated surface, and no ``metabolites`` tool.
  - v14_consolidated_no_neighbor: same ``get_features`` as
    ``v14_consolidated``; no ``get_neighbors`` / ``get_neighbors_*`` tools.
  - v15: TRIM-style tools with a stable OpenAI schema surface:
         ``get_mol_properties_and_fg`` + ``compare_similar_mols``.
         Task is injected dynamically at execution time from trace metadata.
  - v15_no_neighbor: v15 single-molecule property evidence only.
  - v15_neighbor_only: v15 local-analog comparison only.
  - v15_neighbor_only_4: v15 local-analog comparison only, but with 4 total
    neighbors (2 positive + 2 negative).
  - v16: four compact grouped buckets over the v14_consolidated evidence
         surface, with a single task-aware ``get_neighbors`` tool.
  - v16_no_neighbor: same ``get_features`` as ``v16``; no ``get_neighbors``
     tool.
  - v17: v16-style grouped ``get_features`` plus the analogue-quality
    ``get_neighbors`` tool.
  - v17_neighbor_only: v17 analogue-quality ``get_neighbors`` only.
  - v17_get_features_only: v17 grouped ``get_features`` only.

Usage::

    from openrlhf.utils.tool_versions import get_version

    ver = get_version("v3")
    schemas = ver["basic_schemas"]          # list of OpenAI tool dicts
    task_map = ver["task_specific_map"]     # {task: [extra tool dicts]}
    callables = ver["callables"]            # {name: callable}
"""

import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


def _schema_tool_name(schema: Dict[str, Any]) -> str:
    if "function" in schema and isinstance(schema["function"], dict):
        return str(schema["function"].get("name", ""))
    return str(schema.get("name", ""))


def _filter_schemas(schemas: List[Dict[str, Any]], *, haydn_passthrough: bool = False) -> List[Dict[str, Any]]:
    """Remove globally-excluded and (by default) Haydn-only tools.

    Set *haydn_passthrough=True* when adding Haydn tools explicitly so that
    only the global exclusion set applies.
    """
    blocked = _EXCLUDED_TOOLS if haydn_passthrough else _ALL_STRIPPED
    return [t for t in schemas if _schema_tool_name(t) not in blocked]


def _filter_callables(callables: Dict[str, Callable], *, haydn_passthrough: bool = False) -> Dict[str, Callable]:
    blocked = _EXCLUDED_TOOLS if haydn_passthrough else _ALL_STRIPPED
    return {k: v for k, v in callables.items() if k not in blocked}


def _tool_name_matches_selector(tool_name: str, selector: str) -> bool:
    """Match a tool name against an exact name or '*' suffix prefix selector."""
    if selector.endswith("*"):
        return tool_name.startswith(selector[:-1])
    return tool_name == selector


def filter_schemas_by_selectors(
    schemas: List[Dict[str, Any]],
    selectors: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Filter OpenAI tool schemas by name selectors.

    Selectors support exact tool names as well as prefix wildcards such as
    ``get_neighbors*`` to capture task-specific variants.
    """
    if not selectors:
        return list(schemas)
    return [
        schema
        for schema in schemas
        if any(_tool_name_matches_selector(_schema_tool_name(schema), selector) for selector in selectors)
    ]


def filter_callables_by_selectors(
    callables: Dict[str, Callable],
    selectors: Optional[List[str]],
) -> Dict[str, Callable]:
    if not selectors:
        return dict(callables)
    return {
        name: fn
        for name, fn in callables.items()
        if any(_tool_name_matches_selector(name, selector) for selector in selectors)
    }


def resolve_tool_metric_metadata(version_or_cfg: str | Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """Resolve registry-declared metric metadata for a tool.

    Versions can optionally declare ``metric_endpoints`` as a list of records
    with:
      - ``selectors``: exact names or ``prefix*`` selectors
      - ``endpoint``: logical endpoint family such as ``features`` or
        ``neighbors``
      - ``count_request_metric``: whether calls should contribute to the
        request-rate metrics (defaults to True)
    """
    version_cfg = get_version(version_or_cfg) if isinstance(version_or_cfg, str) else version_or_cfg
    metric_endpoints = version_cfg.get("metric_endpoints") or []
    for entry in metric_endpoints:
        selectors = entry.get("selectors") or []
        if any(_tool_name_matches_selector(tool_name, selector) for selector in selectors):
            return {
                "endpoint": entry.get("endpoint"),
                "count_request_metric": bool(entry.get("count_request_metric", True)),
            }
    return None


def resolve_tool_metric_endpoint(version_or_cfg: str | Dict[str, Any], tool_name: str) -> Optional[str]:
    """Return the logical metric endpoint family for a tool name."""
    metadata = resolve_tool_metric_metadata(version_or_cfg, tool_name)
    if metadata is None:
        return None
    return metadata.get("endpoint")


def resolve_phase_spec(
    version_cfg: Dict[str, Any],
    current_global_step: int,
    total_training_steps: int,
) -> Optional[Dict[str, Any]]:
    """Resolve the active phase spec for the current training step.

    Phase specs are optional and version-specific. Each phase can declare:
      - ``start_frac``: inclusive lower bound in [0, 1]
      - ``end_frac``: exclusive upper bound in [0, 1], except the last phase
      - ``tool_selectors``: exact names or ``prefix*`` selectors
      - ``hidden_instruction``: extra hidden instruction injected only in phase
    """
    policy = version_cfg.get("phase_policy")
    if not policy:
        return None

    phases = policy.get("phases") or []
    if not phases:
        return None

    if total_training_steps is None or total_training_steps <= 0:
        progress = 0.0
    else:
        clamped_step = min(max(int(current_global_step), 0), int(total_training_steps))
        progress = clamped_step / max(int(total_training_steps), 1)

    last_phase = phases[-1]
    for phase in phases:
        start_frac = float(phase.get("start_frac", 0.0))
        end_frac = float(phase.get("end_frac", 1.0))
        is_last = phase is last_phase
        if progress >= start_frac and (progress < end_frac or (is_last and math.isclose(progress, end_frac))):
            return phase

    return last_phase


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
    V14_GET_FEATURES_TOOL as _V14_GET_FEATURES_SCHEMA,
    V14_GET_NEIGHBORS_TOOL as _V14_GET_NEIGHBORS_SCHEMA,
    V14_TASK_NEIGHBOR_TOOL_SCHEMAS as _V14_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V14_TASK_NEIGHBOR_CALLABLES as _V14_TASK_NEIGHBOR_CALLABLES,
    v14_get_features as _V14_GET_FEATURES_CALLABLE,
    v14_get_neighbors as _V14_GET_NEIGHBORS_CALLABLE,
    V14_NO_NEIGHBOR_GET_FEATURES_TOOL as _V14_NO_NEIGHBOR_GET_FEATURES_SCHEMA,
    v14_no_neighbor_get_features as _V14_NO_NEIGHBOR_GET_FEATURES_CALLABLE,
    V14_CONSOLIDATED_GET_FEATURES_TOOL as _V14_CONSOLIDATED_GET_FEATURES_SCHEMA,
    V14_CONSOLIDATED_GET_NEIGHBORS_TOOL as _V14_CONSOLIDATED_GET_NEIGHBORS_SCHEMA,
    V14_CONSOLIDATED_TASK_NEIGHBOR_TOOL_SCHEMAS as _V14_CONSOLIDATED_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V14_CONSOLIDATED_TASK_NEIGHBOR_CALLABLES as _V14_CONSOLIDATED_TASK_NEIGHBOR_CALLABLES,
    v14_consolidated_get_features as _V14_CONSOLIDATED_GET_FEATURES_CALLABLE,
    v14_consolidated_get_neighbors as _V14_CONSOLIDATED_GET_NEIGHBORS_CALLABLE,
    V14_CONSOLIDATED_NO_NEIGHBOR_GET_FEATURES_TOOL as _V14_CONSOLIDATED_NO_NEIGHBOR_GET_FEATURES_SCHEMA,
    v14_consolidated_no_neighbor_get_features as _V14_CONSOLIDATED_NO_NEIGHBOR_GET_FEATURES_CALLABLE,
    V15_GET_MOL_PROPERTIES_AND_FG_TOOL as _V15_GET_MOL_PROPERTIES_AND_FG_SCHEMA,
    V15_COMPARE_SIMILAR_MOLS_TOOL as _V15_COMPARE_SIMILAR_MOLS_SCHEMA,
    V15_NO_NEIGHBOR_GET_MOL_PROPERTIES_AND_FG_TOOL as _V15_NO_NEIGHBOR_GET_MOL_PROPERTIES_AND_FG_SCHEMA,
    V15_NEIGHBOR_ONLY_COMPARE_SIMILAR_MOLS_TOOL as _V15_NEIGHBOR_ONLY_COMPARE_SIMILAR_MOLS_SCHEMA,
    V15_NEIGHBOR_ONLY_4_COMPARE_SIMILAR_MOLS_TOOL as _V15_NEIGHBOR_ONLY_4_COMPARE_SIMILAR_MOLS_SCHEMA,
    v15_no_neighbor_get_mol_properties_and_fg as _V15_NO_NEIGHBOR_GET_MOL_PROPERTIES_AND_FG_CALLABLE,
    v15_neighbor_only_compare_similar_mols as _V15_NEIGHBOR_ONLY_COMPARE_SIMILAR_MOLS_CALLABLE,
    v15_neighbor_only_4_compare_similar_mols as _V15_NEIGHBOR_ONLY_4_COMPARE_SIMILAR_MOLS_CALLABLE,
    V16_GET_FEATURES_TOOL as _V16_GET_FEATURES_SCHEMA,
    V16_GET_NEIGHBORS_TOOL as _V16_GET_NEIGHBORS_SCHEMA,
    V16_TASK_NEIGHBOR_TOOL_SCHEMAS as _V16_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V16_TASK_NEIGHBOR_CALLABLES as _V16_TASK_NEIGHBOR_CALLABLES,
    v16_get_features as _V16_GET_FEATURES_CALLABLE,
    v16_get_neighbors as _V16_GET_NEIGHBORS_CALLABLE,
    V16_NO_NEIGHBOR_GET_FEATURES_TOOL as _V16_NO_NEIGHBOR_GET_FEATURES_SCHEMA,
    v16_no_neighbor_get_features as _V16_NO_NEIGHBOR_GET_FEATURES_CALLABLE,
    V17_GET_FEATURES_TOOL as _V17_GET_FEATURES_SCHEMA,
    V17_GET_NEIGHBORS_TOOL as _V17_GET_NEIGHBORS_SCHEMA,
    V17_TASK_NEIGHBOR_TOOL_SCHEMAS as _V17_TASK_NEIGHBOR_TOOL_SCHEMAS,
    V17_TASK_NEIGHBOR_CALLABLES as _V17_TASK_NEIGHBOR_CALLABLES,
    v17_get_features as _V17_GET_FEATURES_CALLABLE,
    v17_get_neighbors as _V17_GET_NEIGHBORS_CALLABLE,
    V17_GET_FEATURES_ONLY_GET_FEATURES_TOOL as _V17_GET_FEATURES_ONLY_GET_FEATURES_SCHEMA,
    v17_get_features_only_get_features as _V17_GET_FEATURES_ONLY_GET_FEATURES_CALLABLE,
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
    "metric_endpoints": [
        {"selectors": ["find_similar_molecules*"], "endpoint": "neighbors"},
    ],
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
    "metric_endpoints": [
        {"selectors": ["find_similar_molecules*"], "endpoint": "neighbors"},
    ],
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
    "metric_endpoints": [
        {"selectors": ["find_similar_molecules*"], "endpoint": "neighbors"},
    ],
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
    "metric_endpoints": [
        {"selectors": ["get_molecular_properties"], "endpoint": "features", "count_request_metric": False},
        {"selectors": ["get_similar_neighbors*"], "endpoint": "neighbors"},
    ],
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
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
        {"selectors": ["get_neighbors*"], "endpoint": "neighbors"},
    ],
    "phase_policy": {
        "phases": [
            {
                "name": "features_only",
                "start_frac": 0.0,
                "end_frac": 0.5,
                "tool_selectors": ["get_features"],
            },
            {
                "name": "features_plus_neighbors",
                "start_frac": 0.5,
                "end_frac": 1.0,
                "tool_selectors": ["get_features", "get_neighbors*"],
                "hidden_instruction": (
                    "Please inspect similar training-set neighbors and compare their labels and "
                    "feature patterns before deciding on the final answer."
                ),
            },
        ]
    },
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
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
        {"selectors": ["get_neighbors*"], "endpoint": "neighbors"},
    ],
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
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
        {"selectors": ["get_neighbors*"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v14: v12 vocabulary + extra cache-backed scalars; metabolites removed; no v13 SFT descriptors
# ---------------------------------------------------------------------------
_V14_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V14_GET_FEATURES_SCHEMA,
]

_V14_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    _task: [_schema] for _task, _schema in _V14_TASK_NEIGHBOR_TOOL_SCHEMAS.items()
}

_V14_CALLABLES: Dict[str, Callable] = {
    "get_features": _V14_GET_FEATURES_CALLABLE,
    "get_neighbors": _V14_GET_NEIGHBORS_CALLABLE,
    **_V14_TASK_NEIGHBOR_CALLABLES,
}

_V14_VERSION = {
    "basic_schemas": _V14_BASIC_SCHEMAS,
    "task_specific_map": _V14_TASK_MAP,
    "callables": _V14_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
        {"selectors": ["get_neighbors*"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v14_consolidated: v11-style grouped feature buckets over the v14 surface
# ---------------------------------------------------------------------------
_V14_CONSOLIDATED_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V14_CONSOLIDATED_GET_FEATURES_SCHEMA,
]

_V14_CONSOLIDATED_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {
    _task: [_schema] for _task, _schema in _V14_CONSOLIDATED_TASK_NEIGHBOR_TOOL_SCHEMAS.items()
}

_V14_CONSOLIDATED_CALLABLES: Dict[str, Callable] = {
    "get_features": _V14_CONSOLIDATED_GET_FEATURES_CALLABLE,
    "get_neighbors": _V14_CONSOLIDATED_GET_NEIGHBORS_CALLABLE,
    **_V14_CONSOLIDATED_TASK_NEIGHBOR_CALLABLES,
}

_V14_CONSOLIDATED_VERSION = {
    "basic_schemas": _V14_CONSOLIDATED_BASIC_SCHEMAS,
    "task_specific_map": _V14_CONSOLIDATED_TASK_MAP,
    "callables": _V14_CONSOLIDATED_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
        {"selectors": ["get_neighbors*"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v14_no_neighbor: v14 get_features only (no per-task neighbor tools)
# ---------------------------------------------------------------------------
_V14_NO_NEIGHBOR_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V14_NO_NEIGHBOR_GET_FEATURES_SCHEMA,
]

_V14_NO_NEIGHBOR_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V14_NO_NEIGHBOR_CALLABLES: Dict[str, Callable] = {
    "get_features": _V14_NO_NEIGHBOR_GET_FEATURES_CALLABLE,
}

_V14_NO_NEIGHBOR_VERSION = {
    "basic_schemas": _V14_NO_NEIGHBOR_BASIC_SCHEMAS,
    "task_specific_map": _V14_NO_NEIGHBOR_TASK_MAP,
    "callables": _V14_NO_NEIGHBOR_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
    ],
}

# ---------------------------------------------------------------------------
# v14_consolidated_no_neighbor: v14_consolidated get_features only
# ---------------------------------------------------------------------------
_V14_CONSOLIDATED_NO_NEIGHBOR_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V14_CONSOLIDATED_NO_NEIGHBOR_GET_FEATURES_SCHEMA,
]

_V14_CONSOLIDATED_NO_NEIGHBOR_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V14_CONSOLIDATED_NO_NEIGHBOR_CALLABLES: Dict[str, Callable] = {
    "get_features": _V14_CONSOLIDATED_NO_NEIGHBOR_GET_FEATURES_CALLABLE,
}

_V14_CONSOLIDATED_NO_NEIGHBOR_VERSION = {
    "basic_schemas": _V14_CONSOLIDATED_NO_NEIGHBOR_BASIC_SCHEMAS,
    "task_specific_map": _V14_CONSOLIDATED_NO_NEIGHBOR_TASK_MAP,
    "callables": _V14_CONSOLIDATED_NO_NEIGHBOR_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
    ],
}

# ---------------------------------------------------------------------------
# v15: TRIM-style stable property + local-analog tools
# ---------------------------------------------------------------------------
_V15_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V15_GET_MOL_PROPERTIES_AND_FG_SCHEMA,
    _V15_COMPARE_SIMILAR_MOLS_SCHEMA,
]

_V15_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V15_CALLABLES: Dict[str, Callable] = {
    "get_mol_properties_and_fg": _ALL_CALLABLES["get_mol_properties_and_fg"],
    "compare_similar_mols": _ALL_CALLABLES["compare_similar_mols"],
}

_V15_VERSION = {
    "basic_schemas": _V15_BASIC_SCHEMAS,
    "task_specific_map": _V15_TASK_MAP,
    "callables": _V15_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_mol_properties_and_fg"], "endpoint": "features"},
        {"selectors": ["compare_similar_mols"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v15_no_neighbor: v15 property evidence only
# ---------------------------------------------------------------------------
_V15_NO_NEIGHBOR_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V15_NO_NEIGHBOR_GET_MOL_PROPERTIES_AND_FG_SCHEMA,
]

_V15_NO_NEIGHBOR_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V15_NO_NEIGHBOR_CALLABLES: Dict[str, Callable] = {
    "get_mol_properties_and_fg": _V15_NO_NEIGHBOR_GET_MOL_PROPERTIES_AND_FG_CALLABLE,
}

_V15_NO_NEIGHBOR_VERSION = {
    "basic_schemas": _V15_NO_NEIGHBOR_BASIC_SCHEMAS,
    "task_specific_map": _V15_NO_NEIGHBOR_TASK_MAP,
    "callables": _V15_NO_NEIGHBOR_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_mol_properties_and_fg"], "endpoint": "features"},
    ],
}

# ---------------------------------------------------------------------------
# v15_neighbor_only: v15 local-analog comparison only
# ---------------------------------------------------------------------------
_V15_NEIGHBOR_ONLY_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V15_NEIGHBOR_ONLY_COMPARE_SIMILAR_MOLS_SCHEMA,
]

_V15_NEIGHBOR_ONLY_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V15_NEIGHBOR_ONLY_CALLABLES: Dict[str, Callable] = {
    "compare_similar_mols": _V15_NEIGHBOR_ONLY_COMPARE_SIMILAR_MOLS_CALLABLE,
}

_V15_NEIGHBOR_ONLY_VERSION = {
    "basic_schemas": _V15_NEIGHBOR_ONLY_BASIC_SCHEMAS,
    "task_specific_map": _V15_NEIGHBOR_ONLY_TASK_MAP,
    "callables": _V15_NEIGHBOR_ONLY_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["compare_similar_mols"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v15_neighbor_only_4: v15 local-analog comparison only, 2 pos + 2 neg
# ---------------------------------------------------------------------------
_V15_NEIGHBOR_ONLY_4_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V15_NEIGHBOR_ONLY_4_COMPARE_SIMILAR_MOLS_SCHEMA,
]

_V15_NEIGHBOR_ONLY_4_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V15_NEIGHBOR_ONLY_4_CALLABLES: Dict[str, Callable] = {
    "compare_similar_mols": _V15_NEIGHBOR_ONLY_4_COMPARE_SIMILAR_MOLS_CALLABLE,
}

_V15_NEIGHBOR_ONLY_4_VERSION = {
    "basic_schemas": _V15_NEIGHBOR_ONLY_4_BASIC_SCHEMAS,
    "task_specific_map": _V15_NEIGHBOR_ONLY_4_TASK_MAP,
    "callables": _V15_NEIGHBOR_ONLY_4_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["compare_similar_mols"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v16: four compact grouped buckets over the v14_consolidated surface
# ---------------------------------------------------------------------------
_V16_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V16_GET_FEATURES_SCHEMA,
    _V16_GET_NEIGHBORS_SCHEMA,
]

_V16_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V16_CALLABLES: Dict[str, Callable] = {
    "get_features": _V16_GET_FEATURES_CALLABLE,
    "get_neighbors": _V16_GET_NEIGHBORS_CALLABLE,
}

_V16_VERSION = {
    "basic_schemas": _V16_BASIC_SCHEMAS,
    "task_specific_map": _V16_TASK_MAP,
    "callables": _V16_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
        {"selectors": ["get_neighbors"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v16_no_neighbor: v16 get_features only
# ---------------------------------------------------------------------------
_V16_NO_NEIGHBOR_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V16_NO_NEIGHBOR_GET_FEATURES_SCHEMA,
]

_V16_NO_NEIGHBOR_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V16_NO_NEIGHBOR_CALLABLES: Dict[str, Callable] = {
    "get_features": _V16_NO_NEIGHBOR_GET_FEATURES_CALLABLE,
}

_V16_NO_NEIGHBOR_VERSION = {
    "basic_schemas": _V16_NO_NEIGHBOR_BASIC_SCHEMAS,
    "task_specific_map": _V16_NO_NEIGHBOR_TASK_MAP,
    "callables": _V16_NO_NEIGHBOR_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
    ],
}

# ---------------------------------------------------------------------------
# v17: v16 get_features + new analogue-quality get_neighbors
# ---------------------------------------------------------------------------
_V17_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V17_GET_FEATURES_SCHEMA,
    _V17_GET_NEIGHBORS_SCHEMA,
]

_V17_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V17_CALLABLES: Dict[str, Callable] = {
    "get_features": _V17_GET_FEATURES_CALLABLE,
    "get_neighbors": _V17_GET_NEIGHBORS_CALLABLE,
}

_V17_VERSION = {
    "basic_schemas": _V17_BASIC_SCHEMAS,
    "task_specific_map": _V17_TASK_MAP,
    "callables": _V17_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
        {"selectors": ["get_neighbors"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v17_neighbor_only: v17 analogue-quality get_neighbors only
# ---------------------------------------------------------------------------
_V17_NEIGHBOR_ONLY_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V17_GET_NEIGHBORS_SCHEMA,
]

_V17_NEIGHBOR_ONLY_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V17_NEIGHBOR_ONLY_CALLABLES: Dict[str, Callable] = {
    "get_neighbors": _V17_GET_NEIGHBORS_CALLABLE,
}

_V17_NEIGHBOR_ONLY_VERSION = {
    "basic_schemas": _V17_NEIGHBOR_ONLY_BASIC_SCHEMAS,
    "task_specific_map": _V17_NEIGHBOR_ONLY_TASK_MAP,
    "callables": _V17_NEIGHBOR_ONLY_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_neighbors"], "endpoint": "neighbors"},
    ],
}

# ---------------------------------------------------------------------------
# v17_get_features_only: v17 grouped get_features surface without neighbors
# ---------------------------------------------------------------------------
_V17_GET_FEATURES_ONLY_BASIC_SCHEMAS: List[Dict[str, Any]] = [
    _V17_GET_FEATURES_ONLY_GET_FEATURES_SCHEMA,
]

_V17_GET_FEATURES_ONLY_TASK_MAP: Dict[str, List[Dict[str, Any]]] = {}

_V17_GET_FEATURES_ONLY_CALLABLES: Dict[str, Callable] = {
    "get_features": _V17_GET_FEATURES_ONLY_GET_FEATURES_CALLABLE,
}

_V17_GET_FEATURES_ONLY_VERSION = {
    "basic_schemas": _V17_GET_FEATURES_ONLY_BASIC_SCHEMAS,
    "task_specific_map": _V17_GET_FEATURES_ONLY_TASK_MAP,
    "callables": _V17_GET_FEATURES_ONLY_CALLABLES,
    "metric_endpoints": [
        {"selectors": ["get_features"], "endpoint": "features"},
    ],
}

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
_ALL_VERSIONS = {
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
    "v11", "v12", "v13", "v14", "v14_consolidated", "v14_no_neighbor",
    "v14_consolidated_no_neighbor", "v15", "v15_no_neighbor",
    "v15_neighbor_only", "v15_neighbor_only_4", "v16", "v16_no_neighbor", "v17",
    "v17_neighbor_only",
    "v17_get_features_only",
}

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
    "v14": _V14_VERSION,
    "v14_consolidated": _V14_CONSOLIDATED_VERSION,
    "v14_no_neighbor": _V14_NO_NEIGHBOR_VERSION,
    "v14_consolidated_no_neighbor": _V14_CONSOLIDATED_NO_NEIGHBOR_VERSION,
    "v15": _V15_VERSION,
    "v15_no_neighbor": _V15_NO_NEIGHBOR_VERSION,
    "v15_neighbor_only": _V15_NEIGHBOR_ONLY_VERSION,
    "v15_neighbor_only_4": _V15_NEIGHBOR_ONLY_4_VERSION,
    "v16": _V16_VERSION,
    "v16_no_neighbor": _V16_NO_NEIGHBOR_VERSION,
    "v17": _V17_VERSION,
    "v17_neighbor_only": _V17_NEIGHBOR_ONLY_VERSION,
    "v17_get_features_only": _V17_GET_FEATURES_ONLY_VERSION,
}


def get_version(ver: str) -> Dict[str, Any]:
    """Return the tool version config dict, or raise ValueError."""
    if ver not in _ALL_VERSIONS:
        raise ValueError(
            f"Unknown tool version {ver!r}. "
            f"Available: {sorted(_ALL_VERSIONS)}"
        )
    if ver in (
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v14_consolidated",
        "v14_no_neighbor",
        "v14_consolidated_no_neighbor",
        "v15",
        "v15_no_neighbor",
        "v15_neighbor_only",
        "v15_neighbor_only_4",
        "v16",
        "v16_no_neighbor",
        "v17",
        "v17_neighbor_only",
        "v17_get_features_only",
    ):
        return TOOL_VERSIONS[ver]
    # Lazy-load legacy versions on first access
    legacy = _build_legacy_versions()
    TOOL_VERSIONS.update(legacy)
    return legacy[ver]


__all__ = [
    "TOOL_VERSIONS",
    "get_version",
    "resolve_tool_metric_endpoint",
    "resolve_tool_metric_metadata",
]
