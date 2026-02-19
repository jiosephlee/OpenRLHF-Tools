"""Versioned tool registry for GRPO training.

Versions (incremental):
  - v1: RDKit basic + AccFG
  - v2: v1 + remove_salts (standardize_tools)
  - v3: v2 + predict_pka + estimate_logd + get_3d_exposed_polar_surface
  - v4: v3 + 10 Haydn tools

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
# Ensure Intern-S1-recipe/tools is importable, bypassing __init__.py
# (same trick as tool_calling_turn.py — centralised here so both share it)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_INTERN_S1_ROOT = _PROJECT_ROOT / "Intern-S1-recipe"
assert (_INTERN_S1_ROOT / "tools").is_dir(), (
    f"Intern-S1-recipe/tools not found at {_INTERN_S1_ROOT}/tools. "
    f"Run: git submodule update --init Intern-S1-recipe"
)
if str(_INTERN_S1_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERN_S1_ROOT))
_TOOLS_PATH = str(_INTERN_S1_ROOT / "tools")
_existing_tools_pkg = sys.modules.get("tools")
if _existing_tools_pkg is None:
    _pkg = types.ModuleType("tools")
    _pkg.__path__ = [_TOOLS_PATH]
    _pkg.__package__ = "tools"
    sys.modules["tools"] = _pkg
else:
    _existing_path = list(getattr(_existing_tools_pkg, "__path__", []))
    if _TOOLS_PATH not in _existing_path:
        _existing_tools_pkg.__path__ = [_TOOLS_PATH, *_existing_path]


# ---------------------------------------------------------------------------
# Imports from Intern-S1-recipe (always available — RDKit only)
# ---------------------------------------------------------------------------
from tools.RDKit_tools import (
    RDKIT_BASIC_OPENAI_TOOLS,
    TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
    # basic callables
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
    # task-specific callables
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

# Optional: ePSA (may fail if freesasa not installed)
try:
    from tools.ePSA_3D import get_3d_exposed_polar_surface, SASA_OPENAI_TOOLS
except ImportError:
    get_3d_exposed_polar_surface = None
    SASA_OPENAI_TOOLS = []

# Optional: pKa tools (may fail if molgpka not installed)
try:
    from tools.pka_related_tools import predict_pka, estimate_logd, PKA_TOOL, LOGD_TOOL
except ImportError:
    predict_pka = None
    estimate_logd = None
    PKA_TOOL = None
    LOGD_TOOL = None

# ---------------------------------------------------------------------------
# Shared callables (RDKit basic + AccFG + task-specific)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Version schemas (incremental)
# ---------------------------------------------------------------------------
_V1_SCHEMAS: List[Dict[str, Any]] = RDKIT_BASIC_OPENAI_TOOLS + AccFG_OPENAI_TOOLS
_V2_SCHEMAS: List[Dict[str, Any]] = _V1_SCHEMAS + STANDARDIZE_OPENAI_TOOLS

_V3_EXTRA_SCHEMAS: List[Dict[str, Any]] = []
if PKA_TOOL is not None:
    _V3_EXTRA_SCHEMAS.append(PKA_TOOL)
if LOGD_TOOL is not None:
    _V3_EXTRA_SCHEMAS.append(LOGD_TOOL)
_V3_SCHEMAS: List[Dict[str, Any]] = _V2_SCHEMAS + _V3_EXTRA_SCHEMAS + SASA_OPENAI_TOOLS

# v4: v3 + Haydn (lazy import to avoid hard dep)
try:
    from openrlhf.utils.haydn_wrappers import HAYDN_OPENAI_TOOLS, HAYDN_CALLABLES
except ImportError:
    HAYDN_OPENAI_TOOLS = []
    HAYDN_CALLABLES = {}

_V4_HAYDN_NAMES = {"compute_similarity", "score_structural_alerts", "match_substructure"}
_V4_SCHEMAS: List[Dict[str, Any]] = _V3_SCHEMAS + [
    t for t in HAYDN_OPENAI_TOOLS if t["function"]["name"] in _V4_HAYDN_NAMES
]

# ---------------------------------------------------------------------------
# Version callables (incremental)
# ---------------------------------------------------------------------------
_V1_CALLABLES: Dict[str, Callable] = dict(_RDKIT_ACCFG_CALLABLES)

_V2_CALLABLES: Dict[str, Callable] = {**_V1_CALLABLES, "remove_salts": remove_salts}

_V3_CALLABLES: Dict[str, Callable] = dict(_V2_CALLABLES)
if predict_pka is not None:
    _V3_CALLABLES["predict_pka"] = predict_pka
if estimate_logd is not None:
    _V3_CALLABLES["estimate_logd"] = estimate_logd
if get_3d_exposed_polar_surface is not None:
    _V3_CALLABLES["get_3d_exposed_polar_surface"] = get_3d_exposed_polar_surface

_V4_CALLABLES: Dict[str, Callable] = {
    **_V3_CALLABLES,
    **{k: v for k, v in HAYDN_CALLABLES.items() if k in _V4_HAYDN_NAMES},
}

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
TOOL_VERSIONS: Dict[str, Dict[str, Any]] = {
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
}


def get_version(ver: str) -> Dict[str, Any]:
    """Return the tool version config dict, or raise ValueError."""
    if ver not in TOOL_VERSIONS:
        raise ValueError(
            f"Unknown tool version {ver!r}. "
            f"Available: {sorted(TOOL_VERSIONS.keys())}"
        )
    return TOOL_VERSIONS[ver]


__all__ = ["TOOL_VERSIONS", "get_version"]
