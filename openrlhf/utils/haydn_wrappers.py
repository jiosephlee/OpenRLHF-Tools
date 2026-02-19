"""Thin wrappers around Intern-S1-recipe/tools/haydn_tools_python_311.py for GRPO training.

Each wrapper:
  - Takes simple typed args (str / float / bool / list / dict)
  - Lazily imports via ``_load_haydn_module()`` (no module-level hard dependency)
  - Returns a JSON string suitable for tool-call feedback

OpenAI-compatible tool schemas are exported as ``HAYDN_OPENAI_TOOLS`` (list)
and callable wrappers as ``HAYDN_CALLABLES`` (dict[str, Callable]).
"""

import json
import importlib.util
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold


# ---------------------------------------------------------------------------
# Wrappers (10 functions)
# ---------------------------------------------------------------------------

_HAYDN_MODULE_NAME = "_openrlhf_haydn_tools_python_311"


def _resolve_haydn_module_path() -> Path:
    candidates: list[Path] = []

    # Common project layouts (repo root + runtime cwd)
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(project_root / "Intern-S1-recipe" / "tools" / "haydn_tools_python_311.py")
    candidates.append(Path.cwd() / "Intern-S1-recipe" / "tools" / "haydn_tools_python_311.py")

    # If "tools" package exists, prefer its own search paths.
    tools_pkg = sys.modules.get("tools")
    if tools_pkg is not None:
        for p in getattr(tools_pkg, "__path__", []):
            candidates.append(Path(p) / "haydn_tools_python_311.py")

    # Search import paths for either ".../Intern-S1-recipe" or direct ".../tools".
    for p in sys.path:
        sp = Path(p)
        candidates.append(sp / "Intern-S1-recipe" / "tools" / "haydn_tools_python_311.py")
        candidates.append(sp / "tools" / "haydn_tools_python_311.py")

    seen = set()
    for c in candidates:
        r = c.resolve() if c.is_absolute() else c
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c

    checked = "\n".join(f"- {p}" for p in seen)
    raise FileNotFoundError(
        "Could not locate haydn_tools_python_311.py. Checked:\n"
        f"{checked}"
    )


def _load_haydn_module():
    module = sys.modules.get(_HAYDN_MODULE_NAME)
    if module is not None:
        return module

    try:
        module = importlib.import_module("tools.haydn_tools_python_311")
        sys.modules[_HAYDN_MODULE_NAME] = module
        return module
    except ModuleNotFoundError:
        pass

    module_path = _resolve_haydn_module_path()
    spec = importlib.util.spec_from_file_location(_HAYDN_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Haydn tools module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[_HAYDN_MODULE_NAME] = module
    return module

def compute_similarity_wrapper(
    smiles: str,
    reference_smiles: list,
    fingerprint: str = "morgan",
) -> str:
    haydn = _load_haydn_module()
    result = haydn.compute_similarity(smiles, reference_smiles, haydn.FingerprintType(fingerprint))
    return result.model_dump_json(indent=2)


def find_mcs_wrapper(
    smiles: str,
    reference_smiles: list,
    complete_rings_only: bool = True,
    ring_matches_ring_only: bool = True,
) -> str:
    haydn = _load_haydn_module()
    result = haydn.find_mcs(smiles, reference_smiles, complete_rings_only, ring_matches_ring_only)
    return result.model_dump_json(indent=2)


def score_structural_alerts_wrapper(
    smiles: str,
    alert_library: str = "all",
) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    library_map = {
        "all": FilterCatalogParams.FilterCatalogs.ALL,
        "pains": FilterCatalogParams.FilterCatalogs.PAINS,
        "pains_a": FilterCatalogParams.FilterCatalogs.PAINS_A,
        "pains_b": FilterCatalogParams.FilterCatalogs.PAINS_B,
        "pains_c": FilterCatalogParams.FilterCatalogs.PAINS_C,
        "brenk": FilterCatalogParams.FilterCatalogs.BRENK,
        "nih": FilterCatalogParams.FilterCatalogs.NIH,
        "zinc": FilterCatalogParams.FilterCatalogs.ZINC,
        "chembl": FilterCatalogParams.FilterCatalogs.CHEMBL,
        "chembl_bms": FilterCatalogParams.FilterCatalogs.CHEMBL_BMS,
        "chembl_lint": FilterCatalogParams.FilterCatalogs.CHEMBL_LINT,
        "chembl_mlsmr": FilterCatalogParams.FilterCatalogs.CHEMBL_MLSMR,
    }

    key = alert_library.lower().strip()
    if key not in library_map:
        raise ValueError(
            f"Unknown alert_library '{alert_library}'. "
            f"Valid: {', '.join(library_map.keys())}"
        )

    params = FilterCatalogParams()
    params.AddCatalog(library_map[key])
    catalog = FilterCatalog(params)

    alerts = []
    for entry in catalog.GetMatches(mol):
        props = {name: entry.GetProp(name) for name in entry.GetPropList()}
        alerts.append(
            {
                "description": entry.GetDescription(),
                "filter_set": props.get("FilterSet"),
                "scope": props.get("Scope"),
            }
        )

    return json.dumps(
        {"library": key, "count": len(alerts), "alerts": alerts},
        indent=2,
        ensure_ascii=False,
    )


def extract_pharmacophore_features_wrapper(smiles: str) -> str:
    haydn = _load_haydn_module()
    result = haydn.extract_pharmacophore_features(smiles)
    return result.model_dump_json(indent=2)


def classify_ionization_wrapper(smiles: str, ph: float = 7.4) -> str:
    haydn = _load_haydn_module()
    result = haydn.classify_ionization(smiles, ph)
    return result.model_dump_json(indent=2)


def standardize_smiles_wrapper(
    smiles: str,
    remove_salts: bool = True,
    canonical_tautomer: bool = True,
    neutralize: bool = False,
) -> str:
    haydn = _load_haydn_module()
    return haydn.standardize_smiles(smiles, remove_salts, canonical_tautomer, neutralize)


def compute_descriptors_wrapper(
    smiles: str,
    descriptors: Optional[list] = None,
) -> str:
    haydn = _load_haydn_module()
    result = haydn.compute_descriptors(smiles, descriptors)
    return json.dumps(result, indent=2, ensure_ascii=False)


def match_substructure_wrapper(
    smiles: str,
    patterns: dict,
) -> str:
    haydn = _load_haydn_module()
    result = haydn.match_substructure(smiles, patterns)
    # SubstructureMatchResult is a pydantic model
    return json.dumps(
        {k: v.model_dump() for k, v in result.items()},
        indent=2,
        ensure_ascii=False,
    )


def analyze_ring_systems_wrapper(smiles: str) -> str:
    haydn = _load_haydn_module()
    result = haydn.analyze_ring_systems(smiles)
    return result.model_dump_json(indent=2)


def get_murcko_scaffold_wrapper(
    smiles: str,
    generic: bool = False,
) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    mol_heavy_atoms = mol.GetNumHeavyAtoms()
    core = MurckoScaffold.GetScaffoldForMol(mol)
    has_core = core is not None and core.GetNumAtoms() > 0

    scaffold_smiles = Chem.MolToSmiles(core, canonical=True, isomericSmiles=True) if has_core else ""
    num_scaffold_atoms = core.GetNumHeavyAtoms() if has_core else 0
    num_scaffold_rings = core.GetRingInfo().NumRings() if has_core else 0

    generic_scaffold_smiles = None
    if generic:
        generic_scaffold_smiles = ""
        if has_core:
            generic_core = MurckoScaffold.MakeScaffoldGeneric(core)
            generic_scaffold_smiles = Chem.MolToSmiles(
                generic_core, canonical=True, isomericSmiles=False
            )

    return json.dumps(
        {
            "scaffold_smiles": scaffold_smiles,
            "generic_scaffold_smiles": generic_scaffold_smiles,
            "num_scaffold_atoms": num_scaffold_atoms,
            "num_scaffold_rings": num_scaffold_rings,
            "scaffold_fraction": (
                round(num_scaffold_atoms / mol_heavy_atoms, 4) if mol_heavy_atoms > 0 else 0.0
            ),
        },
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Callables dict  (name used in tool schemas → wrapper function)
# ---------------------------------------------------------------------------
HAYDN_CALLABLES: Dict[str, Callable] = {
    "compute_similarity": compute_similarity_wrapper,
    "find_mcs": find_mcs_wrapper,
    "score_structural_alerts": score_structural_alerts_wrapper,
    "extract_pharmacophore_features": extract_pharmacophore_features_wrapper,
    "classify_ionization": classify_ionization_wrapper,
    "standardize_smiles": standardize_smiles_wrapper,
    "compute_descriptors": compute_descriptors_wrapper,
    "match_substructure": match_substructure_wrapper,
    "analyze_ring_systems": analyze_ring_systems_wrapper,
    "get_murcko_scaffold": get_murcko_scaffold_wrapper,
}


# ---------------------------------------------------------------------------
# OpenAI tool schemas (10 schemas)
# ---------------------------------------------------------------------------

COMPUTE_SIMILARITY_TOOL = {
    "type": "function",
    "function": {
        "name": "compute_similarity",
        "description": (
            "Compute Tanimoto fingerprint similarity between a query molecule "
            "and reference molecules."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the query molecule.",
                },
                "reference_smiles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of reference SMILES to compare against.",
                },
                "fingerprint": {
                    "type": "string",
                    "enum": [
                        "morgan", "rdkit", "maccs",
                        "atom_pair", "topological_torsion",
                    ],
                    "description": "Fingerprint type (default: morgan).",
                },
            },
            "required": ["smiles", "reference_smiles"],
            "additionalProperties": False,
        },
    },
}

FIND_MCS_TOOL = {
    "type": "function",
    "function": {
        "name": "find_mcs",
        "description": (
            "Find the maximum common substructure (MCS) across a query and "
            "reference molecules. Returns SMARTS, atom/bond counts, and coverage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the query molecule.",
                },
                "reference_smiles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of reference SMILES to include in MCS search.",
                },
                "complete_rings_only": {
                    "type": "boolean",
                    "description": "MCS must contain complete rings (default: true).",
                },
                "ring_matches_ring_only": {
                    "type": "boolean",
                    "description": "Ring atoms only match other ring atoms (default: true).",
                },
            },
            "required": ["smiles", "reference_smiles"],
            "additionalProperties": False,
        },
    },
}

SCORE_STRUCTURAL_ALERTS_TOOL = {
    "type": "function",
    "function": {
        "name": "score_structural_alerts",
        "description": (
            "Screen a molecule against RDKit's built-in structural alert "
            "catalogs (PAINS, Brenk, NIH, ZINC, ChEMBL, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                },
                "alert_library": {
                    "type": "string",
                    "enum": [
                        "all", "pains", "pains_a", "pains_b", "pains_c",
                        "brenk", "nih", "zinc",
                        "chembl", "chembl_bms", "chembl_lint", "chembl_mlsmr",
                    ],
                    "description": "Alert library to screen against (default: all).",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}

EXTRACT_PHARMACOPHORE_FEATURES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_pharmacophore_features",
        "description": (
            "Extract pharmacophore-like features (donors, acceptors, "
            "hydrophobes, aromatics, etc.) using RDKit BaseFeatures."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}

CLASSIFY_IONIZATION_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_ionization",
        "description": (
            "Classify the ionization state of a molecule at a target pH "
            "using Dimorphite-DL protonation enumeration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                },
                "ph": {
                    "type": "number",
                    "description": "Target pH for protonation (default: 7.4).",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}

STANDARDIZE_SMILES_TOOL = {
    "type": "function",
    "function": {
        "name": "standardize_smiles",
        "description": (
            "Standardize a SMILES string: remove salts, canonicalize tautomers, "
            "and optionally neutralize charges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string to standardize.",
                },
                "remove_salts": {
                    "type": "boolean",
                    "description": "Remove salts/counterions (default: true).",
                },
                "canonical_tautomer": {
                    "type": "boolean",
                    "description": "Canonicalize tautomers (default: true).",
                },
                "neutralize": {
                    "type": "boolean",
                    "description": "Neutralize formal charges (default: false).",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}

COMPUTE_DESCRIPTORS_TOOL = {
    "type": "function",
    "function": {
        "name": "compute_descriptors",
        "description": (
            "Compute molecular descriptors: masses, atom counts, surface/shape, "
            "ring counts, logP, rotatable bonds, QED, hydrogen bonding, "
            "Lipinski violations, ESOL solubility."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                },
                "descriptors": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "masses", "atom_counts", "surface_shape_props",
                            "ring_counts", "logp", "num_rotatable_bonds",
                            "num_amide_bonds", "formal_charge", "qed",
                            "hydrogen_bonding", "lipinski_violations", "esol",
                        ],
                    },
                    "description": "Descriptor names to compute. Omit for all.",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}

MATCH_SUBSTRUCTURE_TOOL = {
    "type": "function",
    "function": {
        "name": "match_substructure",
        "description": (
            "Test whether a molecule contains the given SMARTS substructures "
            "and count occurrences."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                },
                "patterns": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Mapping of pattern name to SMARTS string.",
                },
            },
            "required": ["smiles", "patterns"],
            "additionalProperties": False,
        },
    },
}

ANALYZE_RING_SYSTEMS_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_ring_systems",
        "description": (
            "Analyze fused ring systems: detect PAH-like systems, macrocycles, "
            "spiro/bridgehead atoms, aromaticity, and heteroatom content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}

GET_MURCKO_SCAFFOLD_TOOL = {
    "type": "function",
    "function": {
        "name": "get_murcko_scaffold",
        "description": (
            "Extract the Bemis-Murcko scaffold from a molecule. "
            "Reports scaffold SMILES, atom/ring counts, and scaffold fraction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule.",
                },
                "generic": {
                    "type": "boolean",
                    "description": "Return generic scaffold (all atoms→C, all bonds→single). Default: false.",
                },
            },
            "required": ["smiles"],
            "additionalProperties": False,
        },
    },
}

HAYDN_OPENAI_TOOLS: List[Dict[str, Any]] = [
    COMPUTE_SIMILARITY_TOOL,
    FIND_MCS_TOOL,
    SCORE_STRUCTURAL_ALERTS_TOOL,
    EXTRACT_PHARMACOPHORE_FEATURES_TOOL,
    CLASSIFY_IONIZATION_TOOL,
    STANDARDIZE_SMILES_TOOL,
    COMPUTE_DESCRIPTORS_TOOL,
    MATCH_SUBSTRUCTURE_TOOL,
    ANALYZE_RING_SYSTEMS_TOOL,
    GET_MURCKO_SCAFFOLD_TOOL,
]

__all__ = ["HAYDN_OPENAI_TOOLS", "HAYDN_CALLABLES"]
