"""Thin wrappers around Intern-S1-recipe/tools/full_Haydn.py for GRPO training.

Each wrapper:
  - Takes simple typed args (str / float / bool / list / dict)
  - Lazily imports from ``tools.full_Haydn`` (no module-level hard dependency)
  - Returns a JSON string suitable for tool-call feedback

OpenAI-compatible tool schemas are exported as ``HAYDN_OPENAI_TOOLS`` (list)
and callable wrappers as ``HAYDN_CALLABLES`` (dict[str, Callable]).
"""

import json
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Wrappers (10 functions)
# ---------------------------------------------------------------------------

def compute_similarity_wrapper(
    smiles: str,
    reference_smiles: list,
    fingerprint: str = "morgan",
) -> str:
    from tools.haydn_tools_python_311 import compute_similarity, FingerprintType
    result = compute_similarity(smiles, reference_smiles, FingerprintType(fingerprint))
    return result.model_dump_json(indent=2)


def find_mcs_wrapper(
    smiles: str,
    reference_smiles: list,
    complete_rings_only: bool = True,
    ring_matches_ring_only: bool = True,
) -> str:
    from tools.haydn_tools_python_311 import find_mcs
    result = find_mcs(smiles, reference_smiles, complete_rings_only, ring_matches_ring_only)
    return result.model_dump_json(indent=2)


def score_structural_alerts_wrapper(
    smiles: str,
    alert_library: str = "all",
) -> str:
    from tools.haydn_tools_python_311 import score_structural_alerts, AlertLibrary
    result = score_structural_alerts(smiles, AlertLibrary(alert_library))
    return result.model_dump_json(indent=2)


def extract_pharmacophore_features_wrapper(smiles: str) -> str:
    from tools.haydn_tools_python_311 import extract_pharmacophore_features
    result = extract_pharmacophore_features(smiles)
    return result.model_dump_json(indent=2)


def classify_ionization_wrapper(smiles: str, ph: float = 7.4) -> str:
    from tools.haydn_tools_python_311 import classify_ionization
    result = classify_ionization(smiles, ph)
    return result.model_dump_json(indent=2)


def standardize_smiles_wrapper(
    smiles: str,
    remove_salts: bool = True,
    canonical_tautomer: bool = True,
    neutralize: bool = False,
) -> str:
    from tools.haydn_tools_python_311 import standardize_smiles
    return standardize_smiles(smiles, remove_salts, canonical_tautomer, neutralize)


def compute_descriptors_wrapper(
    smiles: str,
    descriptors: Optional[list] = None,
) -> str:
    from tools.haydn_tools_python_311 import compute_descriptors
    result = compute_descriptors(smiles, descriptors)
    return json.dumps(result, indent=2, ensure_ascii=False)


def match_substructure_wrapper(
    smiles: str,
    patterns: dict,
) -> str:
    from tools.haydn_tools_python_311 import match_substructure
    result = match_substructure(smiles, patterns)
    # SubstructureMatchResult is a pydantic model
    return json.dumps(
        {k: v.model_dump() for k, v in result.items()},
        indent=2,
        ensure_ascii=False,
    )


def analyze_ring_systems_wrapper(smiles: str) -> str:
    from tools.haydn_tools_python_311 import analyze_ring_systems
    result = analyze_ring_systems(smiles)
    return result.model_dump_json(indent=2)


def get_murcko_scaffold_wrapper(
    smiles: str,
    generic: bool = False,
) -> str:
    from tools.haydn_tools_python_311 import get_murcko_scaffold
    result = get_murcko_scaffold(smiles, generic=generic)
    return result.model_dump_json(indent=2)


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
