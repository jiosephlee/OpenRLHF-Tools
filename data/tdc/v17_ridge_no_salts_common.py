from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

TOOL_VERSION = "v17_ridge_no_salts"
BACKEND = "v17_tools"

DEFAULT_RAW_DIR = SCRIPT_DIR / "deduplicated_no_salts"
DEFAULT_FEATURE_CACHE = REPO_ROOT / "ml_experiments" / "feature_cache_no_salts"
DEFAULT_RESULTS_DIR = REPO_ROOT / "ml_experiments" / "results" / "v17_tools_ridge_dedup_no_salts_wide"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "summary.json"

DEFAULT_LOCAL_ATTR_DEBUG_DIR = SCRIPT_DIR / "debug_v17_ridge_no_salts_local_attribution_prompts"
DEFAULT_REASONING_TRACE_DEBUG_DIR = SCRIPT_DIR / "debug_v17_ridge_no_salts_local_reasoning_traces_from_local_attr"
DEFAULT_REASONING_TRACE_OUTPUT_DIR = SCRIPT_DIR / "openai_format_v17_ridge_no_salts_local_reasoning_trace"
PROMPTS_PATH = SCRIPT_DIR / "metadata" / "prompts.json"

TASKS = [
    "AMES",
    "BBB_Martins",
    "Bioavailability_Ma",
    "Carcinogens_Lagunin",
    "ClinTox",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "DILI",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret",
    "Skin_Reaction",
    "hERG",
]
