from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

TOOL_VERSION = "v17_ridge_official_v15"
BACKEND = "v17_tools"

DEFAULT_RAW_DIR = SCRIPT_DIR / "official_v15_dataset"
DEFAULT_FEATURE_CACHE = REPO_ROOT / "ml_experiments" / "feature_cache_official_v15"
DEFAULT_RESULTS_DIR = REPO_ROOT / "ml_experiments" / "results" / "v17_tools_ridge_official_v15_wide"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "summary.json"

DEFAULT_LOCAL_ATTR_DEBUG_DIR = SCRIPT_DIR / "debug_v17_ridge_official_v15_local_attribution_prompts"
DEFAULT_REASONING_TRACE_DEBUG_DIR = SCRIPT_DIR / "debug_v17_ridge_official_v15_local_reasoning_traces_from_local_attr"
DEFAULT_REASONING_TRACE_OUTPUT_DIR = Path(
    "/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/data/openai_format_v17_ridge_official_v15_local_reasoning_trace"
)
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
