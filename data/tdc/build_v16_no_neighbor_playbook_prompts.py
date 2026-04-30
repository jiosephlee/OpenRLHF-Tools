"""Build task-specific playbook artifacts adapted to the v16_no_neighbor tool surface."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.tdc.ml_prompt_artifacts import playbook_guides_dir, write_playbook_guide


TOOL_VERSION = "v16_no_neighbor"
SOURCE_DIR = Path("/vast/home/j/jojolee/OpenRLHF-Tools/Intern-S1-recipe/playbooks/Haydn_origin")

PLAYBOOK_BLOCK_RE = re.compile(r"\{%\s*set prompt_playbook -?%\}(.*?)\{%[-\s]*endset\s*%\}", re.S)

SOURCE_FILE_MAP = {
    "AMES": "AMES.jinja2",
    "BBB_Martins": "BBB_Martins.jinja2",
    "Bioavailability_Ma": "Bioavailability_Ma.jinja2",
    "CYP2C9_Substrate_CarbonMangels": "CYP2C9_Substrate_CarbonMangels.jinja2",
    "CYP2D6_Substrate_CarbonMangels": "CYP2D6_Substrate_CarbonMangels.jinja2",
    "CYP3A4_Substrate_CarbonMangels": "CYP3A4_Substrate_CarbonMangels.jinja2",
    "Carcinogens_Lagunin": "Carcinogens_Lagunin.jinja2",
    "ClinTox": "ClinTox.jinja2",
    "DILI": "DILI.jinja2",
    "HIA_Hou": "HIA_Hou.jinja2",
    "PAMPA_NCATS": "PAMPA_NCATS.jinja2",
    "Pgp_Broccatelli": "Pgp_Broccatelli.jinja2",
    "SARSCoV2_3CLPro_Diamond": "SARSCOV2_3CLPro_Diamond.jinja2",
    "SARSCoV2_Vitro_Touret": "SARSCoV2_Vitro_Touret.jinja2",
    "Skin_Reaction": "Skin_Reaction.jinja2",
    "hERG": "hERG_Karim.jinja2",
}

TASK_FEATURE_GROUPS = {
    "AMES": ["structure_and_topology", "alert_screening", "molecular_profile", "ionization_and_solubility"],
    "BBB_Martins": ["molecular_profile", "ionization_and_solubility", "structure_and_topology"],
    "Bioavailability_Ma": ["molecular_profile", "ionization_and_solubility", "structure_and_topology", "alert_screening"],
    "CYP2C9_Substrate_CarbonMangels": ["molecular_profile", "ionization_and_solubility", "structure_and_topology"],
    "CYP2D6_Substrate_CarbonMangels": ["molecular_profile", "ionization_and_solubility", "structure_and_topology"],
    "CYP3A4_Substrate_CarbonMangels": ["molecular_profile", "ionization_and_solubility", "structure_and_topology"],
    "Carcinogens_Lagunin": ["alert_screening", "structure_and_topology", "molecular_profile", "ionization_and_solubility"],
    "ClinTox": ["alert_screening", "molecular_profile", "ionization_and_solubility", "structure_and_topology"],
    "DILI": ["alert_screening", "molecular_profile", "ionization_and_solubility", "structure_and_topology"],
    "HIA_Hou": ["molecular_profile", "ionization_and_solubility", "structure_and_topology"],
    "PAMPA_NCATS": ["ionization_and_solubility", "molecular_profile", "structure_and_topology"],
    "Pgp_Broccatelli": ["molecular_profile", "ionization_and_solubility", "structure_and_topology", "alert_screening"],
    "SARSCoV2_3CLPro_Diamond": ["structure_and_topology", "molecular_profile", "ionization_and_solubility", "alert_screening"],
    "SARSCoV2_Vitro_Touret": ["molecular_profile", "ionization_and_solubility", "structure_and_topology", "alert_screening"],
    "Skin_Reaction": ["alert_screening", "structure_and_topology", "molecular_profile", "ionization_and_solubility"],
    "hERG": ["molecular_profile", "ionization_and_solubility", "structure_and_topology", "alert_screening"],
}


def extract_playbook_body(path: Path) -> str:
    text = path.read_text()
    match = PLAYBOOK_BLOCK_RE.search(text)
    if match is None:
        raise ValueError(f"Could not extract prompt_playbook block from {path}")
    return match.group(1).strip()


def split_title_and_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    lines = body.strip().splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        remainder = "\n".join(lines[1:]).strip()
    else:
        remainder = body.strip()

    matches = list(re.finditer(r"^##\s+.+$", remainder, re.M))
    if not matches:
        return title, []

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        heading = match.group(0).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(remainder)
        content = remainder[start:end].strip()
        sections.append((heading, content))
    return title, sections


def clean_title(title: str) -> str:
    cleaned = re.sub(r"^\s*Playbook:\s*", "", title).strip()
    cleaned = re.sub(r"\(.*?0.?100.*?\)", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return f"# {cleaned}" if cleaned else "# Task-specific reasoning guide"


def semantic_label_for_value(value: int) -> str:
    if value <= 10:
        return "very low confidence"
    if value <= 25:
        return "low confidence"
    if value <= 40:
        return "modest confidence"
    if value <= 60:
        return "mixed confidence"
    if value <= 75:
        return "moderately strong confidence"
    if value <= 90:
        return "strong confidence"
    return "very strong confidence"


def semantic_label_for_delta(value: int) -> str:
    magnitude = abs(value)
    if magnitude <= 5:
        strength = "slight"
    elif magnitude <= 10:
        strength = "moderate"
    elif magnitude <= 20:
        strength = "strong"
    else:
        strength = "very strong"
    direction = "upward" if value >= 0 else "downward"
    return f"{strength} {direction} adjustment"


def humanize_key(name: str) -> str:
    text = name.strip().strip('"').strip("'")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+like\b", "-like", text)
    return text.strip()


def render_workflow_content(task: str) -> str:
    prioritized = TASK_FEATURE_GROUPS[task]
    lead = ", ".join(f"`{group}`" for group in prioritized[:2])
    body = [
        "Call `get_features` once and read the returned evidence as a single integrated profile rather than as separate legacy tool calls.",
        f"Start with {lead}, then use the remaining groups to confirm or soften the first impression.",
        "- `molecular_profile`: weight, lipophilicity, polarity, hydrogen-bonding burden, flexibility, aromaticity, complexity, electronics, and 3D-shape surrogates.",
        "- `ionization_and_solubility`: pKa pattern, charge state near physiological pH, neutral fraction, logD, and solubility-related constraints.",
        "- `structure_and_topology`: functional groups, ring systems, scaffold texture, and topology cues that shape recognition and permeability.",
        "- `alert_screening`: broad screening alerts plus more specific liability motifs that deserve extra scrutiny.",
        "Treat the groups as one story: start with the global profile, then ask whether specific motifs or alerts reinforce that direction or create an exception.",
    ]
    return "\n".join(body)


def rewrite_probability_bands(text: str) -> str:
    text = re.sub(r"- \*\*(\d+\s*[–-]\s*\d+)\s*:\*\*\s*([^\n]+)", r"- \2 (~\1)", text)
    text = re.sub(r"(\b\d+\s*[–-]\s*\d+\b)\s*→\s*([A-Za-z][^\n]*)", r"\2 (~\1)", text)
    text = re.sub(r"([Vv]ery unlikely|[Uu]nlikely|[Mm]ixed|[Ll]ikely|[Vv]ery likely)\s*\((\d+\s*[–-]\s*\d+)\)", r"\1 (~\2)", text)
    return text


def rewrite_numeric_confidence_language(text: str) -> str:
    text = text.replace("−", "-")

    def replace_bucket(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return f"{semantic_label_for_value(value)} (~{value})"

    def replace_adjustment(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return f"{semantic_label_for_delta(value)} (~{value:+d})"

    text = re.sub(r"\bP0\s*=\s*(\d{1,3})\b", lambda m: replace_bucket(m), text)
    text = re.sub(r"\bP\s*=\s*(\d{1,3})\b", lambda m: replace_bucket(m), text)
    text = re.sub(r"\bscore\s*=\s*(\d{1,3})\b", lambda m: replace_bucket(m), text)
    text = re.sub(r"(?<=:\s)\*\*(\d{1,3})\*\*", lambda m: replace_bucket(m), text)
    text = re.sub(r"(?<=:\s)\*\*([+-]\d{1,3})\*\*", lambda m: replace_adjustment(m), text)
    text = re.sub(r"(?<=:\s)([+-]\d{1,3})(?=$|\s)", lambda m: replace_adjustment(m), text)
    text = re.sub(
        r":\s*P\s*([+-]=)\s*(\d{1,3})",
        lambda m: f": {semantic_label_for_delta((1 if m.group(1) == '+=' else -1) * int(m.group(2)))} (~{('+' if m.group(1) == '+=' else '-')}{m.group(2)})",
        text,
    )
    text = re.sub(
        r"cap probability at (\d{1,3}(?:\s*[–-]\s*\d{1,3})?)",
        lambda m: f"treat the ceiling as {semantic_band_from_range(m.group(1))}",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"probability at (\d{1,3}(?:\s*[–-]\s*\d{1,3})?)",
        lambda m: semantic_band_from_range(m.group(1)),
        text,
        flags=re.I,
    )
    return rewrite_probability_bands(text)


def semantic_band_from_range(raw: str) -> str:
    parts = [int(part) for part in re.split(r"\s*[–-]\s*", raw)]
    midpoint = round(sum(parts) / len(parts))
    return f"{semantic_label_for_value(midpoint)} (~{raw})"


def rewrite_json_code_blocks(text: str) -> str:
    def replace_block(match: re.Match[str]) -> str:
        code = match.group(1)
        keys = re.findall(r'"([^"]+)":', code)
        if not keys:
            return ""
        bullets = ["Look explicitly for motifs such as:"]
        bullets.extend(f"- {humanize_key(key)}" for key in keys)
        return "\n".join(bullets)

    return re.sub(r"```(?:json|python)?\n(.*?)```", replace_block, text, flags=re.S)


def rewrite_similarity_section() -> str:
    return (
        "Use read-across qualitatively rather than as a separate tool call. If the returned motifs, ring systems, and alert patterns closely resemble a known high-risk or high-confidence chemotype for this task, let that reinforce the decision. If the chemistry looks unusual and the signals conflict, stay away from extreme confidence."
    )


def rewrite_output_section(content: str) -> str:
    cleaned = content
    cleaned = re.sub(r"\*\*Final output.*?\n", "", cleaned, flags=re.I)
    cleaned = re.sub(r"output an \*\*integer 0[–-]100\*\*", "make a categorical decision between `(A)` and `(B)`", cleaned, flags=re.I)
    cleaned = re.sub(r"single integer\s*`?0[–-]100`?\s*\*\*only\*\*", "the final answer should be a categorical decision between `(A)` and `(B)`", cleaned, flags=re.I)
    cleaned = re.sub(r"Final output constraint:.*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"semantic confidence judgment", "categorical decision between `(A)` and `(B)`", cleaned, flags=re.I)
    intro = (
        "Use the confidence language in this guide as internal calibration while deciding between `(A)` and `(B)`.\n"
        "Semantic anchors: very unlikely (~0-15), unlikely (~15-35), mixed (~35-65), likely (~65-85), very likely (~85-100).\n\n"
    )
    rendered = rewrite_general_content(cleaned).strip()
    rendered = rendered.replace("- - Use this guidance internally and then give the required label.", "Use this guidance internally and then give the required label.")
    rendered = re.sub(
        r"^\s*-\s*Use this guidance internally and then give the required label\.\s*$",
        "Use this guidance internally and then give the required label.",
        rendered,
        flags=re.M,
    )
    return intro + rendered


def rewrite_alert_section(content: str) -> str:
    text = content
    text = text.replace("Run `get_features` and also separately with:", "Within `alert_screening`, pay special attention to broad alert families such as:")
    text = text.replace("Use `get_features` and request at least:", "Within the returned feature profile, focus especially on:")
    text = text.replace("Also run (recommended):", "Also read from the same returned profile:")
    text = text.replace("Use `match_substructure` with patterns like below (tune as needed).", "Read the alert and motif outputs in a more literal way. Separate broad alert-library hits from a smaller set of high-signal motifs.")
    text = text.replace("Use `match_substructure` with a hand-curated set of **high-signal motifs**. Suggested patterns (examples; expand over time):", "Separate the alert readout into a few easy-to-scan motif families:")
    text = text.replace("Use `match_substructure` with patterns like below", "Look directly for motifs like the following")
    text = rewrite_json_code_blocks(text)
    text = rewrite_general_content(text)
    text = re.sub(r"^- ([^:\n]+):\s*`[^`]+`(?:,\s*`[^`]+`)*(.*)$", r"- \1\2", text, flags=re.M)
    return text.strip()


def rewrite_scoring_section(content: str) -> str:
    intro = (
        "Use the scoring language below as an internal confidence ladder rather than a literal numeric output.\n"
        "Translate the end state into the task label while keeping the approximate numeric anchors in mind.\n\n"
    )
    text = rewrite_general_content(content)
    text = rewrite_numeric_confidence_language(text)
    text = re.sub(r"output as integer\.?", "use that as an internal confidence anchor.", text, flags=re.I)
    text = re.sub(r"Output \*\*integer\*\*.*", "", text, flags=re.I)
    return intro + text.strip()


def rewrite_general_content(content: str) -> str:
    text = content.strip()
    text = text.replace("−", "-")
    text = text.replace("---", "")
    replacements = {
        "`standardize_smiles`": "`get_features`",
        "`compute_descriptors`": "`get_features`",
        "`classify_ionization`": "`get_features`",
        "`estimate_logd`": "`get_features`",
        "`analyze_ring_systems`": "`get_features`",
        "`get_functional_groups`": "`get_features`",
        "`score_structural_alerts`": "`get_features`",
        "`match_substructure`": "`get_features`",
        "`find_mcs`": "a qualitative scaffold-overlap check",
        "0–100 probability": "confidence ladder",
        "0-100 probability": "confidence ladder",
        "probability (0–100)": "confidence bands",
        "probability (0-100)": "confidence bands",
        "integer 0–100": "semantic confidence judgment",
        "integer 0-100": "semantic confidence judgment",
        "output only the integer": "use the confidence language internally and then give the required label",
        "Final output must be a single integer 0–100": "Use these bands as internal calibration only",
        "Final output must be a single integer 0-100": "Use these bands as internal calibration only",
        "use tools in this order": "read the consolidated evidence in this order",
        "Use `get_features` and request at least:": "Within the returned feature profile, focus especially on:",
        "Use `get_features`.": "Read the relevant fields from `get_features`.",
        "Run `get_features` and also separately with:": "Within `alert_screening`, pay special attention to broad alert families such as:",
        "Also run (recommended):": "Also read from the same returned profile:",
        "estimate_logd(...).logd": "the logD field in ionization_and_solubility",
        "estimate_logd(...).most_basic_pka": "the strongest basic pKa in ionization_and_solubility",
        "estimate_logd(...).num_basic_sites": "the number of basic sites in ionization_and_solubility",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"PMIDs?:", "PMIDs:", text)
    text = re.sub(r"`score_structural_alerts\(.*?\)`", "`get_features`", text)
    text = re.sub(r"`match_substructure\(.*?\)`", "`get_features`", text)
    text = re.sub(r"`compute_descriptors\(.*?\)`", "`get_features`", text)
    text = re.sub(r"`classify_ionization\(.*?\)`", "`get_features`", text)
    text = re.sub(r"`estimate_logd\(.*?\)`", "`get_features`", text)
    text = re.sub(r"`analyze_ring_systems\(.*?\)`", "`get_features`", text)
    text = re.sub(r"`get_functional_groups\(.*?\)`", "`get_features`", text)
    text = re.sub(r"`standardize_smiles\(.*?\)`", "`get_features`", text)
    text = re.sub(r"Use `get_features` and request at least:\s*", "Within the returned feature profile, focus especially on:\n", text)
    text = re.sub(r"Run `get_features` and also separately with:\s*", "Within `alert_screening`, pay special attention to broad alert families such as:\n", text)
    text = re.sub(r"Use `get_features`\.\s*", "Read the relevant fields from `get_features`.\n", text)
    text = re.sub(
        r"Also read from the same returned profile:\s*\n- `get_features`\s*\n\s*- ",
        "Also read from the same returned profile:\n- ",
        text,
    )
    text = re.sub(r"Do all reasoning internally; output \*\*only the integer\*\*\.", "Use this guidance internally and then give the required label.", text)
    text = text.replace("- - Use this guidance internally and then give the required label.", "Use this guidance internally and then give the required label.")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def rewrite_section(task: str, heading: str, content: str) -> str:
    heading_lower = heading.lower()
    scoring_heading = any(
        token in heading_lower
        for token in (
            "scoring rubric",
            "probability interpretation",
            "interpretation bands",
            "convert score",
            "convert evidence",
            "turn features",
            "probability calculation",
            "calibration guide",
            "final probability mapping",
            "confidence annotation",
            "scoring →",
            "scoring ->",
        )
    )
    output_heading = any(
        token in heading_lower
        for token in (
            "output rule",
            "output contract",
            "output constraint",
            "required output",
            "what you are predicting",
            "goal and definition",
            "goal, output",
            "scope & label",
            "definitions and scope",
        )
    )
    if any(token in heading_lower for token in ("workflow", "tool workflow", "minimal execution checklist", "minimal tool recipe", "input handling", "preprocess", "preprocessing", "standardize input", "step-by-step tool workflow")):
        body = render_workflow_content(task)
    elif "similarity" in heading_lower or "read-across" in heading_lower:
        body = rewrite_similarity_section()
    elif scoring_heading or output_heading:
        if scoring_heading and not output_heading:
            body = rewrite_scoring_section(content)
        else:
            body = rewrite_output_section(content)
    elif "alert" in heading_lower or "smarts" in heading_lower or "substructure" in heading_lower or "motif" in heading_lower:
        body = rewrite_alert_section(content)
    else:
        body = rewrite_general_content(content)
        body = rewrite_numeric_confidence_language(body)
    return f"{heading}\n{body}".strip()


def render_task_guide(task: str, source_body: str) -> str:
    title, sections = split_title_and_sections(source_body)
    rendered_sections = [rewrite_section(task, heading, content) for heading, content in sections]
    return f"{clean_title(title)}\n\n" + "\n\n".join(rendered_sections).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build minimally adapted v16_no_neighbor playbook artifacts")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific tasks to build")
    args = parser.parse_args()

    tasks = args.tasks or sorted(SOURCE_FILE_MAP)
    out_dir = playbook_guides_dir(TOOL_VERSION)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_tasks = []
    for task in tasks:
        source_path = SOURCE_DIR / SOURCE_FILE_MAP[task]
        source_body = extract_playbook_body(source_path)
        content = render_task_guide(task, source_body)
        out_path = write_playbook_guide(TOOL_VERSION, task, content)
        manifest_tasks.append(
            {
                "task": task,
                "artifact": str(out_path),
                "source_file": str(source_path),
                "feature_groups": TASK_FEATURE_GROUPS[task],
            }
        )
        print(f"  {task}: {out_path}")

    manifest = {
        "tool_version": TOOL_VERSION,
        "artifact_type": "playbook_guides",
        "source_dir": str(SOURCE_DIR),
        "tasks": manifest_tasks,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nPlaybook guides written to {out_dir}")


if __name__ == "__main__":
    main()
