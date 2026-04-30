from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable


RESULTS_BASE = Path("/vast/projects/myatskar/design-documents/hf_home/ml_experiments/results")
PROMPT_ARTIFACTS_BASE = Path("/vast/projects/myatskar/design-documents/joseph/therapeutic-tuning/prompt_artifacts")
LEGACY_V14_COEFFICIENTS_DIR = RESULTS_BASE / "v14_coefficients"
LEGACY_V14_GLOBAL_GUIDES_DIR = LEGACY_V14_COEFFICIENTS_DIR / "prompts"


def version_artifacts_dir(tool_version: str) -> Path:
    return PROMPT_ARTIFACTS_BASE / tool_version


def global_guides_dir(tool_version: str) -> Path:
    return version_artifacts_dir(tool_version) / "global"


def local_attribution_dir(tool_version: str) -> Path:
    return version_artifacts_dir(tool_version) / "local_attribution"


def sample_prompt_variant_dir(tool_version: str, variant: str) -> Path:
    return version_artifacts_dir(tool_version) / variant


def playbook_variant_dir(tool_version: str, variant: str = "playbook") -> Path:
    return version_artifacts_dir(tool_version) / variant


def playbook_guides_dir(tool_version: str) -> Path:
    return playbook_variant_dir(tool_version, "playbook")


def local_attribution_task_file(tool_version: str, task: str, split: str) -> Path:
    return local_attribution_dir(tool_version) / task / f"{split}.jsonl"


def sample_prompt_variant_task_file(tool_version: str, variant: str, task: str, split: str) -> Path:
    return sample_prompt_variant_dir(tool_version, variant) / task / f"{split}.jsonl"


def playbook_guide_file(tool_version: str, task: str, variant: str = "playbook") -> Path:
    return playbook_variant_dir(tool_version, variant) / f"{task}.md"


def load_global_guide(tool_version: str, task: str) -> str:
    path = global_guides_dir(tool_version) / f"{task}.md"
    if not path.exists():
        return ""
    return path.read_text().strip()


def load_playbook_guide(tool_version: str, task: str, variant: str = "playbook") -> str:
    path = playbook_guide_file(tool_version, task, variant=variant)
    if not path.exists():
        return ""
    return path.read_text().strip()


def copy_legacy_v14_global_guides(tool_version: str, tasks: Iterable[str]) -> list[Path]:
    dst_dir = global_guides_dir(tool_version)
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for task in tasks:
        src = LEGACY_V14_GLOBAL_GUIDES_DIR / f"{task}.md"
        if not src.exists():
            continue
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    all_tasks_src = LEGACY_V14_GLOBAL_GUIDES_DIR / "all_tasks.md"
    if all_tasks_src.exists():
        shutil.copy2(all_tasks_src, dst_dir / all_tasks_src.name)

    manifest = {
        "tool_version": tool_version,
        "artifact_type": "global_guides",
        "source": str(LEGACY_V14_GLOBAL_GUIDES_DIR),
        "tasks": sorted(path.stem for path in copied),
    }
    (dst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return copied


def write_local_attribution_records(
    tool_version: str,
    task: str,
    split: str,
    records: list[dict],
) -> Path:
    out_path = sample_prompt_variant_task_file(tool_version, "local_attribution", task, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def write_sample_prompt_records(
    tool_version: str,
    variant: str,
    task: str,
    split: str,
    records: list[dict],
) -> Path:
    out_path = sample_prompt_variant_task_file(tool_version, variant, task, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def write_playbook_guide(tool_version: str, task: str, content: str, variant: str = "playbook") -> Path:
    out_path = playbook_guide_file(tool_version, task, variant=variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content.rstrip() + "\n")
    return out_path


def load_local_attribution_map(tool_version: str, task: str, split: str) -> dict[str, dict]:
    return load_sample_prompt_map(tool_version, "local_attribution", task, split)


def load_sample_prompt_map(tool_version: str, variant: str, task: str, split: str) -> dict[str, dict]:
    path = sample_prompt_variant_task_file(tool_version, variant, task, split)
    if not path.exists():
        return {}

    payload: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles = str(row["smiles"])
            payload[smiles] = row
    return payload
