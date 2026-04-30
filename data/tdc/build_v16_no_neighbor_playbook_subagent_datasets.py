"""Build v16_no_neighbor TDC datasets from subagent-authored playbooks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("build_v16_no_neighbor_playbook_datasets.py")
    cmd = [
        sys.executable,
        str(script),
        "--playbook-variant",
        "playbook_subagent",
        "--output-dir",
        str(Path(__file__).with_name("openai_format_v16_no_neighbor_playbook_subagent")),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
