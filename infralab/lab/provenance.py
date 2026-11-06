"""Provenance della campagna: commit git, hash spec, versioni, seed per run."""
import hashlib
import json
import platform
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import infralab


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_provenance(spec_raw_text: str) -> Dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "spec_sha256": hashlib.sha256(spec_raw_text.encode("utf-8")).hexdigest(),
        "infralab_version": infralab.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform(),
    }


def manifest_json(manifest: Dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
