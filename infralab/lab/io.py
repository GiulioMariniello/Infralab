"""Scrittura degli output di campagna: runs/steps tidy + manifest di provenance."""
import logging
import os
from typing import Any, Dict, List

import pandas as pd

from infralab.lab.provenance import manifest_json

logger = logging.getLogger(__name__)


def write_outputs(
    output_dir: str,
    results: Dict[str, Any],
    manifest: Dict[str, Any],
    formats: List[str],
) -> Dict[str, str]:
    """Scrive runs.*, steps.*, collapses.* (registro dei collassi, se il runner lo
    fornisce), manifest.json. Restituisce i path scritti."""
    os.makedirs(output_dir, exist_ok=True)
    written: Dict[str, str] = {}

    frames = {
        "runs": pd.DataFrame(results["runs"]),
        "steps": pd.DataFrame(results["steps"]),
    }
    if "collapses" in results:
        # una riga per collasso; senza collassi resta la sola intestazione
        from infralab.lab.runner import COLLAPSE_COLUMNS
        rows = results["collapses"]
        lead = [c for c in (rows[0] if rows else {}) if c not in COLLAPSE_COLUMNS] or ["run_id", "agent_id", "replicate", "seed"]
        frames["collapses"] = pd.DataFrame(rows, columns=lead + list(COLLAPSE_COLUMNS))
    for name, frame in frames.items():
        for fmt in formats:
            path = os.path.join(output_dir, f"{name}.{fmt}")
            if fmt == "csv":
                frame.to_csv(path, index=False)
            elif fmt == "parquet":
                try:
                    frame.to_parquet(path, index=False)
                except ImportError:
                    logger.warning("pyarrow assente: salto l'output parquet %s", path)
                    continue
            written[f"{name}.{fmt}"] = path

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_json(manifest))
    written["manifest.json"] = manifest_path

    if results.get("errors"):
        errors_path = os.path.join(output_dir, "errors.log")
        with open(errors_path, "w", encoding="utf-8") as f:
            for error in results["errors"]:
                f.write(f"--- run {error['run_id']} ({error['agent_id']}) ---\n{error['error']}\n")
        written["errors.log"] = errors_path
    return written
