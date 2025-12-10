import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ai_scientist.utils.pathing import resolve_output_path


def log_missing_or_corrupt(entries: List[Dict[str, Any]], filename: str = "verification_missing_report_post_run.json") -> Dict[str, str]:
    """
    Write a health report under experiment_results/_health/. Safe to call when nothing is missing.
    """
    base_env = os.environ.get("AISC_EXP_RESULTS", "") or os.environ.get("AISC_BASE_FOLDER", "")
    exp_dir, _, _ = resolve_output_path(subdir="_health", name=filename, run_root=Path(base_env) if base_env else None, allow_quarantine=True, unique=False)
    report_path = exp_dir
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        return {"path": str(report_path)}
    except Exception as exc:
        return {"error": str(exc), "path": str(report_path)}
