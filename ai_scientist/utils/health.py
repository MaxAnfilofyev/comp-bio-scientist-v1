import json
from pathlib import Path
from typing import Any, Dict, List

from ai_scientist.tools.base_tool import BaseTool


def log_missing_or_corrupt(entries: List[Dict[str, Any]], filename: str = "verification_missing_report_post_run.json") -> Dict[str, str]:
    """
    Write a health report under experiment_results/_health/. Safe to call when nothing is missing.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    health_dir = exp_dir / "_health"
    health_dir.mkdir(parents=True, exist_ok=True)
    report_path = health_dir / filename
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        return {"path": str(report_path)}
    except Exception as exc:
        return {"error": str(exc), "path": str(report_path)}
