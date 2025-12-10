import json
import math
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metrics input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        return pd.DataFrame(data if isinstance(data, list) else [data])
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _critical_transport_est(df: pd.DataFrame) -> Optional[float]:
    if "transport" not in df.columns or "frac_failed" not in df.columns:
        return None
    df_sorted = df.sort_values("transport")
    above = df_sorted[df_sorted["frac_failed"] >= 0.5]
    if above.empty:
        return None
    return float(above["transport"].iloc[0])


def _aggregate_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    if "frac_failed" in df.columns:
        metrics["overall_mean_frac_failed"] = float(df["frac_failed"].mean())
        metrics["overall_max_frac_failed"] = float(df["frac_failed"].max())
    crit = _critical_transport_est(df)
    if crit is not None:
        metrics["critical_transport_est"] = crit

    grouped_rows: List[Dict[str, Any]] = []
    key_cols = [c for c in ["transport", "demand", "demand_scale", "baseline"] if c in df.columns]
    if key_cols:
        grouped = df.groupby(key_cols)
        for keys, sub in grouped:
            row: Dict[str, Any] = {}
            if isinstance(keys, tuple):
                for k, v in zip(key_cols, keys):
                    row[k] = v
            else:
                row[key_cols[0]] = keys
            if "frac_failed" in sub.columns:
                row["mean_frac_failed"] = float(sub["frac_failed"].mean())
                row["max_frac_failed"] = float(sub["frac_failed"].max())
            grouped_rows.append(row)
    return pd.DataFrame(grouped_rows), metrics


class ComputeModelMetricsTool(BaseTool):
    """
    Compute domain-specific metrics from sweep CSVs or model outputs.
    Produces sweep_metrics_csv and/or model_metrics_json artifacts.
    """

    def __init__(
        self,
        name: str = "ComputeModelMetrics",
        description: str = (
            "Compute domain-specific metrics from a sweep CSV (transport/demand vs frac_failed) "
            "or model outputs. Emits *_metrics.csv (grouped stats) and optional {model_key}_metrics.json "
            "with summary statistics like critical_transport_est."
        ),
    ):
        parameters = [
            {
                "name": "input_path",
                "type": "str",
                "description": "Path to sweep CSV/JSON with columns e.g., transport, demand, frac_failed.",
            },
            {
                "name": "label",
                "type": "str",
                "description": "Label for output metrics CSV filename (default: stem of input).",
            },
            {
                "name": "model_key",
                "type": "str",
                "description": "Optional model key; if set, also write {model_key}_metrics.json.",
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Override output dir; defaults to experiment_results.",
            },
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs: Any) -> Dict[str, Any]:
        input_path = kwargs.get("input_path")
        if not input_path:
            raise ValueError("input_path is required")
        label = kwargs.get("label")
        model_key = kwargs.get("model_key")
        output_dir = kwargs.get("output_dir")

        in_path = BaseTool.resolve_input_path(input_path)
        df = _load_table(in_path)

        # Compute aggregates
        grouped_df, summary_metrics = _aggregate_metrics(df)

        # Resolve outputs
        run_root = BaseTool.resolve_output_dir(output_dir)
        label_use = label or in_path.stem
        metrics_csv_path, _, _ = resolve_output_path(
            subdir="simulations", name=f"{label_use}_metrics.csv", run_root=run_root, allow_quarantine=False, unique=False
        )
        metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not grouped_df.empty:
            grouped_df.to_csv(metrics_csv_path, index=False)
        else:
            # write header-only if nothing to aggregate
            with metrics_csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["note"])
                writer.writerow(["no grouped metrics available"])

        summary_json_path = None
        if model_key:
            summary_json_path, _, _ = resolve_output_path(
                subdir="models", name=f"{model_key}_metrics.json", run_root=run_root, allow_quarantine=False, unique=False
            )
            summary_payload = {
                "model_key": model_key,
                "source": str(in_path),
                "metrics": summary_metrics,
                "n_rows": int(len(df)),
            }
            summary_json_path.parent.mkdir(parents=True, exist_ok=True)
            summary_json_path.write_text(json.dumps(summary_payload, indent=2))

        return {
            "input": str(in_path),
            "output_csv": str(metrics_csv_path),
            "model_metrics_json": str(summary_json_path) if summary_json_path else None,
            "summary": summary_metrics,
        }
