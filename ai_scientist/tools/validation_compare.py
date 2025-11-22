import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from ai_scientist.tools.base_tool import BaseTool


class RunValidationCompareTool(BaseTool):
    """
    Compare model outputs against lit_summary: compute simple correlations between arbor metrics and collapse fraction.
    """

    def __init__(
        self,
        name: str = "RunValidationCompare",
        description: str = (
            "Compare lit_summary metrics (axon_length, branch_order, node_degree) to model outputs "
            "(frac_failed). Provide paths to lit_summary (csv/json) and sim json."
        ),
    ):
        parameters = [
            {"name": "lit_path", "type": "str", "description": "Path to lit_summary.csv/json"},
            {"name": "sim_path", "type": "str", "description": "Path to simulation json with frac_failed"},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, lit_path: str, sim_path: str) -> Dict[str, Any]:
        lit_p = BaseTool.resolve_input_path(lit_path, allow_dir=False)
        sim_p = BaseTool.resolve_input_path(sim_path, allow_dir=True)

        # If a directory was provided for sim_path, search for a usable JSON
        if sim_p.is_dir():
            candidates = sorted(sim_p.glob("*.json"))
            if not candidates:
                raise FileNotFoundError(f"No simulation JSONs found under directory: {sim_p}")
            picked = None
            for cand in candidates:
                try:
                    with cand.open() as f:
                        sim_probe = json.load(f)
                    if "frac_failed" in sim_probe:
                        picked = cand
                        break
                except Exception:
                    continue
            if picked is None:
                picked = candidates[0]
            sim_p = picked

        # Load lit
        lit_records: List[Dict[str, Any]] = []
        if lit_p.suffix.lower() == ".json":
            with lit_p.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                lit_records = data
            elif isinstance(data, dict):
                lit_records = [data]
        else:
            with lit_p.open() as f:
                reader = csv.DictReader(f)
                lit_records = [dict(row) for row in reader]

        with sim_p.open() as f:
            sim = json.load(f)

        frac_failed = sim.get("frac_failed", None)
        if frac_failed is None:
            raise ValueError("Simulation JSON missing 'frac_failed'")

        def _to_float_or_nan(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        axon_lengths = np.array([_to_float_or_nan(r.get("axon_length")) for r in lit_records])
        branch_orders = np.array([_to_float_or_nan(r.get("branch_order")) for r in lit_records])
        node_degrees = np.array([_to_float_or_nan(r.get("node_degree")) for r in lit_records])

        def corr(vec):
            vec = vec[~np.isnan(vec)]
            if vec.size < 2:
                return None
            return float(np.corrcoef(vec, np.full_like(vec, frac_failed))[0, 1])

        return {
            "frac_failed": frac_failed,
            "corr_axon_length": corr(axon_lengths),
            "corr_branch_order": corr(branch_orders),
            "corr_node_degree": corr(node_degrees),
            "n_lit_records": len(lit_records),
        }
