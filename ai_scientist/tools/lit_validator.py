import json
import csv
from pathlib import Path
from typing import Dict, Any, List

from ai_scientist.tools.base_tool import BaseTool


class LitSummaryValidatorTool(BaseTool):
    """
    Validate a lit_summary CSV/JSON for required fields and report coverage.
    """

    def __init__(
        self,
        name: str = "ValidateLitSummary",
        description: str = (
            "Validate lit_summary.csv/json for required fields: "
            "region, axon_length, branch_orders/degree, transport_rate, mitophagy_rate, "
            "ATP_diffusion, calcium_energy_cost. Outputs a coverage report."
        ),
    ):
        parameters = [
            {
                "name": "path",
                "type": "str",
                "description": "Path to lit_summary.csv or lit_summary.json",
            }
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        records: List[Dict[str, Any]] = []
        if p.suffix.lower() == ".json":
            with p.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
        elif p.suffix.lower() == ".csv":
            with p.open() as f:
                reader = csv.DictReader(f)
                records = [dict(row) for row in reader]
        else:
            raise ValueError("Unsupported file type; use .csv or .json")

        required_fields = [
            "region",
            "axon_length",
            "branch_order",
            "node_degree",
            "transport_rate",
            "mitophagy_rate",
            "atp_diffusion_time",
            "calcium_energy_cost",
        ]

        coverage = {f: 0 for f in required_fields}
        for r in records:
            for f in required_fields:
                if r.get(f) not in (None, "", "NA"):
                    coverage[f] += 1

        n = len(records)
        coverage_pct = {f: (coverage[f] / n * 100.0) if n else 0.0 for f in required_fields}

        return {
            "n_records": n,
            "coverage_counts": coverage,
            "coverage_pct": coverage_pct,
            "missing_fields": [f for f in required_fields if coverage[f] == 0],
        }
