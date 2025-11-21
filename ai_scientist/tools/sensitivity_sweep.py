import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import networkx as nx

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.compartmental_sim import simulate_compartmental


class RunSensitivitySweepTool(BaseTool):
    """
    Run a simple parameter sweep over transport_rate and demand_scale on a graph.
    Saves CSV of frac_failed vs parameters.
    """

    def __init__(
        self,
        name: str = "RunSensitivitySweep",
        description: str = (
            "Run a grid/Latin-hypercube-like sweep over transport_rate and demand_scale "
            "for a graph (.gpickle) using the minimal compartmental simulator. "
            "Outputs CSV in output_dir."
        ),
    ):
        parameters = [
            {"name": "graph_path", "type": "str", "description": "Path to graph .gpickle"},
            {"name": "output_dir", "type": "str", "description": "Output directory (default experiment_results)"},
            {"name": "transport_vals", "type": "list[float]", "description": "Values for transport_rate"},
            {"name": "demand_vals", "type": "list[float]", "description": "Values for demand_scale"},
            {"name": "steps", "type": "int", "description": "Simulation steps (default 150)"},
            {"name": "dt", "type": "float", "description": "Timestep (default 0.1)"},
        ]
        super().__init__(name, description, parameters)

    def use_tool(
        self,
        graph_path: str,
        output_dir: str = "experiment_results",
        transport_vals: List[float] | None = None,
        demand_vals: List[float] | None = None,
        steps: int = 150,
        dt: float = 0.1,
    ) -> Dict[str, Any]:
        transport_vals = transport_vals or [0.02, 0.05, 0.1]
        demand_vals = demand_vals or [0.3, 0.5, 0.7]
        G = nx.read_gpickle(graph_path)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{Path(graph_path).stem}_sensitivity.csv"

        rows = []
        for tr in transport_vals:
            for dm in demand_vals:
                res = simulate_compartmental(
                    G,
                    steps=steps,
                    dt=dt,
                    transport_rate=tr,
                    demand_scale=dm,
                    mitophagy_rate=0.02,
                    noise_std=0.0,
                    seed=0,
                )
                rows.append({"transport_rate": tr, "demand_scale": dm, "frac_failed": res["frac_failed"]})

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["transport_rate", "demand_scale", "frac_failed"])
            writer.writeheader()
            writer.writerows(rows)

        return {"output_csv": str(csv_path), "n_rows": len(rows)}
