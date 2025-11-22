from pathlib import Path
from typing import Dict, Any, List
import json
from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.compartmental_sim import simulate_compartmental, load_graph


class RunInterventionTesterTool(BaseTool):
    """
    Apply parameter interventions (transport_rate, demand_scale) and report effect on frac_failed.
    """

    def __init__(
        self,
        name: str = "RunInterventionTester",
        description: str = (
            "Run multiple interventions on a graph and report frac_failed vs baseline. "
            "Provide graph_path and lists of transport_rate/demand_scale deltas."
        ),
    ):
        parameters = [
            {"name": "graph_path", "type": "str", "description": "Path to graph .gpickle"},
            {"name": "output_dir", "type": "str", "description": "Output directory (default experiment_results)"},
            {"name": "transport_vals", "type": "list[float]", "description": "Absolute transport_rate values"},
            {"name": "demand_vals", "type": "list[float]", "description": "Absolute demand_scale values"},
            {"name": "baseline_transport", "type": "float", "description": "Baseline transport (default 0.05)"},
            {"name": "baseline_demand", "type": "float", "description": "Baseline demand (default 0.5)"},
        ]
        super().__init__(name, description, parameters)

    def use_tool(
        self,
        graph_path: str,
        output_dir: str = "experiment_results",
        transport_vals: List[float] | None = None,
        demand_vals: List[float] | None = None,
        baseline_transport: float = 0.05,
        baseline_demand: float = 0.5,
    ) -> Dict[str, Any]:
        transport_vals = transport_vals or [baseline_transport * 0.5, baseline_transport, baseline_transport * 1.5]
        demand_vals = demand_vals or [baseline_demand * 0.8, baseline_demand, baseline_demand * 1.2]
        gp = Path(graph_path)
        if gp.is_dir():
            raise ValueError(f"graph_path must be a file, not a directory: {graph_path}")
        G = load_graph(gp)

        results = []
        # Baseline
        base = simulate_compartmental(G, transport_rate=baseline_transport, demand_scale=baseline_demand)
        base_ff = base["frac_failed"]
        results.append(
            {
                "transport_rate": baseline_transport,
                "demand_scale": baseline_demand,
                "frac_failed": base_ff,
                "label": "baseline",
            }
        )
        for tr in transport_vals:
            for dm in demand_vals:
                res = simulate_compartmental(G, transport_rate=tr, demand_scale=dm)
                results.append(
                    {
                        "transport_rate": tr,
                        "demand_scale": dm,
                        "frac_failed": res["frac_failed"],
                        "delta_frac_failed": res["frac_failed"] - base_ff,
                    }
                )

        out_dir = BaseTool.resolve_output_dir(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(graph_path).stem}_interventions.json"
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)

        return {"output_json": str(out_path), "baseline_frac_failed": base_ff, "n_runs": len(results)}
