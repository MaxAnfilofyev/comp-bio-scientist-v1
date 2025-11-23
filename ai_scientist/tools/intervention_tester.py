import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.compartmental_sim import (
    simulate_compartmental,
    load_graph,
    write_per_compartment_outputs,
    build_node_index_payload,
    compute_topology_metrics,
)


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
            {
                "name": "failure_threshold",
                "type": "float",
                "description": "Threshold for binary failure state when writing per_compartment outputs (default 0.2)",
            },
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
        failure_threshold: float = 0.2,
    ) -> Dict[str, Any]:
        transport_vals = transport_vals or [baseline_transport * 0.5, baseline_transport, baseline_transport * 1.5]
        demand_vals = demand_vals or [baseline_demand * 0.8, baseline_demand, baseline_demand * 1.2]
        gp = Path(graph_path)
        if gp.is_dir():
            raise ValueError(f"graph_path must be a file, not a directory: {graph_path}")
        G = load_graph(gp)
        nodes = list(G.nodes())
        node_index_payload = build_node_index_payload(nodes)
        topology_metrics = compute_topology_metrics(G, nodes, node_index_payload["ordering_checksum"])

        results = []
        per_compartment_outputs: List[Dict[str, Any]] = []
        # Baseline
        base, base_e, base_m = simulate_compartmental(
            G, transport_rate=baseline_transport, demand_scale=baseline_demand, return_arrays=True
        )
        base_ff = base["frac_failed"]
        results.append(
            {
                "transport_rate": baseline_transport,
                "demand_scale": baseline_demand,
                "frac_failed": base_ff,
                "label": "baseline",
            }
        )
        base_subdir = BaseTool.resolve_output_dir(output_dir) / "baseline"
        base_time = np.array(base["time"], dtype=float)
        per_compartment_outputs.append(
            write_per_compartment_outputs(
                output_dir=base_subdir,
                binary_states=(np.array(base_e) < failure_threshold).astype(np.uint8),
                continuous_states=np.stack([base_e, base_m], axis=-1),
                time_vector=base_time,
                node_index_payload=node_index_payload,
                topology_metrics=topology_metrics,
                status=topology_metrics.get("status", "ok"),
            )
        )
        for tr in transport_vals:
            for dm in demand_vals:
                res, e_arr, m_arr = simulate_compartmental(
                    G, transport_rate=tr, demand_scale=dm, return_arrays=True
                )
                results.append(
                    {
                        "transport_rate": tr,
                        "demand_scale": dm,
                        "frac_failed": res["frac_failed"],
                        "delta_frac_failed": res["frac_failed"] - base_ff,
                    }
                )
                sim_subdir = BaseTool.resolve_output_dir(output_dir) / f"transport_{tr}_demand_{dm}"
                time_arr = np.array(res["time"], dtype=float)
                per_compartment_outputs.append(
                    write_per_compartment_outputs(
                        output_dir=sim_subdir,
                        binary_states=(np.array(e_arr) < failure_threshold).astype(np.uint8),
                        continuous_states=np.stack([e_arr, m_arr], axis=-1),
                        time_vector=time_arr,
                        node_index_payload=node_index_payload,
                        topology_metrics=topology_metrics,
                        status=topology_metrics.get("status", "ok"),
                    )
                )

        out_dir = BaseTool.resolve_output_dir(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(graph_path).stem}_interventions.json"
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)

        return {
            "output_json": str(out_path),
            "baseline_frac_failed": base_ff,
            "n_runs": len(results),
            "per_compartment_outputs": per_compartment_outputs,
        }
