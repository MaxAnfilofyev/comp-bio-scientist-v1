import csv
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.tools.compartmental_sim import (
    simulate_compartmental,
    load_graph,
    write_per_compartment_outputs,
    build_node_index_payload,
    compute_topology_metrics,
)


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
        steps: int = 150,
        dt: float = 0.1,
        failure_threshold: float = 0.2,
    ) -> Dict[str, Any]:
        transport_vals = transport_vals or [0.02, 0.05, 0.1]
        demand_vals = demand_vals or [0.3, 0.5, 0.7]
        gp = Path(graph_path)
        if gp.is_dir():
            raise ValueError(f"graph_path must be a file, not a directory: {graph_path}")
        G = load_graph(gp)

        out_dir = BaseTool.resolve_output_dir(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path, _, _ = resolve_output_path(
            subdir=None,
            name=f"{Path(graph_path).stem}_sensitivity.csv",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )
        node_index_payload = build_node_index_payload(list(G.nodes()))
        topology_metrics = compute_topology_metrics(G, list(G.nodes()), node_index_payload["ordering_checksum"])
        per_comp_outputs: List[Dict[str, Any]] = []

        rows = []
        for tr in transport_vals:
            for dm in demand_vals:
                res, e_arr, m_arr = simulate_compartmental(
                    G,
                    steps=steps,
                    dt=dt,
                    transport_rate=tr,
                    demand_scale=dm,
                    mitophagy_rate=0.02,
                    noise_std=0.0,
                    seed=0,
                    return_arrays=True,
                )
                rows.append({"transport_rate": tr, "demand_scale": dm, "frac_failed": res["frac_failed"]})
                sim_subdir, _, _ = resolve_output_path(
                    subdir=None,
                    name=f"transport_{tr}_demand_{dm}",
                    run_root=out_dir,
                    allow_quarantine=True,
                    unique=False,
                )
                time_arr = np.array(res["time"], dtype=float)
                binary_states = (np.array(e_arr) < failure_threshold).astype(np.uint8)
                continuous_states = np.stack([e_arr, m_arr], axis=-1)
                per_comp_outputs.append(
                    write_per_compartment_outputs(
                        output_dir=sim_subdir,
                        binary_states=binary_states,
                        continuous_states=continuous_states,
                        time_vector=time_arr,
                        node_index_payload=node_index_payload,
                        topology_metrics=topology_metrics,
                        status=topology_metrics.get("status", "ok"),
                    )
                )

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["transport_rate", "demand_scale", "frac_failed"])
            writer.writeheader()
            writer.writerows(rows)

        return {
            "output_csv": str(csv_path),
            "n_rows": len(rows),
            "per_compartment_outputs": per_comp_outputs,
        }
