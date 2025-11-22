# pyright: reportMissingImports=false, reportMissingModuleSource=false
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.compartmental_sim import load_graph


def export_sim_timeseries(
    sim_json_path: Path | str,
    graph_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    failure_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Convert a sim.json into failure_matrix.npy, time_vector.npy, and nodes_order.txt.
    """
    sim_json_path = BaseTool.resolve_input_path(str(sim_json_path), allow_dir=False)
    with sim_json_path.open() as f:
        sim_data = json.load(f)

    time = sim_data.get("time")
    e_vals = sim_data.get("E")
    if time is None or e_vals is None:
        raise ValueError("sim.json missing required keys 'time' and/or 'E'")

    e_arr = np.array(e_vals, dtype=float)
    t_arr = np.array(time, dtype=float)

    # Align lengths defensively
    min_len = min(len(t_arr), e_arr.shape[0])
    e_arr = e_arr[:min_len]
    t_arr = t_arr[:min_len]

    if e_arr.ndim != 2:
        raise ValueError(f"Expected E to be 2D (time x nodes); got shape {e_arr.shape}")
    n_time, n_nodes = e_arr.shape

    nodes: List[Any]
    if graph_path:
        graph = load_graph(Path(graph_path))
        nodes = list(graph.nodes())
        if len(nodes) != n_nodes:
            # Fallback to index order if mismatch
            nodes = list(range(n_nodes))
    else:
        nodes = list(range(n_nodes))

    failure_matrix = (e_arr < failure_threshold).astype(np.uint8)

    out_dir = BaseTool.resolve_output_dir(output_dir) if output_dir else sim_json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = sim_json_path.stem.replace(".json", "")
    failure_path = out_dir / f"{stem}_failure_matrix.npy"
    time_path = out_dir / f"{stem}_time_vector.npy"
    nodes_path = out_dir / f"nodes_order_{stem}.txt"

    np.save(failure_path, failure_matrix)
    np.save(time_path, t_arr)
    with nodes_path.open("w") as nf:
        for node in nodes:
            nf.write(f"{node}\n")

    return {
        "failure_matrix": str(failure_path),
        "time_vector": str(time_path),
        "nodes_order": str(nodes_path),
        "n_timepoints": int(n_time),
        "n_nodes": int(n_nodes),
    }


class SimPostprocessTool(BaseTool):
    """
    Convert sim.json outputs to .npy arrays and node order text.
    """

    def __init__(
        self,
        name: str = "SimPostprocess",
        description: str = (
            "Convert sim.json (with time/E) into failure_matrix.npy, time_vector.npy, and nodes_order.txt. "
            "Optionally provide graph_path to emit node names; otherwise indices are used."
        ),
    ):
        parameters = [
            {"name": "sim_json_path", "type": "str", "description": "Path to sim.json file."},
            {"name": "output_dir", "type": "str", "description": "Output directory (default: sim folder)."},
            {"name": "graph_path", "type": "str", "description": "Optional graph file to derive node ordering."},
            {"name": "failure_threshold", "type": "float", "description": "Energy threshold for failure (default 0.2)."},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs) -> Dict[str, Any]:
        sim_json_path = kwargs.get("sim_json_path")
        if not sim_json_path:
            raise ValueError("sim_json_path is required")
        output_dir = kwargs.get("output_dir")
        graph_path = kwargs.get("graph_path")
        failure_threshold = float(kwargs.get("failure_threshold", 0.2))

        return export_sim_timeseries(
            sim_json_path=sim_json_path,
            graph_path=graph_path,
            output_dir=output_dir,
            failure_threshold=failure_threshold,
        )
