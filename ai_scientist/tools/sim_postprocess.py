# pyright: reportMissingImports=false, reportMissingModuleSource=false
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.compartmental_sim import load_graph
from ai_scientist.utils.per_compartment import derive_per_compartment_from_arrays
from ai_scientist.utils.pathing import resolve_output_path


def export_sim_timeseries(
    sim_json_path: Path | str,
    graph_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    failure_threshold: float = 0.2,
    write_per_compartment: bool = True,
    per_compartment_threshold: Optional[float] = None,
    force_per_compartment: bool = True,
) -> Dict[str, Any]:
    """
    Convert a sim.json into failure_matrix.npy, time_vector.npy, and nodes_order.txt.
    Also emits per_compartment.npz + node_index_map.json + topology_summary.json when requested.
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

    # Route outputs through the sanitized resolver to keep everything under the run root.
    base_dir = BaseTool.resolve_output_dir(output_dir) if output_dir else sim_json_path.parent
    out_dir, _, _ = resolve_output_path(
        subdir=None,
        name="",
        run_root=base_dir,
        allow_quarantine=True,
        unique=False,
    )

    stem = sim_json_path.stem.replace(".json", "")
    failure_path, _, _ = resolve_output_path(
        subdir=None,
        name=f"{stem}_failure_matrix.npy",
        run_root=out_dir,
        allow_quarantine=True,
        unique=True,
    )
    time_path, _, _ = resolve_output_path(
        subdir=None,
        name=f"{stem}_time_vector.npy",
        run_root=out_dir,
        allow_quarantine=True,
        unique=True,
    )
    nodes_path, _, _ = resolve_output_path(
        subdir=None,
        name=f"nodes_order_{stem}.txt",
        run_root=out_dir,
        allow_quarantine=True,
        unique=True,
    )

    np.save(failure_path, failure_matrix)
    np.save(time_path, t_arr)
    with nodes_path.open("w") as nf:
        for node in nodes:
            nf.write(f"{node}\n")

    per_compartment_result: Dict[str, Any] = {}
    if write_per_compartment:
        per_compartment_result = derive_per_compartment_from_arrays(
            failure_matrix=failure_matrix,
            time_vector=t_arr,
            nodes_order=nodes,
            output_dir=out_dir,
            binary_threshold=per_compartment_threshold,
            allow_mismatch=False,
            skip_existing=not force_per_compartment,
            provenance="sim_postprocess",
        )

    return {
        "failure_matrix": str(failure_path),
        "time_vector": str(time_path),
        "nodes_order": str(nodes_path),
        "n_timepoints": int(n_time),
        "n_nodes": int(n_nodes),
        "per_compartment": per_compartment_result or None,
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
            "Optionally provide graph_path to emit node names; otherwise indices are used. "
            "Also writes per_compartment.npz + node_index_map.json + topology_summary.json for downstream validation."
        ),
    ):
        parameters = [
            {"name": "sim_json_path", "type": "str", "description": "Path to sim.json file."},
            {"name": "output_dir", "type": "str", "description": "Output directory (default: sim folder)."},
            {"name": "graph_path", "type": "str", "description": "Optional graph file to derive node ordering."},
            {"name": "failure_threshold", "type": "float", "description": "Energy threshold for failure (default 0.2)."},
            {
                "name": "write_per_compartment",
                "type": "bool",
                "description": "Also emit per_compartment.npz/node_index_map/topology_summary (default True).",
            },
            {
                "name": "per_compartment_threshold",
                "type": "float",
                "description": "Threshold to binarize non-binary failure matrices (default >0 when needed).",
            },
            {
                "name": "force_per_compartment",
                "type": "bool",
                "description": "Overwrite per_compartment artifacts if present (default True).",
            },
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs) -> Dict[str, Any]:
        sim_json_path = kwargs.get("sim_json_path")
        if not sim_json_path:
            raise ValueError("sim_json_path is required")
        output_dir = kwargs.get("output_dir")
        graph_path = kwargs.get("graph_path")
        failure_threshold = float(kwargs.get("failure_threshold", 0.2))
        write_per_compartment = bool(kwargs.get("write_per_compartment", True))
        per_comp_threshold = kwargs.get("per_compartment_threshold")
        force_per_compartment = bool(kwargs.get("force_per_compartment", True))
        pct = float(per_comp_threshold) if per_comp_threshold is not None else None

        return export_sim_timeseries(
            sim_json_path=sim_json_path,
            graph_path=graph_path,
            output_dir=output_dir,
            failure_threshold=failure_threshold,
            write_per_compartment=write_per_compartment,
            per_compartment_threshold=pct,
            force_per_compartment=force_per_compartment,
        )
