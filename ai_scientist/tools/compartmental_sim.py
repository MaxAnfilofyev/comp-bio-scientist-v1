# pyright: reportMissingImports=false, reportMissingModuleSource=false
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import networkx as nx
import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.utils.per_compartment import (
    build_node_index_payload,
    compute_topology_metrics,
    write_per_compartment_outputs,
)


def _resolve_graph_path(p: Path) -> Path:
    """
    Resolve a graph file path, trying:
    - provided path
    - AISC_EXP_RESULTS/<name>
    - AISC_BASE_FOLDER/<name>
    """
    return BaseTool.resolve_input_path(str(p), allow_dir=False)


def load_graph(graph_path: Path | str) -> nx.Graph:
    """
    Load a graph from common formats used in the project.
    """
    graph_path = Path(graph_path)
    if graph_path.is_dir():
        raise ValueError(f"graph_path must be a file, not a directory: {graph_path}")
    graph_path = _resolve_graph_path(graph_path)
    suffix = graph_path.suffix.lower()
    if suffix in {".gpickle", ".pickle", ".pkl"}:
        with graph_path.open("rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.Graph):
            raise ValueError(f"Pickled object is not a networkx Graph: {type(graph)}")
        return graph
    if suffix in {".graphml", ".xml"}:
        return nx.read_graphml(graph_path)
    if suffix == ".gml":
        return nx.read_gml(graph_path)
    if suffix in {".npz", ".npy"}:
        # Assume adjacency matrix saved via numpy
        arr = np.load(graph_path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # grab first array in the archive
            key = arr.files[0]
            arr = arr[key]
        graph = nx.from_numpy_array(arr)
        return graph
    raise ValueError(f"Unsupported graph format: {suffix}")


def simulate_compartmental(
    G: nx.Graph,
    steps: int = 200,
    dt: float = 0.1,
    transport_rate: float = 0.05,
    demand_scale: float = 0.5,
    mitophagy_rate: float = 0.02,
    noise_std: float = 0.0,
    seed: int = 0,
    return_arrays: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], np.ndarray, np.ndarray]]:
    """
    Minimal compartmental simulation on a graph with E (energy) and M (mitochondrial capacity).
    Not biophysically detailed; intended as a placeholder for agent-driven refinement.
    """
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    E = np.ones(n, dtype=float)
    M = np.ones(n, dtype=float)

    e_hist: list[np.ndarray] = []
    m_hist: list[np.ndarray] = []

    for _ in range(steps):
        deg = np.array([nx.degree(G, node) for node in nodes], dtype=float) + 1e-6
        demand = demand_scale * deg

        # Energy dynamics: production from M, consumption from demand, decay when low
        dE = 0.1 * M - demand * E
        if noise_std > 0:
            dE += rng.normal(0, noise_std, size=n)

        # Mitochondrial dynamics: turnover minus load-dependent damage, plus transport diffusion
        dM = 0.05 * (1 - M) - mitophagy_rate * demand * (1 - E)

        # Simple transport: Laplacian diffusion of M
        L = nx.laplacian_matrix(G, nodelist=nodes).astype(float)
        dM -= transport_rate * (L @ M)

        E = np.clip(E + dt * dE, 0.0, 1.5)
        M = np.clip(M + dt * dM, 0.0, 1.5)

        e_hist.append(E.copy())
        m_hist.append(M.copy())

    e_arr = np.array(e_hist)
    m_arr = np.array(m_hist)
    frac_failed = float(np.mean(e_arr[-1] < 0.2))

    result: Dict[str, Any] = {
        "time": (np.arange(steps) * dt).tolist(),
        "E": e_arr.tolist(),
        "M": m_arr.tolist(),
        "frac_failed": frac_failed,
        "n_nodes": n,
    }
    if return_arrays:
        return result, e_arr, m_arr
    return result


class RunCompartmentalSimTool(BaseTool):
    """
    Run a minimal compartmental simulation on a graph file and save outputs.
    """

    def __init__(
        self,
        name: str = "RunCompartmentalSimulation",
        description: str = (
            "Run a simple E/M compartmental simulation on a graph (.gpickle/.graphml/.gml). "
            "Saves JSON with time, E/M trajectories, frac_failed. Writes per_compartment.npz + topology summary for downstream analyses."
        ),
    ):
        parameters = [
            {"name": "graph_path", "type": "str", "description": "Path to graph file (.gpickle/.graphml/.gml)"},
            {"name": "output_dir", "type": "str", "description": "Output directory (default experiment_results)"},
            {"name": "steps", "type": "int", "description": "Number of steps (default 200)"},
            {"name": "dt", "type": "float", "description": "Timestep (default 0.1)"},
            {"name": "transport_rate", "type": "float", "description": "Transport rate (default 0.05)"},
            {"name": "demand_scale", "type": "float", "description": "Demand scaling (default 0.5)"},
            {"name": "mitophagy_rate", "type": "float", "description": "Mitophagy rate (default 0.02)"},
            {"name": "noise_std", "type": "float", "description": "Noise std (default 0.0)"},
            {"name": "seed", "type": "int", "description": "Random seed (default 0)"},
            {"name": "store_timeseries", "type": "bool", "description": "Store full time series (default True)"},
            {"name": "downsample", "type": "int", "description": "Store every Nth step (default 1)"},
            {"name": "max_elements", "type": "int", "description": "Safety limit: steps * nodes (default 5000000)"},
            {"name": "status_path", "type": "str", "description": "Optional status json path to track run state"},
            {"name": "export_arrays", "type": "bool", "description": "Also write failure_matrix.npy/time_vector.npy/nodes_order.txt (default False)"},
            {"name": "failure_threshold", "type": "float", "description": "Failure threshold for export_arrays (default 0.2)"},
            {"name": "export_output_dir", "type": "str", "description": "Optional output dir for exported arrays (default sim folder)"},
            {"name": "write_per_compartment", "type": "bool", "description": "Write per_compartment.npz + node_index_map + topology summary (default True)"},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs) -> Dict[str, Any]:
        graph_path = kwargs.get("graph_path")
        if graph_path is None or str(graph_path).strip() == "":
            raise ValueError("graph_path is required (build one with BuildGraphsTool or provide a path)")

        base_out = BaseTool.resolve_output_dir(kwargs.get("output_dir"))
        output_dir = base_out
        steps = int(kwargs.get("steps", 200))
        dt = float(kwargs.get("dt", 0.1))
        transport_rate = float(kwargs.get("transport_rate", 0.05))
        demand_scale = float(kwargs.get("demand_scale", 0.5))
        mitophagy_rate = float(kwargs.get("mitophagy_rate", 0.02))
        noise_std = float(kwargs.get("noise_std", 0.0))
        seed = int(kwargs.get("seed", 0))
        store_timeseries = bool(kwargs.get("store_timeseries", True))
        downsample = int(kwargs.get("downsample", 1))
        max_elements = int(kwargs.get("max_elements", 5_000_000))
        status_path = kwargs.get("status_path")
        export_arrays = bool(kwargs.get("export_arrays", False))
        failure_threshold = float(kwargs.get("failure_threshold", 0.2))
        export_output_dir = kwargs.get("export_output_dir") or output_dir
        export_output_dir = resolve_output_path(
            subdir=None,
            name="",
            run_root=BaseTool.resolve_output_dir(export_output_dir),
            allow_quarantine=True,
            unique=False,
        )[0]

        graph = load_graph(Path(graph_path))
        n_nodes = graph.number_of_nodes()
        if steps * max(1, n_nodes) > max_elements:
            raise ValueError(
                f"Simulation too large (steps * nodes = {steps * n_nodes} > {max_elements}). "
                "Reduce steps, downsample, or increase max_elements if intentional."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        status_file = (
            Path(status_path)
            if status_path
            else resolve_output_path(
                subdir=None,
                name=f"{Path(graph_path).stem}_sim.status.json",
                run_root=output_dir,
                allow_quarantine=True,
                unique=False,
            )[0]
        )
        # Write running status so agents see progress
        try:
            with status_file.open("w") as sf:
                json.dump(
                    {
                        "status": "running",
                        "graph_path": str(graph_path),
                        "output_dir": str(output_dir),
                        "steps": steps,
                        "dt": dt,
                        "transport_rate": transport_rate,
                        "demand_scale": demand_scale,
                        "start_time": time.time(),
                    },
                    sf,
                    indent=2,
                )
        except Exception:
            pass

        sim_result, e_arr, m_arr = simulate_compartmental(
            graph,
            steps=steps,
            dt=dt,
            transport_rate=transport_rate,
            demand_scale=demand_scale,
            mitophagy_rate=mitophagy_rate,
            noise_std=noise_std,
            seed=seed,
            return_arrays=True,
        )

        if downsample > 1:
            e_arr = e_arr[::downsample]
            m_arr = m_arr[::downsample]
            sim_result["time"] = sim_result["time"][::downsample]
            sim_result["E"] = e_arr.tolist()
            sim_result["M"] = m_arr.tolist()
        else:
            sim_result["E"] = e_arr.tolist()
            sim_result["M"] = m_arr.tolist()

        result = sim_result

        if not store_timeseries:
            result = {
                "frac_failed": result["frac_failed"],
                "n_nodes": result.get("n_nodes"),
                "final_E": result["E"][-1] if result.get("E") else None,
                "final_M": result["M"][-1] if result.get("M") else None,
                "transport_rate": transport_rate,
                "demand_scale": demand_scale,
                "mitophagy_rate": mitophagy_rate,
                "noise_std": noise_std,
                "seed": seed,
            }
        out_path = resolve_output_path(
            subdir=None,
            name=f"{Path(graph_path).stem}_sim.json",
            run_root=output_dir,
            allow_quarantine=True,
            unique=True,
        )[0]
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

        try:
            with status_file.open("w") as sf:
                json.dump(
                    {
                        "status": "completed",
                        "output_json": str(out_path),
                        "frac_failed": result.get("frac_failed"),
                        "export_arrays": export_arrays,
                        "end_time": time.time(),
                        "duration_sec": None,  # can be filled by caller if desired
                    },
                    sf,
                    indent=2,
                )
        except Exception:
            pass

        export_result = None
        per_compartment_output = bool(kwargs.get("write_per_compartment", True))
        per_comp_status: Dict[str, Any] = {}
        if per_compartment_output:
            time_arr = np.array(sim_result["time"], dtype=float)
            binary_states = (e_arr < failure_threshold).astype(np.uint8)
            continuous_states = np.stack([e_arr, m_arr], axis=-1)
            nodes = list(graph.nodes())
            node_index_payload = build_node_index_payload(nodes)
            topology_metrics = compute_topology_metrics(graph, nodes, node_index_payload["ordering_checksum"])
            per_comp_status = write_per_compartment_outputs(
                output_dir=Path(export_output_dir),
                binary_states=binary_states,
                continuous_states=continuous_states,
                time_vector=time_arr,
                node_index_payload=node_index_payload,
                topology_metrics=topology_metrics,
                status=topology_metrics.get("status", "ok"),
            )

        if export_arrays:
            try:
                from ai_scientist.tools.sim_postprocess import export_sim_timeseries

                export_result = export_sim_timeseries(
                    sim_json_path=out_path,
                    graph_path=graph_path,
                    output_dir=export_output_dir,
                    failure_threshold=failure_threshold,
                )
            except Exception as exc:  # pragma: no cover - defensive
                export_result = {"error": f"export_arrays failed: {exc}"}

        return {
            "output_json": str(out_path),
            "frac_failed": result["frac_failed"],
            "exported": export_result,
            "per_compartment": per_comp_status or None,
        }
