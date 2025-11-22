# pyright: reportMissingImports=false, reportMissingModuleSource=false
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np

from ai_scientist.tools.base_tool import BaseTool


def load_graph(graph_path: Path) -> nx.Graph:
    """
    Load a graph from common formats used in the project.
    """
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
) -> Dict[str, Any]:
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

    return {
        "time": (np.arange(steps) * dt).tolist(),
        "E": e_arr.tolist(),
        "M": m_arr.tolist(),
        "frac_failed": frac_failed,
        "n_nodes": n,
    }


class RunCompartmentalSimTool(BaseTool):
    """
    Run a minimal compartmental simulation on a graph file and save outputs.
    """

    def __init__(
        self,
        name: str = "RunCompartmentalSimulation",
        description: str = (
            "Run a simple E/M compartmental simulation on a graph (.gpickle/.graphml/.gml). "
            "Saves JSON with time, E/M trajectories, frac_failed."
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
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs) -> Dict[str, Any]:
        graph_path = kwargs.get("graph_path")
        if graph_path is None:
            raise ValueError("graph_path is required")

        output_dir = kwargs.get("output_dir", "experiment_results")
        steps = int(kwargs.get("steps", 200))
        dt = float(kwargs.get("dt", 0.1))
        transport_rate = float(kwargs.get("transport_rate", 0.05))
        demand_scale = float(kwargs.get("demand_scale", 0.5))
        mitophagy_rate = float(kwargs.get("mitophagy_rate", 0.02))
        noise_std = float(kwargs.get("noise_std", 0.0))
        seed = int(kwargs.get("seed", 0))

        graph = load_graph(Path(graph_path))
        result = simulate_compartmental(
            graph,
            steps=steps,
            dt=dt,
            transport_rate=transport_rate,
            demand_scale=demand_scale,
            mitophagy_rate=mitophagy_rate,
            noise_std=noise_std,
            seed=seed,
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(graph_path).stem}_sim.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

        return {"output_json": str(out_path), "frac_failed": result["frac_failed"]}
