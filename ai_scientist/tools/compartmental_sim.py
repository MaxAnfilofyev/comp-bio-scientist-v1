import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import networkx as nx

from ai_scientist.tools.base_tool import BaseTool


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
    n = G.number_of_nodes()
    E = np.ones(n, dtype=float)
    M = np.ones(n, dtype=float)

    history = {"E": [], "M": []}

    for _ in range(steps):
        deg = np.array([G.degree(i) for i in range(n)], dtype=float) + 1e-6
        demand = demand_scale * deg

        # Energy dynamics: production from M, consumption from demand, decay when low
        dE = 0.1 * M - demand * E
        if noise_std > 0:
            dE += rng.normal(0, noise_std, size=n)

        # Mitochondrial dynamics: turnover minus load-dependent damage, plus transport diffusion
        dM = 0.05 * (1 - M) - mitophagy_rate * demand * (1 - E)

        # Simple transport: Laplacian diffusion of M
        L = nx.laplacian_matrix(G).astype(float)
        dM -= transport_rate * (L @ M)

        E = np.clip(E + dt * dE, 0.0, 1.5)
        M = np.clip(M + dt * dM, 0.0, 1.5)

        history["E"].append(E.copy())
        history["M"].append(M.copy())

    history["E"] = np.array(history["E"])
    history["M"] = np.array(history["M"])
    frac_failed = float(np.mean(history["E"][-1] < 0.2))

    return {
        "time": (np.arange(steps) * dt).tolist(),
        "E": history["E"].tolist(),
        "M": history["M"].tolist(),
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
            "Run a simple E/M compartmental simulation on a graph (.gpickle). "
            "Saves JSON with time, E/M trajectories, frac_failed."
        ),
    ):
        parameters = [
            {"name": "graph_path", "type": "str", "description": "Path to graph .gpickle"},
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

    def use_tool(
        self,
        graph_path: str,
        output_dir: str = "experiment_results",
        steps: int = 200,
        dt: float = 0.1,
        transport_rate: float = 0.05,
        demand_scale: float = 0.5,
        mitophagy_rate: float = 0.02,
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> Dict[str, Any]:
        G = nx.read_gpickle(graph_path)
        result = simulate_compartmental(
            G,
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
