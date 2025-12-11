import pickle
import random
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path


def safe_random_tree(n_nodes: int, seed: int = 0) -> nx.Graph:
    try:
        return nx.random_tree(n_nodes, seed=seed)
    except Exception:
        rng = random.Random(seed)
        K = nx.complete_graph(n_nodes)
        for u, v in K.edges():
            K[u][v]["weight"] = rng.random()
        return nx.minimum_spanning_tree(K, weight="weight")


class BuildGraphsTool(BaseTool):
    """
    Build canonical graphs (binary, heavy-tailed, random tree) and save to disk.
    """

    def __init__(
        self,
        name: str = "BuildGraphs",
        description: str = (
            "Build canonical graphs: binary tree, heavy-tailed (preferential attachment), "
            "and random spanning tree. Saves gpickle + adjacency .npy."
        ),
    ):
        parameters = [
            {"name": "n_nodes", "type": "int", "description": "Number of nodes (default 100)"},
            {"name": "output_dir", "type": "str", "description": "Directory to save graphs (default experiment_results/morphologies)"},
            {"name": "seed", "type": "int", "description": "Random seed (default 0)"},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, n_nodes: int = 100, output_dir: str = "experiment_results/morphologies", seed: int = 0) -> Dict[str, Any]:
        root_dir = BaseTool.resolve_output_dir(output_dir)
        out_dir, _, _ = resolve_output_path(subdir=None, name="", run_root=root_dir, allow_quarantine=False, unique=False)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs: Dict[str, Any] = {}

        def _write_gpickle(graph: nx.Graph, path: Path) -> None:
            with path.open("wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Binary tree (balanced)
        bin_tree = nx.balanced_tree(r=2, h=int(np.ceil(np.log2(n_nodes))) if n_nodes > 1 else 1)
        bin_tree = nx.convert_node_labels_to_integers(bin_tree)  # relabel
        bin_pkl, _, _ = resolve_output_path(
            subdir=None,
            name=f"binary_tree_{n_nodes}.gpickle",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )
        adj_bin, _, _ = resolve_output_path(
            subdir=None,
            name=f"binary_tree_{n_nodes}.npy",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )
        _write_gpickle(bin_tree, bin_pkl)
        np.save(adj_bin, nx.to_numpy_array(bin_tree))
        outputs["binary_tree"] = {"gpickle": str(bin_pkl), "adjacency": str(adj_bin)}

        # Heavy-tailed (preferential attachment)
        m = max(1, min(4, n_nodes // 10))
        heavy = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)
        heavy_pkl, _, _ = resolve_output_path(
            subdir=None,
            name=f"heavy_tailed_{n_nodes}.gpickle",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )
        adj_heavy, _, _ = resolve_output_path(
            subdir=None,
            name=f"heavy_tailed_{n_nodes}.npy",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )
        _write_gpickle(heavy, heavy_pkl)
        np.save(adj_heavy, nx.to_numpy_array(heavy))
        outputs["heavy_tailed"] = {"gpickle": str(heavy_pkl), "adjacency": str(adj_heavy)}

        # Random spanning tree
        tree = safe_random_tree(n_nodes, seed=seed)
        tree_pkl, _, _ = resolve_output_path(
            subdir=None,
            name=f"random_tree_{n_nodes}.gpickle",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )
        adj_tree, _, _ = resolve_output_path(
            subdir=None,
            name=f"random_tree_{n_nodes}.npy",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )
        _write_gpickle(tree, tree_pkl)
        np.save(adj_tree, nx.to_numpy_array(tree))
        outputs["random_tree"] = {"gpickle": str(tree_pkl), "adjacency": str(adj_tree)}

        return outputs
