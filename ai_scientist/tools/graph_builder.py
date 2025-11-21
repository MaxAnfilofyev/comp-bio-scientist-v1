import random
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any

from ai_scientist.tools.base_tool import BaseTool


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
            {"name": "output_dir", "type": "str", "description": "Directory to save graphs (default graphs)"},
            {"name": "seed", "type": "int", "description": "Random seed (default 0)"},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, n_nodes: int = 100, output_dir: str = "graphs", seed: int = 0) -> Dict[str, Any]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)

        outputs: Dict[str, Any] = {}

        # Binary tree (balanced)
        bin_tree = nx.balanced_tree(r=2, h=int(np.ceil(np.log2(n_nodes))) if n_nodes > 1 else 1)
        bin_tree = nx.convert_node_labels_to_integers(bin_tree)  # relabel
        bin_pkl = out_dir / f"binary_tree_{n_nodes}.gpickle"
        adj_bin = out_dir / f"binary_tree_{n_nodes}.npy"
        nx.write_gpickle(bin_tree, bin_pkl)
        np.save(adj_bin, nx.to_numpy_array(bin_tree))
        outputs["binary_tree"] = {"gpickle": str(bin_pkl), "adjacency": str(adj_bin)}

        # Heavy-tailed (preferential attachment)
        m = max(1, min(4, n_nodes // 10))
        heavy = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)
        heavy_pkl = out_dir / f"heavy_tailed_{n_nodes}.gpickle"
        adj_heavy = out_dir / f"heavy_tailed_{n_nodes}.npy"
        nx.write_gpickle(heavy, heavy_pkl)
        np.save(adj_heavy, nx.to_numpy_array(heavy))
        outputs["heavy_tailed"] = {"gpickle": str(heavy_pkl), "adjacency": str(adj_heavy)}

        # Random spanning tree
        tree = safe_random_tree(n_nodes, seed=seed)
        tree_pkl = out_dir / f"random_tree_{n_nodes}.gpickle"
        adj_tree = out_dir / f"random_tree_{n_nodes}.npy"
        nx.write_gpickle(tree, tree_pkl)
        np.save(adj_tree, nx.to_numpy_array(tree))
        outputs["random_tree"] = {"gpickle": str(tree_pkl), "adjacency": str(adj_tree)}

        return outputs
