# pyright: reportMissingImports=false, reportMissingModuleSource=false
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.compartmental_sim import load_graph


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - best-effort import
        plt = None
    return plt


def diagnose_graph(
    graph_path: str,
    output_dir: Optional[str] = None,
    make_plots: bool = True,
    max_nodes_for_layout: int = 2000,
) -> Dict[str, Any]:
    """
    Compute basic diagnostics and optionally emit plots for a graph.
    """
    g_path = BaseTool.resolve_input_path(graph_path, allow_dir=False)
    G: Any = load_graph(g_path)

    stats: Dict[str, Any] = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed": G.is_directed(),
    }

    degrees = [deg for _, deg in G.degree] if hasattr(G, "degree") else []
    if degrees:
        stats.update(
            {
                "degree_min": min(degrees),
                "degree_max": max(degrees),
                "degree_avg": sum(degrees) / len(degrees),
                "degree_median": float(np.median(degrees)),
            }
        )

    out_dir = BaseTool.resolve_output_dir(output_dir) if output_dir else g_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[str] = []

    # Attempt plots if requested
    if make_plots:
        plt = _safe_import_matplotlib()
        if plt:
            # Update median (already set if degrees exist)
            if degrees:
                stats["degree_median"] = float(np.median(degrees))

            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(degrees, bins=min(50, max(10, int(len(set(degrees)) * 1.5))), color="steelblue", edgecolor="black")
                ax.set_xlabel("Degree")
                ax.set_ylabel("Count")
                ax.set_title(f"Degree Histogram: {g_path.stem}")
                hist_path = out_dir / f"{g_path.stem}_degree_hist.png"
                fig.tight_layout()
                fig.savefig(hist_path, dpi=150)
                plt.close(fig)
                artifacts.append(str(hist_path))
            except Exception:
                pass

            # Spring layout (avoid extremely large graphs)
            if G.number_of_nodes() <= max_nodes_for_layout:
                try:
                    pos = nx.spring_layout(G, seed=0)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    nx.draw_networkx(
                        G,
                        pos=pos,
                        ax=ax,
                        node_size=10,
                        width=0.3,
                        with_labels=False,
                        node_color="darkorange",
                        edge_color="gray",
                    )
                    ax.set_title(f"Graph Layout: {g_path.stem}")
                    layout_path = out_dir / f"{g_path.stem}_layout.png"
                    fig.tight_layout()
                    fig.savefig(layout_path, dpi=150)
                    plt.close(fig)
                    artifacts.append(str(layout_path))
                except Exception:
                    pass

    stats_path = out_dir / f"{g_path.stem}_graph_stats.json"
    try:
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2)
        artifacts.append(str(stats_path))
    except Exception as exc:
        return {"error": f"Failed to write stats: {exc}", "stats": stats}

    return {"stats": stats, "artifacts": artifacts}


class GraphDiagnosticsTool(BaseTool):
    """
    Compute graph diagnostics and optionally write plots.
    """

    def __init__(
        self,
        name: str = "GraphDiagnostics",
        description: str = "Compute graph diagnostics (nodes/edges/degree stats) and optionally plots.",
    ):
        parameters = [
            {"name": "graph_path", "type": "str", "description": "Path to graph file (.gpickle/.graphml/.gml/.npz/.npy)."},
            {"name": "output_dir", "type": "str", "description": "Where to write diagnostics (default: alongside graph)."},
            {"name": "make_plots", "type": "bool", "description": "Whether to generate PNG plots (default True)."},
            {"name": "max_nodes_for_layout", "type": "int", "description": "Skip layout plot if graph is larger (default 2000)."},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs) -> Dict[str, Any]:
        graph_path = kwargs.get("graph_path")
        if not graph_path:
            raise ValueError("graph_path is required")
        output_dir = kwargs.get("output_dir")
        make_plots = bool(kwargs.get("make_plots", True))
        max_nodes_for_layout = int(kwargs.get("max_nodes_for_layout", 2000))

        return diagnose_graph(
            graph_path=graph_path,
            output_dir=output_dir,
            make_plots=make_plots,
            max_nodes_for_layout=max_nodes_for_layout,
        )
