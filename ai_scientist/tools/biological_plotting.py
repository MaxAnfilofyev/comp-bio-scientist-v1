from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Sequence, Optional, Union

import matplotlib
# Use a non-interactive backend for batch/scripted runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path


class BiologicalPlotter:
    """Plotting utilities for computational biology models."""

    def __init__(self, figures_dir: str = "figures"):
        self.figures_dir = figures_dir
        self._ensure_dir(self.figures_dir)

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        """Create a filesystem-friendly filename stem."""
        if not name:
            return "figure"
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
        return stem or "figure"

    def _save_fig(self, fig: Figure, filename: Optional[str]) -> str:
        stem = self._sanitize_filename(filename or "figure")
        path = os.path.join(self.figures_dir, f"{stem}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_time_series(
        self,
        time: Union[Sequence[float], np.ndarray],
        trajectories: np.ndarray,
        labels: Optional[Iterable[str]] = None,
        title: str = "Time Series",
        xlabel: str = "Time",
        ylabel: str = "Value",
        filename: Optional[str] = None,
    ) -> str:
        """Plot trajectories over time for multiple variables."""
        arr = np.asarray(trajectories)
        if arr.ndim == 1:
            arr = arr[:, None]
        t = np.asarray(time)

        fig, ax = plt.subplots(figsize=(6, 4))
        num_series = arr.shape[1]

        if labels is None:
            labels = [f"Series {i+1}" for i in range(num_series)]

        for idx in range(num_series):
            ax.plot(t, arr[:, idx], label=list(labels)[idx], linewidth=2)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        return self._save_fig(fig, filename or title)

    def plot_phase_portrait(
        self,
        x: Union[Sequence[float], NDArray[np.floating[Any]]],
        y: Union[Sequence[float], NDArray[np.floating[Any]]],
        title: str = "Phase Portrait",
        xlabel: str = "Variable 1",
        ylabel: str = "Variable 2",
        filename: Optional[str] = None,
        annotate_start: bool = True,
        annotate_end: bool = True,
    ) -> str:
        """
        Plot a 2D trajectory (e.g., cooperation vs. defection).

        Args:
            x: Values for the horizontal axis.
            y: Values for the vertical axis.
            annotate_start: Mark the starting point.
            annotate_end: Mark the ending point.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(x, y, color="tab:blue", linewidth=2)

        if annotate_start and len(x) > 0:
            ax.scatter(x[0], y[0], color="green", s=60, label="start", zorder=3)
        if annotate_end and len(x) > 0:
            ax.scatter(x[-1], y[-1], color="red", s=60, label="end", zorder=3)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        return self._save_fig(fig, filename or title)

    def plot_bifurcation(
        self,
        params: Sequence[float],
        observable: Sequence[float],
        title: str = "Bifurcation Diagram",
        xlabel: str = "Parameter",
        ylabel: str = "Observable",
        filename: Optional[str] = None,
    ) -> str:
        """Plot a simple bifurcation/parameter sweep curve."""
        params_arr = np.asarray(params)
        obs_arr = np.asarray(observable)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(params_arr, obs_arr, marker="o", linestyle="-", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return self._save_fig(fig, filename or title)

    def plot_heatmap(
        self,
        data: np.ndarray,
        x_labels: Optional[Sequence[str]] = None,
        y_labels: Optional[Sequence[str]] = None,
        title: str = "Heatmap",
        xlabel: str = "",
        ylabel: str = "",
        filename: Optional[str] = None,
        cmap: str = "viridis",
    ) -> str:
        """Plot a heatmap (useful for parameter sweeps over two axes)."""
        arr = np.asarray(data)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(arr, aspect="auto", cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if x_labels is not None:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
        if y_labels is not None:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return self._save_fig(fig, filename or title)


class RunBiologicalPlottingTool(BaseTool):
    """
    Plot time-series (and optionally phase portrait) from a saved model solution JSON.
    """

    def __init__(
        self,
        name: str = "RunBiologicalPlotting",
        description: str = (
            "Plot time-series (and optional phase portrait) from a model solution JSON "
            "produced by RunBiologicalModel. Saves PNGs into output_dir."
        ),
    ):
        parameters = [
            {
                "name": "solution_path",
                "type": "str",
                "description": "Path to JSON produced by RunBiologicalModel.",
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to save plots (default experiment_results).",
            },
            {
                "name": "make_phase_portrait",
                "type": "bool",
                "description": "Whether to create a 2D phase portrait if two variables are present.",
            },
            {
                "name": "make_combined_svg",
                "type": "bool",
                "description": "Whether to emit a combined SVG embedding time-series and phase images.",
            },
            {
                "name": "downsample",
                "type": "int",
                "description": "Use every Nth timepoint for plotting (default 1).",
            },
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs: Any) -> Dict[str, Any]:
        solution_path = kwargs.get("solution_path")
        if solution_path is None:
            raise ValueError("solution_path is required")
        root_dir = BaseTool.resolve_output_dir(kwargs.get("output_dir"))
        output_dir, _, _ = resolve_output_path(subdir=None, name="", run_root=root_dir, allow_quarantine=False, unique=False)
        make_phase_portrait = bool(kwargs.get("make_phase_portrait", True))
        make_combined_svg = bool(kwargs.get("make_combined_svg", False))
        downsample = int(kwargs.get("downsample", 1))

        path = BaseTool.resolve_input_path(solution_path, allow_dir=True)
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
            if not candidates:
                raise FileNotFoundError(f"No solution JSON found under directory: {path}")
            path = candidates[0]
        if path.suffix.lower() != ".json":
            raise ValueError(f"run_biological_plotting expects a JSON solution file; got {path.suffix} at {path}")

        with path.open() as f:
            sol = json.load(f)

        time = np.array(sol.get("time", []))
        solutions = sol.get("solutions")
        data = np.array(solutions) if solutions is not None else np.array([])
        variables: List[str] = sol.get("variables", []) or []

        # Fallback: if 'solutions' missing, try E/M matrices and aggregate
        if data.size == 0 and "E" in sol:
            E = np.array(sol.get("E", []))
            M = np.array(sol.get("M", []))
            if E.ndim == 2:
                mean_E = E.mean(axis=1)
                series = [mean_E]
                labels = ["mean_E"]
                if M.ndim == 2 and M.shape[0] == E.shape[0]:
                    mean_M = M.mean(axis=1)
                    series.append(mean_M)
                    labels.append("mean_M")
                data = np.vstack(series).T  # shape (T, n_series)
                variables = labels
            else:
                data = np.array([])
        
        # If time is missing but we have steps, approximate
        if time.size == 0 and data.size > 0:
            # Try to guess DT or just use indices
            time = np.arange(data.shape[0])

        if downsample > 1:
            time = time[::downsample]
            if data.size:
                data = data[::downsample]

        if data.size and not variables:
            variables = [f"var{i}" for i in range(data.shape[1])]

        output_dir.mkdir(parents=True, exist_ok=True)
        plotter = BiologicalPlotter(figures_dir=str(output_dir))

        outputs: Dict[str, Any] = {}
        if data.size:
            ts_path = plotter.plot_time_series(
                time=time,
                trajectories=data,
                labels=variables,
                title=f"{sol.get('model','model')} time series",
                xlabel="time",
                ylabel="value",
                filename=f"{Path(solution_path).stem}_timeseries",
            )
            outputs["time_series"] = ts_path

            if make_phase_portrait and data.shape[1] >= 2:
                pp_path = plotter.plot_phase_portrait(
                    x=data[:, 0],
                    y=data[:, 1],
                    title=f"{sol.get('model','model')} phase portrait",
                    xlabel=variables[0] if variables else "x",
                    ylabel=variables[1] if len(variables) > 1 else "y",
                    filename=f"{Path(solution_path).stem}_phase",
                )
                outputs["phase_portrait"] = pp_path

            if make_combined_svg and "time_series" in outputs:
                try:
                    import base64
                    ts_file = Path(outputs["time_series"])
                    pp_file = Path(outputs.get("phase_portrait", ""))
                    svg_name = f"{Path(solution_path).stem}_timeseries_phase.svg"
                    svg_path = output_dir / svg_name
                    svg_lines = [
                        "<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='600'>",
                        "  <rect width='100%' height='100%' fill='white'/>",
                    ]
                    def _embed_image(img_path: Path, x: int) -> str:
                        data_str = base64.b64encode(img_path.read_bytes()).decode("ascii")
                        return f"  <image href='data:image/png;base64,{data_str}' x='{x}' y='0' width='600' height='600'/>"
                    svg_lines.append(_embed_image(ts_file, 0))
                    if pp_file.exists():
                        svg_lines.append(_embed_image(pp_file, 600))
                    svg_lines.append(
                        f"  <text x='20' y='20' font-size='14' fill='black'>{Path(solution_path).stem}: timeseries (left) and phase (right)</text>"
                    )
                    svg_lines.append("</svg>")
                    svg_path.write_text("\n".join(svg_lines))
                    outputs["combined_svg"] = str(svg_path)
                except Exception:
                    pass

        return outputs
