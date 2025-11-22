"""
Lightweight plotting helpers for theoretical and computational biology workflows.

Provides a `BiologicalPlotter` with common utilities:
- time-series plots for ODE / replicator dynamics
- phase portraits for 2D systems
- simple bifurcation or parameter sweep visualizations

Designed to be backend-agnostic (uses Agg) and safe in headless environments.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence
import matplotlib

# Use a non-interactive backend for batch/scripted runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

mpl_config_dir = os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
cache_dir = os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.makedirs(mpl_config_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)


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

    def _save_fig(self, fig: Figure, filename: str | None) -> str:
        stem = self._sanitize_filename(filename or "figure")
        path = os.path.join(self.figures_dir, f"{stem}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_time_series(
        self,
        time: Sequence[float],
        trajectories: np.ndarray,
        labels: Iterable[str] | None = None,
        title: str = "Time Series",
        xlabel: str = "Time",
        ylabel: str = "Value",
        filename: str | None = None,
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
        x: Sequence[float] | NDArray[np.floating[Any]],
        y: Sequence[float] | NDArray[np.floating[Any]],
        title: str = "Phase Portrait",
        xlabel: str = "Variable 1",
        ylabel: str = "Variable 2",
        filename: str | None = None,
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
        filename: str | None = None,
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
        x_labels: Sequence[str] | None = None,
        y_labels: Sequence[str] | None = None,
        title: str = "Heatmap",
        xlabel: str = "",
        ylabel: str = "",
        filename: str | None = None,
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
