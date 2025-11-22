# pyright: reportMissingImports=false
from pathlib import Path
from typing import Any, Dict, List
import json
import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.perform_biological_plotting import BiologicalPlotter


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
        output_dir = BaseTool.resolve_output_dir(kwargs.get("output_dir"))
        make_phase_portrait = bool(kwargs.get("make_phase_portrait", True))
        downsample = int(kwargs.get("downsample", 1))

        path = BaseTool.resolve_input_path(solution_path, allow_dir=True)
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
            if not candidates:
                raise FileNotFoundError(f"No solution JSON found under directory: {path}")
            path = candidates[0]

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

        return outputs
