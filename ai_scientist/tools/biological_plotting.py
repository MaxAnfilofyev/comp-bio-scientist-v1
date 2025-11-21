from pathlib import Path
from typing import Dict, Any, List
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
        ]
        super().__init__(name, description, parameters)

    def use_tool(
        self,
        solution_path: str,
        output_dir: str = "experiment_results",
        make_phase_portrait: bool = True,
    ) -> Dict[str, Any]:
        path = Path(solution_path)
        if not path.exists():
            raise FileNotFoundError(f"Solution file not found: {solution_path}")

        with path.open() as f:
            sol = json.load(f)

        time = np.array(sol.get("time", []))
        data = np.array(sol.get("solutions", []))
        variables: List[str] = sol.get("variables", []) or [f"var{i}" for i in range(data.shape[1])] if data.size else []

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plotter = BiologicalPlotter(figures_dir=str(out_dir))

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
