from pathlib import Path
from typing import Dict, Any
import json
import numpy as np

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.perform_biological_modeling import (
    create_sample_models,
    solve_biological_model,
)


class RunBiologicalModelTool(BaseTool):
    """
    Run a small library biological model (ODE/replicator) and store outputs.
    """

    def __init__(
        self,
        name: str = "RunBiologicalModel",
        description: str = (
            "Run a built-in biological model (e.g., cooperation_evolution, predator_prey, epidemiology) "
            "and save the solution to JSON under output_dir."
        ),
    ):
        parameters = [
            {
                "name": "model_key",
                "type": "str",
                "description": "Model to run: cooperation_evolution | predator_prey | epidemiology",
            },
            {
                "name": "time_end",
                "type": "float",
                "description": "End time for simulation (default 20.0)",
            },
            {
                "name": "num_points",
                "type": "int",
                "description": "Number of time points (default 200)",
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to store results JSON (default experiment_results)",
            },
        ]
        super().__init__(name, description, parameters)

    def use_tool(
        self,
        model_key: str = "cooperation_evolution",
        time_end: float = 20.0,
        num_points: int = 200,
        output_dir: str = "experiment_results",
    ) -> Dict[str, Any]:
        models = create_sample_models()
        if model_key not in models:
            raise ValueError(f"Unknown model_key '{model_key}'. Available: {list(models.keys())}")

        model = models[model_key]
        t = np.linspace(0, time_end, num_points)

        if hasattr(model, "replicator_dynamics"):
            sol = model.replicator_dynamics([0.01, 0.99], t)
        else:
            sol = solve_biological_model(model, t)

        out_dir = BaseTool.resolve_output_dir(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path, _, _ = resolve_output_path(
            subdir=None,
            name=f"{model_key}_solution.json",
            run_root=out_dir,
            allow_quarantine=True,
            unique=True,
        )

        payload: Dict[str, Any] = {
            "model": model_key,
            "time": sol.get("time", []),
            "variables": sol.get("variables", []),
            "solutions": sol.get("solutions", []),
            "success": sol.get("success", False),
            "message": sol.get("message", ""),
        }
        # Convert numpy types to native for JSON
        def _to_native(x):
            if hasattr(x, "tolist"):
                return x.tolist()
            return x

        payload = {k: _to_native(v) for k, v in payload.items()}

        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)

        return {"output_json": str(out_path), "success": payload["success"]}
