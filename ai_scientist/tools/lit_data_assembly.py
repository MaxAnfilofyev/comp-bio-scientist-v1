import os
from pathlib import Path
from typing import List, Dict, Any

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.perform_lit_data_assembly import assemble_lit_data


class LitDataAssemblyTool(BaseTool):
    """
    Tool wrapper to assemble literature data into CSV/JSON artifacts.
    """

    def __init__(
        self,
        name: str = "AssembleLitData",
        description: str = (
            "Assemble literature data into normalized CSV/JSON. "
            "Provide optional seed paths and semantic scholar queries. "
            "Outputs are always written to experiment_results/lit_summary.csv and lit_summary.json under the active run."
        ),
    ):
        parameters = [
            {
                "name": "queries",
                "type": "list[str]",
                "description": "Optional list of Semantic Scholar queries to run.",
            },
            {
                "name": "seed_paths",
                "type": "list[str]",
                "description": "Optional list of local CSV/JSON seed files to merge.",
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Max results per query (default 25).",
            },
            {
                "name": "use_semantic_scholar",
                "type": "bool",
                "description": "Whether to hit Semantic Scholar (default True).",
            },
        ]
        super().__init__(name, description, parameters)

    def _resolve_out_dir(self) -> Path:
        """
        Resolve a fixed output directory for lit summaries:
        - If AISC_EXP_RESULTS is set (by orchestrator), use it.
        - Otherwise default to experiment_results.
        """
        base_env = os.environ.get("AISC_EXP_RESULTS", "")
        out = Path(base_env) if base_env else Path("experiment_results")
        return out

    def use_tool(
        self,
        queries: List[str] | None = None,
        seed_paths: List[str] | None = None,
        max_results: int = 25,
        use_semantic_scholar: bool = False,
    ) -> Dict[str, Any]:
        out_dir = self._resolve_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "lit_summary.csv"
        json_path = out_dir / "lit_summary.json"

        result = assemble_lit_data(
            output_csv=str(csv_path),
            output_json=str(json_path),
            queries=queries,
            local_seed_paths=seed_paths,
            max_results=max_results,
            use_semantic_scholar=use_semantic_scholar,
        )
        return result
