from pathlib import Path
from typing import Any, Dict, List

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.perform_lit_data_assembly import assemble_lit_data
from ai_scientist.utils.pathing import resolve_output_path


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
        Resolve a fixed output directory for lit summaries using the canonical run folder.
        """
        return BaseTool.resolve_output_dir(None)

    def use_tool(self, **kwargs: Any) -> Dict[str, Any]:
        out_dir = self._resolve_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path, quarantined_csv, _ = resolve_output_path(
            subdir=None, name="lit_summary.csv", run_root=out_dir, allow_quarantine=True, unique=True
        )
        json_path, quarantined_json, _ = resolve_output_path(
            subdir=None, name="lit_summary.json", run_root=out_dir, allow_quarantine=True, unique=True
        )

        queries: List[str] | None = kwargs.get("queries")
        seed_paths: List[str] | None = kwargs.get("seed_paths")
        max_results = int(kwargs.get("max_results", 25))
        use_semantic_scholar = bool(kwargs.get("use_semantic_scholar", False))

        result = assemble_lit_data(
            output_csv=str(csv_path),
            output_json=str(json_path),
            queries=queries,
            local_seed_paths=seed_paths,
            max_results=max_results,
            use_semantic_scholar=use_semantic_scholar,
        )
        if quarantined_csv or quarantined_json:
            result["quarantined"] = True
        return result
