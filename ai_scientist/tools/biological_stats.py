from typing import Dict, Any, Sequence, Iterable, Mapping

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.perform_biological_stats import (
    adjust_pvalues,
    enrichment_analysis,
)


class RunBiologicalStatsTool(BaseTool):
    """
    Lightweight stats helper: multiple-testing correction or enrichment analysis.
    """

    def __init__(
        self,
        name: str = "RunBiologicalStats",
        description: str = (
            "Run basic stats: multiple-testing correction (BH) or enrichment analysis (Fisher + BH). "
            "Mode is selected via 'task': 'adjust_pvalues' or 'enrichment'."
        ),
    ):
        parameters = [
            {
                "name": "task",
                "type": "str",
                "description": "adjust_pvalues | enrichment",
            },
            {
                "name": "pvalues",
                "type": "list[float]",
                "description": "List of p-values (for adjust_pvalues).",
            },
            {
                "name": "alpha",
                "type": "float",
                "description": "Significance level (default 0.05).",
            },
            {
                "name": "test_ids",
                "type": "list[str]",
                "description": "IDs of interest (for enrichment).",
            },
            {
                "name": "background_ids",
                "type": "list[str]",
                "description": "Background IDs (for enrichment).",
            },
            {
                "name": "term_to_ids",
                "type": "dict[str, list[str]]",
                "description": "Mapping term -> IDs (for enrichment).",
            },
        ]
        super().__init__(name, description, parameters)

    def use_tool(
        self,
        task: str,
        pvalues: Sequence[float] | None = None,
        alpha: float = 0.05,
        test_ids: Iterable[str] | None = None,
        background_ids: Iterable[str] | None = None,
        term_to_ids: Mapping[str, Iterable[str]] | None = None,
    ) -> Dict[str, Any]:
        # Allow common aliases for BH correction
        if task in {"adjust_pvalues", "bh_correction", "bh", "benjamini_hochberg"}:
            if pvalues is None:
                raise ValueError("pvalues are required for adjust_pvalues")
            res = adjust_pvalues(pvalues=pvalues, alpha=alpha)
            return {
                "raw_pvalues": res.raw_pvalues.tolist(),
                "adjusted_pvalues": res.adjusted_pvalues.tolist(),
                "rejected": res.rejected.tolist(),
                "alpha": res.alpha,
                "method": res.method,
            }

        if task == "enrichment":
            if test_ids is None or background_ids is None or term_to_ids is None:
                raise ValueError("test_ids, background_ids, and term_to_ids are required for enrichment")
            res = enrichment_analysis(
                test_ids=test_ids,
                background_ids=background_ids,
                term_to_ids=term_to_ids,
                alpha=alpha,
            )
            return {
                "alpha": res.alpha,
                "method": res.method,
                "terms": [
                    {
                        "term_id": t.term_id,
                        "description": t.description,
                        "overlap": t.overlap,
                        "term_size": t.term_size,
                        "background_size": t.background_size,
                        "p_value": t.p_value,
                        "adjusted_p_value": t.adjusted_p_value,
                        "significant": t.significant,
                        "contingency_table": t.contingency_table,
                    }
                    for t in res.terms
                ],
            }

        raise ValueError(f"Unknown task '{task}'. Use 'adjust_pvalues' or 'enrichment'.")
