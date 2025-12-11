from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.stats import fisher_exact

from ai_scientist.tools.base_tool import BaseTool


@dataclass
class MultipleTestingResult:
    """Result of a multiple testing correction."""

    raw_pvalues: np.ndarray
    adjusted_pvalues: np.ndarray
    rejected: np.ndarray  # boolean mask of hypotheses rejected at given alpha
    alpha: float
    method: str


def adjust_pvalues(
    pvalues: Union[Sequence[float], np.ndarray],
    alpha: float = 0.05,
    method: str = "benjamini_hochberg",
) -> MultipleTestingResult:
    """
    Adjust a collection of p-values for multiple testing.

    Currently implemented:
        - Benjamini-Hochberg FDR control ("benjamini_hochberg", "bh")

    Args:
        pvalues: Iterable of raw p-values (0 <= p <= 1).
        alpha: Desired FDR / family-wise error rate threshold.
        method: Multiple testing method. Currently supports:
            - "benjamini_hochberg" (alias: "bh")

    Returns:
        MultipleTestingResult with adjusted p-values and rejection mask.
    """
    pvals = np.asarray(pvalues, dtype=float)
    if pvals.ndim != 1:
        raise ValueError("pvalues must be a 1D sequence")

    m = pvals.size
    if m == 0:
        return MultipleTestingResult(
            raw_pvalues=pvals,
            adjusted_pvalues=np.array([], dtype=float),
            rejected=np.array([], dtype=bool),
            alpha=alpha,
            method=method,
        )

    method_normalized = method.lower()
    if method_normalized in {"benjamini_hochberg", "bh"}:
        # Benjamini-Hochberg procedure
        order = np.argsort(pvals)
        ranked_pvals = pvals[order]
        ranks = np.arange(1, m + 1)

        # Compute BH adjusted p-values in sorted order
        bh_factors = m / ranks
        adj_sorted = ranked_pvals * bh_factors
        # Enforce monotonicity (from largest to smallest)
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

        # Map back to original order
        adjusted = np.empty_like(adj_sorted)
        adjusted[order] = adj_sorted

        rejected = adjusted <= alpha

        return MultipleTestingResult(
            raw_pvalues=pvals,
            adjusted_pvalues=adjusted,
            rejected=rejected,
            alpha=alpha,
            method="benjamini_hochberg",
        )
    else:
        raise ValueError(f"Unsupported multiple testing method: {method}")


@dataclass
class EnrichmentTermResult:
    """Result for a single enrichment term."""

    term_id: str
    description: Optional[str]
    overlap: int
    term_size: int
    background_size: int
    p_value: float
    adjusted_p_value: float
    significant: bool
    contingency_table: Tuple[int, int, int, int]


@dataclass
class EnrichmentResult:
    """Collection of enrichment results sorted by adjusted p-value."""

    terms: List[EnrichmentTermResult]
    alpha: float
    method: str


def enrichment_analysis(
    test_ids: Iterable[str],
    background_ids: Iterable[str],
    term_to_ids: Mapping[str, Iterable[str]],
    term_descriptions: Optional[Mapping[str, str]] = None,
    alpha: float = 0.05,
    multiple_testing_method: str = "benjamini_hochberg",
    min_term_size: int = 5,
    max_term_size: int = 5000,
) -> EnrichmentResult:
    """
    Perform generic enrichment analysis for a set of IDs (e.g., genes, proteins).

    This function is intentionally generic and can be used for:
      - Gene Ontology enrichment
      - Pathway enrichment (KEGG/Reactome)
      - Custom term sets (e.g., protein complexes, motifs)

    It constructs a 2x2 contingency table for each term and uses Fisher's exact test
    (one-sided, enrichment) to compute p-values, followed by multiple testing
    correction (e.g., Benjamini-Hochberg FDR).

    Args:
        test_ids: IDs of interest (e.g., differentially expressed genes).
        background_ids: Background universe of IDs (must contain all test IDs).
        term_to_ids: Mapping from term ID -> iterable of associated IDs.
        term_descriptions: Optional mapping from term ID -> human-readable description.
        alpha: Significance threshold after multiple testing correction.
        multiple_testing_method: Multiple testing method, passed to adjust_pvalues.
        min_term_size: Minimum number of background IDs a term must cover to be tested.
        max_term_size: Maximum number of background IDs a term may cover to be tested.

    Returns:
        EnrichmentResult containing a list of EnrichmentTermResult, sorted by
        adjusted p-value.
    """
    test_set: Set[str] = set(test_ids)
    background_set: Set[str] = set(background_ids)

    if not test_set:
        raise ValueError("test_ids is empty")
    if not background_set:
        raise ValueError("background_ids is empty")
    if not test_set.issubset(background_set):
        raise ValueError("All test_ids must be contained in background_ids")

    # Convert term_to_ids into sets restricted to the background
    clean_term_to_ids: Dict[str, Set[str]] = {}
    for term, ids in term_to_ids.items():
        s = set(ids) & background_set
        if min_term_size <= len(s) <= max_term_size:
            clean_term_to_ids[term] = s

    if not clean_term_to_ids:
        return EnrichmentResult(terms=[], alpha=alpha, method=multiple_testing_method)

    m = len(clean_term_to_ids)
    raw_pvals = np.empty(m, dtype=float)
    term_keys: List[str] = []
    contingency_tables: List[Tuple[int, int, int, int]] = []
    overlaps: List[int] = []
    term_sizes: List[int] = []
    bg_size = len(background_set)

    # Compute Fisher's exact test p-values for each term
    for idx, (term, term_ids) in enumerate(clean_term_to_ids.items()):
        term_keys.append(term)

        # Overlap between term and test set
        overlap_ids = test_set & term_ids
        a = len(overlap_ids)  # in term & in test
        b = len(test_set - term_ids)  # not in term & in test
        c = len(term_ids - test_set)  # in term & not in test
        d = bg_size - a - b - c      # not in term & not in test

        contingency = (a, b, c, d)
        contingency_tables.append(contingency)
        overlaps.append(a)
        term_sizes.append(len(term_ids))

        table = np.array([[a, b], [c, d]], dtype=int)
        # One-sided Fisher's exact test, enrichment in test set (greater alternative)
        oddsratio, pval = fisher_exact(table, alternative="greater")
        raw_pvals[idx] = pval

    # Multiple testing correction
    mt_res = adjust_pvalues(
        pvalues=raw_pvals,
        alpha=alpha,
        method=multiple_testing_method,
    )

    results: List[EnrichmentTermResult] = []
    for i, term in enumerate(term_keys):
        desc = None
        if term_descriptions is not None:
            desc = term_descriptions.get(term)
        res = EnrichmentTermResult(
            term_id=term,
            description=desc,
            overlap=overlaps[i],
            term_size=term_sizes[i],
            background_size=bg_size,
            p_value=float(mt_res.raw_pvalues[i]),
            adjusted_p_value=float(mt_res.adjusted_pvalues[i]),
            significant=bool(mt_res.rejected[i]),
            contingency_table=contingency_tables[i],
        )
        results.append(res)

    # Sort by adjusted p-value
    results.sort(key=lambda r: r.adjusted_p_value)

    return EnrichmentResult(
        terms=results,
        alpha=alpha,
        method=multiple_testing_method,
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
        pvalues: Optional[Sequence[float]] = None,
        alpha: float = 0.05,
        test_ids: Optional[Iterable[str]] = None,
        background_ids: Optional[Iterable[str]] = None,
        term_to_ids: Optional[Mapping[str, Iterable[str]]] = None,
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

__all__ = [
    "MultipleTestingResult",
    "adjust_pvalues",
    "EnrichmentTermResult",
    "EnrichmentResult",
    "enrichment_analysis",
    "RunBiologicalStatsTool"
]
