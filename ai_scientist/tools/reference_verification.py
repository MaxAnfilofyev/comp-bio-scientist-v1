import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.crossref import CrossRefSearchTool


def _normalize_title(title: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for fuzzy matching."""
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_authors(raw_authors: Any) -> List[str]:
    """
    Normalize authors to lowercase last names for overlap checks.
    Accepts comma/semicolon-delimited strings or lists of dicts/strings.
    """
    if isinstance(raw_authors, str):
        parts = re.split(r"[;,]", raw_authors)
        names = [p.strip() for p in parts if p.strip()]
    elif isinstance(raw_authors, list):
        names = []
        for item in raw_authors:
            if isinstance(item, dict):
                name_val = item.get("name") or ""
            else:
                name_val = str(item)
            if name_val:
                names.append(name_val.strip())
    else:
        names = []

    last_names: List[str] = []
    for name in names:
        tokens = name.split()
        if tokens:
            last_names.append(tokens[-1].lower())
    return last_names


def _score_candidate(
    target_title: str,
    candidate_title: str,
    target_authors: List[str],
    candidate_authors: List[str],
    target_year: Optional[int],
    candidate_year: Optional[int],
) -> float:
    """Blend title similarity with author/year agreement."""
    title_score = SequenceMatcher(None, _normalize_title(target_title), _normalize_title(candidate_title)).ratio()

    author_score = 0.0
    if target_authors and candidate_authors:
        overlap = len(set(target_authors) & set(candidate_authors))
        author_score = overlap / max(len(target_authors), len(candidate_authors))

    year_penalty = 0.0
    if target_year and candidate_year:
        try:
            if abs(int(target_year) - int(candidate_year)) > 1:
                year_penalty = 0.1
        except Exception:
            pass

    score = (title_score * 0.7) + (author_score * 0.3) - year_penalty
    return max(0.0, min(1.0, score))


class ReferenceVerificationTool(BaseTool):
    """
    Verify literature references against Semantic Scholar.
    Outputs both CSV and JSON tables with match scores and DOI status.
    """

    def __init__(
        self,
        name: str = "ReferenceVerificationTool",
        description: str = (
            "Verify references from lit_summary against Semantic Scholar. "
            "Checks existence, DOI, and title/author similarity. "
            "Outputs lit_reference_verification.csv/json under experiment_results."
        ),
        default_max_results: int = 5,
        default_score_threshold: float = 0.65,
    ):
        parameters = [
            {
                "name": "lit_path",
                "type": "str",
                "description": "Path to lit_summary.json/csv. Defaults to experiment_results/lit_summary.json (or .csv).",
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": "Output directory for verification artifacts. Defaults to experiment_results/literature.",
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Number of Semantic Scholar hits to consider per reference (default 5).",
            },
            {
                "name": "score_threshold",
                "type": "float",
                "description": "Minimum match score to mark a reference as found (default 0.65).",
            },
        ]
        super().__init__(name, description, parameters)
        self.default_max_results = default_max_results
        self.default_score_threshold = default_score_threshold

    def _resolve_lit_path(self, lit_path: Optional[str]) -> Path:
        """
        Resolve the lit summary path, preferring JSON then CSV under the run folder.
        """
        if lit_path:
            return BaseTool.resolve_input_path(lit_path)

        exp_dir = BaseTool.resolve_output_dir(None)
        default_json = exp_dir / "lit_summary.json"
        default_csv = exp_dir / "lit_summary.csv"

        if default_json.exists():
            return default_json
        if default_csv.exists():
            return default_csv
        raise FileNotFoundError("No lit_summary.json or lit_summary.csv found in experiment_results.")

    def _load_records(self, path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".json":
            with path.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            raise ValueError("lit_summary.json must be a list or object.")

        if path.suffix.lower() == ".csv":
            with path.open() as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]

        raise ValueError("Unsupported lit_path extension; use .json or .csv.")

    def _extract_fields(self, record: Dict[str, Any], idx: int) -> Tuple[str, str, List[str], Optional[int], Optional[str]]:
        ref_id = str(record.get("ref_id") or record.get("id") or record.get("paper_id") or idx)
        title = str(record.get("title") or record.get("paper_title") or "").strip()
        authors = _normalize_authors(record.get("authors"))
        year_raw = record.get("year") or record.get("publication_year")
        year: Optional[int] = None
        try:
            if year_raw not in (None, ""):
                year = int(str(year_raw))
        except Exception:
            year = None
        doi = record.get("doi") or record.get("DOI") or record.get("doi_id")
        doi = str(doi).strip() if doi else None
        return ref_id, title, authors, year, doi

    def _verify_record(
        self,
        record: Dict[str, Any],
        idx: int,
        search_tool: SemanticScholarSearchTool,
        score_threshold: float,
    ) -> Dict[str, Any]:
        ref_id, title, authors, year, doi = self._extract_fields(record, idx)
        if not title:
            return {
                "ref_id": ref_id,
                "title": title,
                "authors": record.get("authors"),
                "year": year,
                "doi": doi,
                "found": False,
                "match_score": 0.0,
                "notes": "Missing title in lit_summary.",
            }

        query_parts = [title]
        if authors:
            query_parts.append(authors[0])
        if year:
            query_parts.append(str(year))
        query = " ".join([p for p in query_parts if p])

        best_score = 0.0
        best_hit: Optional[Dict[str, Any]] = None
        notes = ""

        try:
            results = search_tool.search_for_papers(query) or []
        except Exception as exc:
            return {
                "ref_id": ref_id,
                "title": title,
                "authors": record.get("authors"),
                "year": year,
                "doi": doi,
                "found": False,
                "match_score": 0.0,
                "notes": f"Semantic Scholar lookup failed: {exc}",
            }

        if not results:
            notes = "No Semantic Scholar results."
        else:
            candidate_authors = []
            for hit in results:
                cand_title = hit.get("title") or ""
                cand_authors = _normalize_authors(hit.get("authors"))
                cand_year = hit.get("year")
                score = _score_candidate(title, cand_title, authors, cand_authors, year, cand_year)
                if score > best_score:
                    best_score = score
                    best_hit = hit
                    candidate_authors = cand_authors

            if best_hit:
                hit_doi = (
                    best_hit.get("doi")
                    or best_hit.get("externalIds", {}).get("DOI")
                    if isinstance(best_hit.get("externalIds"), dict)
                    else None
                )
                notes = f"Best hit: '{best_hit.get('title', '')[:80]}', authors={', '.join(candidate_authors) or 'NA'}, year={best_hit.get('year')}, doi={hit_doi or 'NA'}"
                if not doi and hit_doi:
                    doi = hit_doi
            
        # --- CrossRef Fallback ---
        # If (not found) OR (found but missing DOI), try CrossRef
        current_found = bool(best_score >= score_threshold)
        if (not current_found) or (current_found and not doi):
            try:
                cr_tool = CrossRefSearchTool(max_results=5)
                cr_results = cr_tool.search_works(query) or []
            except Exception:
                cr_results = []
                # Keep calm and carry on
                
            cr_best_score = 0.0
            cr_best_hit = None
            
            for hit in cr_results:
                c_title = hit.get("title") or ""
                c_authors = [] 
                if hit.get("authors"):
                     c_authors = _normalize_authors(hit.get("authors"))
                
                c_year = hit.get("year")
                
                score = _score_candidate(title, c_title, authors, c_authors, year, c_year)
                if score > cr_best_score:
                    cr_best_score = score
                    cr_best_hit = hit
            
            # Decision logic
            if cr_best_score >= score_threshold:
                # CrossRef found a match
                cr_doi = cr_best_hit.get("doi")
                
                # scenario 1: S2 failed, CrossRef succeeded
                if not current_found:
                    best_score = cr_best_score
                    # We don't overwrite best_hit (S2) structure completely, but we take the DOI and status
                    # constructing a note
                    short_title = cr_best_hit.get('title', '')[:50]
                    notes = f"Verified via CrossRef: '{short_title}...', doi={cr_doi}"
                    if cr_doi:
                        doi = cr_doi
                
                # scenario 2: S2 succeeded but missed DOI, CrossRef found it (and matches sufficiently)
                elif current_found and not doi and cr_doi:
                    # Ensure it's likely the same paper (double check score didn't drop massively? 
                    # Actually we just checked cr_best_score >= threshold)
                    doi = cr_doi
                    notes += " (DOI added via CrossRef)"

        found = bool(best_score >= score_threshold)
        return {
            "ref_id": ref_id,
            "title": title,
            "authors": record.get("authors"),
            "year": year,
            "doi": doi,
            "found": found,
            "match_score": round(best_score, 3),
            "notes": notes,
        }

    def use_tool(self, **kwargs: Any) -> Dict[str, Any]:
        lit_path_arg = kwargs.get("lit_path") or kwargs.get("path")
        output_dir_arg = kwargs.get("output_dir")
        max_results = int(kwargs.get("max_results", self.default_max_results))
        score_threshold = float(kwargs.get("score_threshold", self.default_score_threshold))

        lit_path = self._resolve_lit_path(lit_path_arg)
        records = self._load_records(lit_path)

        if output_dir_arg:
            out_dir = BaseTool.resolve_output_dir(output_dir_arg)
        else:
            out_dir = BaseTool.resolve_output_dir(None) / "literature"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "lit_reference_verification.csv"
        json_path = out_dir / "lit_reference_verification.json"

        search_tool = SemanticScholarSearchTool(max_results=max_results)

        rows: List[Dict[str, Any]] = []
        n_total_recs = len(records)
        print(f"Verifying {n_total_recs} references...")
        
        for idx, record in enumerate(records):
            # Optimization: If we just fetched it from S2 AND it has a DOI, don't re-verify
            if record.get("source") == "semantic_scholar" and record.get("doi"):
                ref_id, title, authors, year, doi = self._extract_fields(record, idx)
                rows.append({
                    "ref_id": ref_id,
                    "title": title,
                    "authors": record.get("authors"),
                    "year": year,
                    "doi": doi,
                    "found": True,
                    "match_score": 1.0,
                    "notes": "Sourced directly from Semantic Scholar.",
                })
            else:
                rows.append(self._verify_record(record, idx, search_tool, score_threshold))
            
            # Incremental save every 5 records
            if (idx + 1) % 5 == 0:
                with json_path.open("w") as f:
                    json.dump(rows, f, indent=2)
                print(f"Verified {idx + 1}/{n_total_recs}...")

        with json_path.open("w") as f:
            json.dump(rows, f, indent=2)

        fieldnames = ["ref_id", "title", "authors", "year", "doi", "found", "match_score", "notes"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        n_total = len(rows)
        n_found = sum(1 for r in rows if r.get("found"))
        n_low_score = sum(1 for r in rows if r.get("match_score", 0.0) < score_threshold)

        return {
            "n_references": n_total,
            "n_found": n_found,
            "n_low_score": n_low_score,
            "score_threshold": score_threshold,
            "csv": str(csv_path),
            "json": str(json_path),
        }
