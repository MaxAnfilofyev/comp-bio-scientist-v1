from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path

# Semantic Scholar helper (optional)
try:
    from ai_scientist.tools.semantic_scholar import search_for_papers
except Exception:
    search_for_papers = None


def _load_seed_file(path: Path) -> List[Dict[str, Any]]:
    """Load local seed data from CSV or JSON; ignore unknown extensions."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    try:
        if path.suffix.lower() == ".csv":
            with path.open() as f:
                reader = csv.DictReader(f)
                records.extend(dict(row) for row in reader)
        elif path.suffix.lower() == ".json":
            with path.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
    except Exception as e:
        print(f"[warn] Failed to load seed file {path}: {e}", file=sys.stderr)
    return records


def _fetch_semantic_scholar(
    queries: Iterable[str], max_results: int = 25
) -> List[Dict[str, Any]]:
    """Fetch papers from Semantic Scholar, if the helper is available."""
    if search_for_papers is None:
        return []

    out: List[Dict[str, Any]] = []

    for q in queries:
        try:
            # NB: current helper signature is search_for_papers(query, result_limit=10)
            papers = search_for_papers(q, result_limit=max_results)
        except Exception as e:
            print(f"[warn] Semantic Scholar query failed for '{q}': {e}", file=sys.stderr)
            continue

        # Some backends may return None/empty on errors; guard to keep the tool resilient.
        if not papers:
            continue
        if not isinstance(papers, list):
            print(f"[warn] Unexpected Semantic Scholar response type {type(papers)} for query '{q}'", file=sys.stderr)
            continue

        for p in papers:
            # Normalize fields; fall back to None if missing
            out.append(
                {
                    "query": q,
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "abstract": p.get("abstract"),
                    "authors": ", ".join(a.get("name", "") for a in p.get("authors", []))
                    if isinstance(p.get("authors"), list)
                    else None,
                    "venue": p.get("venue"),
                    "url": p.get("url"),
                    "doi": p.get("externalIds", {}).get("DOI") or p.get("doi"),
                    "tldr": p.get("tldr", {}).get("text") if isinstance(p.get("tldr"), dict) else None,
                    "paperId": p.get("paperId"),
                    "corpusId": p.get("corpusId"),
                }
            )
    return out


def _write_outputs(
    records: List[Dict[str, Any]], output_csv: Path, output_json: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    with output_json.open("w") as f:
        json.dump(records, f, indent=2)

    # CSV: collect all keys to avoid missing columns
    keys: List[str] = sorted({k for r in records for k in r.keys()}) if records else []
    with output_csv.open("w", newline="") as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        else:
            f.write("")  # create empty file if no records


def assemble_lit_data(
    output_csv: str,
    output_json: str,
    queries: Optional[Iterable[str]] = None,
    local_seed_paths: Optional[Iterable[str]] = None,
    max_results: int = 25,
    use_semantic_scholar: bool = False,
    search_history_path: Optional[str] = None,
    excluded_ids: Optional[Iterable[str]] = None,
    exclusion_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble literature data into normalized CSV/JSON artifacts.

    Returns a dict with paths and record counts.
    """
    output_csv_path = Path(output_csv)
    output_json_path = Path(output_json)
    records: List[Dict[str, Any]] = []
    
    # Normalize exclude list
    excluded_set = set()
    if excluded_ids:
        for ex in excluded_ids:
            if ex:
                excluded_set.add(ex.lower().strip())

    # Load from exclusion file if provided
    if exclusion_file_path:
        ex_path = Path(exclusion_file_path)
        if ex_path.exists():
            try:
                with ex_path.open() as f:
                    file_excludes = json.load(f)
                    if isinstance(file_excludes, list):
                        for ex in file_excludes:
                             if ex:
                                excluded_set.add(str(ex).lower().strip())
                    elif isinstance(file_excludes, dict) and "excluded_ids" in file_excludes:
                         # Handle {"excluded_ids": [...]} format too
                         for ex in file_excludes["excluded_ids"]:
                             if ex:
                                excluded_set.add(str(ex).lower().strip())
            except Exception as e:
                print(f"[warn] Failed to load exclusion file {ex_path}: {e}")
    
    def _is_excluded(r: Dict[str, Any]) -> bool:
        if not excluded_set:
            return False
        # Check standard fields
        identifiers = [
            r.get("paperId"),
            r.get("corpusId"),
            r.get("doi"),
            r.get("url"),
            r.get("title")
        ]
        # Also check externalIds dict if present
        ext = r.get("externalIds") or {}
        if isinstance(ext, dict):
             identifiers.extend(ext.values())

        for ident in identifiers:
            if ident and str(ident).lower().strip() in excluded_set:
                return True
        return False

    # Load existing output if it exists (for incremental restart)
    out_json_path = Path(output_json)

    if out_json_path.exists():
        try:
            with out_json_path.open() as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    # Filter existing records against exclusion list
                    filtered_existing = [r for r in existing if not _is_excluded(r)]
                    records.extend(filtered_existing)
        except Exception as e:
            print(f"[warn] Could not load existing lit_summary: {e}")

    # Load local seeds
    for path_str in local_seed_paths or []:
        seeds = _load_seed_file(Path(path_str))
        filtered_seeds = [r for r in seeds if not _is_excluded(r)]
        records.extend(filtered_seeds)

    # Fetch from Semantic Scholar
    if use_semantic_scholar and queries:
        queries_list = list(queries)
        
        # Filter queries using history
        completed_queries = set()
        history_file = Path(search_history_path) if search_history_path else None
        
        if history_file and history_file.exists():
            try:
                with history_file.open() as f:
                    hdata = json.load(f)
                    completed_queries = set(hdata.get("completed_queries", []))
            except Exception as e:
                print(f"[warn] Failed to load search history: {e}")

        queries_to_run = [q for q in queries_list if q not in completed_queries]
        if len(queries_to_run) < len(queries_list):
            print(f"Skipping {len(queries_list) - len(queries_to_run)} already completed queries.")
            
        # Save progress incrementally
        total_queries = len(queries_to_run)
        if total_queries > 0:
            print(f"Starting S2 search for {total_queries} new queries...")
            
            # We need to maintain the deduped list as we go
            # Deduplicate by (title, doi, url) tuple
            deduped: Dict[tuple, Dict[str, Any]] = {}
            
            def _get_key(r):
                return (
                    r.get("title") or "",
                    r.get("doi") or "",
                    r.get("url") or "",
                )

            for r in records:
                deduped[_get_key(r)] = r

            for i, q in enumerate(queries_to_run):
                new_records = _fetch_semantic_scholar([q], max_results=max_results)
                
                # Filter new records
                filtered_new = []
                for r in new_records:
                    if not _is_excluded(r):
                        filtered_new.append(r)
                
                for r in filtered_new:
                    r["source"] = "semantic_scholar"
                    deduped[_get_key(r)] = r
                
                # Update history
                completed_queries.add(q)
                if history_file:
                    try:
                        with history_file.open("w") as f:
                            json.dump({"completed_queries": list(completed_queries)}, f, indent=2)
                    except Exception as e:
                        print(f"[warn] Failed to save search history: {e}")
                
                # Incremental save
                current_records = list(deduped.values())
                _write_outputs(current_records, output_csv_path, output_json_path)
                print(f"Query {i+1}/{total_queries} processed. Saved {len(current_records)} records.")

            records = list(deduped.values())
        else:
            print("All queries already completed.")

    else:
        # Just dedupe local records
        deduped: Dict[tuple, Dict[str, Any]] = {}
        for r in records:
            key = (
                r.get("title") or "",
                r.get("doi") or "",
                r.get("url") or "",
            )
            deduped[key] = r
        records = list(deduped.values())
        _write_outputs(records, output_csv_path, output_json_path)

    final_records = records


    return {
        "n_records": len(final_records),
        "csv": str(output_csv_path),
        "json": str(output_json_path),
    }


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
            {
                "name": "excluded_ids",
                "type": "list[str]",
                "description": "Optional list of paper IDs, DOIs, or titles to exclude from the results.",
            },
        ]
        super().__init__(name, description, parameters)

    def _resolve_out_dir(self) -> Path:
        """
        Resolve a fixed output directory for lit summaries using the canonical run folder.
        """
        return BaseTool.resolve_output_dir(None) / "literature"

    def use_tool(self, **kwargs: Any) -> Dict[str, Any]:
        out_dir = self._resolve_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path, quarantined_csv, _ = resolve_output_path(
            subdir=None, name="lit_summary.csv", run_root=out_dir, allow_quarantine=True, unique=False
        )
        json_path, quarantined_json, _ = resolve_output_path(
            subdir=None, name="lit_summary.json", run_root=out_dir, allow_quarantine=True, unique=False
        )
        
        # Sidecar history file (not an artifact per se, but internal state)
        history_path = out_dir / "lit_search_history.json"
        
        # Persistent exclusion list
        exclusion_file_path = out_dir / "lit_exclusions.json"

        queries: Optional[List[str]] = kwargs.get("queries")
        seed_paths: Optional[List[str]] = kwargs.get("seed_paths")
        max_results = int(kwargs.get("max_results", 25))
        use_semantic_scholar = bool(kwargs.get("use_semantic_scholar", False))
        excluded_ids: Optional[List[str]] = kwargs.get("excluded_ids")

        result = assemble_lit_data(
            output_csv=str(csv_path),
            output_json=str(json_path),
            queries=queries,
            local_seed_paths=seed_paths,
            max_results=max_results,
            use_semantic_scholar=use_semantic_scholar,
            search_history_path=str(history_path),
            excluded_ids=excluded_ids,
            exclusion_file_path=str(exclusion_file_path),
        )
        if quarantined_csv or quarantined_json:
            result["quarantined"] = True
        return result
