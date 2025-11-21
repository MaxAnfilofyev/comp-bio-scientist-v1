"""
Lightweight literature data assembly utility for computational biology runs.

Goals:
- Optionally query Semantic Scholar (if network + API key available) for domain-specific
  metrics (e.g., axonal arbor size, transport/mitophagy rates).
- Merge with any local seed data the user places in data/ (CSV or JSON).
- Normalize into a simple tabular schema and write both CSV and JSON artifacts.

Usage (library):
    from ai_scientist.perform_lit_data_assembly import assemble_lit_data
    assemble_lit_data(
        output_csv="experiment_results/lit_summary.csv",
        output_json="experiment_results/lit_summary.json",
        queries=["substantia nigra axonal arborization", "mitochondrial transport rates"],
        local_seed_paths=["data/snc_vta_lit_seed.csv"],
        max_results=25,
        use_semantic_scholar=True,
    )

Usage (CLI):
    python ai_scientist/perform_lit_data_assembly.py --output-dir experiment_results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

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
    api_key = os.environ.get("S2_API_KEY", "")

    for q in queries:
        try:
            papers = search_for_papers(q, num_results=max_results, api_key=api_key)
        except Exception as e:
            print(f"[warn] Semantic Scholar query failed for '{q}': {e}", file=sys.stderr)
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
                    "doi": p.get("doi"),
                    "tldr": p.get("tldr", {}).get("text") if isinstance(p.get("tldr"), dict) else None,
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
    use_semantic_scholar: bool = True,
) -> Dict[str, Any]:
    """
    Assemble literature data into normalized CSV/JSON artifacts.

    Returns a dict with paths and record counts.
    """
    records: List[Dict[str, Any]] = []

    # Load local seeds
    for path_str in local_seed_paths or []:
        records.extend(_load_seed_file(Path(path_str)))

    # Fetch from Semantic Scholar
    if use_semantic_scholar and queries:
        records.extend(_fetch_semantic_scholar(queries, max_results=max_results))

    # Deduplicate by (title, doi, url) tuple when available
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for r in records:
        key = (
            r.get("title") or "",
            r.get("doi") or "",
            r.get("url") or "",
        )
        deduped[key] = r

    final_records = list(deduped.values())
    output_csv_path = Path(output_csv)
    output_json_path = Path(output_json)
    _write_outputs(final_records, output_csv_path, output_json_path)

    return {
        "n_records": len(final_records),
        "csv": str(output_csv_path),
        "json": str(output_json_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assemble literature data (Semantic Scholar + local seeds) into CSV/JSON."
    )
    parser.add_argument("--output-dir", default="experiment_results", help="Output directory for artifacts.")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=[
            "substantia nigra dopaminergic axonal arborization reconstruction",
            "ventral tegmental area dopaminergic axon morphology",
            "mitochondrial transport rate neuron",
            "mitophagy rate substantia nigra neuron",
            "ATP diffusion time axon",
            "calcium pacemaking energy cost dopamine neuron",
        ],
        help="Semantic Scholar queries",
    )
    parser.add_argument(
        "--seed",
        nargs="*",
        default=[],
        help="Paths to local seed CSV/JSON files to merge",
    )
    parser.add_argument("--max-results", type=int, default=25, help="Max results per query")
    parser.add_argument(
        "--no-semantic-scholar",
        action="store_true",
        help="Disable Semantic Scholar queries (use only local seeds)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lit_summary.csv"
    json_path = out_dir / "lit_summary.json"

    result = assemble_lit_data(
        output_csv=str(csv_path),
        output_json=str(json_path),
        queries=args.queries,
        local_seed_paths=args.seed,
        max_results=args.max_results,
        use_semantic_scholar=not args.no_semantic_scholar,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
