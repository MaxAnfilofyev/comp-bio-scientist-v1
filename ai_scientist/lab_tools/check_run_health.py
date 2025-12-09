"""
Manifest health checker (VI-05).

Usage:
    python -m ai_scientist.lab_tools.check_run_health --base-folder <run_root>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Minimal artifact registry for validation (mirrors orchestrator VI-01).
ARTIFACT_TYPE_REGISTRY: Dict[str, Dict[str, str]] = {
    "lit_summary_main": {"rel_dir": "experiment_results", "pattern": "lit_summary.json"},
    "lit_summary_csv": {"rel_dir": "experiment_results", "pattern": "lit_summary.csv"},
    "claim_graph_main": {"rel_dir": "experiment_results", "pattern": "claim_graph.json"},
    "graph_pickle": {"rel_dir": "experiment_results/graphs", "pattern": "{graph_id}.gpickle"},
    "graph_topology_json": {"rel_dir": "experiment_results/graphs", "pattern": "{graph_id}_topology.json"},
    "parameter_set": {"rel_dir": "experiment_results/parameters", "pattern": "{name}_params.json"},
    "biological_model_solution": {"rel_dir": "experiment_results/models", "pattern": "{model_key}_solution.json"},
    "transport_manifest": {"rel_dir": "experiment_results/simulations/transport_runs", "pattern": "manifest.json"},
    "transport_sim_json": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "{baseline}_sim.json"},
    "transport_sim_status": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "{baseline}_sim.status.json"},
    "transport_failure_matrix": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "{baseline}_sim_failure_matrix.npy"},
    "transport_time_vector": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "{baseline}_sim_time_vector.npy"},
    "transport_nodes_order": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "nodes_order_{baseline}_sim.txt"},
    "transport_per_compartment": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "per_compartment.npz"},
    "transport_node_index_map": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "node_index_map.json"},
    "transport_topology_summary": {"rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}", "pattern": "topology_summary.json"},
    "sensitivity_sweep_table": {"rel_dir": "experiment_results/simulations/sensitivity_sweep", "pattern": "sweep__{label}.csv"},
    "intervention_table": {"rel_dir": "experiment_results/simulations/interventions", "pattern": "intervention__{label}.csv"},
    "plot_intermediate": {"rel_dir": "experiment_results/figures", "pattern": "{slug}.png"},
    "manuscript_figure_png": {"rel_dir": "experiment_results/figures_for_manuscript", "pattern": "fig_{figure_id}.png"},
    "manuscript_figure_svg": {"rel_dir": "experiment_results/figures_for_manuscript", "pattern": "fig_{figure_id}.svg"},
    "phase_portrait": {"rel_dir": "experiment_results/figures", "pattern": "phase_portrait_{label}.png"},
    "energetic_landscape": {"rel_dir": "experiment_results/figures", "pattern": "energy_landscape_{label}.png"},
    "ac_sweep": {"rel_dir": "experiment_results/simulations/ac_sweep", "pattern": "ac_sweep__{run_id}.csv"},
    "figures_readme": {"rel_dir": "experiment_results/figures_for_manuscript", "pattern": "README.md"},
    "interpretation_json": {"rel_dir": "experiment_results", "pattern": "interpretation.json"},
    "interpretation_md": {"rel_dir": "experiment_results", "pattern": "interpretation.md"},
    "verification_note": {"rel_dir": "experiment_results", "pattern": "{artifact_id}_verification.md"},
    "writeup_pdf": {"rel_dir": "experiment_results", "pattern": "manuscript.pdf"},
    "manuscript_pdf": {"rel_dir": "experiment_results", "pattern": "manuscript.pdf"},
}


def _pattern_to_regex(pattern: str):
    import re
    from string import Formatter

    regex_parts: List[str] = []
    formatter = Formatter()
    for literal, field, _format_spec, _conv in formatter.parse(pattern):
        if literal:
            regex_parts.append(re.escape(literal))
        if field:
            regex_parts.append(r"(?P<%s>[A-Za-z0-9._-]+)" % field)
    regex = "".join(regex_parts)
    return re.compile(rf"^{regex}$")

SCRATCH_DIRS = {"_unrouted", "_temp", "_scratch"}
MAX_STRING_FIELD_LEN = 500  # catch verbose notes sneaking into manifest


def _load_manifest(base_folder: Path) -> List[Dict]:
    manifest_dir = base_folder / "experiment_results" / "manifest"
    index_path = manifest_dir / "manifest_index.json"
    entries: List[Dict] = []
    if not index_path.exists():
        return entries
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        return entries
    shards = index.get("shards", [])
    for shard_meta in shards:
        shard_path = Path(shard_meta.get("path", ""))
        if not shard_path.exists():
            continue
        try:
            with shard_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            entries.append(obj)
                    except Exception:
                        continue
        except Exception:
            continue
    return entries


def _check_manifest_entries(base_folder: Path, entries: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    missing: List[str] = []
    naming_errors: List[str] = []
    verbose_fields: List[str] = []
    registry = ARTIFACT_TYPE_REGISTRY

    for entry in entries:
        path = entry.get("path")
        kind = entry.get("kind")
        status = entry.get("status", "ok")
        name = entry.get("name") or (Path(path).name if path else "")
        if path:
            p = Path(path)
            try:
                exists = p.exists()
            except Exception:
                exists = False
            if not exists and status == "ok":
                missing.append(path)
        if kind:
            reg = registry.get(kind)
            if not reg:
                naming_errors.append(f"{path or name}: unknown kind '{kind}'")
            else:
                regex = _pattern_to_regex(reg["pattern"])
                if not regex.fullmatch(name):
                    naming_errors.append(f"{path or name}: name does not match pattern {reg['pattern']}")
        for val in entry.values():
            if isinstance(val, str) and len(val) > MAX_STRING_FIELD_LEN:
                verbose_fields.append(f"{path or name}: field too long ({len(val)} chars)")
    return missing, naming_errors, verbose_fields


def _check_files_covered(base_folder: Path, entries: List[Dict]) -> List[str]:
    exp_dir = base_folder / "experiment_results"
    manifest_paths = {str(Path(e["path"]).resolve()) for e in entries if e.get("path")}
    uncovered: List[str] = []
    if not exp_dir.exists():
        return []
    for p in exp_dir.rglob("*"):
        if p.is_dir():
            continue
        rel_parts = p.relative_to(exp_dir).parts
        if rel_parts and rel_parts[0] in SCRATCH_DIRS:
            continue
        if "manifest" in p.parts:
            continue
        try:
            resolved = str(p.resolve())
        except Exception:
            resolved = str(p)
        if resolved not in manifest_paths:
            uncovered.append(str(p))
    return uncovered


def check_run_health(base_folder: Path) -> Tuple[bool, List[str]]:
    entries = _load_manifest(base_folder)
    missing, naming_errors, verbose_fields = _check_manifest_entries(base_folder, entries)
    uncovered = _check_files_covered(base_folder, entries)

    errors: List[str] = []
    if missing:
        errors.append(f"Missing files (status=ok): {missing}")
    if naming_errors:
        errors.append(f"Naming/registry violations: {naming_errors}")
    if verbose_fields:
        errors.append(f"Verbose manifest fields (> {MAX_STRING_FIELD_LEN} chars): {verbose_fields}")
    if uncovered:
        errors.append(f"Files not covered by manifest (outside scratch): {uncovered}")
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Manifest health check (VI-05).")
    parser.add_argument("--base-folder", help="Run root (folder containing experiment_results). Defaults to AISC_BASE_FOLDER.")
    args = parser.parse_args()

    default_root = Path(os.environ.get("AISC_BASE_FOLDER", "."))
    base = Path(args.base_folder) if args.base_folder else default_root
    ok, errors = check_run_health(base)
    if ok:
        print("Health check passed.")
        return 0
    print("Health check FAILED:")
    for err in errors:
        print(f"- {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
