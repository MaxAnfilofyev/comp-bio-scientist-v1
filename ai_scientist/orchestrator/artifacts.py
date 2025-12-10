import json
import os
import re
from string import Formatter
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.manifest import append_or_update, load_entries
from ai_scientist.utils.pathing import resolve_output_path

# Canonical artifact registry (VI-01)
ARTIFACT_TYPE_REGISTRY: Dict[str, Dict[str, str]] = {
    "lit_summary_main": {
        "rel_dir": "experiment_results",
        "pattern": "lit_summary.json",
        "description": "Primary literature summary.",
    },
    "lit_summary_csv": {
        "rel_dir": "experiment_results",
        "pattern": "lit_summary.csv",
        "description": "CSV-formatted literature summary.",
    },
    "lit_reference_verification_table": {
        "rel_dir": "experiment_results",
        "pattern": "lit_reference_verification.csv",
        "description": "Reference verification table (CSV).",
    },
    "lit_reference_verification_json": {
        "rel_dir": "experiment_results",
        "pattern": "lit_reference_verification.json",
        "description": "Reference verification table (JSON).",
    },
    "claim_graph_main": {
        "rel_dir": "experiment_results",
        "pattern": "claim_graph.json",
        "description": "Claim graph JSON.",
    },
    "graph_pickle": {
        "rel_dir": "experiment_results/graphs",
        "pattern": "{graph_id}.gpickle",
        "description": "Pickled graph for simulations.",
    },
    "graph_topology_json": {
        "rel_dir": "experiment_results/graphs",
        "pattern": "{graph_id}_topology.json",
        "description": "Graph topology summary JSON.",
    },
    "parameter_set": {
        "rel_dir": "experiment_results/parameters",
        "pattern": "{name}_params.json",
        "description": "Parameter set definition for simulations or models.",
    },
    "biological_model_solution": {
        "rel_dir": "experiment_results/models",
        "pattern": "{model_key}_solution.json",
        "description": "Solution from built-in biological model.",
    },
    "transport_manifest": {
        "rel_dir": "experiment_results/simulations/transport_runs",
        "pattern": "manifest.json",
        "description": "Transport run manifest index.",
    },
    "transport_sim_json": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim.json",
        "description": "Simulation JSON for a transport run.",
    },
    "transport_sim_status": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim.status.json",
        "description": "Status JSON for a transport run.",
    },
    "transport_failure_matrix": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim_failure_matrix.npy",
        "description": "Failure matrix array.",
    },
    "transport_time_vector": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "{baseline}_sim_time_vector.npy",
        "description": "Time vector array.",
    },
    "transport_nodes_order": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "nodes_order_{baseline}_sim.txt",
        "description": "Node ordering text file.",
    },
    "transport_per_compartment": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "per_compartment.npz",
        "description": "Per-compartment export (binary/continuous/time).",
    },
    "transport_node_index_map": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "node_index_map.json",
        "description": "Node index map for per-compartment outputs.",
    },
    "transport_topology_summary": {
        "rel_dir": "experiment_results/simulations/transport_runs/{baseline}/transport_{transport}/seed_{seed}",
        "pattern": "topology_summary.json",
        "description": "Topology summary for per-compartment outputs.",
    },
    "sensitivity_sweep_table": {
        "rel_dir": "experiment_results/simulations/sensitivity_sweep",
        "pattern": "sweep__{label}.csv",
        "description": "Sensitivity sweep CSV output.",
    },
    "intervention_table": {
        "rel_dir": "experiment_results/simulations/interventions",
        "pattern": "intervention__{label}.csv",
        "description": "Intervention tester CSV output.",
    },
    "plot_intermediate": {
        "rel_dir": "experiment_results/figures",
        "pattern": "{slug}.png",
        "description": "Intermediate plot or diagnostic figure.",
    },
    "manuscript_figure_png": {
        "rel_dir": "experiment_results/figures_for_manuscript",
        "pattern": "fig_{figure_id}.png",
        "description": "PNG figure for manuscript.",
    },
    "manuscript_figure_svg": {
        "rel_dir": "experiment_results/figures_for_manuscript",
        "pattern": "fig_{figure_id}.svg",
        "description": "SVG figure for manuscript.",
    },
    "manuscript_input_text": {
        "rel_dir": "experiment_results",
        "pattern": "manuscript_input.txt",
        "description": "Raw manuscript text ingested for run initialization.",
    },
    "seed_idea_from_manuscript": {
        "rel_dir": "experiment_results",
        "pattern": "seed_idea_from_manuscript.json",
        "description": "Idea seed JSON derived from manuscript ingestion.",
    },
    "provenance_summary_md": {
        "rel_dir": "experiment_results",
        "pattern": "provenance_summary.md",
        "description": "Provenance summary for the run.",
    },
    "hypothesis_trace_json": {
        "rel_dir": "experiment_results",
        "pattern": "hypothesis_trace.json",
        "description": "Hypothesis trace JSON for run.",
    },
    "release_manifest": {
        "rel_dir": "experiment_results/releases/{tag}",
        "pattern": "release_manifest.json",
        "description": "Release bundle manifest (checksums + metadata).",
    },
    "code_release_archive": {
        "rel_dir": "experiment_results/releases/{tag}",
        "pattern": "code_release.zip",
        "description": "Code snapshot archive for release.",
    },
    "env_manifest": {
        "rel_dir": "experiment_results/releases/{tag}",
        "pattern": "env_manifest.json",
        "description": "Environment manifest JSON for release.",
    },
    "release_diff_patch": {
        "rel_dir": "experiment_results/releases/{tag}",
        "pattern": "diff.patch",
        "description": "Git diff patch captured for a dirty working tree at release time.",
    },
    "release_repro_status_md": {
        "rel_dir": "experiment_results/releases/{tag}",
        "pattern": "repro_status.md",
        "description": "Reproducibility check summary for a release tag.",
    },
    "repro_methods_md": {
        "rel_dir": "experiment_results/releases/{tag}",
        "pattern": "reproduction_methods.md",
        "description": "Methods-level reproduction instructions for a release.",
    },
    "repro_protocol_md": {
        "rel_dir": "experiment_results/releases/{tag}",
        "pattern": "reproduction_protocol.md",
        "description": "Supplementary reproduction protocol for a release.",
    },
}


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    regex_parts: List[str] = []
    formatter = Formatter()
    for literal, field, _format_spec, _conv in formatter.parse(pattern):
        if literal:
            regex_parts.append(re.escape(literal))
        if field:
            regex_parts.append(r"(?P<%s>[A-Za-z0-9._-]+)" % field)
    regex = "".join(regex_parts)
    return re.compile(rf"^{regex}$")


def _format_artifact_path(kind: str, meta: Dict[str, Any]) -> Tuple[str, str]:
    entry = ARTIFACT_TYPE_REGISTRY.get(kind)
    if not entry:
        raise KeyError(f"Unknown artifact kind '{kind}'")
    rel_dir_template = entry["rel_dir"]
    pattern_template = entry["pattern"]
    try:
        rel_dir = rel_dir_template.format(**meta)
    except KeyError as exc:
        raise KeyError(f"Missing placeholder '{exc.args[0]}' for rel_dir of kind '{kind}'") from exc
    try:
        name = pattern_template.format(**meta)
    except KeyError as exc:
        raise KeyError(f"Missing placeholder '{exc.args[0]}' for pattern of kind '{kind}'") from exc
    if "/" in name or name.strip() == "":
        raise ValueError("Pattern must resolve to a filename without directories.")
    regex = _pattern_to_regex(pattern_template)
    if not regex.fullmatch(name):
        raise ValueError(f"Resolved name '{name}' does not match pattern '{pattern_template}'.")
    return rel_dir, name


def _artifact_kind_catalog() -> str:
    parts = []
    for kind in sorted(ARTIFACT_TYPE_REGISTRY.keys()):
        entry = ARTIFACT_TYPE_REGISTRY[kind]
        parts.append(f"{kind}: {entry['rel_dir']}/{entry['pattern']}")
    return "; ".join(parts)


def list_artifacts_by_kind(kind: str, limit: int = 100):
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = load_entries(base_folder=exp_dir, limit=None)
    filtered = [e for e in entries if e.get("kind") == kind]
    return {"kind": kind, "paths": [e.get("path") for e in filtered[:limit]], "total": len(filtered)}


def _reserve_typed_artifact_impl(kind: str, meta_json: Optional[str], unique: bool) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if meta_json:
        try:
            parsed = json.loads(meta_json)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid meta_json: {exc}", "kind": kind}
        if not isinstance(parsed, dict):
            return {"error": "meta_json must decode to an object/dict.", "kind": kind}
        meta = parsed
    try:
        rel_dir, name = _format_artifact_path(kind, meta)
    except Exception as exc:
        return {"error": str(exc), "kind": kind}
    try:
        target, quarantined, note = resolve_output_path(
            subdir=rel_dir, name=name, allow_quarantine=True, unique=unique
        )
    except Exception as exc:
        return {"error": f"Failed to reserve path: {exc}", "kind": kind}
    result: Dict[str, Any] = {
        "reserved_path": str(target),
        "kind": kind,
        "name": name,
        "rel_dir": rel_dir,
        "quarantined": quarantined,
    }
    if note:
        result["note"] = note
    return result


def reserve_typed_artifact(kind: str, meta_json: Optional[str] = None, unique: bool = True):
    return _reserve_typed_artifact_impl(kind=kind, meta_json=meta_json, unique=unique)


def reserve_and_register_artifact(
    kind: str, meta_json: Optional[str] = None, status: str = "pending", unique: bool = True
):
    meta: Dict[str, Any] = {}
    if meta_json:
        try:
            parsed = json.loads(meta_json)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid meta_json: {exc}", "kind": kind}
        if not isinstance(parsed, dict):
            return {"error": "meta_json must decode to an object/dict.", "kind": kind}
        meta = parsed

    reserve = _reserve_typed_artifact_impl(kind=kind, meta_json=meta_json, unique=unique)
    if reserve.get("error"):
        return reserve
    path = reserve.get("reserved_path")
    if not path:
        return {"error": "failed_to_reserve", "kind": kind}
    entry = {
        "path": path,
        "name": reserve.get("name") or os.path.basename(path),
        "kind": kind,
        "created_by": meta.get("created_by") or meta.get("actor") or os.environ.get("AISC_ACTIVE_ROLE") or "unknown",
        "status": status,
    }
    manifest_res = append_or_update(entry, base_folder=BaseTool.resolve_output_dir(None))
    if manifest_res.get("error"):
        reserve["manifest_error"] = manifest_res["error"]
    else:
        manifest_idx = manifest_res.get("manifest_index")
        if manifest_idx:
            reserve["manifest_index"] = manifest_idx
    return reserve


def reserve_output(name: str, subdir: Optional[str] = None):
    target, quarantined, note = resolve_output_path(subdir=subdir, name=name)
    result = {"reserved_path": str(target), "quarantined": quarantined}
    if note:
        result["note"] = note
    return result
