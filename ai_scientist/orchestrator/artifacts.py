import json
import os
import re
from datetime import datetime
from string import Formatter
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.manifest import append_or_update, load_entries
from ai_scientist.utils.pathing import resolve_output_path

from ai_scientist.orchestrator.summarization import maybe_generate_module_summary

from ai_scientist.orchestrator.context_specs import (
    active_role,
    ensure_write_permission,
    record_context_access,
)

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
    "project_snapshot_md": {
        "rel_dir": "experiment_results",
        "pattern": "project_snapshot.md",
        "description": "Full project snapshot summary.",
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
    "integration_memo_md": {
        "rel_dir": "experiment_results/summaries",
        "pattern": "integrated_memo_{module}.md",
        "description": "Module-level integrated memo summarizing artifacts.",
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
    "verification_note": {
        "rel_dir": "experiment_results/verification",
        "pattern": "{experiment_id}_verification.md",
        "description": "Verification note or proof-of-work for an experiment.",
    },
    "model_metrics_json": {
        "rel_dir": "experiment_results/metrics",
        "pattern": "{model_key}_metrics.json",
        "description": "Computed metrics for a model (JSON).",
    },
    "model_metrics_csv": {
        "rel_dir": "experiment_results/metrics",
        "pattern": "{model_key}_metrics.csv",
        "description": "Computed metrics for a model (CSV).",
    },
    "review_report": {
        "rel_dir": "experiment_results/reviews",
        "pattern": "review_report_{manuscript_version}.md",
        "description": "Holistic review report for a manuscript version.",
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


def _load_meta_dict(meta_json: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not meta_json:
        return {}, None
    try:
        parsed = json.loads(meta_json)
    except json.JSONDecodeError as exc:
        return {}, f"Invalid meta_json: {exc}"
    if not isinstance(parsed, dict):
        return {}, "meta_json must decode to an object/dict."
    return parsed, None


def _prepare_artifact_metadata(meta: Dict[str, Any], kind: str) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(meta)
    now = datetime.utcnow().isoformat()
    normalized.setdefault("id", f"{kind}_{now.replace(':', '').replace('-', '')}_{uuid4().hex[:6]}")
    normalized.setdefault("type", kind)
    normalized.setdefault("version", "v1")
    normalized.setdefault("parent_version", None)
    normalized.setdefault("status", "pending")
    normalized.setdefault("module", normalized.get("module") or "general")
    summary = normalized.get("summary")
    if not summary or not str(summary).strip():
        normalized["summary"] = f"Auto-generated {kind} artifact"
    content = normalized.get("content")
    if content is None:
        normalized["content"] = {"description": f"Auto-generated placeholder for {kind}"}
    metadata = normalized.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    metadata.setdefault("created_at", now)
    metadata.setdefault("created_by", metadata.get("created_by") or os.environ.get("AISC_ACTIVE_ROLE") or "unknown")
    metadata.setdefault("tags", [])
    metadata.setdefault("tools_used", [])
    
    # Handle change_summary if it was passed at top level or in metadata
    cs = normalized.pop("change_summary", None)
    if cs:
        metadata["change_summary"] = cs
        
    normalized["metadata"] = metadata
    return normalized


def list_artifacts_by_kind(kind: str, limit: int = 100):
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = load_entries(base_folder=exp_dir, limit=None)
    filtered = [e for e in entries if e.get("kind") == kind]
    return {"kind": kind, "paths": [e.get("path") for e in filtered[:limit]], "total": len(filtered)}


def _reserve_typed_artifact_impl(
    kind: str,
    meta_json: Optional[str],
    unique: bool,
    *,
    meta_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if meta_dict is not None:
        normalized_meta = _prepare_artifact_metadata(meta_dict, kind)
    else:
        parsed, err = _load_meta_dict(meta_json)
        if err:
            return {"error": err, "kind": kind}
        normalized_meta = _prepare_artifact_metadata(parsed, kind)
    meta = normalized_meta
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
    result["metadata"] = normalized_meta
    if note:
        result["note"] = note
    record_context_access(
        active_role(), 
        kind, 
        result["reserved_path"], 
        "write", 
        artifact_id=normalized_meta.get("id"),
        version=normalized_meta.get("version")
    )
    return result


def reserve_typed_artifact(kind: str, meta_json: Optional[str] = None, unique: bool = True):
    role = active_role()
    allowed, reason = ensure_write_permission(kind, role)
    if not allowed:
        return {"error": reason, "kind": kind}
    return _reserve_typed_artifact_impl(kind=kind, meta_json=meta_json, unique=unique)


def reserve_and_register_artifact(
    kind: str,
    meta_json: Optional[str] = None,
    status: str = "pending",
    unique: bool = True,
    skip_summary: bool = False,
    change_summary: str = "",
):
    role = active_role()
    allowed, reason = ensure_write_permission(kind, role)
    if not allowed:
        return {"error": reason, "kind": kind}
    parsed_meta, err = _load_meta_dict(meta_json)
    if err:
        return {"error": err, "kind": kind}
    
    # Auto-versioning logic
    # If version is NOT provided in meta, try to increment
    if not parsed_meta.get("version"):
        module = parsed_meta.get("module")
        latest = _get_latest_artifact_entry(kind, module)
        if latest:
            latest_ver = latest.get("version") or latest.get("metadata", {}).get("version")
            parsed_meta["parent_version"] = latest_ver
            parsed_meta["version"] = _next_version(latest_ver)
        else:
            parsed_meta["version"] = "v1"
            parsed_meta["parent_version"] = None

    if change_summary:
        parsed_meta["change_summary"] = change_summary

    try:
        normalized_meta = _prepare_artifact_metadata(parsed_meta, kind)
    except ValueError as exc:
        return {"error": str(exc), "kind": kind}
    normalized_meta["status"] = status

    reserve = _reserve_typed_artifact_impl(
        kind=kind,
        meta_json=None,
        unique=unique,
        meta_dict=normalized_meta,
    )
    if reserve.get("error"):
        return reserve
    path = reserve.get("reserved_path")
    if not path:
        return {"error": "failed_to_reserve", "kind": kind}
    entry = {
        "path": path,
        "name": reserve.get("name") or os.path.basename(path),
        "kind": kind,
        "created_by": normalized_meta.get("metadata", {}).get("created_by")
        or os.environ.get("AISC_ACTIVE_ROLE")
        or "unknown",
        "status": normalized_meta.get("status") or status,
        "version": normalized_meta.get("version"),
        "parent_version": normalized_meta.get("parent_version"),
        "module": normalized_meta.get("module"),
        "summary": normalized_meta.get("summary"),
        "content": normalized_meta.get("content"),
        "metadata": normalized_meta,
    }
    manifest_res = append_or_update(entry, base_folder=BaseTool.resolve_output_dir(None))
    if manifest_res.get("error"):
        reserve["manifest_error"] = manifest_res["error"]
    else:
        manifest_idx = manifest_res.get("manifest_index")
        if manifest_idx:
            reserve["manifest_index"] = manifest_idx
    if not skip_summary:
        module_name = normalized_meta.get("module")
        if module_name:
            base_dir = str(BaseTool.resolve_output_dir(None))
            threshold_value = max(
                1, int(os.environ.get("AISC_SUMMARY_THRESHOLD") or "5")
            )
            _maybe_trigger_module_summary(module_name, base_dir, threshold_value)
    return reserve


def reserve_output(name: str, subdir: Optional[str] = None):
    target, quarantined, note = resolve_output_path(subdir=subdir, name=name)
    result = {"reserved_path": str(target), "quarantined": quarantined}
    if note:
        result["note"] = note
    return result


def _next_version(parent_version: Optional[str]) -> str:
    if parent_version and isinstance(parent_version, str) and parent_version.startswith("v"):
        digits = parent_version[1:]
        if digits.isdigit():
            return f"v{int(digits) + 1}"
    return "v1"


def _next_summary_version(parent_version: Optional[str]) -> str:
    return _next_version(parent_version)


def _get_latest_artifact_entry(kind: str, module: Optional[str] = None) -> Optional[Dict[str, Any]]:
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = load_entries(base_folder=exp_dir, limit=None)
    # Filter by kind
    candidates = [e for e in entries if (e.get("kind") or e.get("metadata", {}).get("kind")) == kind]
    # Filter by module if specified
    if module:
        candidates = [e for e in candidates if (e.get("module") or e.get("metadata", {}).get("module")) == module]
    
    if not candidates:
        return None
    
    # Sort by created_at (most recent first)
    def sort_key(e):
        ts = e.get("created_at")
        if not ts or ts == "null":
            return ""
        return str(ts)
    
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def promote_artifact_to_canonical(name: str, kind: str, notes: str = ""):
    """
    Promote an artifact to 'canonical' status.
    Updates the manifest entry AND reserves a new manifest entry to log the status change.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    from ai_scientist.utils import manifest as manifest_utils
    entry = manifest_utils.find_manifest_entry(name, base_folder=exp_dir)
    if not entry:
        return {"error": f"Artifact '{name}' not found in manifest."}
    
    if entry.get("kind") != kind:
        return {"error": f"Artifact kind mismatch: found '{entry.get('kind')}', expected '{kind}'"}

    new_entry = dict(entry)
    new_entry["status"] = "canonical"
    
    # Add promotion notes to metadata
    meta = new_entry.get("metadata", {})
    if isinstance(meta, dict):
        promotions = meta.get("promotions", [])
        promotions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "actor": os.environ.get("AISC_ACTIVE_ROLE", "unknown"),
            "notes": notes
        })
        meta["promotions"] = promotions
        new_entry["metadata"] = meta
        
    manifest_res = append_or_update(new_entry, base_folder=exp_dir)
    return {
        "status": "promoted", 
        "name": name, 
        "version": new_entry.get("version"),
        "manifest_result": manifest_res
    }


def _maybe_trigger_module_summary(module_name: str, base_folder: str, threshold: int) -> None:
    try:
        summary_payload = maybe_generate_module_summary(
            module_name, base_folder=base_folder, threshold=threshold
        )
    except Exception:
        record_context_access(active_role(), "integration_memo_md", None, "summary_generation_failed")
        return
    if not summary_payload:
        return
    _register_module_summary(summary_payload, base_folder)


def _register_module_summary(summary_payload: Dict[str, Any], base_folder: str) -> None:
    module_name = summary_payload["module"]
    summary_text = summary_payload["summary_text"]
    summary_meta = {
        "module": module_name,
        "summary": summary_text.splitlines()[0] if summary_text else f"Integrated memo for {module_name}",
        "content": {
            "text": summary_text,
            "entries": summary_payload["entries"],
        },
        "metadata": {
            "summarized_up_to": summary_payload["summarized_up_to"],
            "summarized_count": summary_payload["summary_count"],
            "entries": [entry["name"] for entry in summary_payload["entries"]],
        },
        "version": _next_summary_version(summary_payload.get("parent_version")),
        "parent_version": summary_payload.get("parent_version"),
        "status": "canonical",
    }
    summary_meta["id"] = f"integration_memo_{module_name}_{summary_payload['summary_count']}_{summary_meta['version']}"
    reserve = _reserve_typed_artifact_impl(
        kind="integration_memo_md",
        meta_json=None,
        unique=False,
        meta_dict=summary_meta,
    )
    if reserve.get("error"):
        record_context_access(active_role(), "integration_memo_md", None, "summary_reserve_failed")
        return
    path = reserve.get("reserved_path")
    if not path:
        record_context_access(active_role(), "integration_memo_md", None, "summary_notify_nopath")
        return
    dir_name = os.path.dirname(path)
    try:
        os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(summary_text)
    except Exception:
        record_context_access(active_role(), "integration_memo_md", path, "summary_write_failed")
    reserve_metadata = reserve.get("metadata") or {}
    entry = {
        "path": path,
        "name": reserve.get("name") or os.path.basename(path),
        "kind": "integration_memo_md",
        "created_by": reserve_metadata.get("created_by") or os.environ.get("AISC_ACTIVE_ROLE") or "automated_summarizer",
        "status": summary_meta["status"],
        "version": summary_meta["version"],
        "parent_version": summary_meta.get("parent_version"),
        "module": module_name,
        "summary": summary_meta["summary"],
        "content": summary_meta["content"],
        "metadata": reserve_metadata,
    }
    append_or_update(entry, base_folder=base_folder)
