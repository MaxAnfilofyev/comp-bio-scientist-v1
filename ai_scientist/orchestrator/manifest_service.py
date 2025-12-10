"""
Manifest service module to handle orchestrator-facing manifest operations,
watcher logic, and project knowledge management.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_scientist.utils.health import log_missing_or_corrupt
from ai_scientist.utils import manifest as manifest_utils
from ai_scientist.utils.notes import append_run_note, read_note_file, write_note_file

from agents import function_tool


def _scan_and_auto_update_manifest(exp_dir: Path, skip: bool = False) -> List[str]:
    """
    Background Watcher: Scans experiment_results for orphaned files (files not in manifest).
    Adds them with inferred types and 'auto_watcher' annotation.
    Returns list of added filenames.
    """
    if (
        skip
        or os.environ.get("AISC_SKIP_WATCHER", "").strip().lower()
        in {"1", "true", "yes"}
    ):
        # Explicit opt-out to avoid long manifest scans during troubleshooting.
        return []
    added_files = []

    manifest_root = exp_dir / "manifest"

    # Preload manifest entries so we only write new files and avoid slow rewrites.
    known_paths: set[str] = set()
    try:
        existing_entries = manifest_utils.load_entries(base_folder=exp_dir)
        for entry in existing_entries:
            raw_path = entry.get("path")
            if not raw_path:
                continue
            p = Path(raw_path)
            known_paths.add(str(p))
            try:
                known_paths.add(str(p.resolve()))
            except Exception:
                pass
            if not p.is_absolute():
                try:
                    known_paths.add(str((exp_dir.parent / p).resolve()))
                except Exception:
                    pass
    except Exception:
        known_paths = set()

    for root, _, files in os.walk(exp_dir):
        if manifest_root in Path(root).parents or Path(root) == manifest_root:
            continue
        for name in files:
            if name == "file_manifest.json":
                continue
            if Path(root) == manifest_root:
                continue

            full_path = Path(root) / name
            path_str = str(full_path)
            resolved_str = None
            try:
                resolved_str = str(full_path.resolve())
            except Exception:
                resolved_str = None

            # Skip anything already indexed (by stored or resolved path).
            if path_str in known_paths or (resolved_str and resolved_str in known_paths):
                continue

            # Infer type
            suffix = full_path.suffix.lower()
            etype = "unknown"
            if suffix in [".png", ".pdf", ".svg"]:
                etype = "figure"
            elif suffix in [".csv", ".json", ".npy", ".npz"]:
                etype = "data"
            elif suffix in [".py"]:
                etype = "code"
            elif suffix in [".md", ".txt", ".log"]:
                etype = "text"

            # Create Entry (manifest v2 lean schema)
            try:
                size_bytes = full_path.stat().st_size
            except Exception:
                size_bytes = None
            entry = {
                "name": name,
                "path": path_str,
                "kind": etype,
                "created_by": "auto_watcher",
                "status": "ok",
                "size_bytes": size_bytes,
                "created_at": json.dumps(None),  # Will be converted to timestamp by manifest utils
            }
            res = manifest_utils.append_or_update(entry, base_folder=exp_dir)
            if not res.get("error"):
                added_files.append(name)
                known_paths.add(path_str)
                if resolved_str:
                    known_paths.add(resolved_str)

    return added_files


def _build_metadata_for_compat(entry_type: Optional[str], annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if entry_type:
        meta["type"] = entry_type
    return meta


def _normalize_manifest_entry(entry: Dict[str, Any], fallback_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    def _add_annotation(raw: Any, annotations_list: List[Dict[str, Any]]):
        if not isinstance(raw, dict):
            return
        cleaned = {k: v for k, v in raw.items() if k not in {"type", "annotations"}}
        if cleaned and cleaned not in annotations_list:
            annotations_list.append(cleaned)

    path = entry.get("path") or fallback_path or entry.get("name")
    if not path:
        return None
    name = os.path.basename(entry.get("name") or path)
    base_type = entry.get("type")
    annotations: List[Dict[str, Any]] = []

    meta = entry.get("metadata")
    if isinstance(meta, dict):
        base_type = meta.get("type", base_type)
        if not annotations:
            _add_annotation(meta, annotations)
        nested = meta.get("annotations")
        if isinstance(nested, list):
            for ann in nested:
                _add_annotation(ann, annotations)
    elif isinstance(meta, list):
        for m in meta:
            if isinstance(m, dict):
                if m.get("type") and not base_type:
                    base_type = m.get("type")
                if not annotations:
                    _add_annotation(m, annotations)
                nested = m.get("annotations")
                if isinstance(nested, list):
                    for ann in nested:
                        _add_annotation(ann, annotations)

    existing_annotations = entry.get("annotations")
    if isinstance(existing_annotations, list):
        for ann in existing_annotations:
            _add_annotation(ann, annotations)

    normalized = {
        "name": name,
        "path": path,
        "type": base_type,
        "annotations": annotations,
        "timestamp": entry.get("timestamp")
    }
    compat_meta = _build_metadata_for_compat(base_type, annotations)
    if compat_meta:
        normalized["metadata"] = compat_meta
    return normalized


def _load_manifest_map(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Backward-compatible manifest loader. Prefers sharded manifest via manifest_utils, and
    falls back to the legacy file_manifest.json if needed.
    """
    try:
        entries = manifest_utils.load_entries(base_folder=Path(manifest_path).parent.parent)
        manifest_map = {e["path"]: e for e in entries if isinstance(e, dict) and e.get("path")}
        if manifest_map:
            return manifest_map
    except Exception:
        pass

    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    manifest_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                norm = _normalize_manifest_entry(val, fallback_path=key)
                if norm:
                    manifest_map[norm["path"]] = norm
        return manifest_map

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                norm = _normalize_manifest_entry(item)
                if norm:
                    manifest_map[norm["path"]] = norm
    return manifest_map


def _append_manifest_entry(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False):
    from ai_scientist.tools.base_tool import BaseTool
    exp_dir = BaseTool.resolve_output_dir(None)

    try:
        target_path = BaseTool.resolve_input_path(name, allow_dir=True)
    except FileNotFoundError:
        if not allow_missing:
            return {"error": f"Referenced file not found: {name}. Use reserve_typed_artifact/reserve_output + append_manifest after creation, or set allow_missing=True if intentional."}
        target_path = BaseTool.resolve_output_dir(None) / name

    meta: Dict[str, Any] = {}
    if metadata_json:
        if len(metadata_json) > 400:
            return {"error": "metadata_json too long; keep kind/created_by/status short", "raw_len": len(metadata_json)}
        try:
            parsed_meta = json.loads(metadata_json)
            if isinstance(parsed_meta, dict):
                meta = parsed_meta
            else:
                return {"error": "metadata_json must be a JSON object", "raw": metadata_json}
        except Exception as exc:
            return {"error": f"Invalid metadata_json: {exc}", "raw": metadata_json}

    path_str = str(target_path)
    name_only = os.path.basename(name or path_str)
    try:
        size_bytes = target_path.stat().st_size
    except Exception:
        size_bytes = None
    entry: Dict[str, Any] = {
        "name": name_only,
        "path": path_str,
        "kind": meta.get("kind") or meta.get("type"),
        "created_by": meta.get("created_by") or meta.get("source") or meta.get("actor"),
        "status": meta.get("status") or "ok",
        "size_bytes": size_bytes,
        "created_at": meta.get("created_at") or json.dumps(None),  # Will be converted by manifest utils
    }
    res = manifest_utils.append_or_update(entry, base_folder=exp_dir)
    if res.get("error"):
        return res
    return {
        "manifest_index": res.get("manifest_index"),
        "shard": res.get("shard"),
        "n_entries": res.get("count"),
        "deduped": res.get("deduped", False),
    }


def _append_artifact_from_result(result: Any, key: str, metadata_json: Optional[str], allow_missing: bool = True):
    if not metadata_json or not isinstance(result, dict):
        return
    out = result.get(key)
    if isinstance(out, str):
        _append_manifest_entry(name=out, metadata_json=metadata_json, allow_missing=allow_missing)


def _append_figures_from_result(result: Any, metadata_json: Optional[str]):
    if not metadata_json or not isinstance(result, dict):
        return
    for _, v in result.items():
        if isinstance(v, str) and v.endswith((".png", ".pdf", ".svg")):
            _append_manifest_entry(name=v, metadata_json=metadata_json, allow_missing=True)


@function_tool
def check_project_state(base_folder: str) -> str:
    """
    Reads the project state to see what artifacts exist.
    UPDATED: Automatically scans for orphaned files and updates the manifest.
    Set env AISC_SKIP_WATCHER=1 or pass skip_watcher=True to skip the manifest scan.
    """
    status_msg = "Folder existed"

    if not os.path.exists(base_folder):
        try:
            os.makedirs(base_folder, exist_ok=True)
            exp_results = os.path.join(base_folder, "experiment_results")
            os.makedirs(exp_results, exist_ok=True)
            status_msg = f"Created new directory: {base_folder}"
        except Exception as e:
            return json.dumps({"error": f"Failed to create folder {base_folder}: {str(e)}"})

    exists = os.listdir(base_folder)
    exp_results = os.path.join(base_folder, "experiment_results")

    # --- AUTO-WATCHER TRIGGER ---
    orphans = []
    if os.path.exists(exp_results):
        # Default to skipping the watcher for speed; can be overridden via env or caller args.
        orphans = _scan_and_auto_update_manifest(
            Path(exp_results),
            skip=os.environ.get("AISC_SKIP_WATCHER", "").strip().lower()
            not in {"0", "false", "no"},
        )

    artifacts = os.listdir(exp_results) if os.path.exists(exp_results) else []
    has_plots = False
    has_data = any(x.endswith('.csv') for x in artifacts)
    has_lit_review = "lit_summary.json" in artifacts or "lit_summary.csv" in artifacts
    if os.path.exists(exp_results):
        for root, dirs, files in os.walk(exp_results):
            for f in files:
                lf = f.lower()
                if lf.endswith((".png", ".svg", ".pdf")):
                    has_plots = True
                if lf.endswith(".csv"):
                    has_data = True
                if lf in {"lit_summary.json", "lit_summary.csv"}:
                    has_lit_review = True
            if has_plots and has_data and has_lit_review:
                break

    return json.dumps({
        "status_message": status_msg,
        "orphaned_files_recovered": len(orphans),
        "root_files": exists,
        "artifacts": artifacts,
        "has_lit_review": has_lit_review,
        "has_data": has_data,
        "has_plots": has_plots,
        "has_draft": "manuscript.pdf" in exists or "manuscript.tex" in exists
    })


# Project knowledge management
@function_tool
def manage_project_knowledge(
    action: str,
    category: str = "general",
    observation: str = "",
    solution: str = "",
    actor: str = "",
) -> str:
    """
    Manage the persistent Project Knowledge Base (project_knowledge.md).
    Use this to store constraints, decisions, failure patterns, and REFLECTIONS that persist across sessions.

    Args:
        action: 'add' to log new info, 'read' to retrieve all knowledge.
        category: 'constraint', 'decision', 'failure_pattern', or 'reflection'.
        observation: Context of the problem, inefficiency, or constraint (Required for 'add').
        solution: The fix, decision, or proposed improvement (Required for 'add').
        actor: Optional role/agent name to auto-log who created the record. If omitted, falls back to env AISC_ACTIVE_ROLE or 'unknown'.
    """
    from ai_scientist.tools.base_tool import BaseTool
    base = BaseTool.resolve_output_dir(None).parent
    kb_path = os.path.join(base, "project_knowledge.md")
    actor_name = (
        actor.strip()
        or os.environ.get("AISC_ACTIVE_ROLE", "").strip()
        or "unknown"
    )

    if action == "read":
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading knowledge base: {str(e)}"
        return "Project Knowledge Base is empty."

    if action == "add":
        if not observation or not solution:
            return "Error: Both 'observation' and 'solution' are required for 'add' action."

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"\n## [{category.upper()}] {timestamp}\n"
            f"**Actor:** {actor_name}\n"
            f"**Observation/Problem:** {observation}\n"
            f"**Solution/Insight:** {solution}\n"
            f"{'-'*40}\n"
        )

        try:
            with open(kb_path, 'a') as f:
                f.write(entry)
            return f"Added new {category} entry to project_knowledge.md"
        except Exception as e:
            return f"Error writing to knowledge base: {str(e)}"

    return "Invalid action. Use 'add' or 'read'."


@function_tool
def inspect_recent_manifest_entries(base_folder: str) -> str:
    """
    Load recent manifest entries from experiment_results for agents.
    """
    from ai_scientist.utils import manifest as manifest_utils
    try:
        result = manifest_utils.inspect_manifest(
            base_folder=base_folder,
            limit=50,  # Cap for agent use
        )
        return json.dumps(result.get("entries", result))
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@function_tool
def inspect_manifest(base_folder: str) -> str:
    """
    Inspect manifest for agents, returning summary and recent entries.
    """
    from ai_scientist.utils import manifest as manifest_utils
    try:
        result = manifest_utils.inspect_manifest(
            base_folder=base_folder,
            summary_only=False,
            limit=100,
        )
        return json.dumps(result.get("entries", result))
    except Exception as exc:
        return json.dumps({"error": str(exc)})
