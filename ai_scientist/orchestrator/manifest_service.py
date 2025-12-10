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
from ai_scientist.utils.notes import append_run_note

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
def append_manifest(name: str, metadata_json: Optional[str] = None, allow_missing: bool = False):
    """
    Append an entry to the run's sharded manifest (experiment_results/manifest/...).
    Pass metadata as a JSON string (e.g., '{"type":"figure","source":"analyst"}').
    Creates the manifest file if missing.
    """
    return _append_manifest_entry(name=name, metadata_json=metadata_json, allow_missing=allow_missing)


def read_manifest_entry(path_or_name: str):
    """
    Read a single manifest entry by path key or filename (basename).
    """
    from ai_scientist.utils import manifest as manifest_utils
    from ai_scientist.tools.base_tool import BaseTool
    entry = manifest_utils.find_manifest_entry(path_or_name, base_folder=BaseTool.resolve_output_dir(None))
    if entry:
        return {"entry": entry}
    return {"error": "Not found", "path_or_name": path_or_name}


def check_manifest():
    """
    Validate manifest entries: report missing files, entries lacking type, and duplicate basenames.
    """
    from ai_scientist.utils import manifest as manifest_utils
    from ai_scientist.tools.base_tool import BaseTool
    from pathlib import Path
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = manifest_utils.load_entries(base_folder=exp_dir)
    if not entries:
        return {"error": "Manifest empty or missing", "path": str(exp_dir / 'manifest')}

    missing = []
    missing_type = []
    by_name: Dict[str, list[str]] = {}
    duplicates_by_path: Dict[str, int] = {}
    for entry in entries:
        path = entry.get("path", "")
        try:
            exists = Path(path).exists()
        except Exception:
            exists = False
        if not exists:
            missing.append(path)
        if not entry.get("kind"):
            missing_type.append(path)
        name = entry.get("name") or os.path.basename(path or "")
        by_name.setdefault(name, []).append(path)
        duplicates_by_path[path] = duplicates_by_path.get(path, 0) + 1

    duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    duplicate_paths = [p for p, count in duplicates_by_path.items() if count > 1]
    health_entries: List[Dict[str, Any]] = []
    if missing:
        health_entries.append({"missing_files": missing})
    if missing_type:
        health_entries.append({"missing_type": missing_type})
    if duplicates:
        health_entries.append({"duplicate_names": duplicates})
    if duplicate_paths:
        health_entries.append({"duplicate_paths": duplicate_paths})
    if health_entries:
        log_missing_or_corrupt(health_entries)
    return {
        "manifest_index": str(BaseTool.resolve_output_dir(None) / "manifest" / "manifest_index.json"),
        "n_entries": len(entries),
        "missing_files": missing,
        "missing_type": missing_type,
        "duplicate_names": duplicates,
        "duplicate_paths": duplicate_paths,
    }


def read_manifest():
    """Read the run's manifest with a capped entry list; use inspect_manifest for filtered views."""
    from ai_scientist.utils import manifest as manifest_utils
    from ai_scientist.tools.base_tool import BaseTool
    data = manifest_utils.inspect_manifest(
        base_folder=BaseTool.resolve_output_dir(None),
        summary_only=False,
        limit=500,
    )
    return {
        "manifest_index": data.get("manifest_index"),
        "entries": data.get("entries", []),
        "summary": data.get("summary", {}),
        "note": "Entries capped at 500; use inspect_manifest for filtered views.",
    }


def check_manifest_unique_paths():
    """
    Validate that manifest paths are unique (COUNT(DISTINCT path) == COUNT(*)).
    """
    from ai_scientist.utils import manifest as manifest_utils
    from ai_scientist.tools.base_tool import BaseTool
    result = manifest_utils.unique_path_check(base_folder=BaseTool.resolve_output_dir(None))
    ok = result["total"] == result["distinct_paths"]
    result["ok"] = ok
    return result


def list_artifacts(suffix: Optional[str] = None, subdir: Optional[str] = None):
    """
    List artifacts under experiment_results (optionally a subdir) with optional suffix filter.
    Agents should use this before selecting files.
    """
    from ai_scientist.tools.base_tool import BaseTool
    from pathlib import Path
    from ai_scientist.utils import manifest as manifest_utils

    exp_dir = BaseTool.resolve_output_dir(None)
    roots: List[Path] = []
    if subdir:
        sub_path = Path(subdir)
        roots.append(sub_path if sub_path.is_absolute() else exp_dir / sub_path)
        # Also try under base folder if provided
        base = os.environ.get("AISC_BASE_FOLDER", "")
        if base:
            roots.append(Path(base) / sub_path)
            roots.append(Path(base) / "experiment_results" / sub_path)
    else:
        roots.append(exp_dir)

    root = next((r for r in roots if r.exists()), roots[0])

    # If manifest has entries, return from manifest first
    manifest_entries = manifest_utils.load_entries(base_folder=exp_dir)
    manifest_paths = [e.get("path") for e in manifest_entries if e.get("path") and isinstance(e.get("path"), str)]
    manifest_paths_normalized = [str(Path(p).relative_to(exp_dir)) if p and Path(p).is_absolute() else str(p) for p in manifest_paths]

    files: List[str] = manifest_paths_normalized or []

    if not files:
        # Fallback to directory scan
        try:
            for p in root.rglob("*"):
                if p.is_file():
                    if suffix and not str(p).endswith(suffix):
                        continue
                    try:
                        rel = p.relative_to(root)
                        files.append(str(rel))
                    except Exception:
                        files.append(str(p))
        except Exception as exc:
            return {"root": str(root), "files": files, "error": f"list_artifacts failed: {exc}"}

    return {"root": str(root), "files": files, "note": "Files from manifest" if manifest_paths else "Scanned directory"}


def list_artifacts_by_kind(kind: str, limit: int = 100):
    """
    List artifacts from manifest v2 filtered by kind.
    """
    from ai_scientist.utils import manifest as manifest_utils
    from ai_scientist.tools.base_tool import BaseTool
    entries = manifest_utils.load_entries(base_folder=BaseTool.resolve_output_dir(None), limit=None)
    filtered = [e for e in entries if e.get("kind") == kind]
    return {"kind": kind, "paths": [e.get("path") for e in filtered[:limit]], "total": len(filtered)}


def get_artifact_index(max_entries: int = 2000):
    """
    Build a lightweight index of artifacts under experiment_results and include manifest entries if present.
    """
    from ai_scientist.tools.base_tool import BaseTool
    from ai_scientist.utils import manifest as manifest_utils

    exp_dir = BaseTool.resolve_output_dir(None)
    manifest = manifest_utils.inspect_manifest(
        base_folder=exp_dir,
        summary_only=False,
        limit=min(500, max_entries),
    )
    files: List[Dict[str, Any]] = []
    try:
        count = 0
        for p in exp_dir.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(exp_dir))
                try:
                    size = p.stat().st_size
                except Exception:
                    size = None
                files.append({"path": rel, "suffix": p.suffix.lower(), "size": size})
                count += 1
                if count >= max_entries:
                    break
    except Exception as exc:
        return {"root": str(exp_dir), "manifest": manifest, "error": f"index failed: {exc}"}
    return {"root": str(exp_dir), "manifest": manifest, "files": files}


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
