"""Bulk repair utility for simulation outputs that are missing exported arrays."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.tools.per_compartment_validator import validate_per_compartment_outputs
from ai_scientist.tools.sim_postprocess import export_sim_timeseries
from ai_scientist.utils import manifest as manifest_utils


def _acquire_lock(target: Path, timeout: float = 5.0, poll: float = 0.2) -> Optional[Path]:
    """
    Simple per-artifact lock to avoid concurrent repair of the same file.
    """
    lock_path = target.with_suffix(target.suffix + ".repair.lock")
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(poll)
    return None


def _release_lock(lock_path: Optional[Path]):
    if lock_path and lock_path.exists():
        try:
            lock_path.unlink()
        except Exception:
            pass


def _normalize_entry(entry: Dict[str, Any], fallback_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Normalize manifest entry shape (path-keyed dict).
    """
    path = entry.get("path") or fallback_path or entry.get("name")
    if not path:
        return None
    name = os.path.basename(entry.get("name") or path)
    entry_type = entry.get("type")
    annotations: List[Dict[str, Any]] = []

    meta = entry.get("metadata")
    if isinstance(meta, dict):
        entry_type = entry_type or meta.get("type")
        ann = {k: v for k, v in meta.items() if k != "type"}
        if ann:
            annotations.append(ann)
        nested = meta.get("annotations")
        if isinstance(nested, list):
            for ann in nested:
                if isinstance(ann, dict):
                    annotations.append({k: v for k, v in ann.items() if k != "type"})
    if isinstance(entry.get("annotations"), list):
        for ann in entry["annotations"]:
            if isinstance(ann, dict):
                annotations.append(ann)

    normalized = {
        "name": name,
        "path": path,
        "type": entry_type,
        "annotations": annotations,
        "timestamp": entry.get("timestamp"),
    }
    if entry_type:
        normalized["metadata"] = {"type": entry_type}
    return normalized


def _resolve_manifest_base(manifest_path: Optional[Path], run_root: Optional[Path]) -> Path:
    """
    Resolve the experiment_results directory for manifest loading based on inputs.
    """
    if run_root:
        exp_dir = run_root / "experiment_results"
        return exp_dir if exp_dir.exists() else run_root
    if manifest_path:
        if manifest_path.is_dir():
            return manifest_path
        if manifest_path.name == "manifest_index.json":
            if manifest_path.parent.name == "manifest":
                return manifest_path.parent.parent
            return manifest_path.parent
        if manifest_path.name == "file_manifest.json":
            return manifest_path.parent
        return manifest_path.parent
    return BaseTool.resolve_output_dir(None)


def _load_manifest_map(base_folder: Path) -> Dict[str, Dict[str, Any]]:
    entries = manifest_utils.load_entries(base_folder=base_folder)
    return {e["path"]: e for e in entries if isinstance(e, dict) and e.get("path")}


def _atomic_write_json(target: Path, data: Any) -> Optional[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(target)
    if lock is None:
        return "manifest_locked"
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, target)
    except Exception as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            _release_lock(lock)
        return f"write_failed: {exc}"
    _release_lock(lock)
    return None


def _expected_postprocess_paths(sim_path: Path) -> Dict[str, Path]:
    stem = sim_path.stem.replace(".json", "")
    out_dir = sim_path.parent
    return {
        "failure_matrix": out_dir / f"{stem}_failure_matrix.npy",
        "time_vector": out_dir / f"{stem}_time_vector.npy",
        "nodes_order": out_dir / f"nodes_order_{stem}.txt",
    }


def _append_annotation(entry: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
    annotations = entry.get("annotations") or []
    if annotation and annotation not in annotations:
        annotations.append(annotation)
    entry["annotations"] = annotations
    if entry.get("type"):
        entry["metadata"] = {"type": entry["type"]}
    entry["timestamp"] = datetime.now().isoformat()
    return entry


def _log_tool_summary(exp_dir: Path, lines: List[str]) -> str:
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "tool_summary.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = [f"[{ts}] repair_sim_outputs"] + lines
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for line in payload:
                f.write(line + "\n")
    except Exception:
        return str(log_path)
    return str(log_path)


def _collect_targets(manifest_map: Dict[str, Dict[str, Any]], manifest_paths: Optional[List[str]]) -> List[Path]:
    targets: List[Path] = []
    seen: set[str] = set()

    if manifest_paths:
        for raw in manifest_paths:
            try:
                p = BaseTool.resolve_input_path(raw, allow_dir=False)
            except FileNotFoundError:
                p = Path(raw)
            if p.suffix.lower() != ".json":
                continue
            if str(p) not in seen:
                seen.add(str(p))
                targets.append(p)
    else:
        for path_str, entry in manifest_map.items():
            if not path_str.lower().endswith(".json"):
                continue
            if "sim" not in Path(path_str).name.lower():
                continue
            if str(path_str) in seen:
                continue
            seen.add(str(path_str))
            targets.append(Path(path_str))
    targets.sort(key=lambda p: str(p))
    return targets


def _process_sim(sim_path: Path, force: bool, failure_threshold: float = 0.2) -> Dict[str, Any]:
    lock = _acquire_lock(sim_path)
    if lock is None:
        return {"path": str(sim_path), "status": "skipped", "reason": "locked"}
    result: Dict[str, Any] = {"path": str(sim_path)}
    try:
        if not sim_path.exists():
            result["status"] = "error"
            result["reason"] = "missing"
            return result

        try:
            with sim_path.open() as f:
                sim_data = json.load(f)
        except Exception as exc:
            result["status"] = "error"
            result["reason"] = f"json_load_failed: {exc}"
            return result

        if sim_data.get("time") is None or sim_data.get("E") is None:
            result["status"] = "error"
            result["reason"] = "sim_json_missing_time_or_E"
            return result

        expected = _expected_postprocess_paths(sim_path)
        missing_before = [k for k, p in expected.items() if not p.exists()]
        result["missing_before"] = missing_before

        post_status: Dict[str, Any] = {}
        if missing_before or force:
            try:
                export_result = export_sim_timeseries(
                    sim_json_path=sim_path,
                    graph_path=None,
                    output_dir=sim_path.parent,
                    failure_threshold=failure_threshold,
                )
                post_status = {"generated": True, "export_result": export_result}
            except Exception as exc:
                post_status = {"generated": False, "error": str(exc)}
        else:
            post_status = {"generated": False, "skipped": "arrays_present"}
        result["postprocess"] = post_status

        missing_after = [k for k, p in expected.items() if not p.exists()]
        result["missing_after"] = missing_after

        validation = validate_per_compartment_outputs(sim_path.parent)
        result["per_compartment_validation"] = validation

        if post_status.get("error"):
            result["status"] = "error"
            result["reason"] = post_status.get("error")
        elif missing_after:
            result["status"] = "partial"
            result["reason"] = "outputs_still_missing"
        else:
            result["status"] = "repaired" if missing_before else "unchanged"
    finally:
        _release_lock(lock)
    return result


def _ensure_entry(manifest_map: Dict[str, Dict[str, Any]], path: Path, entry_type: str) -> Dict[str, Any]:
    path_str = str(path)
    entry = manifest_map.get(path_str, {"name": path.name, "path": path_str, "type": entry_type, "annotations": []})
    entry["name"] = entry.get("name") or path.name
    entry["path"] = path_str
    entry["type"] = entry.get("type") or entry_type
    return entry


def repair_sim_outputs(
    manifest_paths: Optional[List[str]] = None,
    batch_size: int = 10,
    force: bool = False,
    manifest_path: Optional[str] = None,
    run_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bulk repair simulation outputs: runs sim_postprocess on sim.json files lacking exported arrays,
    validates per-compartment artifacts, and updates the manifest/tool_summary with outcomes.
    """
    run_root_path = Path(run_root) if run_root else None
    manifest_path_obj = Path(manifest_path) if manifest_path else None
    exp_dir = _resolve_manifest_base(manifest_path_obj, run_root_path)

    manifest_map = _load_manifest_map(exp_dir)
    targets = _collect_targets(manifest_map, manifest_paths)
    if not targets:
        return {
            "manifest_path": str(exp_dir / "manifest" / "manifest_index.json"),
            "processed": 0,
            "repaired": 0,
            "errors": [],
            "skipped": [],
            "warning": "No candidate sim.json files found.",
        }

    results: List[Dict[str, Any]] = []
    repaired = 0
    errors: List[str] = []
    skipped: List[str] = []
    updated_entries: List[Dict[str, Any]] = []

    for sim_path in targets[: max(batch_size, 1)]:
        res = _process_sim(sim_path, force=force)
        results.append(res)
        status = res.get("status")
        if status == "repaired":
            repaired += 1
        elif status == "error":
            errors.append(res.get("reason", str(res.get("path"))))
        elif status == "skipped":
            skipped.append(res.get("reason", "skipped"))

        # Update manifest entries for this sim and its outputs
        sim_entry = _ensure_entry(manifest_map, sim_path, entry_type=res.get("type") or "data")
        sim_annotation = {
            "source": "repair_sim_outputs",
            "status": status,
            "reason": res.get("reason"),
        }
        _append_annotation(sim_entry, sim_annotation)
        manifest_map[sim_entry["path"]] = sim_entry
        updated_entries.append(sim_entry)

        expected = _expected_postprocess_paths(sim_path)
        for label, p in expected.items():
            if p.exists():
                entry = _ensure_entry(manifest_map, p, entry_type="data")
                _append_annotation(
                    entry,
                    {
                        "source": "repair_sim_outputs",
                        "derived_from": sim_path.name,
                        "kind": label,
                    },
                )
                manifest_map[entry["path"]] = entry
                updated_entries.append(entry)

    for entry in updated_entries:
        manifest_utils.append_manifest_entry(entry, base_folder=exp_dir, dedupe_key=entry.get("path"))

    log_lines = [
        f"manifest: {exp_dir/'manifest'/'manifest_index.json'}",
        f"processed: {len(results)} (limit {batch_size})",
        f"repaired: {repaired}",
        f"errors: {len(errors)}",
        f"skipped: {len(skipped)}",
    ]
    log_path = _log_tool_summary(exp_dir, log_lines)

    summary: Dict[str, Any] = {
        "manifest_path": str(exp_dir / "manifest" / "manifest_index.json"),
        "processed": len(results),
        "repaired": repaired,
        "errors": errors,
        "skipped": skipped,
        "results": results,
        "log_path": log_path,
    }
    return summary


class RepairSimOutputsTool(BaseTool):
    """
    Bulk-repair sim outputs: find sim.json entries (or provided paths), run sim_postprocess to emit arrays,
    validate per-compartment artifacts, and update the manifest/tool_summary. Idempotent and lock-aware.
    """

    def __init__(
        self,
        name: str = "repair_sim_outputs",
        description: str = (
            "Bulk repair sim outputs: given manifest paths (or auto-scan manifest for sim.json), "
            "run sim_postprocess when arrays are missing, validate per-compartment artifacts, and update the manifest."
        ),
    ):
        parameters = [
            {
                "name": "manifest_paths",
                "type": "list[str]",
                "description": "Optional explicit sim.json paths to repair (defaults to manifest scan).",
            },
            {"name": "batch_size", "type": "int", "description": "Max entries to process in one call (default 10)."},
            {"name": "force", "type": "bool", "description": "Re-run postprocess even if arrays exist (default False)."},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, **kwargs) -> Dict[str, Any]:
        manifest_paths = kwargs.get("manifest_paths")
        batch_size = int(kwargs.get("batch_size", 10))
        force = bool(kwargs.get("force", False))
        manifest_path = kwargs.get("manifest_path")
        run_root = kwargs.get("run_root")

        return repair_sim_outputs(
            manifest_paths=manifest_paths,
            batch_size=batch_size,
            force=force,
            manifest_path=manifest_path,
            run_root=run_root,
        )


__all__ = ["RepairSimOutputsTool", "repair_sim_outputs"]
