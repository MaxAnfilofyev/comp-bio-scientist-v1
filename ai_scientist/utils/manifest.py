import fnmatch
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.health import log_missing_or_corrupt

SCHEMA_VERSION = 2
DEFAULT_SHARD_SIZE = 10000
MAX_TOOL_LIMIT = 2000


def _iso_now() -> str:
    return datetime.now().isoformat()


def _manifest_paths(base_folder: Optional[str | Path]) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Resolve manifest directories relative to the active run.
    """
    exp_dir = Path(base_folder) if base_folder else BaseTool.resolve_output_dir(None)
    manifest_dir = exp_dir / "manifest"
    index_path = manifest_dir / "manifest_index.json"
    quarantine_dir = manifest_dir / "_quarantine"
    legacy_path = exp_dir / "file_manifest.json"
    return exp_dir, manifest_dir, index_path, quarantine_dir, legacy_path


def _acquire_lock(target: Path, timeout: float = 5.0, poll: float = 0.2) -> Optional[Path]:
    lock_path = target.with_suffix(target.suffix + ".lock")
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(poll)
    return None


def _release_lock(lock_path: Optional[Path]) -> None:
    if lock_path and lock_path.exists():
        try:
            lock_path.unlink()
        except Exception:
            pass


def _atomic_write_json(target: Path, data: Any) -> Optional[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(target)
    if lock is None:
        return "manifest_locked"
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            _release_lock(lock)
        return f"write_failed: {exc}"
    _release_lock(lock)
    return None


def _normalize_entry(entry: Dict[str, Any], fallback_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Normalize a manifest entry to the expected schema.
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

    timestamp = entry.get("timestamp") or _iso_now()
    normalized = {
        "name": name,
        "path": path,
        "type": entry_type,
        "annotations": annotations,
        "timestamp": timestamp,
        "metadata": {"type": entry_type} if entry_type else {},
        "schema_version": SCHEMA_VERSION,
    }
    return normalized


def _normalize_v2_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Lean schema for manifest v2: one row per artifact path.
    Required: path; Optional: name (defaults to basename), kind, created_by, status, created_at, size_bytes.
    """
    path = entry.get("path") or entry.get("name")
    if not path:
        return None
    name = os.path.basename(entry.get("name") or path)
    kind = entry.get("kind") or entry.get("type")
    created_by = entry.get("created_by") or entry.get("source") or entry.get("actor")
    status = entry.get("status") or "ok"
    created_at = entry.get("created_at") or _iso_now()
    size_bytes = entry.get("size_bytes")
    if size_bytes is None:
        try:
            size_bytes = Path(path).stat().st_size
        except Exception:
            size_bytes = None
    return {
        "path": path,
        "name": name,
        "kind": kind,
        "created_by": created_by,
        "status": status,
        "created_at": created_at,
        "size_bytes": size_bytes,
        "schema_version": SCHEMA_VERSION,
    }


def unique_path_check(*, base_folder: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Return counts for total vs distinct paths to ensure manifest deduplication.
    """
    entries = load_entries(base_folder=base_folder, limit=None)
    total = len(entries)
    paths = [str(e.get("path")) for e in entries if e.get("path")]
    distinct = len(set(paths))
    dup_paths = []
    seen: set[str] = set()
    for p in paths:
        if p in seen:
            dup_paths.append(p)
        else:
            seen.add(p)
    return {"total": total, "distinct_paths": distinct, "duplicate_paths": dup_paths}


def _load_legacy_manifest_map(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
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
                norm = _normalize_entry(val, fallback_path=key)
                if norm:
                    manifest_map[norm["path"]] = norm
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                norm = _normalize_entry(item)
                if norm:
                    manifest_map[norm["path"]] = norm
    return manifest_map


def _write_shard(shard_path: Path, entries: Iterable[Dict[str, Any]]) -> Tuple[int, Optional[str]]:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry))
                f.write("\n")
                count += 1
        os.replace(tmp_path, shard_path)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return count, f"write_failed: {exc}"
    return count, None


def _read_shard(shard_path: Path) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """
    Return entries, status, and parse errors for a shard.
    """
    if not shard_path.exists():
        return [], "missing", [f"missing:{shard_path}"]
    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    status = "ok"
    try:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        entries.append(obj)
                except Exception as exc:
                    status = "corrupt"
                    errors.append(f"{shard_path.name}:{exc}")
    except Exception as exc:
        status = "corrupt"
        errors.append(f"{shard_path.name}:{exc}")
    return entries, status, errors


def _quarantine_shard(shard_path: Path, quarantine_dir: Path) -> Optional[str]:
    try:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = quarantine_dir / f"{shard_path.name}.corrupt"
        shard_path.replace(dest)
        return str(dest)
    except Exception as exc:
        return f"quarantine_failed:{exc}"


def _bootstrap_index(index_path: Path, shard_size: int) -> Dict[str, Any]:
    base_index = {
        "schema_version": SCHEMA_VERSION,
        "shard_size": shard_size,
        "shards": [],
        "updated_at": _iso_now(),
    }
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(index_path, base_index)
        return base_index
    try:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("index_not_dict")
    except Exception:
        _atomic_write_json(index_path, base_index)
        return base_index

    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("shard_size", shard_size)
    data.setdefault("shards", [])
    data["updated_at"] = _iso_now()
    _atomic_write_json(index_path, data)
    return data


def _migrate_legacy_manifest(legacy_path: Path, manifest_dir: Path, index: Dict[str, Any]) -> Optional[str]:
    legacy_map = _load_legacy_manifest_map(legacy_path)
    if not legacy_map:
        return None

    entries = list(legacy_map.values())
    if not entries:
        return None

    shard_size = int(index.get("shard_size") or DEFAULT_SHARD_SIZE)
    shards_meta: List[Dict[str, Any]] = []
    chunk: List[Dict[str, Any]] = []
    shard_idx = 1
    for entry in entries:
        chunk.append(entry)
        if len(chunk) >= shard_size:
            shard_path = manifest_dir / f"manifest_shard_{shard_idx:04d}.ndjson"
            count, err = _write_shard(shard_path, chunk)
            if err:
                return err
            shards_meta.append(
                {
                    "path": str(shard_path),
                    "count": count,
                    "ts_min": min(e.get("timestamp", "") for e in chunk),
                    "ts_max": max(e.get("timestamp", "") for e in chunk),
                    "status": "ok",
                }
            )
            shard_idx += 1
            chunk = []

    if chunk:
        shard_path = manifest_dir / f"manifest_shard_{shard_idx:04d}.ndjson"
        count, err = _write_shard(shard_path, chunk)
        if err:
            return err
        shards_meta.append(
            {
                "path": str(shard_path),
                "count": count,
                "ts_min": min(e.get("timestamp", "") for e in chunk),
                "ts_max": max(e.get("timestamp", "") for e in chunk),
                "status": "ok",
            }
        )

    index["shards"] = shards_meta
    index["updated_at"] = _iso_now()
    _atomic_write_json(manifest_dir / "manifest_index.json", index)
    return None


def bootstrap_manifest(base_folder: Optional[str | Path] = None, shard_size: int = DEFAULT_SHARD_SIZE) -> Dict[str, Any]:
    """
    Ensure manifest/index exist; migrate legacy file_manifest.json if present.
    """
    _, manifest_dir, index_path, _, legacy_path = _manifest_paths(base_folder)
    index = _bootstrap_index(index_path, shard_size)
    if not index.get("shards"):
        migrate_err = _migrate_legacy_manifest(legacy_path, manifest_dir, index)
        if migrate_err:
            log_missing_or_corrupt([{"error": migrate_err, "path": str(legacy_path)}])
        # Fallback: hydrate shards from existing ndjson files if index is empty.
        if not index.get("shards"):
            shard_files = sorted(manifest_dir.glob("manifest_shard_*.ndjson"))
            shards_meta: List[Dict[str, Any]] = []
            for shard_path in shard_files:
                entries, status, _ = _read_shard(shard_path)
                shard_meta: Dict[str, Any] = {
                    "path": str(shard_path),
                    "count": len(entries),
                    "status": status,
                }
                if entries:
                    shard_meta["ts_min"] = min(e.get("created_at") or e.get("timestamp") or "" for e in entries)
                    shard_meta["ts_max"] = max(e.get("created_at") or e.get("timestamp") or "" for e in entries)
                else:
                    shard_meta["ts_min"] = None
                    shard_meta["ts_max"] = None
                shards_meta.append(shard_meta)
            if shards_meta:
                index["shards"] = shards_meta
                index["updated_at"] = _iso_now()
                _atomic_write_json(index_path, index)
    return index


def _update_shard_meta(shard_meta: Dict[str, Any], entries: List[Dict[str, Any]]) -> None:
    shard_meta["count"] = len(entries)
    if entries:
        shard_meta["ts_min"] = min(e.get("created_at") or e.get("timestamp") or "" for e in entries)
        shard_meta["ts_max"] = max(e.get("created_at") or e.get("timestamp") or "" for e in entries)
    else:
        shard_meta["ts_min"] = None
        shard_meta["ts_max"] = None


def _active_shard(index: Dict[str, Any], manifest_dir: Path, shard_size: int) -> Dict[str, Any]:
    shards = index.get("shards", [])
    if shards:
        last = shards[-1]
        if int(last.get("count", 0)) < shard_size:
            return last
    shard_path = manifest_dir / f"manifest_shard_{len(shards) + 1:04d}.ndjson"
    shard_meta = {"path": str(shard_path), "count": 0, "ts_min": None, "ts_max": None, "status": "ok"}
    shards.append(shard_meta)
    index["shards"] = shards
    return shard_meta


def append_manifest_entry(
    entry: Dict[str, Any],
    *,
    base_folder: Optional[str | Path] = None,
    shard_size: Optional[int] = None,
    dedupe_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Legacy append path. Delegates to append_or_update (manifest v2 lean schema).
    """
    return append_or_update(entry, base_folder=base_folder, shard_size=shard_size)


def append_or_update(
    entry: Dict[str, Any],
    *,
    base_folder: Optional[str | Path] = None,
    shard_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Manifest v2: upsert by path with a lean schema (path, name, kind, created_by, created_at, status, size_bytes).
    Deduplicates across shards to keep one row per artifact.
    """
    exp_dir, manifest_dir, index_path, quarantine_dir, _ = _manifest_paths(base_folder)
    index = bootstrap_manifest(base_folder, shard_size or DEFAULT_SHARD_SIZE)
    norm = _normalize_v2_entry(entry)
    if not norm:
        return {"error": "invalid_entry", "entry": entry}

    path_key = norm["path"]
    shard_sz = int(index.get("shard_size") or shard_size or DEFAULT_SHARD_SIZE)

    lock = _acquire_lock(index_path)
    if lock is None:
        return {"error": "manifest_locked"}
    try:
        # First, remove any existing entry for this path across all shards.
        for shard_meta in list(index.get("shards", [])):
            shard_path = Path(shard_meta.get("path", ""))
            if not shard_path.exists():
                continue
            shard_entries, shard_status, errors = _read_shard(shard_path)
            if shard_status == "corrupt":
                quarantine_res = _quarantine_shard(shard_path, quarantine_dir)
                shard_meta["status"] = "quarantined"
                _atomic_write_json(
                    exp_dir / "_health" / "manifest_health.json",
                    {"errors": errors, "quarantine": quarantine_res, "shard": str(shard_path)},
                )
                continue
            filtered = [e for e in shard_entries if e.get("path") != path_key]
            if len(filtered) != len(shard_entries):
                filtered.sort(key=lambda e: e.get("created_at") or e.get("timestamp") or "")
                _write_shard(shard_path, filtered)
                _update_shard_meta(shard_meta, filtered)

        # Append/update in the active shard
        shard_meta = _active_shard(index, manifest_dir, shard_sz)
        shard_path = Path(shard_meta["path"])
        shard_entries: List[Dict[str, Any]] = []
        shard_status = shard_meta.get("status", "ok")
        if shard_path.exists():
            shard_entries, shard_status, errors = _read_shard(shard_path)
            if shard_status == "corrupt":
                quarantine_res = _quarantine_shard(shard_path, quarantine_dir)
                shard_meta["status"] = "quarantined"
                _atomic_write_json(
                    exp_dir / "_health" / "manifest_health.json",
                    {"errors": errors, "quarantine": quarantine_res, "shard": str(shard_path)},
                )
                shard_meta = _active_shard(index, manifest_dir, shard_sz)
                shard_path = Path(shard_meta["path"])
                shard_entries = []
                shard_status = "ok"
        shard_entries = [e for e in shard_entries if e.get("path") != path_key]
        shard_entries.append(norm)
        shard_entries.sort(key=lambda e: e.get("created_at") or e.get("timestamp") or "")
        count, err = _write_shard(shard_path, shard_entries)
        if err:
            return {"error": err, "shard": str(shard_path)}

        shard_meta["path"] = str(shard_path)
        shard_meta["status"] = shard_status
        _update_shard_meta(shard_meta, shard_entries)
        index["updated_at"] = _iso_now()
        _atomic_write_json(index_path, index)
        return {
            "manifest_index": str(index_path),
            "shard": str(shard_path),
            "count": count,
            "deduped": True,
            "schema_version": SCHEMA_VERSION,
        }
    finally:
        _release_lock(lock)


def _matches_filters(entry: Dict[str, Any], role: Optional[str], path_glob: Optional[str], since_ts: Optional[datetime]) -> bool:
    if since_ts:
        try:
            ts = datetime.fromisoformat(entry.get("created_at") or entry.get("timestamp", ""))
            if ts < since_ts:
                return False
        except Exception:
            pass
    if role:
        src = str(entry.get("created_by") or "").lower()
        if role.lower() not in src:
            return False
    if path_glob:
        if not (fnmatch.fnmatch(entry.get("path", ""), path_glob) or fnmatch.fnmatch(entry.get("name", ""), path_glob)):
            return False
    return True


def _iter_entries(
    shards: List[Dict[str, Any]],
    role: Optional[str],
    path_glob: Optional[str],
    since_ts: Optional[datetime],
    reverse: bool = True,
) -> Iterable[Dict[str, Any]]:
    ordered = list(shards)
    ordered.sort(key=lambda s: s.get("path", ""))
    if reverse:
        ordered = list(reversed(ordered))
    for shard_meta in ordered:
        shard_path = Path(shard_meta.get("path", ""))
        entries, _, _ = _read_shard(shard_path)
        for entry in entries:
            if _matches_filters(entry, role, path_glob, since_ts):
                yield entry


def inspect_manifest(
    *,
    base_folder: Optional[str | Path] = None,
    role: Optional[str] = None,
    path_glob: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 200,
    summary_only: bool = True,
    include_samples: int = 3,
) -> Dict[str, Any]:
    """
    Filtered/summary manifest reader for agents. Defaults to summary-only to avoid large payloads.
    """
    exp_dir, _, index_path, _, _ = _manifest_paths(base_folder)
    index = bootstrap_manifest(base_folder)
    shards = index.get("shards", [])
    since_ts = None
    if since:
        try:
            since_ts = datetime.fromisoformat(since)
        except Exception:
            since_ts = None

    hard_limit = max(0, min(limit, MAX_TOOL_LIMIT))
    sample_limit = max(1, include_samples)
    total = 0
    count_by_role: Dict[str, int] = {}
    count_by_ext: Dict[str, int] = {}
    entries_out: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    for entry in _iter_entries(shards, role, path_glob, since_ts, reverse=True):
        total += 1
        ext = os.path.splitext(entry.get("name", ""))[1].lower()
        if ext:
            count_by_ext[ext] = count_by_ext.get(ext, 0) + 1
        src = str(entry.get("created_by") or "unknown").lower() or "unknown"
        count_by_role[src] = count_by_role.get(src, 0) + 1

        if summary_only and len(samples) < sample_limit:
            samples.append(entry)
        if not summary_only and len(entries_out) < hard_limit:
            entries_out.append(entry)
        if not summary_only and len(entries_out) >= hard_limit:
            continue

    summary = {
        "total": total,
        "count_by_role": count_by_role,
        "count_by_ext": count_by_ext,
    }
    result_entries = samples if summary_only else entries_out
    return {
        "schema_version": index.get("schema_version", SCHEMA_VERSION),
        "manifest_index": str(index_path),
        "summary": summary,
        "entries": result_entries,
        "shard_info": shards,
    }


def find_manifest_entry(path_or_name: str, *, base_folder: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """
    Locate a single manifest entry by exact path or basename. Returns latest match.
    """
    index = bootstrap_manifest(base_folder)
    shards = index.get("shards", [])
    target_lower = path_or_name.lower()
    for entry in _iter_entries(shards, role=None, path_glob=None, since_ts=None, reverse=True):
        if entry.get("path", "").lower() == target_lower or entry.get("name", "").lower() == target_lower:
            return entry
    return None


def load_entries(
    *,
    base_folder: Optional[str | Path] = None,
    limit: Optional[int] = None,
    role: Optional[str] = None,
    path_glob: Optional[str] = None,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Internal helper to stream manifest entries with optional limit and filters.
    """
    index = bootstrap_manifest(base_folder)
    shards = index.get("shards", [])
    since_ts = None
    if since:
        try:
            since_ts = datetime.fromisoformat(since)
        except Exception:
            since_ts = None
    out: List[Dict[str, Any]] = []
    max_items = limit or float("inf")
    for entry in _iter_entries(shards, role, path_glob, since_ts, reverse=True):
        out.append(entry)
        if len(out) >= max_items:
            break
    return out


def compact_manifest_shard(shard_path: Path) -> Dict[str, Any]:
    """
    Deduplicate a shard in place by path, keeping the newest entry.
    """
    entries, status, errors = _read_shard(shard_path)
    if status != "ok":
        return {"error": "corrupt_shard", "details": errors}
    dedup: Dict[str, Dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True):
        key = entry.get("path") or entry.get("name")
        if key and key not in dedup:
            dedup[key] = entry
    keep = list(reversed(list(dedup.values())))
    count, err = _write_shard(shard_path, keep)
    if err:
        return {"error": err}
    return {"shard": str(shard_path), "count": count}
