# pyright: reportMissingImports=false
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.pathing import resolve_output_path
from ai_scientist.utils.transport_index import index_transport_runs


def transport_manifest_path() -> Path:
    exp_dir = BaseTool.resolve_output_dir(None)
    manifest_path = exp_dir / "simulations" / "transport_runs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return manifest_path


def acquire_manifest_lock(manifest_path: Path, timeout: float = 5.0, poll: float = 0.2) -> Optional[Path]:
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(poll)
    return None


def atomic_write_json(target: Path, data: Any) -> Optional[str]:
    lock = acquire_manifest_lock(target)
    if lock is None:
        return "Failed to acquire manifest lock; concurrent write in progress."

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            try:
                lock.unlink()
            except Exception:
                pass
        return f"Failed to write manifest: {exc}"

    try:
        lock.unlink()
    except Exception:
        pass
    return None


def load_transport_manifest() -> Dict[str, Any]:
    manifest_path = transport_manifest_path()
    if not manifest_path.exists():
        return {"schema_version": 1, "runs": []}
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"schema_version": 1, "runs": []}
        data.setdefault("schema_version", 1)
        data.setdefault("runs", [])
        return data
    except Exception:
        return {"schema_version": 1, "runs": []}


def upsert_transport_manifest_entry(
    baseline: str,
    transport: float,
    seed: int,
    status: str,
    paths: Dict[str, Optional[str]],
    notes: str = "",
    actor: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = load_transport_manifest()
    runs: List[Dict[str, Any]] = manifest.get("runs", [])
    actor_name = actor or os.environ.get("AISC_ACTIVE_ROLE", "") or "unknown"

    found = None
    for entry in runs:
        if (
            entry.get("baseline") == baseline
            and entry.get("transport") == transport
            and entry.get("seed") == seed
        ):
            found = entry
            break

    if found is None:
        found = {"baseline": baseline, "transport": transport, "seed": seed}
        runs.append(found)

    found.update(
        {
            "status": status,
            "paths": paths,
            "updated_at": datetime.now().isoformat(),
            "notes": notes or "",
            "actor": actor_name,
        }
    )
    manifest["runs"] = runs
    manifest.setdefault("schema_version", 1)

    err = atomic_write_json(transport_manifest_path(), manifest)
    if err:
        return {"error": err}
    return {"manifest_path": str(transport_manifest_path()), "entry": found}


def scan_transport_runs(root: Path) -> List[Dict[str, Any]]:
    run_root = root.parent.parent if root.name == "transport_runs" else root
    idx = index_transport_runs(base_dir=run_root)
    entries_dict = idx.get("entries", {}) if isinstance(idx, dict) else {}
    entries: List[Dict[str, Any]] = []
    for entry in entries_dict.values():
        if isinstance(entry, dict):
            entry.setdefault("actor", "system-scan")
            entry.setdefault("updated_at", datetime.now().isoformat())
            entries.append(entry)
    return entries


def build_seed_dir(baseline: str, transport: float, seed: int) -> Path:
    root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
    seed_dir = root / baseline / f"transport_{transport}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def resolve_run_paths(seed_dir: Path, baseline: str) -> Dict[str, Path]:
    return {
        "failure_matrix": seed_dir / f"{baseline}_sim_failure_matrix.npy",
        "time_vector": seed_dir / f"{baseline}_sim_time_vector.npy",
        "nodes_order": seed_dir / f"nodes_order_{baseline}_sim.txt",
        "sim_json": seed_dir / f"{baseline}_sim.json",
        "sim_status": seed_dir / f"{baseline}_sim.status.json",
        "verification": seed_dir / f"{baseline}_sim_verification.md",
    }


def status_from_paths(paths: Dict[str, Optional[str]], required_keys: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    required = required_keys or ["failure_matrix", "time_vector", "nodes_order", "sim_json", "sim_status"]
    missing = [k for k in required if not paths.get(k)]
    status = "complete" if not missing else "partial"
    return status, missing


def write_verification(seed_dir: Path, baseline: str, transport: float, seed: int, status: str, notes: str):
    ver_path = resolve_run_paths(seed_dir, baseline)["verification"]
    lines = [
        f"baseline: {baseline}",
        f"transport: {transport}",
        f"seed: {seed}",
        f"status: {status}",
        f"notes: {notes}",
    ]
    try:
        ver_path.write_text("\n".join(lines))
    except Exception:
        pass


def generate_run_recipe(base_folder: str):
    base_path = Path(base_folder)
    morph_dir, _, _ = resolve_output_path(subdir="morphologies", name="", run_root=base_path / "experiment_results", allow_quarantine=False, unique=False)
    transport_runs_dir, _, _ = resolve_output_path(subdir="simulations/transport_runs", name="", run_root=base_path / "experiment_results", allow_quarantine=False, unique=False)
    recipe_path = transport_runs_dir / "run_recipe.json"
    transport_runs_dir.mkdir(parents=True, exist_ok=True)

    allowed_suffixes = [".npy", ".npz", ".graphml", ".gpickle", ".gml"]
    entries: List[Dict[str, str]] = []

    def _find_template(baseline: str) -> Tuple[Optional[str], Optional[str]]:
        root = transport_runs_dir / baseline
        if not root.exists():
            return None, None
        for tdir in root.iterdir():
            if not tdir.is_dir() or not tdir.name.startswith("transport_"):
                continue
            seed_dir = tdir / "seed_0"
            if not seed_dir.exists():
                continue
            sim_json = seed_dir / f"{baseline}_sim.json"
            nodes = seed_dir / f"nodes_order_{baseline}_sim.txt"
            sim_str = str(sim_json) if sim_json.exists() else None
            nodes_str = str(nodes) if nodes.exists() else None
            if sim_str or nodes_str:
                return sim_str, nodes_str
        return None, None

    if morph_dir.exists():
        for f in morph_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in allowed_suffixes:
                continue
            baseline = f.stem
            tpl_json, tpl_nodes = _find_template(baseline)
            entries.append(
                {
                    "baseline": baseline,
                    "morphology_path": str(f),
                    "template_json": tpl_json or "",
                    "nodes_order_template": tpl_nodes or "",
                    "output_root": str(
                        base_path
                        / "experiment_results"
                        / "simulations"
                        / "transport_sweep"
                        / "transport_{transport}"
                        / "seed_{seed}"
                        / ""
                    ),
                }
            )

    try:
        with open(recipe_path, "w") as fp:
            json.dump(entries, fp, indent=2)
    except Exception:
        pass


def resolve_baseline_path_internal(baseline: str) -> Tuple[Optional[Path], List[str], Optional[str]]:
    allowed_suffixes = [".npy", ".npz", ".graphml", ".gpickle", ".gml"]
    try:
        p = BaseTool.resolve_input_path(baseline, allow_dir=False)
        if p.suffix.lower() in allowed_suffixes:
            return p, [], None
        return None, [], f"Unsupported baseline format '{p.suffix}'. Allowed: {', '.join(allowed_suffixes)}"
    except Exception:
        pass

    morph_dir = BaseTool.resolve_output_dir(None) / "morphologies"
    candidates: List[Path] = []
    if morph_dir.exists():
        for f in morph_dir.iterdir():
            if f.is_file():
                candidates.append(f)
    available = sorted({c.stem for c in candidates})

    for suff in allowed_suffixes:
        candidate = morph_dir / f"{baseline}{suff}"
        if candidate.exists():
            return candidate, available, None

    return None, available, f"Baseline '{baseline}' not found. Provide a valid graph path or one of: {', '.join(available)}"


def resolve_baseline_path(baseline: str) -> Dict[str, Any]:
    path, available, err = resolve_baseline_path_internal(baseline)
    if path:
        return {"path": str(path)}
    return {"error": err or "baseline not found", "available_baselines": available}


def resolve_sim_path(baseline: str, transport: float, seed: int) -> Dict[str, Any]:
    data = load_transport_manifest()
    runs = data.get("runs", [])
    if not runs:
        root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
        runs = scan_transport_runs(root)

    candidates = [r for r in runs if r.get("baseline") == baseline]
    available_transports = sorted(
        [t for t in (r.get("transport") for r in candidates if "transport" in r) if isinstance(t, (int, float))]
    )
    available_pairs = sorted(
        [
            (t, s)
            for t, s in (
                (r.get("transport"), r.get("seed")) for r in candidates if "transport" in r and "seed" in r
            )
            if isinstance(t, (int, float)) and isinstance(s, (int, float))
        ]
    )

    entry = next(
        (r for r in candidates if r.get("transport") == transport and r.get("seed") == seed),
        None,
    )
    if entry is None:
        return {
            "error": f"Run not found for baseline={baseline}, transport={transport}, seed={seed}",
            "available_transports": available_transports,
            "available_pairs": available_pairs,
        }

    paths = entry.get("paths", {}) if isinstance(entry, dict) else {}
    expected = (
        BaseTool.resolve_output_dir(None)
        / "simulations"
        / "transport_runs"
        / baseline
        / f"transport_{transport}"
        / f"seed_{seed}"
        / f"{baseline}_sim.json"
    )
    resolved_path: Optional[Path] = None
    sim_path = paths.get("sim_json")
    if sim_path:
        p = Path(sim_path)
        if p.exists():
            resolved_path = p
    if resolved_path is None and expected.exists():
        resolved_path = expected

    missing = [k for k in ["failure_matrix", "time_vector", "nodes_order", "sim_json", "sim_status"] if not paths.get(k)]
    if resolved_path is None:
        return {
            "error": f"sim.json missing for baseline={baseline}, transport={transport}, seed={seed}",
            "status": entry.get("status"),
            "paths": paths,
            "missing": missing,
            "available_transports": available_transports,
            "available_pairs": available_pairs,
        }

    if missing:
        note = f"warning: missing {', '.join(missing)}"
    else:
        note = ""
    return {
        "path": str(resolved_path),
        "status": entry.get("status"),
        "missing": missing,
        "note": note,
        "available_transports": available_transports,
        "available_pairs": available_pairs,
    }


def scan_transport_manifest(write: bool = True) -> Dict[str, Any]:
    root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
    idx = index_transport_runs(base_dir=root.parent.parent if root.name == "transport_runs" else root)
    entries_dict = idx.get("entries", {}) if isinstance(idx, dict) else {}
    entries = list(entries_dict.values())
    manifest = {
        "schema_version": 1,
        "runs": entries,
        "updated_at": datetime.now().isoformat(),
        "index_path": idx.get("index_path") if isinstance(idx, dict) else None,
    }
    if write:
        err = atomic_write_json(transport_manifest_path(), manifest)
        if err:
            return {"error": err, "manifest_path": str(transport_manifest_path())}
    return {"manifest_path": str(transport_manifest_path()), "runs": entries}


def read_transport_manifest(baseline: Optional[str] = None, transport: Optional[float] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    data = load_transport_manifest()
    runs = data.get("runs", [])
    filtered: List[Dict[str, Any]] = []
    for entry in runs:
        if baseline is not None and entry.get("baseline") != baseline:
            continue
        if transport is not None and entry.get("transport") != transport:
            continue
        if seed is not None and entry.get("seed") != seed:
            continue
        filtered.append(entry)
    if not runs:
        root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
        runs = scan_transport_runs(root)
        filtered = [e for e in runs if (baseline is None or e["baseline"] == baseline) and (transport is None or e["transport"] == transport) and (seed is None or e["seed"] == seed)]
    return {"manifest_path": str(transport_manifest_path()), "runs": filtered, "schema_version": data.get("schema_version", 1)}


def update_transport_manifest(
    baseline: str,
    transport: float,
    seed: int,
    status: str,
    paths_json: Optional[str] = None,
    notes: str = "",
    actor: str = "",
) -> Dict[str, Any]:
    root = BaseTool.resolve_output_dir(None) / "simulations" / "transport_runs"
    seed_dir = root / baseline / f"transport_{transport}" / f"seed_{seed}"

    def _infer(path_name: str) -> Optional[str]:
        try:
            parsed_paths = json.loads(paths_json) if paths_json else {}
        except Exception:
            parsed_paths = {}
        if isinstance(parsed_paths, dict) and path_name in parsed_paths:
            return parsed_paths[path_name]
        candidates = {
            "failure_matrix": seed_dir / f"{baseline}_sim_failure_matrix.npy",
            "time_vector": seed_dir / f"{baseline}_sim_time_vector.npy",
            "nodes_order": seed_dir / f"nodes_order_{baseline}_sim.txt",
            "sim_json": seed_dir / f"{baseline}_sim.json",
            "sim_status": seed_dir / f"{baseline}_sim.status.json",
        }
        p = candidates.get(path_name)
        return str(p) if p and p.exists() else None

    resolved_paths = {
        "failure_matrix": _infer("failure_matrix"),
        "time_vector": _infer("time_vector"),
        "nodes_order": _infer("nodes_order"),
        "sim_json": _infer("sim_json"),
        "sim_status": _infer("sim_status"),
    }
    missing = [k for k, v in resolved_paths.items() if v is None]
    if status == "complete" and missing:
        status = "partial"
        suffix = f"; missing: {', '.join(missing)}" if notes else f"missing: {', '.join(missing)}"
        notes = notes + suffix if notes else suffix

    actor_name = actor or os.environ.get("AISC_ACTIVE_ROLE", "") or "unknown"
    return upsert_transport_manifest_entry(
        baseline=baseline,
        transport=transport,
        seed=seed,
        status=status,
        paths=resolved_paths,
        notes=notes,
        actor=actor_name,
    )
