import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ai_scientist.utils.health import log_missing_or_corrupt
from ai_scientist.utils.pathing import resolve_output_path


def _run_root(base_dir: Optional[str | Path]) -> Path:
    if base_dir:
        return Path(base_dir)
    env_dir = os.environ.get("AISC_EXP_RESULTS", "")
    if env_dir:
        return Path(env_dir)
    base = os.environ.get("AISC_BASE_FOLDER", "")
    if base:
        return Path(base) / "experiment_results"
    return Path("experiment_results")


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> Optional[str]:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:  # pragma: no cover - defensive
        return str(exc)
    return None


def _maybe_file(path: Path) -> Optional[str]:
    try:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return str(path)
    except Exception:
        return None
    return None


def _entry_key(baseline: str, transport: float, seed: int, variant: Optional[str]) -> str:
    return f"{baseline}|{transport}|{seed}|{variant or ''}"


def index_transport_runs(base_dir: Optional[str | Path] = None) -> Dict[str, object]:
    """
    Scan simulations/transport_runs for sim outputs and write an index.json.
    Returns mapping keyed by baseline|transport|seed|variant.
    """
    root_dir, _, _ = resolve_output_path(
        subdir="simulations/transport_runs",
        name="",
        run_root=_run_root(base_dir),
        allow_quarantine=True,
        unique=False,
    )
    index_path = root_dir / "index.json"
    entries: Dict[str, Dict[str, object]] = {}
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)

    for baseline_dir in root_dir.iterdir():
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        for transport_dir in baseline_dir.iterdir():
            if not transport_dir.is_dir() or not transport_dir.name.startswith("transport_"):
                continue
            t_str = transport_dir.name.split("transport_", 1)[-1]
            try:
                transport_val = float(t_str)
            except Exception:
                continue
            for seed_dir in transport_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                s_str = seed_dir.name.split("seed_", 1)[-1]
                try:
                    seed_val = int(s_str)
                except Exception:
                    continue

                base_prefix = baseline
                entry: Dict[str, object] = {
                    "baseline": baseline,
                    "transport": transport_val,
                    "seed": seed_val,
                    "variant": None,
                    "paths": {
                        "sim_json": _maybe_file(seed_dir / f"{base_prefix}_sim.json"),
                        "sim_status": _maybe_file(seed_dir / f"{base_prefix}_sim.status.json"),
                        "failure_matrix": _maybe_file(seed_dir / f"{base_prefix}_sim_failure_matrix.npy"),
                        "time_vector": _maybe_file(seed_dir / f"{base_prefix}_sim_time_vector.npy"),
                        "nodes_order": _maybe_file(seed_dir / f"nodes_order_{base_prefix}_sim.txt"),
                        "per_compartment": _maybe_file(seed_dir / "per_compartment.npz"),
                        "node_index_map": _maybe_file(seed_dir / "node_index_map.json"),
                        "topology_summary": _maybe_file(seed_dir / "topology_summary.json"),
                    },
                }
                missing = [k for k, v in entry["paths"].items() if not v]
                entry["status"] = "complete" if not missing else "partial"
                if missing:
                    entry["notes"] = f"missing: {', '.join(missing)}"
                key = _entry_key(baseline, transport_val, seed_val, None)
                entries[key] = entry

    err = _atomic_write_json(index_path, {"entries": entries, "schema_version": 1})
    result: Dict[str, object] = {"index_path": str(index_path), "entries": entries}
    if err:
        # quarantine write if primary path failed
        q_path, _, _ = resolve_output_path(
            subdir="_unrouted/transport_runs",
            name="index.json",
            run_root=_run_root(base_dir),
            allow_quarantine=True,
            unique=True,
        )
        _atomic_write_json(q_path, {"entries": entries, "schema_version": 1, "note": f"primary write failed: {err}"})
        result["error"] = err
        result["index_path"] = str(q_path)
    return result


def resolve_transport_sim(
    baseline: str,
    transport: str | float,
    seed: str | int,
    *,
    variant: Optional[str] = None,
    base_dir: Optional[str | Path] = None,
    refresh: bool = True,
) -> Dict[str, object]:
    """
    Resolve transport sim outputs via index.json, refreshing if needed.
    Returns paths or an error with health note logged.
    """
    root_dir = _run_root(base_dir)
    index_path = root_dir / "simulations" / "transport_runs" / "index.json"
    key = _entry_key(baseline, float(transport), int(seed), variant)

    def _load() -> Tuple[Dict[str, object], Optional[str]]:
        if not index_path.exists():
            return {}, "missing"
        try:
            with index_path.open() as f:
                data = json.load(f)
            return data.get("entries", {}), None
        except Exception as exc:
            return {}, str(exc)

    entries, load_err = _load()
    if (not entries or key not in entries) and refresh:
        idx = index_transport_runs(base_dir)
        index_path = Path(idx["index_path"])
        entries = idx.get("entries", {})  # type: ignore[assignment]

    if key not in entries:
        note = {"transport_missing": {"baseline": baseline, "transport": float(transport), "seed": int(seed)}}
        log_missing_or_corrupt([note])
        return {"error": "sim outputs not found for key", "key": key, "index_path": str(index_path), "load_error": load_err}

    return {"entry": entries[key], "index_path": str(index_path)}
