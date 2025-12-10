"""Shared helpers for standardized per-compartment simulation outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

STATE_SCHEMA_VERSION = "1.0"


def _ordering_checksum(ordering: Sequence[str]) -> str:
    joined = "|".join(ordering)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_node_index_payload(nodes: Sequence[Any]) -> Dict[str, Any]:
    """
    Build a canonical node index mapping with checksum and dual lookups.
    """
    ordering = [str(n) for n in nodes]
    checksum = _ordering_checksum(ordering)
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "ordering": ordering,
        "ordering_checksum": checksum,
        "node_index_map": [{"index": i, "node_id": node_id} for i, node_id in enumerate(ordering)],
        "index_to_node": ordering,
        "node_to_index": {node_id: i for i, node_id in enumerate(ordering)},
    }


def compute_topology_metrics(graph: Any, nodes: Sequence[Any], ordering_checksum: str) -> Dict[str, Any]:
    """
    Compute simple topology metrics. If no graph is provided, degrade gracefully.
    """
    metrics: Dict[str, Any] = {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "status": "ok" if graph is not None else "no_morphology",
        "N": len(nodes),
        "ordering_checksum": ordering_checksum,
    }
    if graph is None:
        metrics.update(
            {
                "leaf_count": 0,
                "mean_depth": 0.0,
                "max_depth": 0.0,
                "total_path_length": 0.0,
                "notes": "topology metrics unavailable (graph not provided)",
            }
        )
        return metrics

    try:
        import networkx as nx  # type: ignore

        degrees = [graph.degree(n) for n in nodes]
        metrics["leaf_count"] = int(sum(1 for d in degrees if d == 1))
        metrics["mean_depth"] = 0.0
        metrics["max_depth"] = 0.0
        metrics["total_path_length"] = 0.0
        if nodes:
            root = nodes[0]
            lengths = nx.single_source_shortest_path_length(graph, root)
            depths = [lengths.get(n, 0) for n in nodes]
            if depths:
                metrics["mean_depth"] = float(np.mean(depths))
                metrics["max_depth"] = float(np.max(depths))
                metrics["total_path_length"] = float(np.sum(list(lengths.values())))
    except Exception as exc:  # pragma: no cover - defensive
        metrics["status"] = "no_morphology"
        metrics["notes"] = f"failed to compute topology metrics: {exc}"
        metrics.setdefault("leaf_count", 0)
        metrics.setdefault("mean_depth", 0.0)
        metrics.setdefault("max_depth", 0.0)
        metrics.setdefault("total_path_length", 0.0)
    return metrics


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def write_per_compartment_outputs(
    output_dir: Path,
    binary_states: np.ndarray,
    continuous_states: np.ndarray,
    time_vector: np.ndarray,
    node_index_payload: Dict[str, Any],
    topology_metrics: Dict[str, Any],
    status: str = "ok",
) -> Dict[str, Any]:
    """
    Standardized writer for per-compartment outputs (binary + continuous + mapping + topology summary).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "per_compartment.npz"
    map_path = output_dir / "node_index_map.json"
    topo_path = output_dir / "topology_summary.json"

    payload_status = status
    npz_error: Optional[str] = None
    tmp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    try:
        try:
            with tmp_npz.open("wb") as f:
                np.savez_compressed(
                    f,
                    binary_states=binary_states.astype(np.uint8),
                    continuous_states=continuous_states.astype(np.float32),
                    time=time_vector.astype(np.float32),
                )
        except Exception:
            # Fallback to uncompressed if compressed fails (e.g., unusual dtypes)
            with tmp_npz.open("wb") as f:
                np.savez(
                    f,
                    binary_states=binary_states.astype(np.uint8),
                    continuous_states=continuous_states.astype(np.float32),
                    time=time_vector.astype(np.float32),
                )
        os.replace(tmp_npz, npz_path)
    except Exception as exc:
        payload_status = "corrupt"
        npz_error = str(exc)
        topology_metrics = {**topology_metrics, "notes": f"npz write failed: {exc}"}
        try:
            if tmp_npz.exists():
                tmp_npz.unlink()
        except Exception:
            pass

    try:
        payload = {"state_schema_version": STATE_SCHEMA_VERSION, **node_index_payload}
        _atomic_write_json(map_path, payload)
    except Exception:
        payload_status = payload_status if payload_status != "ok" else "corrupt"

    topology_payload = {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "status": payload_status,
        **topology_metrics,
        "binary_shape": list(binary_states.shape),
        "continuous_shape": list(continuous_states.shape),
    }
    try:
        _atomic_write_json(topo_path, topology_payload)
    except Exception:
        pass

    result = {
        "per_compartment_npz": str(npz_path),
        "node_index_map": str(map_path),
        "topology_summary": str(topo_path),
        "status": payload_status,
    }
    if npz_error:
        result["npz_error"] = npz_error
    return result


def derive_per_compartment_from_arrays(
    failure_matrix: np.ndarray,
    time_vector: np.ndarray,
    nodes_order: Sequence[Any],
    output_dir: Path,
    *,
    binary_threshold: Optional[float] = None,
    allow_mismatch: bool = False,
    skip_existing: bool = False,
    provenance: str | None = None,
) -> Dict[str, Any]:
    """
    Build per_compartment outputs from failure_matrix/time_vector/nodes_order.
    Returns status + written paths; does not raise on validation failures.
    """
    result: Dict[str, Any] = {"output_dir": str(output_dir)}
    output_dir = Path(output_dir)
    npz_path = output_dir / "per_compartment.npz"
    map_path = output_dir / "node_index_map.json"
    topo_path = output_dir / "topology_summary.json"
    if skip_existing and npz_path.exists() and map_path.exists() and topo_path.exists():
        result.update(
            {
                "status": "skipped",
                "reason": "per_compartment_artifacts_present",
                "per_compartment_npz": str(npz_path),
            }
        )
        return result

    try:
        fm = np.array(failure_matrix)
        tv = np.array(time_vector).reshape(-1)
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"array_conversion_failed: {exc}"
        return result

    if fm.ndim != 2:
        result["status"] = "error"
        result["reason"] = f"failure_matrix_expected_2d_got_{fm.ndim}"
        return result

    n_time, n_nodes = fm.shape
    if tv.shape[0] != n_time:
        if not allow_mismatch:
            result["status"] = "error"
            result["reason"] = f"time_len_{tv.shape[0]}_mismatch_matrix_{n_time}"
            return result
        min_len = min(tv.shape[0], n_time)
        fm = fm[:min_len]
        tv = tv[:min_len]
        result["warning"] = f"trimmed_to_{min_len}_steps_due_to_mismatch"

    ordering = [str(n).strip() for n in nodes_order if str(n).strip() != ""]
    if len(ordering) != n_nodes:
        result["status"] = "error"
        result["reason"] = f"node_count_mismatch_ordering_{len(ordering)}_matrix_{n_nodes}"
        return result

    binary_states = fm
    unique_vals = np.unique(binary_states)
    warning: Optional[str] = result.get("warning")
    if not set(unique_vals.tolist()).issubset({0, 1}):
        threshold = binary_threshold if binary_threshold is not None else 0.0
        binary_states = (fm > threshold).astype(np.uint8)
        if binary_threshold is None:
            warning = (warning + "; " if warning else "") + "non_binary_values_threshold_gt0_applied"
    else:
        binary_states = binary_states.astype(np.uint8)

    continuous_states = fm.astype(np.float32)
    node_index_payload = build_node_index_payload(ordering)
    topology_metrics = compute_topology_metrics(None, ordering, node_index_payload["ordering_checksum"])
    if provenance:
        topology_metrics["provenance"] = provenance

    try:
        write_status = write_per_compartment_outputs(
            output_dir=output_dir,
            binary_states=binary_states,
            continuous_states=continuous_states,
            time_vector=tv,
            node_index_payload=node_index_payload,
            topology_metrics=topology_metrics,
            status=topology_metrics.get("status", "ok"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        result["status"] = "error"
        result["reason"] = f"write_failed: {exc}"
        if warning:
            result["warning"] = warning
        return result

    result.update(write_status)
    write_status_value = write_status.get("status", "ok")
    if write_status_value in {"corrupt", "error"}:
        result["status"] = "error"
        result["reason"] = result.get("reason") or write_status.get("npz_error") or write_status_value
    else:
        result["status"] = "ok"
        if write_status_value != "ok":
            result["warning"] = (result.get("warning") + "; " if result.get("warning") else "") + str(write_status_value)
    npz_path = Path(write_status.get("per_compartment_npz", output_dir / "per_compartment.npz"))
    if not npz_path.exists():
        result["status"] = "error"
        result["reason"] = write_status.get("npz_error") or "per_compartment_npz_missing_after_write"
    if warning:
        result["warning"] = warning
    return result


def derive_per_compartment_from_files(
    failure_matrix_path: Path | str,
    time_vector_path: Path | str,
    nodes_order_path: Path | str,
    output_dir: Path | str,
    *,
    binary_threshold: Optional[float] = None,
    allow_mismatch: bool = False,
    skip_existing: bool = False,
    provenance: str | None = None,
) -> Dict[str, Any]:
    """
    Load failure_matrix/time_vector/nodes_order files and emit per_compartment outputs.
    """
    fm_path = Path(failure_matrix_path)
    tv_path = Path(time_vector_path)
    nodes_path = Path(nodes_order_path)

    result: Dict[str, Any] = {
        "failure_matrix": str(fm_path),
        "time_vector": str(tv_path),
        "nodes_order": str(nodes_path),
        "output_dir": str(output_dir),
    }
    missing = [p.name for p in (fm_path, tv_path, nodes_path) if not p.exists()]
    if missing:
        result["status"] = "error"
        result["reason"] = f"missing_inputs: {missing}"
        return result

    try:
        fm = np.load(fm_path, allow_pickle=False)
        tv = np.load(tv_path, allow_pickle=False)
        with nodes_path.open() as f:
            nodes = [line.strip() for line in f if line.strip() != ""]
    except Exception as exc:  # pragma: no cover - defensive
        result["status"] = "error"
        result["reason"] = f"load_failed: {exc}"
        return result

    derived = derive_per_compartment_from_arrays(
        failure_matrix=fm,
        time_vector=tv,
        nodes_order=nodes,
        output_dir=Path(output_dir),
        binary_threshold=binary_threshold,
        allow_mismatch=allow_mismatch,
        skip_existing=skip_existing,
        provenance=provenance,
    )
    return {**result, **derived}


__all__ = [
    "STATE_SCHEMA_VERSION",
    "build_node_index_payload",
    "compute_topology_metrics",
    "write_per_compartment_outputs",
    "derive_per_compartment_from_arrays",
    "derive_per_compartment_from_files",
]
