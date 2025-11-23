"""Validation utilities for standardized per-compartment simulation outputs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_scientist.tools.base_tool import BaseTool


def _sha256_join(ordering: List[str]) -> str:
    joined = "|".join(ordering)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def validate_per_compartment_outputs(sim_dir: str | Path) -> Dict[str, Any]:
    """
    Validate standardized per-compartment outputs under a simulation folder.
    Expects per_compartment.npz (binary_states, continuous_states, time), node_index_map.json, topology_summary.json.
    Returns shapes/status and any detected errors.
    """
    sim_path = BaseTool.resolve_input_path(str(sim_dir), allow_dir=True)
    npz_path = sim_path / "per_compartment.npz"
    topo_path = sim_path / "topology_summary.json"
    map_path = sim_path / "node_index_map.json"

    info: Dict[str, Any] = {"sim_dir": str(sim_path)}
    errors: List[str] = []
    warnings: List[str] = []

    npz_shapes: Dict[str, List[int]] = {}
    ordering_checksum: Optional[str] = None
    topo_checksum: Optional[str] = None
    topo_binary_shape: Optional[List[int]] = None
    topo_continuous_shape: Optional[List[int]] = None
    schema_version: Optional[str] = None

    if not npz_path.exists():
        errors.append("per_compartment.npz missing")
    else:
        try:
            import numpy as np  # type: ignore

            archive = np.load(npz_path, allow_pickle=False)
            required = ["binary_states", "continuous_states", "time"]
            missing = [k for k in required if k not in archive.files]
            if missing:
                errors.append(f"npz missing keys: {missing}")
            else:
                npz_shapes["binary_shape"] = list(np.shape(archive["binary_states"]))
                npz_shapes["continuous_shape"] = list(np.shape(archive["continuous_states"]))
                npz_shapes["time_shape"] = list(np.shape(archive["time"]))

                if len(npz_shapes["binary_shape"]) != 2:
                    errors.append(f"binary_states expected 2D [time, nodes], got {npz_shapes['binary_shape']}")
                if len(npz_shapes["continuous_shape"]) not in (2, 3):
                    errors.append(
                        f"continuous_states expected 2D/3D [time, nodes(, vars)], got {npz_shapes['continuous_shape']}"
                    )
                if npz_shapes.get("time_shape") and len(npz_shapes["time_shape"]) != 1:
                    errors.append(f"time expected 1D, got {npz_shapes['time_shape']}")

                if (
                    npz_shapes.get("binary_shape")
                    and npz_shapes.get("time_shape")
                    and npz_shapes["binary_shape"][0] != npz_shapes["time_shape"][0]
                ):
                    errors.append(
                        f"time length {npz_shapes['time_shape'][0]} != binary_states time {npz_shapes['binary_shape'][0]}"
                    )
                if (
                    npz_shapes.get("continuous_shape")
                    and npz_shapes.get("binary_shape")
                    and npz_shapes["continuous_shape"][0] != npz_shapes["binary_shape"][0]
                ):
                    errors.append(
                        f"continuous_states time {npz_shapes['continuous_shape'][0]} != binary_states time {npz_shapes['binary_shape'][0]}"
                    )
                if npz_shapes.get("binary_shape") and npz_shapes["binary_shape"][0] <= 0:
                    warnings.append("binary_states has zero time steps")

                info.update(npz_shapes)
        except Exception as exc:  # pragma: no cover - best effort guard
            errors.append(f"failed to read npz: {exc}")

    if map_path.exists():
        try:
            with map_path.open() as f:
                mapping = json.load(f)
            if isinstance(mapping, dict):
                schema_version = mapping.get("state_schema_version")
                ordering = mapping.get("ordering") or []
                ordering_checksum = mapping.get("ordering_checksum")
                info["node_index_count"] = len(mapping.get("node_index_map", []))
                if not ordering_checksum and ordering:
                    ordering_checksum = _sha256_join(ordering)
                    warnings.append("ordering_checksum missing; computed locally")
            else:
                errors.append("node_index_map.json has unexpected structure (expected dict)")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"failed to read node_index_map.json: {exc}")
    else:
        errors.append("node_index_map.json missing")

    if topo_path.exists():
        try:
            with topo_path.open() as f:
                topo = json.load(f)
            schema_version = schema_version or topo.get("state_schema_version")
            topo_checksum = topo.get("ordering_checksum")
            topo_binary_shape = topo.get("binary_shape")
            topo_continuous_shape = topo.get("continuous_shape")
            info["topology_status"] = topo.get("status")
            info["topology_metrics"] = {
                k: topo.get(k) for k in ["N", "leaf_count", "mean_depth", "max_depth", "total_path_length"]
            }
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"failed to read topology_summary.json: {exc}")
    else:
        errors.append("topology_summary.json missing")

    if schema_version and schema_version != "1.0":
        warnings.append(f"unexpected state_schema_version {schema_version}")

    if topo_binary_shape and npz_shapes.get("binary_shape") and topo_binary_shape != npz_shapes["binary_shape"]:
        errors.append(f"topology_summary binary_shape {topo_binary_shape} != npz {npz_shapes['binary_shape']}")
    if (
        topo_continuous_shape
        and npz_shapes.get("continuous_shape")
        and topo_continuous_shape != npz_shapes["continuous_shape"]
    ):
        errors.append(
            f"topology_summary continuous_shape {topo_continuous_shape} != npz {npz_shapes['continuous_shape']}"
        )

    if ordering_checksum and topo_checksum and ordering_checksum != topo_checksum:
        errors.append("ordering_checksum mismatch between node_index_map.json and topology_summary.json")

    if errors:
        info["errors"] = errors
    if warnings:
        info["warnings"] = warnings
    if not errors:
        info["status"] = "ok"
    return info


__all__ = ["validate_per_compartment_outputs"]
