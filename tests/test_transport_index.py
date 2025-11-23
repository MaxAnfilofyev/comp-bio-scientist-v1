import json
from pathlib import Path

import pytest

from ai_scientist.utils.transport_index import index_transport_runs, resolve_transport_sim


def _seed_paths(root: Path, baseline: str, transport: float, seed: int) -> Path:
    seed_dir = root / "simulations" / "transport_runs" / baseline / f"transport_{transport}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / f"{baseline}_sim.json").write_text("{}", encoding="utf-8")
    (seed_dir / f"{baseline}_sim_failure_matrix.npy").write_bytes(b"\x93NUMPY")
    (seed_dir / f"{baseline}_sim_time_vector.npy").write_bytes(b"\x93NUMPY")
    (seed_dir / f"nodes_order_{baseline}_sim.txt").write_text("0\n1\n", encoding="utf-8")
    return seed_dir


def test_index_and_resolve(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment_results"
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_root))
    seed_dir = _seed_paths(exp_root, "baseA", 0.1, 3)

    idx = index_transport_runs()
    assert "entries" in idx
    key = "baseA|0.1|3|"
    assert key in idx["entries"]
    entry = idx["entries"][key]
    assert entry["paths"]["sim_json"].endswith("baseA_sim.json")

    resolved = resolve_transport_sim("baseA", 0.1, 3, refresh=False)
    assert "entry" in resolved
    assert resolved["entry"]["paths"]["failure_matrix"].endswith("_failure_matrix.npy")
    # ensure index_path returned exists
    assert Path(resolved["index_path"]).exists()
    # sanity: the files are under the seed_dir
    assert seed_dir.exists()


def test_resolve_missing_triggers_health(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment_results"
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_root))
    exp_root.mkdir(parents=True, exist_ok=True)

    result = resolve_transport_sim("missing", 0.2, 1, refresh=True)
    assert "error" in result
    assert "index_path" in result
    # index should still be created even if empty
    assert Path(result["index_path"]).exists()
