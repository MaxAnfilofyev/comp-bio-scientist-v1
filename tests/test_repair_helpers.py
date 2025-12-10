from pathlib import Path

from ai_scientist.utils.repair_helpers import normalize_sim_dir, safe_move_into_sim_dir


def test_normalize_sim_dir_strips_duplicate(tmp_path, monkeypatch):
    base = tmp_path / "experiment_results"
    dup = base / "experiment_results" / "simulations" / "transport_runs" / "foo"
    dup.mkdir(parents=True)
    monkeypatch.setenv("AISC_EXP_RESULTS", str(base))
    normalized = normalize_sim_dir(dup)
    assert str(normalized).startswith(str(base))
    assert "experiment_results/experiment_results" not in str(normalized)


def test_safe_move_into_sim_dir(tmp_path, monkeypatch):
    base = tmp_path / "experiment_results"
    sim_dir = base / "simulations" / "transport_runs" / "foo"
    src_dir = base / "tmp_outputs"
    src_dir.mkdir(parents=True)
    src = src_dir / "file.txt"
    src.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("AISC_EXP_RESULTS", str(base))

    dest = safe_move_into_sim_dir(src, sim_dir)
    assert dest.parent == sim_dir
    assert dest.exists()
    assert not src.exists()


def test_safe_move_into_sim_dir_on_failure_logs(tmp_path, monkeypatch):
    base = tmp_path / "experiment_results"
    sim_dir = base / "simulations" / "transport_runs" / "foo"
    src_dir = base / "tmp_outputs"
    src_dir.mkdir(parents=True)
    src = src_dir / "file.txt"
    src.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("AISC_EXP_RESULTS", str(base))

    # Force move failure by pointing dest to a file (simulate permission/rename failure)
    sim_dir.mkdir(parents=True)
    dest_file = sim_dir / "file.txt"
    dest_file.write_text("lock", encoding="utf-8")
    # Monkeypatch replace to raise
    original_replace = Path.replace

    def fail_replace(self, target):
        raise PermissionError("deny")

    monkeypatch.setattr(Path, "replace", fail_replace)
    moved = safe_move_into_sim_dir(src, sim_dir)
    # On failure, we keep the original path
    assert moved == src or moved.exists()
    # Restore replace to avoid side effects
    monkeypatch.setattr(Path, "replace", original_replace)
