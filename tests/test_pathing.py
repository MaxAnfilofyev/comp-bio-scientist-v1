from pathlib import Path

import pytest

from ai_scientist.utils import pathing


def test_resolve_output_path_basic(tmp_path, monkeypatch):
    base = tmp_path / "experiment_results"
    base.mkdir()
    monkeypatch.setenv("AISC_EXP_RESULTS", str(base))
    target, quarantined, note = pathing.resolve_output_path(subdir="simulations", name="run.txt")
    assert target.parent == base / "simulations"
    assert target.name == "run.txt"
    assert quarantined is False
    assert note is None


def test_resolve_output_path_unique(tmp_path, monkeypatch):
    base = tmp_path / "experiment_results"
    sim_dir = base / "simulations"
    sim_dir.mkdir(parents=True)
    existing = sim_dir / "run.txt"
    existing.write_text("seed")
    monkeypatch.setenv("AISC_EXP_RESULTS", str(base))
    target, _, _ = pathing.resolve_output_path(subdir="simulations", name="run.txt")
    assert target != existing
    assert target.parent == sim_dir
    assert target.exists() is False  # not created until write


def test_resolve_output_path_reject_traversal(tmp_path, monkeypatch):
    base = tmp_path / "experiment_results"
    base.mkdir()
    monkeypatch.setenv("AISC_EXP_RESULTS", str(base))
    with pytest.raises(ValueError):
        pathing.resolve_output_path(subdir="simulations/../evil", name="run.txt")


def test_resolve_output_path_quarantine_on_failure(tmp_path, monkeypatch):
    base = tmp_path / "experiment_results"
    base.mkdir()
    monkeypatch.setenv("AISC_EXP_RESULTS", str(base))
    orig_mkdir = pathing.Path.mkdir

    def flaky_mkdir(self, *args, **kwargs):  # type: ignore[override]
        if "_unrouted" not in str(self):
            raise PermissionError("denied")
        return orig_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathing.Path, "mkdir", flaky_mkdir)
    target, quarantined, note = pathing.resolve_output_path(subdir="simulations", name="run.txt")
    assert quarantined is True
    assert "_unrouted" in str(target)
    assert target.parent.exists()
    assert note is not None
