import json
from pathlib import Path

from ai_scientist.lab_tools.check_run_health import check_run_health


def _write_manifest(base: Path, entries: list[dict]):
    manifest_dir = base / "experiment_results" / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shard_path = manifest_dir / "manifest_shard_0001.ndjson"
    with shard_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry))
            f.write("\n")
    index = {
        "schema_version": 2,
        "shard_size": 10000,
        "shards": [
            {
                "path": str(shard_path),
                "count": len(entries),
                "ts_min": entries[0].get("created_at", "") if entries else "",
                "ts_max": entries[-1].get("created_at", "") if entries else "",
                "status": "ok",
            }
        ],
    }
    (manifest_dir / "manifest_index.json").write_text(json.dumps(index))


def test_health_passes_when_manifest_covers_artifact(tmp_path: Path):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    artifact = exp_dir / "lit_summary.json"
    artifact.write_text("ok")
    entry = {
        "path": str(artifact),
        "name": "lit_summary.json",
        "kind": "lit_summary_main",
        "status": "ok",
        "created_at": "2024-01-01T00:00:00",
    }
    _write_manifest(tmp_path, [entry])

    ok, errors = check_run_health(tmp_path)
    assert ok, f"Expected health check to pass, got errors: {errors}"


def test_health_fails_on_uncovered_file(tmp_path: Path):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    artifact = exp_dir / "lit_summary.json"
    artifact.write_text("ok")
    _write_manifest(tmp_path, [])  # no entries, so file is uncovered

    ok, errors = check_run_health(tmp_path)
    assert not ok
    assert any("lit_summary.json" in err for err in errors)
