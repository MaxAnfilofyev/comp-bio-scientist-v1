import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch

from ai_scientist.orchestrator.artifacts import (
    reserve_and_register_artifact,
    promote_artifact_to_canonical,
    _get_latest_artifact_entry
)
from ai_scientist.utils.manifest import load_entries

@pytest.fixture
def temp_workspace():
    # Create a temp dir
    temp_dir = tempfile.mkdtemp()
    
    # Mock AISC_BASE_FOLDER and BaseTool.resolve_output_dir
    with patch.dict(os.environ, {"AISC_BASE_FOLDER": temp_dir, "AISC_ACTIVE_ROLE": "Tester"}), \
         patch("ai_scientist.tools.base_tool.BaseTool.resolve_output_dir", return_value=os.path.join(temp_dir, "experiment_results")):
        
        # Create experiment_results
        os.makedirs(os.path.join(temp_dir, "experiment_results"), exist_ok=True)
        yield temp_dir
        
    shutil.rmtree(temp_dir)

def test_versioning_flow(temp_workspace):
    exp_dir = os.path.join(temp_workspace, "experiment_results")
    
    # 1. Create first version
    res1 = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"content": {"text": "v1 content"}, "module": "lit"}),
        status="draft",
        change_summary="Initial draft"
    )
    assert not res1.get("error"), f"Res1 error: {res1.get('error')}"
    assert res1["metadata"]["version"] == "v1"
    # Ensure nested metadata contains change_summary
    assert res1["metadata"]["metadata"]["change_summary"] == "Initial draft"
    
    # Simulate file creation so unique=True works for next call
    try:
        with open(res1["reserved_path"] if "reserved_path" in res1 else res1["path"], "w") as f:
            f.write("v1")
    except Exception:
        pass # Handle if path key varies

    # Verify manifest entry 1
    entries = load_entries(base_folder=exp_dir)
    assert len(entries) == 1
    # Manifest v2 schema stores extra fields in metadata
    assert entries[0]["metadata"]["version"] == "v1"
    
    # 2. Create second version (auto-increment)
    res2 = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"content": {"text": "v2 content"}, "module": "lit"}),
        status="draft",
        change_summary="Updated content"
    )
    assert not res2.get("error")
    assert res2["metadata"]["version"] == "v2"
    assert res2["metadata"]["parent_version"] == "v1"
    assert res2["metadata"]["metadata"]["change_summary"] == "Updated content"
    
    # Verify manifest entry 2
    entries = load_entries(base_folder=exp_dir)
    assert len(entries) == 2
    # Ensure getting latest works
    latest = _get_latest_artifact_entry("lit_summary_main", "lit")
    assert latest["metadata"]["version"] == "v2"

def test_promotion_flow(temp_workspace):
    exp_dir = os.path.join(temp_workspace, "experiment_results")
    
    # Create an artifact
    res = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"content": {"text": "final content"}, "module": "lit"}),
        status="draft"
    )
    name = res["name"]
    
    # Promote it
    promo_res = promote_artifact_to_canonical(
        name=name,
        kind="lit_summary_main",
        notes="Approved by PI"
    )
    assert not promo_res.get("error")
    assert promo_res["status"] == "promoted"
    
    # Verify manifest has new entry with canonical status
    entries = load_entries(base_folder=exp_dir)
    # reserve_and_register writes one entry. promote updates it in place (as path is same).
    assert len(entries) == 1 
    latest = entries[-1]
    assert latest["status"] == "canonical"
    assert latest["name"] == name
    assert "promotions" in latest["metadata"]
    assert latest["metadata"]["promotions"][0]["notes"] == "Approved by PI"

def test_manual_version_override(temp_workspace):
    # If user provides version, it should be respected
    res = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"version": "v10", "content": {}, "module": "lit"}),
    )
    assert res["metadata"]["version"] == "v10"
