import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch

from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact
from ai_scientist.orchestrator.lifecycle import refresh_project_spine, deprecate_stale_drafts
from ai_scientist.orchestrator.snapshots import generate_project_snapshot

@pytest.fixture
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    with patch.dict(os.environ, {"AISC_BASE_FOLDER": temp_dir, "AISC_ACTIVE_ROLE": "Tester"}), \
         patch("ai_scientist.tools.base_tool.BaseTool.resolve_output_dir", return_value=os.path.join(temp_dir, "experiment_results")):
        
        os.makedirs(os.path.join(temp_dir, "experiment_results"), exist_ok=True)
        yield temp_dir
        
    shutil.rmtree(temp_dir)

def test_spine_and_snapshot(temp_workspace):
    # Setup artifacts
    # 1. Lit Summary (canonical)
    res_lit = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"content": "Lit Content", "module": "lit"}),
        status="canonical"
    )
    with open(res_lit["reserved_path"], "w") as f:
        f.write("Lit Content")
    
    # 2. Integration Memo (canonical)
    res_memo = reserve_and_register_artifact(
        kind="integration_memo_md",
        meta_json=json.dumps({"content": "Memo content", "module": "modeling"}),
        status="canonical"
    )
    assert not res_memo.get("error"), f"Memo reservation failed: {res_memo}"
    with open(res_memo["reserved_path"], "w") as f:
        f.write("Memo content")
    
    # 3. Non-canonical draft (should be ignored by spine)
    res_draft = reserve_and_register_artifact(
        kind="integration_memo_md",
        meta_json=json.dumps({"content": "Draft memo", "module": "modeling_draft"}),
        status="draft"
    )
    with open(res_draft["reserved_path"], "w") as f:
        f.write("Draft memo")
    
    
    import time
    time.sleep(1.0)
    
    # Refresh Spine
    spine = refresh_project_spine()
    assert spine["lit_summary"] is not None
    assert len(spine["memos"]) == 1
    
    # Generate Snapshot
    snapshot_path = generate_project_snapshot()
    assert os.path.exists(snapshot_path)
    
    with open(snapshot_path, "r") as f:
        content = f.read()
        
    assert "Literature Summary" in content
    assert "Memo content" in content
    assert "Draft memo" not in content

def test_deprecation(temp_workspace):
    from datetime import datetime, timedelta
    
    # Create old draft
    old_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()
    
    res = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"content": {"text": "Old draft"}, "module": "lit"}),
        status="draft"
    )
    
    # Manually backdate manifest entry
    exp_dir = os.path.join(temp_workspace, "experiment_results")
    from ai_scientist.utils.manifest import load_entries, append_or_update
    entries = load_entries(base_folder=exp_dir)
    entry = entries[0]
    entry["created_at"] = old_time
    append_or_update(entry, base_folder=exp_dir)
    
    # Run deprecation
    deprecated = deprecate_stale_drafts(age_hours=24)
    assert len(deprecated) == 1
    # Check against reserved path (which is absolute or relative depending on return value)
    # Manifest paths are relative to exp_dir usually? No, existing impl uses absolute.
    # Let's check string continuity.
    target_path = res.get("reserved_path")
    assert target_path
    assert deprecated[0] == target_path
    
    # Verify status in manifest
    entries = load_entries(base_folder=exp_dir)
    latest = entries[-1]
    assert latest["status"] == "deprecated"
    
