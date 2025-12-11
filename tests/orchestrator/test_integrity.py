import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch

from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact
from ai_scientist.orchestrator.integrity import check_dependency_staleness

@pytest.fixture
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    with patch.dict(os.environ, {"AISC_BASE_FOLDER": temp_dir, "AISC_ACTIVE_ROLE": "Tester"}), \
         patch("ai_scientist.tools.base_tool.BaseTool.resolve_output_dir", return_value=os.path.join(temp_dir, "experiment_results")):
        
        os.makedirs(os.path.join(temp_dir, "experiment_results"), exist_ok=True)
        yield temp_dir
        
    shutil.rmtree(temp_dir)

def test_dependency_staleness(temp_workspace):
    # Create an input artifact (v1)
    dep_v1 = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"content": {"text": "v1"}, "module": "lit"}),
        status="canonical"
    )
    
    # Create dependent artifact using v1
    # Note: 'tools_used' usually populated by agents, we mock it via metadata
    dependent = reserve_and_register_artifact(
        kind="lit_summary_csv",
        meta_json=json.dumps({
            "content": {"text": "based on v1"}, 
            "module": "lit",
            "tools_used": [
                {"name": dep_v1["name"], "version": "v1"}
            ]
        }),
        status="draft"
    )
    
    # Check staleness - should be fresh (v1 == v1)
    assert not dependent.get("error"), f"Reservation failed: {dependent}"
    assert dependent.get("metadata", {}).get("tools_used"), "Tools used missing in metadata"
    
    fresh_check = check_dependency_staleness(dependent["name"])
    assert not fresh_check["stale"]
    assert len(fresh_check["stale_dependencies"]) == 0
    
    # Create new version of dependency (v2)
    # File must exist for uniqueness
    with open(dep_v1["reserved_path"], "w") as f:
        f.write("v1")
    
    import time
    time.sleep(1.1)
    
    dep_v2 = reserve_and_register_artifact(
        kind="lit_summary_main",
        meta_json=json.dumps({"content": {"text": "v2"}, "module": "lit"}),
        status="canonical"
    )
    assert not dep_v2.get("error")
    assert dep_v2["metadata"]["version"] == "v2"
    
    # Check staleness again - should be stale (v1 != v2)
    stale_check = check_dependency_staleness(dependent["name"])
    assert stale_check["stale"]
    assert len(stale_check["stale_dependencies"]) == 1
    # Relax assertion to debug or fix if v3 is valid (it shouldn't be)
    current_ver = stale_check["stale_dependencies"][0]["current_version"]
    assert current_ver == "v2", f"Expected v2, got {current_ver}"
    assert stale_check["stale_dependencies"][0]["used_version"] == "v1"
