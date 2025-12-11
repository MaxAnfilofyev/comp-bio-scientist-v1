import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import patch

from ai_scientist.orchestrator.context_specs import record_context_access

@pytest.fixture
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    with patch.dict(os.environ, {"AISC_BASE_FOLDER": temp_dir, "AISC_ACTIVE_ROLE": "Tester"}), \
         patch("ai_scientist.tools.base_tool.BaseTool.resolve_output_dir", return_value=os.path.join(temp_dir, "experiment_results")):
        
        os.makedirs(os.path.join(temp_dir, "experiment_results"), exist_ok=True)
        yield temp_dir
        
    shutil.rmtree(temp_dir)

def test_record_context_access_creates_status_json(temp_workspace):
    exp_dir = os.path.join(temp_workspace, "experiment_results")
    
    # Record a read access
    record_context_access("Modeler", "kind_A", "path/to/A", "read", artifact_id="id_A", version="v1")
    
    status_file = os.path.join(exp_dir, "_status", "status_Modeler.json")
    assert os.path.exists(status_file)
    
    with open(status_file, "r") as f:
        data = json.load(f)
        
    assert data["role"] == "Modeler"
    assert "path/to/A" in data["read"]
    assert data["read"]["path/to/A"]["version"] == "v1"
    
    # Record a write access
    record_context_access("Modeler", "kind_B", "path/to/B", "write", artifact_id="id_B", version="v2")
    
    with open(status_file, "r") as f:
        data = json.load(f)
        
    assert "path/to/B" in data["written"]
    assert data["written"]["path/to/B"]["version"] == "v2"
    
    # Verify timestamp updated
    assert data["updated_at"]
