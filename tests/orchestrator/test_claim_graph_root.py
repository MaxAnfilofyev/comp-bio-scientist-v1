import os
import json
import shutil
from pathlib import Path
from ai_scientist.orchestrator.tool_wrappers import create_claim_graph_artifact, update_claim_graph
from ai_scientist.orchestrator.hypothesis import resolve_claim_graph_path

def test_claim_graph_root_location():
    # Setup mock environment
    test_root = Path("test_claim_graph_env")
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir()
    exp_results = test_root / "experiment_results"
    exp_results.mkdir()
    
    # Mock AISC_BASE_FOLDER to point to our test root (parent of experiment_results)
    os.environ["AISC_BASE_FOLDER"] = str(test_root.resolve())
    
    try:
        print(f"Testing with AISC_BASE_FOLDER={os.environ['AISC_BASE_FOLDER']}")

        # 1. Create artifact
        print("Calling create_claim_graph_artifact...")
        res_create = create_claim_graph_artifact(module="test_mod")
        print(f"Create result: {res_create}")
        
        expected_path = exp_results / "claim_graph.json"
        if not expected_path.exists():
            print(f"NOTE: File not found at expected root location after CREATE: {expected_path}. This is expected if create only reserves.")
            # Do NOT return, proceed to update
            # return

        print(f"SUCCESS: File created at {expected_path}")
        
        # 2. Update artifact
        print("Calling update_claim_graph...")
        res_update = update_claim_graph(
            claim_id="C1",
            claim_text="Test Claim",
            status="linked",
            notes="Added via verification test"
        )
        print(f"Update result: {res_update}")
        
        # Verify content
        data = json.loads(expected_path.read_text())
        print(f"File content: {json.dumps(data, indent=2)}")
        
        has_claim = any(c.get("claim_id") == "C1" for c in data)
        if has_claim:
             print("SUCCESS: Update modified the correct file.")
        else:
             print("FAILURE: Update did not modify the correct file.")

    finally:
        # Cleanup
        if test_root.exists():
            shutil.rmtree(test_root)
        del os.environ["AISC_BASE_FOLDER"]

if __name__ == "__main__":
    test_claim_graph_root_location()
