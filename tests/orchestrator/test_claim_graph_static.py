import ast
from pathlib import Path

def verify_artifacts_change():
    print("Verifying artifacts.py...")
    p = Path("ai_scientist/orchestrator/artifacts.py")
    if not p.exists():
        print("FAIL: artifacts.py not found")
        return False
    
    tree = ast.parse(p.read_text())
    
    # Find ARTIFACT_TYPE_REGISTRY
    registry = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ARTIFACT_TYPE_REGISTRY":
                    registry = node.value
                    break
    
    if not registry:
        print("FAIL: ARTIFACT_TYPE_REGISTRY not found")
        return False

    # Find 'claim_graph_main' entry
    claim_graph_entry = None
    if isinstance(registry, ast.Dict):
        for i, key in enumerate(registry.keys):
            if isinstance(key, ast.Constant) and key.value == "claim_graph_main":
                claim_graph_entry = registry.values[i]
                break
    
    if not claim_graph_entry:
        print("FAIL: claim_graph_main key not found in registry")
        return False
        
    # Check 'rel_dir' value
    rel_dir_val = None
    if isinstance(claim_graph_entry, ast.Dict):
        for i, key in enumerate(claim_graph_entry.keys):
             if isinstance(key, ast.Constant) and key.value == "rel_dir":
                 val_node = claim_graph_entry.values[i]
                 if isinstance(val_node, ast.Constant):
                     rel_dir_val = val_node.value
                 break
    
    print(f"Found rel_dir for claim_graph_main: '{rel_dir_val}'")
    
    if rel_dir_val == ".":
        print("SUCCESS: rel_dir is correctly set to '.'")
        return True
    else:
        print(f"FAIL: rel_dir is '{rel_dir_val}', expected '.'")
        return False

def verify_hypothesis_change():
    print("\nVerifying hypothesis.py...")
    p = Path("ai_scientist/orchestrator/hypothesis.py")
    if not p.exists():
         print("FAIL: hypothesis.py not found")
         return False
         
    content = p.read_text()
    
    # Simple text check for the logic we added
    # We expect: if exp_dir.name == "experiment_results":
    
    required_snippet = 'if exp_dir.name == "experiment_results":'
    if required_snippet in content:
        print(f"SUCCESS: Found snippet '{required_snippet}'")
    else:
        print(f"FAIL: Did not find snippet '{required_snippet}'")
        return False
        
    # We expect it to return ... / "claim_graph.json"
    if 'return exp_dir / "claim_graph.json"' in content:
         print("SUCCESS: Found return statement using exp_dir")
         return True
    else:
         print("FAIL: Did not find expected return statement")
         return False

if __name__ == "__main__":
    ok1 = verify_artifacts_change()
    ok2 = verify_hypothesis_change()
    
    if ok1 and ok2:
        print("\nOVERALL SUCCESS: Static verification passed.")
        exit(0)
    else:
        print("\nOVERALL FAILURE: Static verification failed.")
        exit(1)
