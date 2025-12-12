
import json
import os
import sys

# Mock environment setup if needed
os.environ["AISC_ACTIVE_ROLE"] = "Modeler"
os.environ["AISC_BASE_FOLDER"] = "."

# Mocking missing modules
from types import ModuleType
for mod_name in ["pandas", "agents", "agents.types", "matplotlib", "matplotlib.pyplot", "matplotlib.figure", "seaborn"]:
    m = ModuleType(mod_name)
    sys.modules[mod_name] = m
    if mod_name == "agents":
        m.Agent = type("Agent", (), {})
        m.function_tool = lambda f=None, **kwargs: f if f else lambda x: x
        m.FunctionTool = type("FunctionTool", (), {})
    if mod_name == "agents.types":
        m.RunResult = dict
    if mod_name == "pandas":
        m.DataFrame = type("DataFrame", (), {})
    if mod_name == "matplotlib":
        m.use = lambda *args, **kwargs: None

    if mod_name == "matplotlib.figure":
        m.Figure = type("Figure", (), {})



# Import the relevant functions
try:
    from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact
    from ai_scientist.orchestrator.tool_wrappers import save_model_spec

except ImportError:
    # Adjust path if needed
    sys.path.append(os.getcwd())
    from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact
    from ai_scientist.orchestrator.tool_wrappers import save_model_spec
    try:
        from ai_scientist.orchestrator.tool_wrappers import create_model_spec_artifact
    except ImportError:
        create_model_spec_artifact = None


def test_save_model_spec():
    model_key = "topo_tipping_v1_E3_inline"
    content_json = json.dumps({
        "model": "topo_tipping_v1",
        "params": {"k1": 1.0, "k2": 0.5},
        "A_range": [0.2, 1.2],
        "C_range": [0.1, 1.0],
        "tspan": [0, 100],
        "ic_low": [0.9, 0.9],
        "ic_high": [1.1, 1.1],
        "tolerance": 1e-6,
        "noise_sigma": 0.01
    })
    readme = "Test readme"

    print(f"Testing save_model_spec with key: {model_key}")
    
    # Direct call to tool wrapper
    result = save_model_spec(model_key, content_json, readme)
    print("Result:", result)
    
    if "error" in result:
        print("FAIL: Error returned")
    elif "reserved_path" in result: # Wait, save_model_spec returns reserve result? 
        # No, save_model_spec calls write_text_artifact which returns... wait.
        # Line 2890: write_res_json = write_text_artifact(...)
        # It returns write_res_json.
        print("SUCCESS? Result:", result)
    else:
        # save_model_spec returns the result of write_text_artifact usually
        # but if it fails at reserve, it returns error.
        pass

def debug_reserve_direct():
    print("\n--- Direct Reserve Debug ---")
    model_key = "topo_tipping_v1_E3_inline"
    meta_json = {
        "model_key": model_key,
        "version": "v1", 
        "module": "modeling", 
        "summary": f"Model specification for {model_key}"
    }
    
    res = reserve_and_register_artifact(
        kind="model_spec",
        meta_json=json.dumps(meta_json),
        unique=True
    )
    print("Reserve Result:", res)


def test_create_model_spec_artifact():
    if not create_model_spec_artifact:
        print("\n--- create_model_spec_artifact NOT FOUND ---")
        return
        
    print("\n--- create_model_spec_artifact Debug ---")
    model_key = "topo_tipping_v1_E3_inline"
    content_json = json.dumps({
        "model": "topo_tipping_v1",
        "params": {}
    })
    
    res = create_model_spec_artifact(model_key=model_key, content_json=content_json)
    print("Create Result:", res)

if __name__ == "__main__":
    try:
        debug_reserve_direct()
        test_save_model_spec()
        test_create_model_spec_artifact()

    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
