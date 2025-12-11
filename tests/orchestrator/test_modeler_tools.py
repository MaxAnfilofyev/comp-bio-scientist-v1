import sys
from unittest.mock import MagicMock, patch
import os

# Mock 'agents' module before importing orchestrator agents
mock_agents = MagicMock()
# Make function_tool a pass-through decorator so we can see the original function names
def identity_decorator(*args, **kwargs):
    # If used as @function_tool(arg=val), it returns a decorator
    if len(args) == 0 and len(kwargs) > 0:
        def real_decorator(f):
             return f
        return real_decorator
    # If used as @function_tool on a function
    if len(args) == 1 and callable(args[0]):
        return args[0]
    # If used with strict_mode=False as positional? Unlikely.
    # Fallback
    return args[0] if args else None

mock_agents.function_tool = identity_decorator

sys.modules["agents"] = mock_agents
sys.modules["agents.types"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
# Mock high-level modules that import heavy dependencies
sys.modules["ai_scientist.perform_writeup"] = MagicMock()
sys.modules["ai_scientist.perform_biological_interpretation"] = MagicMock()
sys.modules["ai_scientist.perform_vlm_review"] = MagicMock()
sys.modules["ai_scientist.perform_llm_review"] = MagicMock()

# Now we can safely import
from ai_scientist.orchestrator.agents import build_team  # noqa: E402

def test_modeler_tools_configuration():
    # Mock idea and dirs
    idea = {
        "Title": "Test Project",
        "Abstract": "Test Abstract",
        "Short Hypothesis": "Test Hypothesis",
        "Experiments": ["Exp 1"],
        "Risk Factors and Limitations": []
    }
    dirs = {
        "base": "/tmp/test",
        "results": "/tmp/test/results"
    }

    # Patch environment to avoid path issues if necessary
    with patch.dict(os.environ, {"AISC_BASE_FOLDER": "/tmp/test"}):
        
        # We intercept _make_agent to inspect arguments passed to Modeler constructor
        with patch("ai_scientist.orchestrator.agents._make_agent") as mock_make:
            
            build_team("test-model", idea, dirs)
            
            # Find the call for Modeler
            modeler_call = None
            for call in mock_make.call_args_list:
                # Check kwargs for name="Modeler"
                if "name" in call.kwargs and call.kwargs["name"] == "Modeler":
                    modeler_call = call
                # Check args if positional
                elif len(call.args) > 0 and call.args[0] == "Modeler":
                    modeler_call = call
            
            assert modeler_call, "Modeler was not created via _make_agent"
            
            # Extract tools list
            tools_list = modeler_call.kwargs.get("tools")
            if not tools_list and len(modeler_call.args) > 2:
                tools_list = modeler_call.args[2]
            
            assert tools_list, "No tools passed to Modeler"
            
            # Extract tool names. Depends on how tools are passed (functions or objects)
            # The tool wrappers are usually decorated functions, so __name__ works.
            # If they are bound methods or objects, we check appropriately.
            tool_names = []
            for t in tools_list:
                if hasattr(t, "tool_name"): # BaseTool or configured tool
                    tool_names.append(t.tool_name)
                elif hasattr(t, "__name__"):
                    tool_names.append(t.__name__)
                else:
                    tool_names.append(str(t))

            print(f"Found Modeler tools: {tool_names}")

            # 1. Check for Required Wrappers
            required = [
                "create_transport_artifact",
                "create_sensitivity_table_artifact",
                "create_intervention_table_artifact",
                "create_verification_note_artifact",
                "create_model_spec_artifact",
                "list_model_specs",
                "get_latest_model_spec",
                "list_experiment_results",
                "get_latest_metrics",
                "read_model_spec",
                "read_experiment_config",
                "read_metrics",
                "read_artifact", 
            ]
            for req in required:
                assert req in tool_names, f"Modeler missing required tool: {req}"
            
            # 2. Check for Forbidden Tools
            forbidden = [
                "get_run_paths",
                "resolve_path",
                "get_artifact_index",
                "list_artifacts", 
                "list_artifacts_by_kind",
                "reserve_typed_artifact",
                "reserve_and_register_artifact",
                "append_manifest",
                "read_manifest",
                "read_manifest_entry",
                "check_manifest",
                "check_manifest_unique_paths",
                "mirror_artifacts",
                "summarize_artifact"
            ]
            for forb in forbidden:
                assert forb not in tool_names, f"Modeler has forbidden tool: {forb}"
                
            print("Modeler tools verification passed!")
