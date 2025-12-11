import json
import os
import shutil
import tempfile
import pytest
import sys
from unittest.mock import patch, MagicMock

# --- MOCK AGENTS MODULE BEFORE IMPORTS ---
mock_agents = MagicMock()

# Mock function_tool decorator
def mock_function_tool(func=None, **kwargs):
    if func and callable(func):
        return func
    def wrapper(f):
        return f
    return wrapper

mock_agents.function_tool = mock_function_tool
mock_agents.Agent = MagicMock
mock_agents.ModelSettings = MagicMock

# Inject mocks
sys.modules["agents"] = mock_agents
sys.modules["agents.types"] = MagicMock()

# Mock heavy/missing dependencies
def mock_modules(modules):
    for mod in modules:
        sys.modules[mod] = MagicMock()

base_mocks = [
    "anthropic", "openai", "backoff", "pymupdf", "fitz", "pymupdf4llm", 
    "pypdf", "seaborn", "transformers", "datasets", "wandb", "rich", 
    "igraph", "boto3", "plotly", "coolname", "funcy", "tiktoken",
    "humanize", "dataclasses_json", "genson", "shutup", "jsonschema",
    "botocore", "dotenv", "black", "requests"
]
mock_modules(base_mocks)

# Packages with submodules that are imported via 'from X.Y import Z'
complex_mocks = {
    "matplotlib": ["pyplot", "figure", "colors", "cm"],
    "scipy": ["stats", "optimize", "signal", "cluster"],
    "Bio": ["SeqIO", "Entrez", "Align", "Data"],
    "sklearn": ["cluster", "decomposition", "metrics", "preprocessing", "manifold"],
    "networkx": ["algorithms", "drawing", "generators"],
    "omegaconf": ["errors"],
    "rich": ["syntax", "console", "logging", "progress"],
    "transformers": ["AutoModel", "AutoTokenizer", "pipeline"]
}

for pkg, subs in complex_mocks.items():
    sys.modules[pkg] = MagicMock()
    for sub in subs:
        sys.modules[f"{pkg}.{sub}"] = MagicMock()

# SPECIAL HANDLING: dataclasses_json must be a real class for @dataclass inheritance
class MockMixin:
    pass
sys.modules["dataclasses_json"] = MagicMock()
sys.modules["dataclasses_json"].DataClassJsonMixin = MockMixin
# -----------------------------------------

from ai_scientist.orchestrator.tool_wrappers import (
    create_lit_summary_artifact,
    create_claim_graph_artifact,
    list_lit_summaries,
    list_claim_graphs,
    read_archivist_artifact,
    # Helper to setup environment akin to tests/orchestrator/test_integrity.py
    reserve_and_register_artifact 
)
from ai_scientist.orchestrator.agents import build_team, ModelSettings

@pytest.fixture
def temp_workspace():
    temp_dir = tempfile.mkdtemp()
    with patch.dict(os.environ, {
        "AISC_BASE_FOLDER": temp_dir, 
        "AISC_ACTIVE_ROLE": "Archivist",
        "AISC_EXP_RESULTS": os.path.join(temp_dir, "experiment_results")
    }), \
         patch("ai_scientist.tools.base_tool.BaseTool.resolve_output_dir", return_value=os.path.join(temp_dir, "experiment_results")):
        
        os.makedirs(os.path.join(temp_dir, "experiment_results"), exist_ok=True)
        yield temp_dir
        
    shutil.rmtree(temp_dir)

def test_archivist_create_artifacts(temp_workspace):
    # Test create_lit_summary_artifact
    res1 = create_lit_summary_artifact(module="lit")
    assert not res1.get("error")
    assert res1["kind"] == "lit_summary_main"
    assert "reserved_path" in res1
    
    # Check it exists in manifest
    listing = list_lit_summaries(module="lit")
    assert listing["total"] == 1
    assert listing["paths"][0] == res1["reserved_path"]

    # Test create_claim_graph_artifact
    res2 = create_claim_graph_artifact(module="lit")
    assert not res2.get("error")
    assert res2["kind"] == "claim_graph_main"
    
    listing_cg = list_claim_graphs(module="lit")
    assert listing_cg["total"] == 1

def test_archivist_read_allowed(temp_workspace):
    # Create a lit summary first
    res = create_lit_summary_artifact()
    path = res["reserved_path"]
    
    # Write some content to it so we can read it
    with open(path, "w") as f:
        json.dump({"test": "content"}, f)
        
    # Read back using restricted tool
    read_res = read_archivist_artifact(name=res["name"])
    assert read_res == {"test": "content"}

def test_archivist_read_blocked(temp_workspace):
    # Create a forbidden artifact by bypassing write permissions for setup
    with patch("ai_scientist.orchestrator.artifacts.ensure_write_permission", return_value=(True, "")):
        forbidden = reserve_and_register_artifact(
            kind="integration_memo_md",
            meta_json=json.dumps({"name": "forbidden", "module": "model"}),
            status="ok"
        )
    if forbidden.get("error"):
        pytest.fail(f"Setup failed: {forbidden['error']}")
        
    path = forbidden["reserved_path"]
    with open(path, "w") as f:
        f.write("secret code")
        
    # Attempt to read with archivist tool
    read_res = read_archivist_artifact(name=forbidden["name"])
    assert "error" in read_res
    assert "Permission denied" in read_res["error"]

def test_archivist_agent_instantiation():
    # Verify we can build the team without error and Archivist has correct tools
    idea = {"Title": "Test", "Abstract": "Test", "Experiments": [], "Risk Factors and Limitations": []}
    dirs = {"base": "/tmp", "results": "/tmp/res"}
    
    # Patching make_agent to avoid actually needing an LLM
    with patch("ai_scientist.orchestrator.agents._make_agent") as mock_make:
        build_team("gpt-4", idea, dirs)
        
        # Find Archivist call
        archivist_call = None
        for call in mock_make.call_args_list:
            if call.kwargs.get("name") == "Archivist":
                archivist_call = call
                break
        
        assert archivist_call is not None
        tools = archivist_call.kwargs["tools"]
        
        # Verify tool list contents
        tool_names = [t.__name__ for t in tools]
        assert "create_lit_summary_artifact" in tool_names
        assert "read_archivist_artifact" in tool_names
        assert "get_run_paths" not in tool_names
        assert "reserve_typed_artifact" not in tool_names
