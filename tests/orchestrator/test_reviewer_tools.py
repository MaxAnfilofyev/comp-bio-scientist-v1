
import sys
import json
from unittest.mock import MagicMock, patch

# Mock agents module
mock_agents = MagicMock()
def identity_decorator(func=None, **kwargs):
    if func and callable(func):
        return func
    def _wrapper(f):
        return f
    return _wrapper
mock_agents.function_tool = identity_decorator
sys.modules["agents"] = mock_agents
sys.modules["ai_scientist.llm"] = MagicMock()
sys.modules["ai_scientist.perform_writeup"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["omegaconf"] = MagicMock()
sys.modules["ai_scientist.perform_biological_interpretation"] = MagicMock()

# Import tools to test
from ai_scientist.orchestrator.tool_wrappers import (  # noqa: E402
    create_review_note_artifact,
    check_parameter_sources_for_manuscript,
    check_metrics_for_referenced_models,
    check_hypothesis_trace_consistency,
    check_proof_of_work_for_results,
    check_references_completeness,
)

@patch("ai_scientist.orchestrator.tool_wrappers.reserve_and_register_artifact")
@patch("pathlib.Path.write_text")
@patch("pathlib.Path.parent")
def test_create_review_note_artifact(mock_parent, mock_write, mock_reserve):
    # Setup
    mock_reserve.return_value = {"reserved_path": "/tmp/review.md"}
    mock_parent.mkdir.return_value = None
    
    # Test review_report
    res = create_review_note_artifact(kind="review_report", content="Report content")
    assert res["status"] == "success"
    assert res["path"] == "/tmp/review.md"
    mock_reserve.assert_called_with(
        kind="review_report",
        meta_json=json.dumps({"manuscript_version": "v1", "module": "review"}),
        status="canonical",
        unique=True,
        change_summary=""
    )
    mock_write.assert_called_with("Report content", encoding="utf-8")

    # Test verification_note
    res = create_review_note_artifact(kind="verification_note", content="Note content", experiment_id="exp123")
    assert res["status"] == "success"
    mock_reserve.assert_called_with(
        kind="verification_note",
        meta_json=json.dumps({"experiment_id": "exp123", "module": "modeling"}),
        status="canonical",
        unique=True,
        change_summary=""
    )

    # Test invalid kind
    res = create_review_note_artifact(kind="invalid", content="x")
    assert "error" in res

@patch("ai_scientist.orchestrator.tool_wrappers.list_artifacts_by_kind")
@patch("pathlib.Path.read_text")
def test_check_parameter_sources_for_manuscript(mock_read, mock_list):
    # Setup artifacts
    mock_list.return_value = {"paths": ["/tmp/params.json"]}
    
    # helper to mock read based on path
    def side_effect_read(*args, **kwargs):
        return json.dumps({
            "param1": {"value": 10, "source_type": "lit_value", "lit_claim_id": "C1"},
            "param2": {"value": 20, "source_type": "assumption"},
            "param3": {"value": 30} # Missing source_type
        })
    mock_read.side_effect = side_effect_read
    
    report = check_parameter_sources_for_manuscript()
    assert "Missing source_type for 'param3'" in report
    assert "Valid" not in report # Should fail for param3

@patch("ai_scientist.orchestrator.tool_wrappers.list_artifacts_by_kind")
def test_check_metrics_for_referenced_models(mock_list):
    # Setup
    mock_list.side_effect = [
        {"paths": ["model_A_metrics.json"]}, # json
        {"paths": ["model_B_metrics.csv"]},  # csv
    ]
    
    report = check_metrics_for_referenced_models()
    assert "Found 2 metrics artifacts" in report
    assert "model_A_metrics.json" in report
    assert "model_B_metrics.csv" in report

@patch("os.path.exists")
@patch("pathlib.Path.read_text")
def test_check_hypothesis_trace_consistency(mock_read, mock_exists):
    # Setup trace
    trace = {
        "hypotheses": [
            {
                "id": "H1",
                "experiments": [
                    {
                        "id": "E1",
                        "status": "supported",
                        "runs": ["run1"],
                        "figures": ["fig1.png"]
                    }
                ]
            }
        ]
    }
    mock_read.return_value = json.dumps(trace)
    
    # Case 1: All exist
    mock_exists.return_value = True
    report = check_hypothesis_trace_consistency()
    assert "Figures verified" in report
    
    # Case 2: Figure missing
    # We need strict control over exist check.
    # checking trace path, then figure path.
    # Let's just mock specific paths.
    
    def side_effect_exists(path):
        if "hypothesis_trace.json" in str(path):
            return True
        if "fig1.png" in str(path):
            return False
        return False
    
    mock_exists.side_effect = side_effect_exists
    report = check_hypothesis_trace_consistency()
    # It should fail on fig1.png
    assert "Missing referenced figures" in report

@patch("ai_scientist.orchestrator.tool_wrappers.list_artifacts_by_kind")
def test_check_proof_of_work_for_results(mock_list):
    mock_list.side_effect = [
        {"paths": ["fig1.png", "fig2.png"]}, # figures
        {"paths": ["note1.md"]},             # notes
    ]
    
    report = check_proof_of_work_for_results()
    assert "Found 2 manuscript figures" in report
    assert "1 verification notes" in report

@patch("ai_scientist.orchestrator.tool_wrappers.list_artifacts_by_kind")
@patch("pathlib.Path.read_text")
def test_check_references_completeness(mock_read, mock_list):
    # Setup
    mock_list.return_value = {"paths": ["verification.csv"]} # first call for CSV
    
    mock_read.return_value = "citation_key,found,match_score\nref1,True,1.0\nref2,False,0.0"
    
    report = check_references_completeness()
    assert "Found 1 missing references" in report
