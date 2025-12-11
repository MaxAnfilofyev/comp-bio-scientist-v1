import sys
from unittest.mock import MagicMock, patch

# Mock agents module to avoid ImportError
mock_agents = MagicMock()
# Make function_tool a pass-through decorator so we can test the wrapped functions
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




from ai_scientist.orchestrator.tool_wrappers import (  # noqa: E402
    create_plot_artifact,
    publish_figure_to_manuscript_gallery,
    list_available_runs_for_plotting,
    get_metrics_for_plotting
)

@patch("ai_scientist.orchestrator.tool_wrappers.reserve_and_register_artifact")
def test_create_plot_artifact_valid(mock_reserve):
    mock_reserve.return_value = {"reserved_path": "fig.png"}
    
    result = create_plot_artifact(
        kind="manuscript_figure_png",
        figure_name="fig1",
        change_summary="Initial plot"
    )
    
    assert result == {"reserved_path": "fig.png"}
    mock_reserve.assert_called_once()
    call_args = mock_reserve.call_args[1]
    assert call_args["kind"] == "manuscript_figure_png"
    assert "fig1" in call_args["meta_json"]

def test_create_plot_artifact_invalid_kind():
    result = create_plot_artifact(kind="invalid_kind", figure_name="fig1")
    assert "error" in result
    assert "Invalid plot kind" in result["error"]

@patch("ai_scientist.orchestrator.tool_wrappers.mirror_artifacts")
def test_publish_figure_to_manuscript_gallery(mock_mirror):
    mock_mirror.return_value = {"copied": ["fig1.png"]}
    
    result = publish_figure_to_manuscript_gallery(
        artifact_id="fig1.png",
        name_suffix="_v2"
    )
    
    assert result == {"copied": ["fig1.png"]}
    mock_mirror.assert_called_once_with(
        src_paths=["fig1.png"],
        dest_dir="experiment_results/figures_for_manuscript",
        suffix="_v2",
        mode="copy"
    )

@patch("ai_scientist.orchestrator.tool_wrappers.read_transport_manifest")
def test_list_available_runs_for_plotting(mock_read):
    mock_read.return_value = {
        "runs": [
            {"baseline": "b1", "transport": 0.1, "seed": 1, "status": "complete"},
            {"baseline": "b1", "transport": 0.2, "seed": 2, "status": "failed"}
        ]
    }
    
    result = list_available_runs_for_plotting(experiment_id="exp1") # experiment_id might be ignored if we just read manifest
    
    # We expect a simplified list of successful runs
    assert len(result) == 1
    assert result[0] == "b1_transport_0.1_seed_1"

@patch("ai_scientist.orchestrator.tool_wrappers.list_artifacts_by_kind")
def test_get_metrics_for_plotting(mock_list):
    mock_list.return_value = {
        "paths": [
            "experiment_results/metrics/model_A_metrics.json",
            "experiment_results/metrics/model_B_metrics.json"
        ]
    }
    
    result = get_metrics_for_plotting(experiment_id="exp1", model_key="model_A")
    assert result == "experiment_results/metrics/model_A_metrics.json"
    
    result = get_metrics_for_plotting(experiment_id="exp1", model_key="model_C")
    assert result is None
