
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before import
mock_agents = MagicMock()
def identity_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return getattr(args[0], "__func__", args[0]) if args else lambda x: x

mock_agents.function_tool = identity_decorator
sys.modules["agents"] = mock_agents
sys.modules["agents.types"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["pymupdf"] = MagicMock()
sys.modules["pymupdf4llm"] = MagicMock()
sys.modules["omegaconf"] = MagicMock()
sys.modules["omegaconf.errors"] = MagicMock()
sys.modules["coolname"] = MagicMock()
sys.modules["shutup"] = MagicMock()
sys.modules["igraph"] = MagicMock()
sys.modules["humanize"] = MagicMock()
sys.modules["ai_scientist.tools.biological_interpretation"] = MagicMock()
sys.modules["ai_scientist.treesearch"] = MagicMock()
sys.modules["ai_scientist.treesearch.utils"] = MagicMock()
sys.modules["ai_scientist.treesearch.utils.config"] = MagicMock()
sys.modules["ai_scientist.treesearch.utils.tree_export"] = MagicMock()
sys.modules["ai_scientist.treesearch.journal"] = MagicMock()
sys.modules["ai_scientist.treesearch.interpreter"] = MagicMock()

# Mock underlying tools
sys.modules["ai_scientist.tools.intervention_tester"] = MagicMock()

from ai_scientist.orchestrator.tool_wrappers import run_intervention_tests # noqa: E402

class TestInterventionWrapper(unittest.TestCase):
    
    @patch("ai_scientist.orchestrator.tool_wrappers.RunInterventionTesterTool")
    @patch("ai_scientist.orchestrator.tool_wrappers._fill_output_dir")
    def test_run_intervention_tests_canonical_path(self, mock_fill_output_dir, mock_tool_cls):
        # Setup mocks
        mock_tool_instance = MagicMock()
        mock_tool_cls.return_value = mock_tool_instance
        mock_fill_output_dir.side_effect = lambda x: f"/resolved/{x}" if x else "/resolved/default"
        
        # Call wrapper with experiment_id
        run_intervention_tests(
            graph_path="graph.gpickle",
            experiment_id="TEST_EXP_001",
            skip_lit_gate=True
        )
        
        # Verify _fill_output_dir called with correct subdir
        mock_fill_output_dir.assert_called_with("simulations/TEST_EXP_001")
        
        # Verify underlying tool called with resolved path
        mock_tool_instance.use_tool.assert_called_once()
        call_kwargs = mock_tool_instance.use_tool.call_args.kwargs
        self.assertEqual(call_kwargs["output_dir"], "/resolved/simulations/TEST_EXP_001")

    @patch("ai_scientist.orchestrator.tool_wrappers.RunInterventionTesterTool")
    @patch("ai_scientist.orchestrator.tool_wrappers._fill_output_dir")
    def test_run_intervention_tests_fallback_path(self, mock_fill_output_dir, mock_tool_cls):
        # Setup mocks
        mock_tool_instance = MagicMock()
        mock_tool_cls.return_value = mock_tool_instance
        mock_fill_output_dir.side_effect = lambda x: f"/resolved/{x}" if x else "/resolved/default"
        
        # Call wrapper WITHOUT experiment_id
        run_intervention_tests(
            graph_path="graph.gpickle",
            skip_lit_gate=True
        )
        
        # Verify _fill_output_dir called with adhoc fallback
        mock_fill_output_dir.assert_called_with("simulations/adhoc_interventions")
        
        # Verify underlying tool called with resolved path
        mock_tool_instance.use_tool.assert_called_once()
        call_kwargs = mock_tool_instance.use_tool.call_args.kwargs
        self.assertEqual(call_kwargs["output_dir"], "/resolved/simulations/adhoc_interventions")

if __name__ == "__main__":
    unittest.main()
