"""
Unit tests for PI planning helpers.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_scientist.orchestrator.pi_planning_helpers import (  # noqa: E402
    get_or_create_implementation_plan,
    update_implementation_plan_from_state,
    log_status_to_user_inbox,
    ExperimentPlan,
    TaskPlan,
    PlanState,
)


class TestPIPlanningHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.exp = self.base / "experiment_results"
        self.exp.mkdir(parents=True, exist_ok=True)
        
        self.prev_env = {
            k: os.environ.get(k) 
            for k in ("AISC_BASE_FOLDER", "AISC_EXP_RESULTS", "AISC_ACTIVE_ROLE")
        }
        os.environ["AISC_BASE_FOLDER"] = str(self.base)
        os.environ["AISC_EXP_RESULTS"] = str(self.exp)
        os.environ["AISC_ACTIVE_ROLE"] = "PI"

    def tearDown(self) -> None:
        for k, v in self.prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_get_or_create_creates_skeleton(self) -> None:
        """Test that get_or_create creates a skeleton plan when none exists."""
        plan_path, created = get_or_create_implementation_plan(self.base)
        
        self.assertTrue(created, "Should report plan was created")
        self.assertTrue(plan_path.exists(), "Plan file should exist")
        self.assertEqual(plan_path.name, "implementation_plan.md")
        
        content = plan_path.read_text()
        self.assertIn("# Implementation Plan", content)
        self.assertIn("## Overview", content)
        self.assertIn("## Experiments", content)
        self.assertIn("## Tasks", content)
        self.assertIn("## Decisions / Changes", content)
        self.assertIn("Hypothesis: TBD", content)
        self.assertIn("Current phase: planning", content)

    def test_get_or_create_idempotent(self) -> None:
        """Test that get_or_create doesn't overwrite existing plan."""
        # Create initial plan
        plan_path1, created1 = get_or_create_implementation_plan(self.base)
        self.assertTrue(created1)
        
        # Modify the plan
        custom_content = "# Custom Plan\n\nThis is my custom plan."
        plan_path1.write_text(custom_content)
        
        # Call again
        plan_path2, created2 = get_or_create_implementation_plan(self.base)
        
        self.assertFalse(created2, "Should report plan already exists")
        self.assertEqual(plan_path1, plan_path2, "Should return same path")
        
        # Content should be unchanged
        content = plan_path2.read_text()
        self.assertEqual(content, custom_content, "Should not overwrite existing plan")

    def test_update_implementation_plan_simple(self) -> None:
        """Test updating plan with simple state."""
        plan_path, _ = get_or_create_implementation_plan(self.base)
        
        # Create a simple state
        state = PlanState(
            hypothesis="Test hypothesis",
            current_phase="modeling",
            last_updated="2025-12-10T12:00:00Z",
            experiments=[
                ExperimentPlan(
                    experiment_id="E1",
                    description="First experiment",
                    owner_role="Modeler",
                    status="planned",
                    inputs="None",
                    outputs="sim_results.json",
                    notes="Initial test"
                )
            ],
            tasks=[
                TaskPlan(
                    task_id="T1",
                    experiment_id="E1",
                    description="Run baseline simulation",
                    assigned_to="Modeler",
                    status="pending",
                    linked_artifacts="",
                    last_updated="2025-12-10"
                )
            ],
            decisions=["2025-12-10: Started modeling phase"]
        )
        
        update_implementation_plan_from_state(plan_path, state)
        
        content = plan_path.read_text()
        
        # Check overview
        self.assertIn("Hypothesis: Test hypothesis", content)
        self.assertIn("Current phase: modeling", content)
        self.assertIn("Last updated: 2025-12-10T12:00:00Z", content)
        
        # Check experiment table
        self.assertIn("| E1 |", content)
        self.assertIn("| First experiment |", content)
        self.assertIn("| Modeler |", content)
        self.assertIn("| planned |", content)
        
        # Check task table
        self.assertIn("| T1 |", content)
        self.assertIn("| Run baseline simulation |", content)
        self.assertIn("| pending |", content)
        
        # Check decisions
        self.assertIn("- 2025-12-10: Started modeling phase", content)

    def test_update_implementation_plan_empty_lists(self) -> None:
        """Test updating plan with empty experiments/tasks/decisions."""
        plan_path, _ = get_or_create_implementation_plan(self.base)
        
        state = PlanState(
            hypothesis="Empty state test",
            current_phase="planning",
            last_updated="2025-12-10T12:00:00Z",
            experiments=[],
            tasks=[],
            decisions=[]
        )
        
        # Should not raise an error
        update_implementation_plan_from_state(plan_path, state)
        
        content = plan_path.read_text()
        
        # Should still have headers
        self.assertIn("## Experiments", content)
        self.assertIn("## Tasks", content)
        self.assertIn("## Decisions / Changes", content)
        
        # Tables should have headers but no data rows
        lines = content.split("\n")
        exp_header_idx = next(i for i, line in enumerate(lines) if "| Experiment ID |" in line)
        # Next line should be separator, then either empty or next section
        self.assertIn("---", lines[exp_header_idx + 1])

    def test_update_implementation_plan_atomic_write(self) -> None:
        """Test that update uses atomic write (no partial writes on error)."""
        plan_path, _ = get_or_create_implementation_plan(self.base)
        
        # Write initial content
        initial_content = "# Initial Plan\n\nThis is the initial plan."
        plan_path.write_text(initial_content)
        
        # Create valid state
        state = PlanState(
            hypothesis="Test",
            current_phase="planning",
            last_updated="2025-12-10T12:00:00Z",
            experiments=[],
            tasks=[],
            decisions=[]
        )
        
        # Update should succeed
        update_implementation_plan_from_state(plan_path, state)
        
        # Content should be completely replaced
        content = plan_path.read_text()
        self.assertNotIn("Initial Plan", content)
        self.assertIn("# Implementation Plan", content)

    def test_log_status_to_user_inbox_creates_file(self) -> None:
        """Test that log_status creates user_inbox.md if it doesn't exist."""
        log_status_to_user_inbox(self.base, "Test status update")
        
        inbox_path = self.exp / "user_inbox.md"
        self.assertTrue(inbox_path.exists(), "user_inbox.md should be created")
        
        content = inbox_path.read_text()
        self.assertIn("## Status Update", content)
        self.assertIn("Test status update", content)
        self.assertIn("---", content)

    def test_log_status_to_user_inbox_appends(self) -> None:
        """Test that log_status appends to existing inbox."""
        # Log first status
        log_status_to_user_inbox(self.base, "First update")
        
        # Log second status
        log_status_to_user_inbox(self.base, "Second update")
        
        inbox_path = self.exp / "user_inbox.md"
        content = inbox_path.read_text()
        
        # Both updates should be present
        self.assertIn("First update", content)
        self.assertIn("Second update", content)
        
        # Should have two status update headers
        self.assertEqual(content.count("## Status Update"), 2)
        
        # Should have two separators
        self.assertEqual(content.count("---"), 2)

    def test_log_status_to_user_inbox_timestamped(self) -> None:
        """Test that status updates are timestamped."""
        log_status_to_user_inbox(self.base, "Timestamped update")
        
        inbox_path = self.exp / "user_inbox.md"
        content = inbox_path.read_text()
        
        # Should have ISO timestamp in header
        # Format: ## Status Update (2025-12-10T12:00:00Z)
        import re
        timestamp_pattern = r"## Status Update \(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        self.assertTrue(re.search(timestamp_pattern, content), "Should have timestamped header")

    def test_integration_create_and_update(self) -> None:
        """Integration test: create plan, update it, verify changes."""
        # Create plan
        plan_path, created = get_or_create_implementation_plan(self.base)
        self.assertTrue(created)
        
        # Update with first state
        state1 = PlanState(
            hypothesis="Initial hypothesis",
            current_phase="planning",
            last_updated="2025-12-10T10:00:00Z",
            experiments=[
                ExperimentPlan("E1", "Exp 1", "Archivist", "planned", "", "", "")
            ],
            tasks=[],
            decisions=["2025-12-10: Plan created"]
        )
        update_implementation_plan_from_state(plan_path, state1)
        
        content1 = plan_path.read_text()
        self.assertIn("Initial hypothesis", content1)
        self.assertIn("planning", content1)
        self.assertIn("E1", content1)
        
        # Update with second state (different phase, more experiments)
        state2 = PlanState(
            hypothesis="Initial hypothesis",
            current_phase="modeling",
            last_updated="2025-12-10T14:00:00Z",
            experiments=[
                ExperimentPlan("E1", "Exp 1", "Archivist", "complete", "", "", ""),
                ExperimentPlan("E2", "Exp 2", "Modeler", "in_progress", "", "", "")
            ],
            tasks=[
                TaskPlan("T1", "E2", "Run sims", "Modeler", "in_progress", "", "2025-12-10")
            ],
            decisions=[
                "2025-12-10: Plan created",
                "2025-12-10: Moved to modeling phase"
            ]
        )
        update_implementation_plan_from_state(plan_path, state2)
        
        content2 = plan_path.read_text()
        self.assertIn("modeling", content2)
        self.assertIn("E2", content2)
        self.assertIn("T1", content2)
        self.assertIn("Moved to modeling phase", content2)
        
        # Log status
        log_status_to_user_inbox(self.base, "Modeling phase started")
        
        inbox_path = self.exp / "user_inbox.md"
        inbox_content = inbox_path.read_text()
        self.assertIn("Modeling phase started", inbox_content)

    def test_pi_run_without_writer_tools_creates_inbox_entry(self) -> None:
        """
        Integration test: Simulate a PI run that only calls read-only tools.
        Verify that user_inbox.md gets created with the final message.
        """
        from ai_scientist.orchestrator.pi_orchestration import (
            enforce_pi_writer_tools,
            ToolCallRecord,
        )
        
        final_message = "I analyzed the project state and found 3 missing artifacts."
        tool_calls = [
            ToolCallRecord(name="check_project_state", arguments={}),
            ToolCallRecord(name="list_artifacts", arguments={"kind": "lit_summary"}),
        ]
        
        # Execute enforcement
        enforce_pi_writer_tools(self.base, final_message, tool_calls)
        
        # Verify user_inbox.md was created
        inbox_path = self.exp / "user_inbox.md"
        self.assertTrue(inbox_path.exists(), "user_inbox.md should be created")
        
        content = inbox_path.read_text()
        self.assertIn("Status Update", content)
        self.assertIn(final_message, content)


if __name__ == "__main__":
    unittest.main()
