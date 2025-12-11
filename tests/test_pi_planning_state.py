"""
Unit tests for PI planning state management.

Tests JSON serialization, load/save, merge logic, and markdown rendering.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from ai_scientist.orchestrator.pi_planning_helpers import (
    ExperimentPlan,
    PlanState,
    TaskPlan,
    _generate_markdown_from_state,
    _merge_decisions,
    _merge_experiments,
    _merge_plan_states,
    _merge_tasks,
    load_implementation_plan_state,
    save_implementation_plan_state,
)


class TestDataclassSerialization(unittest.TestCase):
    """Test JSON serialization/deserialization of dataclasses."""

    def test_experiment_plan_round_trip(self):
        """Test ExperimentPlan to_dict and from_dict."""
        exp = ExperimentPlan(
            experiment_id="E1",
            description="Test experiment",
            owner_role="Modeler",
            status="planned",
            inputs=["input1", "input2"],
            outputs=["output1"],
            notes="Test notes",
        )

        # Serialize and deserialize
        data = exp.to_dict()
        exp2 = ExperimentPlan.from_dict(data)

        self.assertEqual(exp, exp2)
        self.assertEqual(exp2.inputs, ["input1", "input2"])
        self.assertEqual(exp2.outputs, ["output1"])

    def test_task_plan_round_trip(self):
        """Test TaskPlan to_dict and from_dict."""
        task = TaskPlan(
            task_id="T1",
            experiment_id="E1",
            description="Test task",
            assigned_to="Modeler",
            status="in_progress",
            linked_artifacts=["artifact1.json", "artifact2.csv"],
            last_updated="2025-12-11T00:00:00",
        )

        data = task.to_dict()
        task2 = TaskPlan.from_dict(data)

        self.assertEqual(task, task2)
        self.assertEqual(task2.linked_artifacts, ["artifact1.json", "artifact2.csv"])

    def test_plan_state_round_trip(self):
        """Test PlanState to_dict and from_dict."""
        state = PlanState(
            hypothesis="Test hypothesis",
            current_phase="modeling",
            last_updated="2025-12-11T00:00:00",
            experiments=[
                ExperimentPlan(
                    experiment_id="E1",
                    description="Exp 1",
                    owner_role="Modeler",
                    status="planned",
                    inputs=["in1"],
                    outputs=["out1"],
                )
            ],
            tasks=[
                TaskPlan(
                    task_id="T1",
                    experiment_id="E1",
                    description="Task 1",
                    assigned_to="Modeler",
                    status="planned",
                    linked_artifacts=["art1"],
                )
            ],
            decisions=["2025-12-11: Decision 1"],
        )

        data = state.to_dict()
        state2 = PlanState.from_dict(data)

        self.assertEqual(state, state2)
        self.assertEqual(len(state2.experiments), 1)
        self.assertEqual(len(state2.tasks), 1)
        self.assertEqual(state2.experiments[0].inputs, ["in1"])


class TestLoadSaveState(unittest.TestCase):
    """Test load and save operations for plan state."""

    def test_save_and_load_state(self):
        """Test saving and loading plan state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a state
            state = PlanState(
                hypothesis="Test hypothesis",
                current_phase="planning",
                last_updated=datetime.utcnow().isoformat(),
                experiments=[
                    ExperimentPlan(
                        experiment_id="E1",
                        description="Test",
                        owner_role="Modeler",
                        status="planned",
                        inputs=["input1"],
                        outputs=["output1"],
                    )
                ],
                tasks=[],
                decisions=["2025-12-11: Created plan"],
            )

            # Save it (we'll mock the artifact system by directly writing)
            json_path = tmp_path / "implementation_plan_state.json"
            json_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

            # Load it back
            loaded_data = json.loads(json_path.read_text(encoding="utf-8"))
            loaded_state = PlanState.from_dict(loaded_data)

            self.assertEqual(loaded_state.hypothesis, state.hypothesis)
            self.assertEqual(loaded_state.current_phase, state.current_phase)
            self.assertEqual(len(loaded_state.experiments), 1)
            self.assertEqual(loaded_state.experiments[0].experiment_id, "E1")


class TestMergeLogic(unittest.TestCase):
    """Test merge functions for experiments, tasks, and decisions."""

    def test_merge_experiments_preserves_old(self):
        """Test that old experiments are preserved when not in new list."""
        old = [
            ExperimentPlan("E1", "Old E1", "Modeler", "planned", ["in1"], ["out1"]),
            ExperimentPlan("E2", "Old E2", "Analyst", "planned", ["in2"], ["out2"]),
            ExperimentPlan("E3", "Old E3", "Modeler", "complete", ["in3"], ["out3"]),
        ]

        new = [
            ExperimentPlan("E1", "New E1", "Modeler", "in_progress", ["in1_new"], ["out1_new"]),
            ExperimentPlan("E4", "New E4", "Analyst", "planned", ["in4"], ["out4"]),
        ]

        merged = _merge_experiments(old, new)

        # Should have E1 (updated), E2 (preserved), E3 (preserved), E4 (new)
        self.assertEqual(len(merged), 4)
        
        by_id = {exp.experiment_id: exp for exp in merged}
        self.assertEqual(by_id["E1"].description, "New E1")  # Updated
        self.assertEqual(by_id["E1"].status, "in_progress")
        self.assertEqual(by_id["E2"].description, "Old E2")  # Preserved
        self.assertEqual(by_id["E3"].description, "Old E3")  # Preserved
        self.assertEqual(by_id["E4"].description, "New E4")  # New

    def test_merge_tasks_preserves_old(self):
        """Test that old tasks are preserved when not in new list."""
        old = [
            TaskPlan("T1", "E1", "Task 1", "Modeler", "planned", ["art1"]),
            TaskPlan("T2", "E1", "Task 2", "Analyst", "complete", ["art2"]),
        ]

        new = [
            TaskPlan("T1", "E1", "Task 1 Updated", "Modeler", "in_progress", ["art1", "art1b"]),
            TaskPlan("T3", "E2", "Task 3", "Modeler", "planned", ["art3"]),
        ]

        merged = _merge_tasks(old, new)

        self.assertEqual(len(merged), 3)
        
        by_id = {task.task_id: task for task in merged}
        self.assertEqual(by_id["T1"].description, "Task 1 Updated")  # Updated
        self.assertEqual(by_id["T1"].linked_artifacts, ["art1", "art1b"])
        self.assertEqual(by_id["T2"].description, "Task 2")  # Preserved
        self.assertEqual(by_id["T3"].description, "Task 3")  # New

    def test_merge_decisions_deduplicates(self):
        """Test that decisions are deduplicated while preserving order."""
        old = [
            "2025-12-10: Decision A",
            "2025-12-10: Decision B",
            "2025-12-11: Decision C",
        ]

        new = [
            "2025-12-10: Decision B",  # Duplicate
            "2025-12-11: Decision D",  # New
            "2025-12-11: Decision C",  # Duplicate
        ]

        merged = _merge_decisions(old, new)

        # Should have A, B, C, D (in that order, no duplicates)
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[0], "2025-12-10: Decision A")
        self.assertEqual(merged[1], "2025-12-10: Decision B")
        self.assertEqual(merged[2], "2025-12-11: Decision C")
        self.assertEqual(merged[3], "2025-12-11: Decision D")

    def test_merge_plan_states_full(self):
        """Test full plan state merge."""
        old_state = PlanState(
            hypothesis="Old hypothesis",
            current_phase="planning",
            last_updated="2025-12-10T00:00:00",
            experiments=[
                ExperimentPlan("E1", "Exp 1", "Modeler", "planned", ["in1"], ["out1"]),
                ExperimentPlan("E2", "Exp 2", "Analyst", "planned", ["in2"], ["out2"]),
            ],
            tasks=[
                TaskPlan("T1", "E1", "Task 1", "Modeler", "planned", ["art1"]),
                TaskPlan("T2", "E1", "Task 2", "Analyst", "planned", ["art2"]),
            ],
            decisions=["2025-12-10: Decision A"],
        )

        new_state = PlanState(
            hypothesis="New hypothesis",
            current_phase="modeling",
            last_updated="2025-12-11T00:00:00",
            experiments=[
                ExperimentPlan("E1", "Exp 1 Updated", "Modeler", "in_progress", ["in1_new"], ["out1_new"]),
            ],
            tasks=[
                TaskPlan("T1", "E1", "Task 1 Updated", "Modeler", "in_progress", ["art1", "art1b"]),
            ],
            decisions=["2025-12-11: Decision B"],
        )

        merged = _merge_plan_states(old_state, new_state)

        # Check hypothesis and phase updated
        self.assertEqual(merged.hypothesis, "New hypothesis")
        self.assertEqual(merged.current_phase, "modeling")
        self.assertEqual(merged.last_updated, "2025-12-11T00:00:00")

        # Check experiments merged
        self.assertEqual(len(merged.experiments), 2)
        by_id = {exp.experiment_id: exp for exp in merged.experiments}
        self.assertEqual(by_id["E1"].description, "Exp 1 Updated")
        self.assertEqual(by_id["E2"].description, "Exp 2")  # Preserved

        # Check tasks merged
        self.assertEqual(len(merged.tasks), 2)
        by_id_task = {task.task_id: task for task in merged.tasks}
        self.assertEqual(by_id_task["T1"].description, "Task 1 Updated")
        self.assertEqual(by_id_task["T2"].description, "Task 2")  # Preserved

        # Check decisions merged
        self.assertEqual(len(merged.decisions), 2)
        self.assertIn("2025-12-10: Decision A", merged.decisions)
        self.assertIn("2025-12-11: Decision B", merged.decisions)

    def test_merge_with_empty_new_hypothesis(self):
        """Test that empty new hypothesis preserves old."""
        old_state = PlanState(
            hypothesis="Old hypothesis",
            current_phase="planning",
            last_updated="2025-12-10T00:00:00",
            experiments=[],
            tasks=[],
            decisions=[],
        )

        new_state = PlanState(
            hypothesis="",  # Empty
            current_phase="modeling",
            last_updated="2025-12-11T00:00:00",
            experiments=[],
            tasks=[],
            decisions=[],
        )

        merged = _merge_plan_states(old_state, new_state)

        self.assertEqual(merged.hypothesis, "Old hypothesis")  # Preserved
        self.assertEqual(merged.current_phase, "modeling")  # Updated


class TestMarkdownRendering(unittest.TestCase):
    """Test markdown generation from plan state."""

    def test_generate_markdown_with_lists(self):
        """Test that List[str] fields are rendered as comma-separated."""
        state = PlanState(
            hypothesis="Test hypothesis",
            current_phase="modeling",
            last_updated="2025-12-11T00:00:00",
            experiments=[
                ExperimentPlan(
                    experiment_id="E1",
                    description="Experiment 1",
                    owner_role="Modeler",
                    status="planned",
                    inputs=["input1", "input2", "input3"],
                    outputs=["output1", "output2"],
                    notes="Test notes",
                )
            ],
            tasks=[
                TaskPlan(
                    task_id="T1",
                    experiment_id="E1",
                    description="Task 1",
                    assigned_to="Modeler",
                    status="planned",
                    linked_artifacts=["artifact1.json", "artifact2.csv"],
                )
            ],
            decisions=["2025-12-11: Decision 1", "2025-12-11: Decision 2"],
        )

        markdown = _generate_markdown_from_state(state)

        # Check that inputs/outputs are comma-separated
        self.assertIn("input1, input2, input3", markdown)
        self.assertIn("output1, output2", markdown)
        self.assertIn("artifact1.json, artifact2.csv", markdown)

        # Check structure
        self.assertIn("# Implementation Plan", markdown)
        self.assertIn("## Overview", markdown)
        self.assertIn("## Experiments", markdown)
        self.assertIn("## Tasks", markdown)
        self.assertIn("## Decisions / Changes", markdown)

    def test_generate_markdown_empty_state(self):
        """Test markdown generation with empty experiments/tasks."""
        state = PlanState(
            hypothesis="",
            current_phase="planning",
            last_updated="2025-12-11T00:00:00",
            experiments=[],
            tasks=[],
            decisions=["2025-12-11: Plan created"],
        )

        markdown = _generate_markdown_from_state(state)

        self.assertIn("# Implementation Plan", markdown)
        self.assertIn("2025-12-11: Plan created", markdown)


if __name__ == "__main__":
    unittest.main()
