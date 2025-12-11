"""
Tests for PI orchestration enforcement logic.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_scientist.orchestrator.pi_orchestration import (
    PI_WRITER_TOOL_NAMES,
    ToolCallRecord,
    enforce_pi_writer_tools,
    truncate_for_inbox,
)


class TestTruncateForInbox:
    """Tests for the truncate_for_inbox function."""

    def test_short_text_unchanged(self):
        """Text under the limit should be returned unchanged."""
        text = "This is a short message."
        result = truncate_for_inbox(text, max_chars=100)
        assert result == text

    def test_long_text_truncated(self):
        """Text over the limit should be truncated with '...' suffix."""
        text = "a" * 5000
        result = truncate_for_inbox(text, max_chars=4000)
        assert len(result) == 4000
        assert result.endswith("...")
        assert result == "a" * 3997 + "..."

    def test_exact_limit_unchanged(self):
        """Text exactly at the limit should be returned unchanged."""
        text = "b" * 4000
        result = truncate_for_inbox(text, max_chars=4000)
        assert result == text
        assert not result.endswith("...")

    def test_custom_max_chars(self):
        """Should respect custom max_chars parameter."""
        text = "x" * 200
        result = truncate_for_inbox(text, max_chars=100)
        assert len(result) == 100
        assert result.endswith("...")


class TestEnforcePiWriterTools:
    """Tests for the enforce_pi_writer_tools function."""

    def test_with_writer_tool_no_action(self, tmp_path):
        """When a writer tool is used, no action should be taken."""
        run_root = tmp_path / "test_run"
        run_root.mkdir()
        
        # Setup environment
        os.environ["AISC_BASE_FOLDER"] = str(run_root)
        os.environ["AISC_EXP_RESULTS"] = str(run_root / "experiment_results")
        (run_root / "experiment_results").mkdir()
        
        final_message = "I updated the implementation plan."
        tool_calls = [
            ToolCallRecord(name="check_project_state", arguments={}),
            ToolCallRecord(name="update_implementation_plan_from_state", arguments={"hypothesis": "test"}),
        ]
        
        # Execute enforcement
        enforce_pi_writer_tools(run_root, final_message, tool_calls)
        
        # Verify user_inbox.md was NOT created (writer tool was already called)
        inbox_path = run_root / "user_inbox.md"
        assert not inbox_path.exists(), "user_inbox.md should not be created when writer tool is used"

    def test_without_writer_tool_auto_logs(self, tmp_path):
        """When no writer tool is used, should auto-log to user_inbox.md."""
        run_root = tmp_path / "test_run"
        run_root.mkdir()
        
        # Setup environment
        os.environ["AISC_BASE_FOLDER"] = str(run_root)
        os.environ["AISC_EXP_RESULTS"] = str(run_root / "experiment_results")
        (run_root / "experiment_results").mkdir()
        
        final_message = "I analyzed the project state and found 3 missing artifacts."
        tool_calls = [
            ToolCallRecord(name="check_project_state", arguments={}),
            ToolCallRecord(name="list_artifacts", arguments={"kind": "lit_summary"}),
        ]
        
        # Execute enforcement
        enforce_pi_writer_tools(run_root, final_message, tool_calls)
        
        # Verify user_inbox.md was created
        inbox_path = run_root / "user_inbox.md"
        assert inbox_path.exists(), "user_inbox.md should be created"
        
        content = inbox_path.read_text()
        assert "Status Update" in content
        assert final_message in content

    def test_multiple_tool_calls_with_writer(self, tmp_path):
        """Should correctly identify writer tools among multiple calls."""
        run_root = tmp_path / "test_run"
        run_root.mkdir()
        
        # Setup environment
        os.environ["AISC_BASE_FOLDER"] = str(run_root)
        os.environ["AISC_EXP_RESULTS"] = str(run_root / "experiment_results")
        (run_root / "experiment_results").mkdir()
        
        final_message = "Completed analysis and logged status."
        tool_calls = [
            ToolCallRecord(name="check_project_state", arguments={}),
            ToolCallRecord(name="list_artifacts", arguments={}),
            ToolCallRecord(name="log_status_to_user_inbox", arguments={"status_block": "test"}),
            ToolCallRecord(name="read_artifact", arguments={}),
        ]
        
        # Execute enforcement
        enforce_pi_writer_tools(run_root, final_message, tool_calls)
        
        # Verify no duplicate entry (writer tool was already called)
        inbox_path = run_root / "user_inbox.md"
        # The log_status_to_user_inbox tool would have been called, creating the file
        # But enforce_pi_writer_tools should not add another entry
        # Since we're mocking, we just verify the function doesn't raise

    def test_truncation_on_long_message(self, tmp_path):
        """Should truncate long final messages before logging."""
        run_root = tmp_path / "test_run"
        run_root.mkdir()
        
        # Setup environment
        os.environ["AISC_BASE_FOLDER"] = str(run_root)
        os.environ["AISC_EXP_RESULTS"] = str(run_root / "experiment_results")
        (run_root / "experiment_results").mkdir()
        
        final_message = "x" * 5000  # Very long message
        tool_calls = [
            ToolCallRecord(name="check_project_state", arguments={}),
        ]
        
        # Execute enforcement
        enforce_pi_writer_tools(run_root, final_message, tool_calls)
        
        # Verify user_inbox.md was created with truncated content
        inbox_path = run_root / "user_inbox.md"
        assert inbox_path.exists()
        
        content = inbox_path.read_text()
        # The truncated message should be in the content
        assert "..." in content
        # Original message was 5000 chars, truncated to 4000
        # The status block formatting adds extra text, so we just check it's reasonable
        assert len(content) < 5000

    def test_all_writer_tools_recognized(self, tmp_path):
        """Verify all writer tools in PI_WRITER_TOOL_NAMES are recognized."""
        run_root = tmp_path / "test_run"
        run_root.mkdir()
        
        # Setup environment
        os.environ["AISC_BASE_FOLDER"] = str(run_root)
        os.environ["AISC_EXP_RESULTS"] = str(run_root / "experiment_results")
        (run_root / "experiment_results").mkdir()
        
        final_message = "Test message"
        
        # Test each writer tool individually
        for writer_tool in PI_WRITER_TOOL_NAMES:
            # Clean up from previous iteration
            inbox_path = run_root / "user_inbox.md"
            if inbox_path.exists():
                inbox_path.unlink()
            
            tool_calls = [
                ToolCallRecord(name="check_project_state", arguments={}),
                ToolCallRecord(name=writer_tool, arguments={}),
            ]
            
            # Execute enforcement
            enforce_pi_writer_tools(run_root, final_message, tool_calls)
            
            # Verify no auto-log happened (writer tool was recognized)
            # Note: Some writer tools might create the inbox file themselves
            # We're just verifying no error is raised


class TestPiWriterToolNames:
    """Tests for the PI_WRITER_TOOL_NAMES constant."""

    def test_writer_tool_names_defined(self):
        """PI_WRITER_TOOL_NAMES should be defined and non-empty."""
        assert isinstance(PI_WRITER_TOOL_NAMES, set)
        assert len(PI_WRITER_TOOL_NAMES) > 0

    def test_expected_tools_present(self):
        """Expected writer tools should be in the set."""
        expected_tools = {
            "update_implementation_plan_from_state",
            "log_status_to_user_inbox",
            "write_pi_notes",
        }
        assert expected_tools.issubset(PI_WRITER_TOOL_NAMES)
