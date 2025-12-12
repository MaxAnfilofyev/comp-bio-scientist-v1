"""
PI Orchestration - Enforcement and orchestration logic for the Principal Investigator agent.

This module provides enforcement mechanisms to ensure PI runs persist their work
through writer tools. If no writer tool is called, the final message is auto-logged
to user_inbox.md.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ai_scientist.orchestrator.pi_planning_helpers import log_status_to_user_inbox


# Tools that count as "persistent writes" for PI
PI_WRITER_TOOL_NAMES: set[str] = {
    "update_implementation_plan_from_state",
    "log_status_to_user_inbox",
    "write_pi_notes",
}


@dataclass
class ToolCallRecord:
    """Record of a single tool call made during an agent run."""
    name: str
    arguments: dict[str, object]





def enforce_pi_writer_tools(
    run_root: Path,
    final_message: str,
    tool_calls: Sequence[ToolCallRecord],
) -> None:
    """
    Enforces that the PI run performed at least one persistent write.
    
    If no writer tool was called, auto-log the final_message to user_inbox.md
    via log_status_to_user_inbox.
    
    Args:
        run_root: Root directory of the run (AISC_BASE_FOLDER)
        final_message: Final assistant message from the PI run
        tool_calls: List of tool calls made during the run
        
    Raises:
        IOError: If auto-logging fails
    """
    used_writer_tools = [
        c for c in tool_calls if c.name in PI_WRITER_TOOL_NAMES
    ]

    if used_writer_tools:
        # PI explicitly called a writer tool, no action needed
        return

    # Fallback: auto-persist final_message to user_inbox
    log_status_to_user_inbox(run_root=run_root, status_block=final_message)
