"""
PI Planning Helpers - Structured planning support for the Principal Investigator agent.

This module provides dataclasses and helper functions for maintaining structured
implementation plans and status updates.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ai_scientist.orchestrator.artifacts import (
    reserve_typed_artifact as _reserve_typed_artifact_impl,
    list_artifacts_by_kind,
)
from ai_scientist.utils.notes import write_note_file


@dataclass
class ExperimentPlan:
    """Represents a planned or in-progress experiment."""
    experiment_id: str  # e.g. "E1"
    description: str
    owner_role: str  # "Archivist", "Modeler", etc.
    status: str  # "planned", "in_progress", "blocked", "complete", etc.
    inputs: str  # human-readable; can mention artifact IDs/paths
    outputs: str
    notes: str = ""


@dataclass
class TaskPlan:
    """Represents a task within an experiment."""
    task_id: str  # e.g. "T1"
    experiment_id: str
    description: str
    assigned_to: str  # role or agent label
    status: str
    linked_artifacts: str  # comma-separated or short description
    last_updated: Optional[str] = None  # ISO date


@dataclass
class PlanState:
    """Complete state of the implementation plan."""
    hypothesis: str
    current_phase: str  # "planning", "modeling", "analysis", "writeup", "publication"
    last_updated: str   # ISO date string
    experiments: List[ExperimentPlan]
    tasks: List[TaskPlan]
    decisions: List[str]  # list of "- YYYY-MM-DD: ..." lines


def get_or_create_implementation_plan(run_root: Path) -> Tuple[Path, bool]:
    """
    Returns the path to implementation_plan.md under the given run_root.
    If it does not exist, creates an empty/skeleton implementation plan via
    the artifact registry under kind='implementation_plan_md'.

    Args:
        run_root: The root directory of the run (typically AISC_BASE_FOLDER)

    Returns:
        (plan_path, created_new): Tuple of the plan file path and whether it was newly created
    """
    # Check if plan already exists
    result = list_artifacts_by_kind(kind="implementation_plan_md", limit=1)
    paths = result.get("paths", [])
    
    if paths and len(paths) > 0:
        # Plan exists, return it
        existing_path = Path(paths[0])
        if existing_path.exists():
            return existing_path, False
    
    # Plan doesn't exist, create it
    now = datetime.utcnow().isoformat()
    meta = {
        "module": "planning",
        "summary": "Structured implementation plan for the current run",
        "status": "active",
        "created_at": now,
        "created_by": os.environ.get("AISC_ACTIVE_ROLE", "PI"),
    }
    
    reserve_result = _reserve_typed_artifact_impl(
        kind="implementation_plan_md",
        meta_json=json.dumps(meta),
        unique=False
    )
    
    if reserve_result.get("error"):
        raise RuntimeError(f"Failed to reserve implementation plan: {reserve_result['error']}")
    
    plan_path = Path(reserve_result["reserved_path"])
    
    # Create skeleton markdown
    skeleton = _create_skeleton_plan(now)
    
    # Write the skeleton
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(skeleton, encoding="utf-8")
    
    return plan_path, True


def _create_skeleton_plan(timestamp: str) -> str:
    """Create a skeleton implementation plan with empty sections."""
    return f"""# Implementation Plan

## Overview
- Hypothesis: TBD
- Current phase: planning
- Last updated: {timestamp}

## Experiments
| Experiment ID | Description | Owner Role | Status | Inputs | Outputs | Notes |
|---------------|-------------|-----------|--------|--------|---------|-------|

## Tasks
| Task ID | Experiment | Task Description | Assigned To | Status | Linked Artifacts | Last Updated |
|---------|-----------|------------------|-------------|--------|------------------|--------------|

## Decisions / Changes
- {timestamp}: Plan created.
"""


def update_implementation_plan_from_state(plan_path: Path, state: PlanState) -> None:
    """
    Overwrites implementation_plan.md with a structured markdown representation
    of the given PlanState.

    Args:
        plan_path: Path to the implementation plan file
        state: Complete plan state to write

    Raises:
        IOError: If the file cannot be written
    """
    markdown = _generate_markdown_from_state(state)
    
    # Atomic write: write to temp file, then rename
    temp_path = plan_path.parent / f".{plan_path.name}.tmp"
    try:
        temp_path.write_text(markdown, encoding="utf-8")
        temp_path.replace(plan_path)
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to update implementation plan: {e}") from e


def _generate_markdown_from_state(state: PlanState) -> str:
    """Generate markdown content from PlanState."""
    lines = [
        "# Implementation Plan",
        "",
        "## Overview",
        f"- Hypothesis: {state.hypothesis}",
        f"- Current phase: {state.current_phase}",
        f"- Last updated: {state.last_updated}",
        "",
        "## Experiments",
        "| Experiment ID | Description | Owner Role | Status | Inputs | Outputs | Notes |",
        "|---------------|-------------|-----------|--------|--------|---------|-------|",
    ]
    
    # Add experiment rows
    for exp in state.experiments:
        row = (
            f"| {exp.experiment_id} | {exp.description} | {exp.owner_role} | "
            f"{exp.status} | {exp.inputs} | {exp.outputs} | {exp.notes} |"
        )
        lines.append(row)
    
    lines.extend([
        "",
        "## Tasks",
        "| Task ID | Experiment | Task Description | Assigned To | Status | Linked Artifacts | Last Updated |",
        "|---------|-----------|------------------|-------------|--------|------------------|--------------|",
    ])
    
    # Add task rows
    for task in state.tasks:
        updated = task.last_updated or ""
        row = (
            f"| {task.task_id} | {task.experiment_id} | {task.description} | "
            f"{task.assigned_to} | {task.status} | {task.linked_artifacts} | {updated} |"
        )
        lines.append(row)
    
    lines.extend([
        "",
        "## Decisions / Changes",
    ])
    
    # Add decisions
    for decision in state.decisions:
        # Ensure decision starts with "- " if it doesn't already
        if not decision.strip().startswith("-"):
            lines.append(f"- {decision}")
        else:
            lines.append(decision)
    
    return "\n".join(lines) + "\n"


def log_status_to_user_inbox(run_root: Path, status_block: str) -> None:
    """
    Appends a timestamped status block to user_inbox.md in the given run_root.

    Args:
        run_root: The root directory of the run (typically AISC_BASE_FOLDER)
        status_block: The status message to append

    Raises:
        IOError: If the inbox cannot be written
    """
    timestamp = datetime.utcnow().isoformat()
    
    # Format the status update
    formatted_status = f"""## Status Update ({timestamp})

{status_block}

---
"""
    
    # Use the existing write_note_file utility with append=True
    result = write_note_file(
        content=formatted_status,
        name="user_inbox.md",
        append=True,
        run_root=run_root
    )
    
    if result.get("error"):
        raise IOError(f"Failed to log status to user inbox: {result['error']}")
