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
    inputs: List[str]  # List of input descriptions or artifact IDs
    outputs: List[str]  # List of output descriptions or artifact IDs
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "owner_role": self.owner_role,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentPlan":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            experiment_id=data["experiment_id"],
            description=data["description"],
            owner_role=data["owner_role"],
            status=data["status"],
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            notes=data.get("notes", ""),
        )



@dataclass
class TaskPlan:
    """Represents a task within an experiment."""
    task_id: str  # e.g. "T1"
    experiment_id: str
    description: str
    assigned_to: str  # role or agent label
    status: str
    linked_artifacts: List[str]  # List of artifact IDs/paths
    last_updated: Optional[str] = None  # ISO date

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "linked_artifacts": self.linked_artifacts,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPlan":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            task_id=data["task_id"],
            experiment_id=data["experiment_id"],
            description=data["description"],
            assigned_to=data["assigned_to"],
            status=data["status"],
            linked_artifacts=data.get("linked_artifacts", []),
            last_updated=data.get("last_updated"),
        )



@dataclass
class PlanState:
    """Complete state of the implementation plan."""
    hypothesis: str
    current_phase: str  # "planning", "modeling", "analysis", "writeup", "publication"
    last_updated: str   # ISO date string
    experiments: List[ExperimentPlan]
    tasks: List[TaskPlan]
    decisions: List[str]  # list of "- YYYY-MM-DD: ..." lines

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hypothesis": self.hypothesis,
            "current_phase": self.current_phase,
            "last_updated": self.last_updated,
            "experiments": [exp.to_dict() for exp in self.experiments],
            "tasks": [task.to_dict() for task in self.tasks],
            "decisions": self.decisions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanState":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            hypothesis=data.get("hypothesis", ""),
            current_phase=data.get("current_phase", "planning"),
            last_updated=data.get("last_updated", datetime.utcnow().isoformat()),
            experiments=[ExperimentPlan.from_dict(exp) for exp in data.get("experiments", [])],
            tasks=[TaskPlan.from_dict(task) for task in data.get("tasks", [])],
            decisions=data.get("decisions", []),
        )





def load_implementation_plan_state(run_root: Path) -> Optional[PlanState]:
    """
    Load the implementation plan state JSON from the run_root if it exists.
    Returns None if no state file is present yet.

    Args:
        run_root: The root directory of the run (typically AISC_BASE_FOLDER)

    Returns:
        PlanState if found and valid, None otherwise

    Raises:
        ValueError: If JSON exists but is invalid/corrupted
    """
    # Look for the JSON state file
    result = list_artifacts_by_kind(kind="implementation_plan_state_json", limit=1)
    paths = result.get("paths", [])
    
    if not paths or len(paths) == 0:
        return None
    
    state_path = Path(paths[0])
    if not state_path.exists():
        return None
    
    try:
        content = state_path.read_text(encoding="utf-8")
        data = json.loads(content)
        
        # Validate required keys
        if not isinstance(data, dict):
            raise ValueError("Plan state JSON must be an object/dict")
        
        return PlanState.from_dict(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in plan state file: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load plan state: {e}") from e


def save_implementation_plan_state(run_root: Path, state: PlanState) -> Path:
    """
    Writes the given PlanState to implementation_plan_state.json under run_root.
    Returns the path to the JSON file.

    Args:
        run_root: The root directory of the run
        state: The plan state to save

    Returns:
        Path to the written JSON file

    Raises:
        IOError: If the file cannot be written
    """
    # Reserve the artifact path
    now = datetime.utcnow().isoformat()
    meta = {
        "module": "planning",
        "summary": "Machine-readable implementation plan state",
        "status": "active",
        "created_at": now,
        "created_by": os.environ.get("AISC_ACTIVE_ROLE", "PI"),
    }
    
    reserve_result = _reserve_typed_artifact_impl(
        kind="implementation_plan_state_json",
        meta_json=json.dumps(meta),
        unique=False  # Only one state file per run
    )
    
    if reserve_result.get("error"):
        raise IOError(f"Failed to reserve plan state path: {reserve_result['error']}")
    
    state_path = Path(reserve_result["reserved_path"])
    
    # Serialize state to JSON
    state_dict = state.to_dict()
    json_content = json.dumps(state_dict, indent=2, ensure_ascii=False)
    
    # Atomic write: temp file + rename
    temp_path = state_path.parent / f".{state_path.name}.tmp"
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json_content, encoding="utf-8")
        temp_path.replace(state_path)
        return state_path
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to save plan state: {e}") from e


def get_or_create_implementation_plan(run_root: Path) -> Tuple[Path, bool]:

    """
    Returns the path to implementation_plan.md under the given run_root.
    If it does not exist, creates an empty/skeleton implementation plan via
    the artifact registry under kind='implementation_plan_md'.
    
    Also ensures that implementation_plan_state.json exists. If markdown exists
    but JSON doesn't, creates a default JSON state.

    Args:
        run_root: The root directory of the run (typically AISC_BASE_FOLDER)

    Returns:
        (plan_path, created_new): Tuple of the plan file path and whether it was newly created
    """
    # Check if plan already exists
    result = list_artifacts_by_kind(kind="implementation_plan_md", limit=1)
    paths = result.get("paths", [])
    
    if paths and len(paths) > 0:
        # Plan exists
        existing_path = Path(paths[0])
        if existing_path.exists():
            # Ensure JSON state also exists
            json_state = load_implementation_plan_state(run_root)
            if json_state is None:
                # Markdown exists but JSON doesn't - create default JSON state
                now = datetime.utcnow().isoformat()
                default_state = PlanState(
                    hypothesis="",
                    current_phase="planning",
                    last_updated=now,
                    experiments=[],
                    tasks=[],
                    decisions=[f"{now}: Plan state initialized from existing markdown."],
                )
                save_implementation_plan_state(run_root, default_state)
            return existing_path, False
    
    # Plan doesn't exist, create both JSON and markdown
    now = datetime.utcnow().isoformat()
    
    # Create default plan state
    default_state = PlanState(
        hypothesis="",
        current_phase="planning",
        last_updated=now,
        experiments=[],
        tasks=[],
        decisions=[f"{now}: Plan created."],
    )
    
    # Save JSON state
    save_implementation_plan_state(run_root, default_state)
    
    # Reserve and create markdown
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
    
    # Generate markdown from state
    markdown = _generate_markdown_from_state(default_state)
    
    # Write the markdown
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(markdown, encoding="utf-8")
    
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


def _merge_experiments(old: List[ExperimentPlan], new: List[ExperimentPlan]) -> List[ExperimentPlan]:
    """
    Merge experiment lists by experiment_id.
    New experiments override old ones with the same ID.
    Experiments only in old are preserved.
    """
    by_id = {exp.experiment_id: exp for exp in old}
    for exp in new:
        by_id[exp.experiment_id] = exp  # New overrides old
    return list(by_id.values())


def _merge_tasks(old: List[TaskPlan], new: List[TaskPlan]) -> List[TaskPlan]:
    """
    Merge task lists by task_id.
    New tasks override old ones with the same ID.
    Tasks only in old are preserved.
    """
    by_id = {task.task_id: task for task in old}
    for task in new:
        by_id[task.task_id] = task  # New overrides old
    return list(by_id.values())


def _merge_decisions(old: List[str], new: List[str]) -> List[str]:
    """
    Merge decision lists by concatenating and deduplicating.
    Preserves order (old decisions first, then new unique ones).
    """
    seen = set()
    merged = []
    for decision in old + new:
        # Normalize whitespace for comparison
        normalized = decision.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(decision)  # Keep original formatting
    return merged


def _merge_plan_states(old_state: Optional[PlanState], new_state: PlanState) -> PlanState:
    """
    Merge old and new plan states with intelligent merge semantics.
    
    Args:
        old_state: Previous plan state (None if this is the first plan)
        new_state: New plan state from current update
    
    Returns:
        Merged plan state
    """
    if old_state is None:
        return new_state
    
    # Merge hypothesis: use new if non-empty, else keep old
    merged_hypothesis = new_state.hypothesis if new_state.hypothesis.strip() else old_state.hypothesis
    
    # Merge current_phase: use new if non-empty, else keep old
    merged_phase = new_state.current_phase if new_state.current_phase.strip() else old_state.current_phase
    
    # Always use new timestamp
    merged_updated = new_state.last_updated
    
    # Merge experiments, tasks, and decisions
    merged_experiments = _merge_experiments(old_state.experiments, new_state.experiments)
    merged_tasks = _merge_tasks(old_state.tasks, new_state.tasks)
    merged_decisions = _merge_decisions(old_state.decisions, new_state.decisions)
    
    return PlanState(
        hypothesis=merged_hypothesis,
        current_phase=merged_phase,
        last_updated=merged_updated,
        experiments=merged_experiments,
        tasks=merged_tasks,
        decisions=merged_decisions,
    )


def update_implementation_plan_from_state(plan_path: Path, state: PlanState) -> None:

    """
    Updates implementation_plan.md and implementation_plan_state.json with merged state.
    
    This function implements merge semantics: it loads the existing plan state (if any),
    merges it with the new state, and writes both the JSON state and markdown rendering.
    
    Merge behavior:
    - Experiments/tasks are merged by ID (new overrides old for same ID)
    - Decisions are concatenated and deduplicated
    - Hypothesis/phase use new value if non-empty, else keep old

    Args:
        plan_path: Path to the implementation plan markdown file
        state: New plan state to merge in

    Raises:
        IOError: If files cannot be written
    """
    # Get run_root from plan_path (plan is in experiment_results/)
    run_root = plan_path.parent.parent
    
    # Load existing state if it exists
    old_state = load_implementation_plan_state(run_root)
    
    # Merge old and new states
    merged_state = _merge_plan_states(old_state, state)
    
    # Save merged state to JSON
    save_implementation_plan_state(run_root, merged_state)
    
    # Generate and save markdown
    markdown = _generate_markdown_from_state(merged_state)
    
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
        # Format lists as comma-separated strings
        inputs_str = ", ".join(exp.inputs) if isinstance(exp.inputs, list) else str(exp.inputs)
        outputs_str = ", ".join(exp.outputs) if isinstance(exp.outputs, list) else str(exp.outputs)
        
        row = (
            f"| {exp.experiment_id} | {exp.description} | {exp.owner_role} | "
            f"{exp.status} | {inputs_str} | {outputs_str} | {exp.notes} |"
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
        # Format linked_artifacts list as comma-separated string
        artifacts_str = ", ".join(task.linked_artifacts) if isinstance(task.linked_artifacts, list) else str(task.linked_artifacts)
        
        row = (
            f"| {task.task_id} | {task.experiment_id} | {task.description} | "
            f"{task.assigned_to} | {task.status} | {artifacts_str} | {updated} |"
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
