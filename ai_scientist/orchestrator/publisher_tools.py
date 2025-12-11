from typing import TypedDict, Optional, Dict, Any, Callable, cast
import json
from agents import function_tool as _function_tool  # type: ignore[reportAttributeAccessIssue]
from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact

function_tool: Callable[..., Any] = cast(Callable[..., Any], _function_tool)

class ArtifactHandle(TypedDict):
    reserved_path: str
    kind: str
    name: str
    rel_dir: str
    quarantined: bool
    metadata: Dict[str, Any]
    manifest_error: Optional[str]
    manifest_index: Optional[int]
    note: Optional[str]

@function_tool
def create_release_manifest_artifact(tag: str) -> Dict[str, Any]:
    """Create and register the release manifest artifact."""
    return reserve_and_register_artifact(kind="release_manifest", unique=False, meta_json=json.dumps({"module": "release", "tag": tag}))

@function_tool
def create_code_release_archive_artifact(tag: str) -> Dict[str, Any]:
    """Create and register the code release archive artifact."""
    return reserve_and_register_artifact(kind="code_release_archive", unique=False, meta_json=json.dumps({"module": "release", "tag": tag}))

@function_tool
def create_env_manifest_artifact(tag: str) -> Dict[str, Any]:
    """Create and register the environment manifest artifact."""
    return reserve_and_register_artifact(kind="env_manifest", unique=False, meta_json=json.dumps({"module": "release", "tag": tag}))

@function_tool
def create_release_diff_patch_artifact(tag: str) -> Dict[str, Any]:
    """Create and register the git diff patch artifact."""
    return reserve_and_register_artifact(kind="release_diff_patch", unique=False, meta_json=json.dumps({"module": "release", "tag": tag}))

@function_tool
def create_release_repro_status_artifact(tag: str) -> Dict[str, Any]:
    """Create and register the release reproducibility status artifact."""
    return reserve_and_register_artifact(kind="release_repro_status_md", unique=False, meta_json=json.dumps({"module": "release", "tag": tag}))

@function_tool
def create_repro_methods_artifact(tag: str) -> Dict[str, Any]:
    """Create and register the reproduction methods artifact."""
    return reserve_and_register_artifact(kind="repro_methods_md", unique=False, meta_json=json.dumps({"module": "release", "tag": tag}))

@function_tool
def create_repro_protocol_artifact(tag: str) -> Dict[str, Any]:
    """Create and register the reproduction protocol artifact."""
    return reserve_and_register_artifact(kind="repro_protocol_md", unique=False, meta_json=json.dumps({"module": "release", "tag": tag}))

@function_tool
def create_manuscript_figure_artifact(figure_id: str) -> Dict[str, Any]:
    """Create and register a manuscript figure artifact (PNG)."""
    return reserve_and_register_artifact(kind="manuscript_figure_png", unique=False, meta_json=json.dumps({"module": "release", "figure_id": figure_id}))

@function_tool
def create_manuscript_figure_svg_artifact(figure_id: str) -> Dict[str, Any]:
    """Create and register a manuscript figure artifact (SVG)."""
    return reserve_and_register_artifact(kind="manuscript_figure_svg", unique=False, meta_json=json.dumps({"module": "release", "figure_id": figure_id}))
