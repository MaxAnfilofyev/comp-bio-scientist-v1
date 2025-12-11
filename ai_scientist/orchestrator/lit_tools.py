from typing import TypedDict, Optional, Dict, Any, Callable, cast
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
def create_lit_review_artifact() -> Dict[str, Any]:
    """
    Create and register the lit_review_md artifact for the current run.

    - kind: 'lit_review_md'
    - module: 'lit'
    - rel_dir: 'experiment_results'
    - pattern: 'lit_review.md'

    Returns:
        ArtifactHandle describing the reserved artifact (id, kind, path, module).

    Invariants:
        - Does not accept arbitrary paths from callers.
        - Uses ARTIFACT_TYPE_REGISTRY for path resolution.
        - Registers the artifact in the manifest.
    """
    return reserve_and_register_artifact(kind="lit_review_md", unique=False, meta_json='{"module": "lit"}')

@function_tool
def create_lit_bibliography_artifact() -> Dict[str, Any]:
    """
    Create and register the lit_bibliography_bib artifact for the current run.

    - kind: 'lit_bibliography_bib'
    - module: 'lit'
    - rel_dir: 'experiment_results'
    - pattern: 'lit_bibliography.bib'

    Returns:
        ArtifactHandle describing the reserved artifact.

    Invariants:
        - Path is derived from ARTIFACT_TYPE_REGISTRY, not caller input.
        - Artifact is registered in the manifest.
    """
    return reserve_and_register_artifact(kind="lit_bibliography_bib", unique=False, meta_json='{"module": "lit"}')

@function_tool
def create_lit_coverage_artifact() -> Dict[str, Any]:
    """
    Create and register the lit_coverage_json artifact for the current run.

    - kind: 'lit_coverage_json'
    - module: 'lit'
    - rel_dir: 'experiment_results'
    - pattern: 'lit_coverage.json'

    Returns:
        ArtifactHandle describing the reserved artifact.

    Invariants:
        - Path is canonical and registry-driven.
        - Manifest is updated.
    """
    return reserve_and_register_artifact(kind="lit_coverage_json", unique=False, meta_json='{"module": "lit"}')

@function_tool
def create_lit_integration_memo_artifact() -> Dict[str, Any]:
    """
    Create and register the integrated memo for the literature module.

    - kind: 'integration_memo_md'
    - module: 'lit'
    - rel_dir: 'experiment_results/summaries'
    - pattern: 'integrated_memo_lit.md'

    Returns:
        ArtifactHandle describing the reserved artifact.

    Invariants:
        - Uses the canonical 'integration_memo_md' registry entry.
        - Does not allow callers to specify custom filenames or paths.
    """
    return reserve_and_register_artifact(kind="integration_memo_md", unique=False, meta_json='{"module": "lit"}')
