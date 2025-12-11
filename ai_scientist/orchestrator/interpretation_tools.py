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
def create_interpretation_json_artifact() -> Dict[str, Any]:
    """
    Create and register the interpretation_json artifact for theoretical biology interpretations.

    - kind: 'interpretation_json'
    - module: 'interpretation'
    - rel_dir: 'experiment_results/interpretation'
    - pattern: 'interpretation.json'

    Returns:
        ArtifactHandle describing the reserved artifact (id, kind, path, module).

    Invariants:
        - Does not accept arbitrary paths from callers.
        - Uses ARTIFACT_TYPE_REGISTRY for path resolution.
        - Registers the artifact in the manifest.
    """
    return reserve_and_register_artifact(kind="interpretation_json", unique=False, meta_json='{"module": "interpretation"}')

@function_tool
def create_interpretation_md_artifact() -> Dict[str, Any]:
    """
    Create and register the interpretation_md artifact for theoretical biology interpretations.

    - kind: 'interpretation_md'
    - module: 'interpretation'
    - rel_dir: 'experiment_results/interpretation'
    - pattern: 'interpretation.md'

    Returns:
        ArtifactHandle describing the reserved artifact.

    Invariants:
        - Path is derived from ARTIFACT_TYPE_REGISTRY, not caller input.
        - Artifact is registered in the manifest.
    """
    return reserve_and_register_artifact(kind="interpretation_md", unique=False, meta_json='{"module": "interpretation"}')
