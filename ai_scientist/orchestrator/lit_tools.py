from typing import TypedDict, Optional, Dict, Any, Callable, cast
from pathlib import Path
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
def create_lit_review_artifact(content: str = "") -> Dict[str, Any]:
    """
    Create and register the lit_review_md artifact for the current run.
    If content is provided, writes it to the file.
    """
    res = reserve_and_register_artifact(kind="lit_review_md", unique=False, meta_json='{"module": "lit"}')
    if "reserved_path" in res:
        Path(res["reserved_path"]).write_text(content, encoding="utf-8")
    return res

@function_tool
def create_lit_bibliography_artifact(content: str = "") -> Dict[str, Any]:
    """
    Create and register the lit_bibliography_bib artifact for the current run.
    If content is provided, writes it to the file.
    """
    res = reserve_and_register_artifact(kind="lit_bibliography_bib", unique=False, meta_json='{"module": "lit"}')
    if "reserved_path" in res:
        Path(res["reserved_path"]).write_text(content, encoding="utf-8")
    return res

@function_tool
def create_lit_coverage_artifact(content: str = "") -> Dict[str, Any]:
    """
    Create and register the lit_coverage_json artifact for the current run.
    If content is provided, writes it to the file.
    """
    res = reserve_and_register_artifact(kind="lit_coverage_json", unique=False, meta_json='{"module": "lit"}')
    if "reserved_path" in res:
        Path(res["reserved_path"]).write_text(content, encoding="utf-8")
    return res

@function_tool
def create_lit_integration_memo_artifact(content: str = "") -> Dict[str, Any]:
    """
    Create and register the integrated memo for the literature module.
    If content is provided, writes it to the file.
    """
    res = reserve_and_register_artifact(kind="integration_memo_md", unique=False, meta_json='{"module": "lit"}')
    if "reserved_path" in res:
        Path(res["reserved_path"]).write_text(content, encoding="utf-8")
    return res
