"""
Orchestrator helper package.

This package exposes shared helpers (artifact registry, path helpers) so
callers can import them without pulling the full CLI entrypoint.
"""

from .artifacts import (
    ARTIFACT_TYPE_REGISTRY,
    _artifact_kind_catalog,
    _format_artifact_path,
    _pattern_to_regex,
    _reserve_typed_artifact_impl,
    list_artifacts_by_kind,
    reserve_and_register_artifact,
    reserve_output,
    reserve_typed_artifact,
)
from .context import _fill_figure_dir, _fill_output_dir

__all__ = [
    "ARTIFACT_TYPE_REGISTRY",
    "_artifact_kind_catalog",
    "_format_artifact_path",
    "_pattern_to_regex",
    "_reserve_typed_artifact_impl",
    "list_artifacts_by_kind",
    "reserve_typed_artifact",
    "reserve_and_register_artifact",
    "reserve_output",
    "_fill_output_dir",
    "_fill_figure_dir",
]
