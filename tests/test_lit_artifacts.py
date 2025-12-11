"""
Unit tests for literature artifact creation and access control.
Tests the new lit-related artifact types and specialized helper tools.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ai_scientist.orchestrator.artifacts import (
    ARTIFACT_TYPE_REGISTRY,
    _format_artifact_path,
)
from ai_scientist.orchestrator.lit_tools import (
    create_lit_review_artifact,
    create_lit_bibliography_artifact,
    create_lit_coverage_artifact,
    create_lit_integration_memo_artifact,
)
from ai_scientist.orchestrator.tool_wrappers import read_archivist_artifact


class TestLitArtifactRegistry:
    """Test that new lit artifact types are properly registered."""

    def test_lit_review_md_in_registry(self):
        """Verify lit_review_md is registered with correct properties."""
        assert "lit_review_md" in ARTIFACT_TYPE_REGISTRY
        entry = ARTIFACT_TYPE_REGISTRY["lit_review_md"]
        assert entry["rel_dir"] == "experiment_results"
        assert entry["pattern"] == "lit_review.md"
        assert "literature review" in entry["description"].lower()

    def test_lit_bibliography_bib_in_registry(self):
        """Verify lit_bibliography_bib is registered with correct properties."""
        assert "lit_bibliography_bib" in ARTIFACT_TYPE_REGISTRY
        entry = ARTIFACT_TYPE_REGISTRY["lit_bibliography_bib"]
        assert entry["rel_dir"] == "experiment_results"
        assert entry["pattern"] == "lit_bibliography.bib"
        assert "bibtex" in entry["description"].lower()

    def test_lit_coverage_json_in_registry(self):
        """Verify lit_coverage_json is registered with correct properties."""
        assert "lit_coverage_json" in ARTIFACT_TYPE_REGISTRY
        entry = ARTIFACT_TYPE_REGISTRY["lit_coverage_json"]
        assert entry["rel_dir"] == "experiment_results"
        assert entry["pattern"] == "lit_coverage.json"
        assert "coverage" in entry["description"].lower()

    def test_integration_memo_md_in_registry(self):
        """Verify integration_memo_md exists for lit module."""
        assert "integration_memo_md" in ARTIFACT_TYPE_REGISTRY
        entry = ARTIFACT_TYPE_REGISTRY["integration_memo_md"]
        assert entry["rel_dir"] == "experiment_results/summaries"
        assert "integrated_memo_{module}.md" in entry["pattern"]

    def test_lit_artifact_path_formatting(self):
        """Test that lit artifact paths format correctly."""
        # Test lit_review_md
        rel_dir, name = _format_artifact_path("lit_review_md", {"module": "lit"})
        assert rel_dir == "experiment_results"
        assert name == "lit_review.md"

        # Test lit_bibliography_bib
        rel_dir, name = _format_artifact_path("lit_bibliography_bib", {"module": "lit"})
        assert rel_dir == "experiment_results"
        assert name == "lit_bibliography.bib"

        # Test lit_coverage_json
        rel_dir, name = _format_artifact_path("lit_coverage_json", {"module": "lit"})
        assert rel_dir == "experiment_results"
        assert name == "lit_coverage.json"

        # Test integration_memo_md with module='lit'
        rel_dir, name = _format_artifact_path("integration_memo_md", {"module": "lit"})
        assert rel_dir == "experiment_results/summaries"
        assert name == "integrated_memo_lit.md"


class TestLitArtifactCreationTools:
    """Test the specialized lit artifact creation helper tools."""

    @patch('ai_scientist.orchestrator.lit_tools.reserve_and_register_artifact')
    def test_create_lit_review_artifact(self, mock_reserve):
        """Test create_lit_review_artifact calls reserve with correct params."""
        mock_reserve.return_value = {
            "reserved_path": "/tmp/experiment_results/lit_review.md",
            "kind": "lit_review_md",
            "name": "lit_review.md",
            "rel_dir": "experiment_results",
            "quarantined": False,
            "metadata": {"module": "lit"},
        }

        result = create_lit_review_artifact()

        mock_reserve.assert_called_once_with(
            kind="lit_review_md",
            unique=False,
            meta_json='{"module": "lit"}'
        )
        assert result["kind"] == "lit_review_md"
        assert "lit_review.md" in result["reserved_path"]

    @patch('ai_scientist.orchestrator.lit_tools.reserve_and_register_artifact')
    def test_create_lit_bibliography_artifact(self, mock_reserve):
        """Test create_lit_bibliography_artifact calls reserve with correct params."""
        mock_reserve.return_value = {
            "reserved_path": "/tmp/experiment_results/lit_bibliography.bib",
            "kind": "lit_bibliography_bib",
            "name": "lit_bibliography.bib",
            "rel_dir": "experiment_results",
            "quarantined": False,
            "metadata": {"module": "lit"},
        }

        result = create_lit_bibliography_artifact()

        mock_reserve.assert_called_once_with(
            kind="lit_bibliography_bib",
            unique=False,
            meta_json='{"module": "lit"}'
        )
        assert result["kind"] == "lit_bibliography_bib"
        assert "lit_bibliography.bib" in result["reserved_path"]

    @patch('ai_scientist.orchestrator.lit_tools.reserve_and_register_artifact')
    def test_create_lit_coverage_artifact(self, mock_reserve):
        """Test create_lit_coverage_artifact calls reserve with correct params."""
        mock_reserve.return_value = {
            "reserved_path": "/tmp/experiment_results/lit_coverage.json",
            "kind": "lit_coverage_json",
            "name": "lit_coverage.json",
            "rel_dir": "experiment_results",
            "quarantined": False,
            "metadata": {"module": "lit"},
        }

        result = create_lit_coverage_artifact()

        mock_reserve.assert_called_once_with(
            kind="lit_coverage_json",
            unique=False,
            meta_json='{"module": "lit"}'
        )
        assert result["kind"] == "lit_coverage_json"
        assert "lit_coverage.json" in result["reserved_path"]

    @patch('ai_scientist.orchestrator.lit_tools.reserve_and_register_artifact')
    def test_create_lit_integration_memo_artifact(self, mock_reserve):
        """Test create_lit_integration_memo_artifact calls reserve with correct params."""
        mock_reserve.return_value = {
            "reserved_path": "/tmp/experiment_results/summaries/integrated_memo_lit.md",
            "kind": "integration_memo_md",
            "name": "integrated_memo_lit.md",
            "rel_dir": "experiment_results/summaries",
            "quarantined": False,
            "metadata": {"module": "lit"},
        }

        result = create_lit_integration_memo_artifact()

        mock_reserve.assert_called_once_with(
            kind="integration_memo_md",
            unique=False,
            meta_json='{"module": "lit"}'
        )
        assert result["kind"] == "integration_memo_md"
        assert "integrated_memo_lit.md" in result["reserved_path"]


class TestArchivistReadRestrictions:
    """Test read_archivist_artifact access control."""

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    @patch('ai_scientist.orchestrator.tool_wrappers.read_artifact')
    def test_read_allowed_lit_review(self, mock_read, mock_find):
        """Test that Archivist can read lit_review_md."""
        mock_find.return_value = {
            "kind": "lit_review_md",
            "path": "/tmp/lit_review.md",
            "module": "lit"
        }
        mock_read.return_value = "# Literature Review\n..."

        result = read_archivist_artifact("lit_review.md")

        assert result == "# Literature Review\n..."
        mock_read.assert_called_once()

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    @patch('ai_scientist.orchestrator.tool_wrappers.read_artifact')
    def test_read_allowed_lit_bibliography(self, mock_read, mock_find):
        """Test that Archivist can read lit_bibliography_bib."""
        mock_find.return_value = {
            "kind": "lit_bibliography_bib",
            "path": "/tmp/lit_bibliography.bib",
            "module": "lit"
        }
        mock_read.return_value = "@article{...}"

        result = read_archivist_artifact("lit_bibliography.bib")

        assert result == "@article{...}"
        mock_read.assert_called_once()

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    @patch('ai_scientist.orchestrator.tool_wrappers.read_artifact')
    def test_read_allowed_lit_coverage(self, mock_read, mock_find):
        """Test that Archivist can read lit_coverage_json."""
        mock_find.return_value = {
            "kind": "lit_coverage_json",
            "path": "/tmp/lit_coverage.json",
            "module": "lit"
        }
        mock_read.return_value = '{"coverage": 0.85}'

        result = read_archivist_artifact("lit_coverage.json")

        assert result == '{"coverage": 0.85}'
        mock_read.assert_called_once()

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    @patch('ai_scientist.orchestrator.tool_wrappers.read_artifact')
    def test_read_allowed_integration_memo_lit(self, mock_read, mock_find):
        """Test that Archivist can read integration_memo_md for module='lit'."""
        mock_find.return_value = {
            "kind": "integration_memo_md",
            "path": "/tmp/integrated_memo_lit.md",
            "module": "lit"
        }
        mock_read.return_value = "# Integrated Memo\n..."

        result = read_archivist_artifact("integrated_memo_lit.md")

        assert result == "# Integrated Memo\n..."
        mock_read.assert_called_once()

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    def test_read_denied_integration_memo_other_module(self, mock_find):
        """Test that Archivist cannot read integration_memo_md for other modules."""
        mock_find.return_value = {
            "kind": "integration_memo_md",
            "path": "/tmp/integrated_memo_modeling.md",
            "module": "modeling"
        }

        result = read_archivist_artifact("integrated_memo_modeling.md")

        assert isinstance(result, dict)
        assert "error" in result
        assert "module='lit'" in result["error"]

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    def test_read_denied_transport_manifest(self, mock_find):
        """Test that Archivist cannot read transport_manifest."""
        mock_find.return_value = {
            "kind": "transport_manifest",
            "path": "/tmp/manifest.json",
            "module": "modeling"
        }

        result = read_archivist_artifact("manifest.json")

        assert isinstance(result, dict)
        assert "error" in result
        assert "Permission denied" in result["error"]
        assert "transport_manifest" in result["error"]

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    def test_read_denied_parameter_set(self, mock_find):
        """Test that Archivist cannot read parameter_set."""
        mock_find.return_value = {
            "kind": "parameter_set",
            "path": "/tmp/params.json",
            "module": "modeling"
        }

        result = read_archivist_artifact("params.json")

        assert isinstance(result, dict)
        assert "error" in result
        assert "Permission denied" in result["error"]

    @patch('ai_scientist.orchestrator.tool_wrappers.manifest_utils.find_manifest_entry')
    def test_read_artifact_not_found(self, mock_find):
        """Test error when artifact is not in manifest."""
        mock_find.return_value = None

        result = read_archivist_artifact("nonexistent.md")

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestArchivistToolsIntegration:
    """Integration tests for Archivist tools with real file system."""

    def test_lit_artifacts_end_to_end(self, tmp_path):
        """Test creating and reading lit artifacts end-to-end."""
        # This would require setting up AISC_BASE_FOLDER and manifest
        # Skipping for now as it requires more complex setup
        pytest.skip("Integration test requires full environment setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
