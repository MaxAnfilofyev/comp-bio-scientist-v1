"""
Manuscript processing utilities for AI Scientist agents.

This module provides functions for extracting ideas and metadata from manuscript files
(PDF, Markdown, Text) to seed research projects.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ai_scientist.tools.manuscript_reader import ManuscriptReaderTool
    from ai_scientist.orchestrator.artifacts import _reserve_typed_artifact_impl
except ImportError:
    # For standalone testing
    ManuscriptReaderTool = None
    _reserve_typed_artifact_impl = None


def _extract_markdown_section(text: str, headers: List[str]) -> str:
    """
    Return the text of the first markdown section matching any header label (case-insensitive).
    """
    if not text or not headers:
        return ""
    header_pattern = "|".join([re.escape(h) for h in headers])
    header_re = re.compile(rf"^\s*#{{1,6}}\s*(?:{header_pattern})\s*$", re.IGNORECASE | re.MULTILINE)
    match = header_re.search(text)
    if not match:
        return ""
    start = match.end()
    remainder = text[start:]
    next_header = re.search(r"^\s*#{1,6}\s+.+$", remainder, re.MULTILINE)
    end = start + next_header.start() if next_header else len(text)
    return remainder[: end - start].strip()


def _extract_markdown_title(text: str) -> str:
    """
    Extract the first H1 markdown title as a fallback.
    """
    match = re.search(r"^\s*#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_subheadings(block: str) -> List[str]:
    """
    Collect subheadings (H2+) from a markdown block.
    """
    return [m.group(1).strip() for m in re.finditer(r"^\s*#{2,6}\s+(.+)$", block, re.MULTILINE)]


def _extract_bullets_or_paragraph(block: str) -> List[str]:
    """
    Prefer bullet items; otherwise return the first paragraph as a single item.
    """
    items: List[str] = []
    for line in block.splitlines():
        bullet = re.match(r"\s*[-*]\s+(.*)", line)
        if bullet:
            cleaned = bullet.group(1).strip()
            if cleaned:
                items.append(cleaned)
    if not items:
        paragraph = " ".join(block.strip().split())
        if paragraph:
            items.append(paragraph)
    return items


def _first_sentence(text: str) -> str:
    """
    Extract the first complete sentence from text.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return parts[0].strip() if parts else cleaned


def _derive_experiments_from_text(text: str) -> List[str]:
    """
    Heuristic extraction of experiment placeholders from manuscript sections.
    """
    experiments: List[str] = []
    results_block = _extract_markdown_section(text, ["results", "experiments"])
    methods_block = _extract_markdown_section(text, ["methods", "materials and methods"])
    for block in [results_block, methods_block]:
        for heading in _extract_subheadings(block):
            experiments.append(heading)
        if not experiments and block:
            experiments.extend(_extract_bullets_or_paragraph(block))
        if experiments:
            break
    if not experiments:
        experiments.append("Extract experiments/figures from manuscript sections (Results/Methods).")
    return experiments


def _derive_idea_from_manuscript(manuscript_path: str, title_override: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a manuscript (md/pdf/txt) and derive an idea dict + context payload for prompts.
    """
    if ManuscriptReaderTool is None:
        raise ImportError("ManuscriptReaderTool not available")

    path = Path(manuscript_path)
    if not path.exists():
        raise FileNotFoundError(f"Manuscript path does not exist: {manuscript_path}")

    try:
        read_result = ManuscriptReaderTool().use_tool(path=str(path))
    except Exception:
        # Fallback for plain-text/markdown reads
        read_result = {"text": path.read_text(encoding="utf-8")}

    raw_text = str(read_result.get("text") or "")
    if not raw_text.strip():
        raise ValueError(f"Manuscript at {manuscript_path} is empty or unreadable.")

    derived_title = title_override or read_result.get("title") or _extract_markdown_title(raw_text) or path.stem
    abstract = _extract_markdown_section(raw_text, ["abstract"])
    intro = _extract_markdown_section(raw_text, ["introduction"])
    short_hypothesis = _first_sentence(abstract) or _first_sentence(intro) or _first_sentence(raw_text) or derived_title
    experiments = _derive_experiments_from_text(raw_text)
    risks_block = _extract_markdown_section(raw_text, ["limitations", "risk factors", "risks"])
    risk_factors = _extract_bullets_or_paragraph(risks_block) if risks_block else []
    if not risk_factors:
        risk_factors = ["Identify limitations and missing data/figures from the manuscript discussion."]

    idea = {
        "Name": derived_title,
        "Title": derived_title,
        "Abstract": abstract or intro or raw_text[:800],
        "Short Hypothesis": short_hypothesis,
        "Experiments": experiments,
        "Risk Factors and Limitations": risk_factors,
        "Related Work": read_result.get("related_work", "None provided."),
        "Source": {"type": "manuscript", "path": str(path)},
    }

    def _truncate_text_response(text: str, path: Optional[str], threshold: int, total_bytes: Optional[int] = None, hint_tool: str = "read_manuscript") -> Dict[str, Any]:
        """Helper for text truncation with standard message."""
        total_chars = len(text)
        snippet = text[:threshold]
        note = (
            f"Content exceeds threshold ({threshold} chars); returned first {threshold} of {total_chars}"
            + (f" (~{total_bytes} bytes)" if total_bytes is not None else "")
            + f". To inspect more, use {hint_tool} or raise return_size_threshold_chars carefully (watch context limits)."
        )
        return {"path": path, "content": snippet, "truncated": True, "note": note, "total_chars": total_chars}

    preview = _truncate_text_response(
        raw_text,
        path=str(path),
        threshold=2000,
        total_bytes=None,
        hint_tool="read_manuscript",
    )

    context = {
        "path": str(path),
        "title": derived_title,
        "abstract": abstract,
        "intro": intro,
        "preview": preview,
        "raw_text": raw_text,
    }
    return idea, context


def _persist_manuscript_seed(manuscript_context: Dict[str, Any], idea: Dict[str, Any]) -> Dict[str, str]:
    """
    Write manuscript-derived seed artifacts into the run folder for provenance.
    """
    if _reserve_typed_artifact_impl is None:
        raise ImportError("_reserve_typed_artifact_impl not available")

    outputs: Dict[str, str] = {}
    try:
        raw_text = manuscript_context.get("raw_text", "")
        if raw_text:
            reserve_text = _reserve_typed_artifact_impl("manuscript_input_text", meta_json=None, unique=False)
            target = reserve_text.get("reserved_path")
            if target:
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_text(raw_text)
                outputs["manuscript_text_path"] = str(target)
        reserve_idea = _reserve_typed_artifact_impl("seed_idea_from_manuscript", meta_json=None, unique=False)
        seed_path = reserve_idea.get("reserved_path")
        if seed_path:
            Path(seed_path).parent.mkdir(parents=True, exist_ok=True)
            Path(seed_path).write_text(json.dumps(idea, indent=2))
            outputs["seed_idea_path"] = str(seed_path)
    except Exception as exc:
        print(f"⚠️ Failed to persist manuscript seed artifacts: {exc}")
    return outputs
