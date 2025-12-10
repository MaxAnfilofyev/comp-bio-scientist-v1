from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.utils.manifest import load_entries

DEFAULT_SUMMARY_THRESHOLD = 5


def _normalize_timestamp(entry: Dict[str, Any]) -> Optional[str]:
    ts = entry.get("timestamp") or entry.get("metadata", {}).get("created_at")
    if not ts:
        return None
    if isinstance(ts, str):
        return ts
    try:
        return str(ts)
    except Exception:
        return None


def _entry_module(entry: Dict[str, Any]) -> Optional[str]:
    module = entry.get("module")
    if isinstance(module, str):
        return module
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        module = metadata.get("module")
        if isinstance(module, str):
            return module
    return None


def _sorted_by_timestamp(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(entries, key=lambda e: _normalize_timestamp(e) or "")


def maybe_generate_module_summary(
    module: str,
    base_folder: Optional[str] = None,
    threshold: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if threshold is None or threshold < 1:
        threshold = int(os.environ.get("AISC_SUMMARY_THRESHOLD") or DEFAULT_SUMMARY_THRESHOLD)
    exp_dir = BaseTool.resolve_output_dir(None) if base_folder is None else base_folder
    entries = load_entries(base_folder=exp_dir, limit=None)
    module_entries = [e for e in entries if _entry_module(e) == module]
    if not module_entries:
        return None

    summary_kind = "integration_memo_md"
    summary_entries = [e for e in module_entries if e.get("kind") == summary_kind]
    artifact_entries = [e for e in module_entries if e.get("kind") != summary_kind]

    last_summary = _sorted_by_timestamp(summary_entries)[-1] if summary_entries else None
    last_summary_cutoff = None
    parent_version = None
    if last_summary:
        last_summary_cutoff = last_summary.get("metadata", {}).get("summarized_up_to") or _normalize_timestamp(last_summary)
        parent_version = last_summary.get("version")

    new_entries: List[Dict[str, Any]] = []
    for entry in _sorted_by_timestamp(artifact_entries):
        entry_ts = _normalize_timestamp(entry)
        if not last_summary_cutoff or (entry_ts and entry_ts > last_summary_cutoff):
            new_entries.append(entry)
    if len(new_entries) < threshold:
        return None

    summarized_up_to = _normalize_timestamp(new_entries[-1]) or last_summary_cutoff
    summary_text = _build_summary_text(module, new_entries, last_summary_cutoff)
    summary_info = [
        {
            "name": entry.get("name"),
            "kind": entry.get("kind"),
            "summary": entry.get("summary") or entry.get("metadata", {}).get("summary") or "No summary provided.",
            "path": entry.get("path"),
            "timestamp": _normalize_timestamp(entry),
        }
        for entry in new_entries
    ]

    return {
        "module": module,
        "summary_text": summary_text,
        "summary_count": len(new_entries),
        "summarized_up_to": summarized_up_to,
        "entries": summary_info,
        "parent_version": parent_version,
        "last_summary_cutoff": last_summary_cutoff,
    }


def _build_summary_text(module: str, entries: List[Dict[str, Any]], last_cutoff: Optional[str]) -> str:
    lines: List[str] = []
    lines.append(f"# Integrated Memo — Module: {module}")
    lines.append("")
    if last_cutoff:
        lines.append(f"**New artifacts since** {last_cutoff}")
    else:
        lines.append("**New artifacts since project inception**")
    lines.append("")
    lines.append(f"Summarized {len(entries)} artifacts:")
    lines.append("")
    for entry in entries:
        summary = entry.get("summary") or entry.get("metadata", {}).get("summary") or "No summary."
        timestamp = _normalize_timestamp(entry) or "unknown time"
        lines.append(f"- **{entry.get('name')}** ({entry.get('kind')} @ {timestamp}): {summary}")
    lines.append("")
    lines.append("## Key takeaways")
    lines.append(f"- {len(entries)} new entries consolidated into this memo.")
    return "\n".join(lines)
