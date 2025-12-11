from typing import Any, Dict, List
import json
from datetime import datetime, timedelta
from ai_scientist.utils.manifest import load_entries, append_or_update
from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact

def refresh_project_spine() -> Dict[str, Any]:
    """
    Identify canonical artifacts that form the "spine" of the project 
    (Brief -> Lit -> Memos -> Summary) and write 'current_spine.json'.
    
    Returns:
        The content of the spine.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = load_entries(base_folder=exp_dir)
    
    # Simple heuristic spine:
    # 1. Project Idea (from brief or init meta)
    # 2. Lit Summary (canonical)
    # 3. Integration Memos (canonical)
    # 4. Project Summary / Manuscript
    
    spine = {
        "idea": None,
        "lit_summary": None,
        "memos": [],
        "manuscript": None,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    # We scan entries. Assuming 'canonical' status is the filter.
    for entry in entries:
        kind = entry.get("kind") or entry.get("metadata", {}).get("kind")
        status = entry.get("status") or entry.get("metadata", {}).get("status")
        version = entry.get("version") or entry.get("metadata", {}).get("version")
        
        # In Value Increment 4, we added promote_artifact_to_canonical.
        # So we look for status="canonical"
        
        if status == "canonical":
            if kind == "lit_summary_main" or kind == "lit_review":
                # Keep latest canonical
                current_ver = spine["lit_summary"].get("version") if spine["lit_summary"] else None
                if not spine["lit_summary"] or (version or "") > (current_ver or ""):
                    spine["lit_summary"] = _entry_ref(entry)
            
            elif kind == "integration_memo_md":
                spine["memos"].append(_entry_ref(entry))
            
            elif kind in ["manuscript_pdf", "manuscript_tex", "manuscript_main"]:
                 current_ver = spine["manuscript"].get("version") if spine["manuscript"] else None
                 if not spine["manuscript"] or (version or "") > (current_ver or ""):
                    spine["manuscript"] = _entry_ref(entry)
                    
    # Idea usually comes from startup, might not be a canonical artifact yet.
    # Check for 'project_brief' or similar if it exists.
    idea_entry = next((e for e in entries if e.get("kind") == "project_brief"), None)
    if idea_entry:
        spine["idea"] = _entry_ref(idea_entry)

    # Sort memos
    spine["memos"].sort(key=lambda x: x.get("version") or "")

    # Write spine artifact
    # We use reserve_and_register to make it first-class
    reserve_and_register_artifact(
        kind="current_spine_json",
        meta_json=json.dumps({"content": spine, "summary": "Project spine snapshot"}),
        status="generated",
        unique=False
    )
    
    return spine


def deprecate_stale_drafts(age_hours: int = 24) -> List[str]:
    """
    Mark drafts older than age_hours as 'deprecated' if they are not canonical.
    Returns list of paths deprecated.
    """
    exp_dir = BaseTool.resolve_output_dir(None)
    entries = load_entries(base_folder=exp_dir)
    
    cutoff = datetime.utcnow() - timedelta(hours=age_hours)
    deprecated_paths = []
    
    for entry in entries:
        status = entry.get("status") or entry.get("metadata", {}).get("status")
        if status == "canonical" or status == "deprecated":
            continue
            
        # Check timestamp
        ts_str = entry.get("created_at") or entry.get("timestamp")
        if not ts_str:
            continue
            
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts < cutoff:
                # Mark as deprecated
                # We do this by appending a new manifest entry with updated status
                new_entry = dict(entry)
                new_entry["status"] = "deprecated"
                
                res = append_or_update(new_entry, base_folder=exp_dir)
                if not res.get("error"):
                    deprecated_paths.append(entry.get("path"))
        except Exception:
            continue
            
    return deprecated_paths

def _entry_ref(entry):
    return {
        "path": entry.get("path"),
        "kind": entry.get("kind"),
        "version": entry.get("version") or entry.get("metadata", {}).get("version"),
        "id": entry.get("metadata", {}).get("id")
    }
