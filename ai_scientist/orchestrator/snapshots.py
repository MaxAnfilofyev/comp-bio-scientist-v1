import os
import json
from datetime import datetime
from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.orchestrator.artifacts import reserve_and_register_artifact
from ai_scientist.orchestrator.lifecycle import refresh_project_spine

def generate_project_snapshot() -> str:
    """
    Aggregates content from the spine into a 'project_snapshot.md' 
    and returns its path.
    """
    spine = refresh_project_spine()
    exp_dir = BaseTool.resolve_output_dir(None)
    
    sections = [f"# Project Snapshot ({datetime.utcnow().isoformat()})\n"]
    
    # Helper to read text content
    def read_text(path):
        if not path:
            return "N/A"
        full_path = path if os.path.isabs(path) else exp_dir / path
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"[Error reading {path}: {e}]"

    # 1. Idea
    if spine.get("idea"):
        sections.append("## Project Brief")
        sections.append(read_text(spine["idea"]["path"]))
    
    # 2. Lit Summary
    if spine.get("lit_summary"):
        sections.append("## Literature Summary")
        # If it's json, might need parsing, but for now specific roles read specific kinds.
        # If text is available in metadata or content, use that.
        # We'll just link to it or try to read if it's md.
        path = spine["lit_summary"]["path"]
        if path.endswith(".json"):
             sections.append(f"See [Literature Summary]({path})")
        else:
             sections.append(read_text(path))

    # 3. Memos
    if spine.get("memos"):
        sections.append("## Integration Memos")
        for memo in spine["memos"]:
             sections.append(f"### {memo.get('kind')} ({memo.get('version')})")
             sections.append(read_text(memo.get("path")))
             
    # 4. Manuscript
    if spine.get("manuscript"):
        sections.append("## Manuscript")
        sections.append(f"Current manuscript at: {spine['manuscript']['path']}")

    content = "\n\n".join(sections)
    
    res = reserve_and_register_artifact(
        kind="project_snapshot_md",
        meta_json=json.dumps({"content": {"text": content}, "summary": "Full project snapshot"}),
        status="generated",
        unique=True
    )
    
    path = res.get("reserved_path")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            
    return path or "Error generating snapshot"
