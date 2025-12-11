import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ai_scientist.tools.base_tool import BaseTool


class ClaimGraphTool(BaseTool):
    """
    Maintain a simple claim ↔ support graph stored as JSON.
    Each claim: claim_id, claim_text, parent_id (None for thesis), support (list), status, notes.
    """

    def __init__(
        self,
        name: str = "UpdateClaimGraph",
        description: str = (
            "Add or update a claim in claim_graph.json. Thesis has parent_id=None; other claims may point to a parent. "
            "All claims should eventually have support references (citations or artifact paths)."
        ),
    ):
        parameters = [
            {"name": "path", "type": "str", "description": "Path to claim_graph.json (will be created if missing)."},
            {"name": "claim_id", "type": "str", "description": "Unique claim id (e.g., thesis, c1)."},
            {"name": "claim_text", "type": "str", "description": "Text of the claim."},
            {"name": "parent_id", "type": "str", "description": "Parent claim id (use null/None for thesis)."},
            {"name": "support", "type": "list[str]", "description": "Support refs (citations or artifact paths)."},
            {"name": "status", "type": "str", "description": "Status (e.g., unlinked, partial, complete)."},
            {"name": "notes", "type": "str", "description": "Optional notes/gaps."},
        ]
        super().__init__(name, description, parameters)

    def use_tool(
        self,
        path: str,
        claim_id: str,
        claim_text: str,
        parent_id: Optional[str] = None,
        support: Optional[List[str]] = None,
        status: str = "unlinked",
        notes: str = "",
    ) -> Dict[str, Any]:
        # Resolve path canonically
        if not path or path == "claim_graph.json":
             path = "claim_graph.json"
             # Resolving input path will check canonical dirs. 
             # But for creation we want to ensure it goes to literature if likely new/default
             out_dir = BaseTool.resolve_output_dir(None) / "literature"
             p = out_dir / "claim_graph.json"
             # If that doesn't exist but we have one elsewhere that resolve_input_path finds, we might want that?
             # Actually, we want to ENFORCE location.
             if not p.exists():
                 # checking if it exists elsewhere to migrate?
                 try:
                     existing = BaseTool.resolve_input_path("claim_graph.json", must_exist=True, default_subdir="literature")
                     p = existing
                 except FileNotFoundError:
                     pass
        else:
             p = Path(path)
             if not p.is_absolute():
                 # If relative, anchor to literature if simple filename, else trust caller?
                 # Safest is to use standard resolution
                 try:
                      p = BaseTool.resolve_input_path(path, must_exist=False, default_subdir="literature")
                 except Exception:
                      p = BaseTool.resolve_output_dir(None) / "literature" / path
        
        # Ensure parent exists
        p.parent.mkdir(parents=True, exist_ok=True)
        
        graph: List[Dict[str, Any]] = []
        if p.exists():
            try:
                graph = json.loads(p.read_text())
            except Exception:
                graph = []

        # Enforce thesis has no parent
        if claim_id.lower() == "thesis":
            parent_id = None

        # Update or append
        updated = False
        for c in graph:
            if c.get("claim_id") == claim_id:
                c.update(
                    {
                        "claim_text": claim_text,
                        "parent_id": parent_id,
                        "support": support or [],
                        "status": status,
                        "notes": notes,
                    }
                )
                updated = True
                break
        if not updated:
            graph.append(
                {
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "parent_id": parent_id,
                    "support": support or [],
                    "status": status,
                    "notes": notes,
                }
            )

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(graph, indent=2))

        return {"path": str(p), "n_claims": len(graph), "updated": claim_id}
