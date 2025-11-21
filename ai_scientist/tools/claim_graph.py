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
        p = Path(path)
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
