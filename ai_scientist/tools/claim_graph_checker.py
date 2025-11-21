import json
from pathlib import Path
from typing import Dict, Any, List

from ai_scientist.tools.base_tool import BaseTool


class ClaimGraphCheckTool(BaseTool):
    """
    Check a claim graph for missing support. Reports claims with no support (and their parents/children).
    """

    def __init__(
        self,
        name: str = "CheckClaimGraph",
        description: str = "Check claim_graph.json for claims (or any descendants) lacking support entries.",
    ):
        parameters = [
            {"name": "path", "type": "str", "description": "Path to claim_graph.json"},
        ]
        super().__init__(name, description, parameters)

    def use_tool(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Claim graph not found: {path}")
        try:
            graph = json.loads(p.read_text())
        except Exception as e:
            raise ValueError(f"Failed to read claim graph: {e}")

        # Build lookup
        claims = {c.get("claim_id"): c for c in graph}
        children: Dict[str, List[str]] = {}
        for c in graph:
            pid = c.get("parent_id")
            if pid:
                children.setdefault(pid, []).append(c.get("claim_id"))

        def has_support(cid: str) -> bool:
            c = claims.get(cid, {})
            return bool(c.get("support"))

        # Find claims with no support and no supported descendants
        missing: List[Dict[str, Any]] = []

        def descendant_supported(cid: str) -> bool:
            if has_support(cid):
                return True
            for ch in children.get(cid, []):
                if descendant_supported(ch):
                    return True
            return False

        for cid, c in claims.items():
            if not descendant_supported(cid):
                missing.append(
                    {
                        "claim_id": cid,
                        "claim_text": c.get("claim_text"),
                        "parent_id": c.get("parent_id"),
                        "support": c.get("support", []),
                        "status": c.get("status", ""),
                    }
                )

        return {"missing_support": missing, "n_missing": len(missing), "n_claims": len(claims)}
