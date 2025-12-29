import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine

from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.database.claim_graph_service import ClaimGraphService


class ClaimGraphTool(BaseTool):
    """
    Maintain a claim graph stored in database.
    Each claim: claim_id, claim_text, parent_id (None for thesis), support (list), status, notes.
    """

    def __init__(
        self,
        name: str = "UpdateClaimGraph",
        description: str = (
            "Add or update a claim in the claim graph database. Thesis has parent_id=None; other claims may point to a parent. "
            "All claims should eventually have support references (citations or artifact paths)."
        ),
    ):
        parameters = [
            {"name": "claim_id", "type": "str", "description": "Unique claim id (e.g., thesis, c1)."},
            {"name": "claim_text", "type": "str", "description": "Text of the claim."},
            {"name": "parent_id", "type": "str", "description": "Parent claim id (use null/None for thesis)."},
            {"name": "support", "type": "list[str]", "description": "Support refs (citations or artifact paths)."},
            {"name": "status", "type": "str", "description": "Status (e.g., unlinked, partial, complete)."},
            {"name": "notes", "type": "str", "description": "Optional notes/gaps."},
            {"name": "project_id", "type": "str", "description": "Project ID (optional, will use current project if not specified)."},
        ]
        super().__init__(name, description, parameters)
        
        # Initialize database connection
        self.engine = create_engine("sqlite:///ai_scientist.sqlite")
        self.service = ClaimGraphService(self.engine)

    def use_tool(
        self,
        claim_id: str,
        claim_text: str,
        parent_id: Optional[str] = None,
        support: Optional[List[str]] = None,
        status: str = "unlinked",
        notes: str = "",
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Get or create project
        if not project_id:
            # Try to infer from current context or use default
            project_id = "DEFAULT_PROJECT"
        
        # Ensure graph exists for project
        graph = self.service.get_graph(project_id)
        if not graph:
            graph_id = self.service.create_graph(project_id)
        
        # Enforce thesis has no parent
        if claim_id.lower() == "thesis":
            parent_id = None
        
        # Check if claim exists in database
        from sqlalchemy import text
        with self.engine.connect() as conn:
            existing_claim = conn.execute(
                text("SELECT claim_id FROM claim WHERE claim_id = :cid"),
                {"cid": claim_id}
            ).fetchone()
            
            if not existing_claim:
                # Create claim in database
                conn.execute(
                    text("""
                        INSERT INTO claim 
                        (claim_id, project_id, module, statement, claim_type, status, 
                         created_by, policy_version)
                        VALUES (:cid, :pid, :module, :stmt, :ctype, :status, :created_by, :policy)
                    """),
                    {
                        "cid": claim_id,
                        "pid": project_id,
                        "module": "claim_graph",
                        "stmt": claim_text,
                        "ctype": "thesis" if claim_id.lower() == "thesis" else "hypothesis",
                        "status": status,
                        "created_by": "claim_graph_tool",
                        "policy": "claim_graph_v1"
                    }
                )
                conn.commit()
            else:
                # Update existing claim
                conn.execute(
                    text("""
                        UPDATE claim 
                        SET statement = :stmt, status = :status
                        WHERE claim_id = :cid
                    """),
                    {"stmt": claim_text, "status": status, "cid": claim_id}
                )
                conn.commit()
        
        # Check if node exists
        nodes = self.service.get_nodes_for_claim(claim_id)
        
        if nodes:
            # Update existing node
            node_id = nodes[0]['node_id']
            
            # Update parent if changed
            if parent_id:
                parent_nodes = self.service.get_nodes_for_claim(parent_id)
                if parent_nodes:
                    parent_node_id = parent_nodes[0]['node_id']
                    with self.engine.connect() as conn:
                        conn.execute(
                            text("""
                                UPDATE claim_graph_node 
                                SET parent_node_id = :parent
                                WHERE node_id = :nid
                            """),
                            {"parent": parent_node_id, "nid": node_id}
                        )
                        conn.commit()
        else:
            # Create new node
            parent_node_id = None
            if parent_id:
                parent_nodes = self.service.get_nodes_for_claim(parent_id)
                if parent_nodes:
                    parent_node_id = parent_nodes[0]['node_id']
            
            node_id = self.service.add_node(claim_id, parent_node_id)
        
        # Update supports
        # Remove old supports and add new ones
        existing_supports = self.service.get_supports(node_id)
        for sup in existing_supports:
            with self.engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM claim_graph_support WHERE support_id = :sid"),
                    {"sid": sup['support_id']}
                )
                conn.commit()
        
        # Add new supports
        if support:
            for sup_ref in support:
                # Determine support type
                if sup_ref.startswith('10.'):  # DOI
                    sup_type = 'citation'
                elif sup_ref.endswith('.json') or sup_ref.endswith('.csv'):
                    sup_type = 'artifact'
                elif sup_ref.startswith('SUP_'):
                    sup_type = 'claim_support'
                else:
                    sup_type = 'citation'  # Default
                
                self.service.add_support(node_id, sup_type, sup_ref, notes)
        
        # Get updated claim info
        claim_data = self.service.get_claim_with_evidence(claim_id)
        
        return {
            "claim_id": claim_id,
            "node_id": node_id,
            "support_count": claim_data.get('support_count', 0),
            "status": "updated"
        }
