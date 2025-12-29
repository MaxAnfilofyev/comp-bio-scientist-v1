"""
Claim Graph Service - Database integration for claim graph functionality.

Provides CRUD operations for claim graph nodes and supports, enabling:
- Hierarchical claim structures
- Evidence linking (S2 pipeline, citations, artifacts)
- Automated evidence gathering
- JSON import/export for backward compatibility
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy import text, Engine

class ClaimGraphService:
    def __init__(self, engine: Engine):
        self.engine = engine
    
    # ==================== Graph Management ====================
    
    def create_graph(self, project_id: str, graph_id: Optional[str] = None) -> str:
        """Create a new claim graph for a project."""
        if not graph_id:
            graph_id = f"GRAPH_{uuid.uuid4().hex[:12]}"
        
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO claim_graph_meta (graph_id, project_id, status)
                    VALUES (:graph_id, :project_id, 'draft')
                """),
                {"graph_id": graph_id, "project_id": project_id}
            )
            conn.commit()
        
        return graph_id
    
    def get_graph(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get graph metadata for a project."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM claim_graph_meta WHERE project_id = :pid"),
                {"pid": project_id}
            ).fetchone()
            
            if result:
                return dict(result._mapping)
        return None
    
    # ==================== Node Management ====================
    
    def add_node(
        self,
        claim_id: str,
        parent_node_id: Optional[str] = None,
        position: int = 0,
        node_id: Optional[str] = None
    ) -> str:
        """Add a claim to the graph as a node."""
        if not node_id:
            node_id = f"NODE_{uuid.uuid4().hex[:12]}"
        
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO claim_graph_node (node_id, claim_id, parent_node_id, position)
                    VALUES (:node_id, :claim_id, :parent_id, :position)
                """),
                {
                    "node_id": node_id,
                    "claim_id": claim_id,
                    "parent_id": parent_node_id,
                    "position": position
                }
            )
            conn.commit()
        
        return node_id
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM vw_claim_graph_status WHERE node_id = :nid"),
                {"nid": node_id}
            ).fetchone()
            
            if result:
                return dict(result._mapping)
        return None
    
    def get_nodes_for_claim(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get all graph nodes for a claim (usually just one)."""
        with self.engine.connect() as conn:
            results = conn.execute(
                text("SELECT * FROM vw_claim_graph_status WHERE claim_id = :cid"),
                {"cid": claim_id}
            ).fetchall()
            
            return [dict(r._mapping) for r in results]
    
    def get_children(self, node_id: str) -> List[Dict[str, Any]]:
        """Get child nodes."""
        with self.engine.connect() as conn:
            results = conn.execute(
                text("""
                    SELECT * FROM vw_claim_graph_status 
                    WHERE parent_node_id = :nid 
                    ORDER BY position
                """),
                {"nid": node_id}
            ).fetchall()
            
            return [dict(r._mapping) for r in results]
    
    # ==================== Support Management ====================
    
    def add_support(
        self,
        node_id: str,
        support_type: str,  # 'citation', 'artifact', 'claim_support', 'experiment'
        reference: str,
        notes: Optional[str] = None,
        support_id: Optional[str] = None
    ) -> str:
        """Add a support reference to a node."""
        if not support_id:
            support_id = f"SUP_{uuid.uuid4().hex[:12]}"
        
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO claim_graph_support 
                    (support_id, node_id, support_type, reference, notes)
                    VALUES (:sup_id, :node_id, :type, :ref, :notes)
                """),
                {
                    "sup_id": support_id,
                    "node_id": node_id,
                    "type": support_type,
                    "ref": reference,
                    "notes": notes
                }
            )
            conn.commit()
        
        return support_id
    
    def get_supports(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all supports for a node."""
        with self.engine.connect() as conn:
            results = conn.execute(
                text("SELECT * FROM claim_graph_support WHERE node_id = :nid"),
                {"nid": node_id}
            ).fetchall()
            
            return [dict(r._mapping) for r in results]
    
    # ==================== Query Helpers ====================
    
    def find_unsupported_claims(self, project_id: str, min_support: int = 2) -> List[Dict[str, Any]]:
        """Find claims in the graph that need more evidence."""
        with self.engine.connect() as conn:
            results = conn.execute(
                text("""
                    SELECT v.* FROM vw_unsupported_claims v
                    JOIN claim c ON v.claim_id = c.claim_id
                    WHERE c.project_id = :pid
                      AND v.support_count < :min_sup
                    ORDER BY v.created_at
                """),
                {"pid": project_id, "min_sup": min_support}
            ).fetchall()
            
            return [dict(r._mapping) for r in results]
    
    def get_claim_with_evidence(self, claim_id: str) -> Dict[str, Any]:
        """Get claim with all graph info and evidence."""
        nodes = self.get_nodes_for_claim(claim_id)
        if not nodes:
            return {}
        
        node = nodes[0]  # Usually just one node per claim
        supports = self.get_supports(node['node_id'])
        children = self.get_children(node['node_id'])
        
        # Get S2 evidence if any claim_support references exist
        s2_evidence = []
        for sup in supports:
            if sup['support_type'] == 'claim_support':
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("""
                            SELECT cs.*, w.title, w.authors, w.year 
                            FROM claim_support cs
                            LEFT JOIN work w ON cs.doi = w.doi
                            WHERE cs.support_id = :sid
                        """),
                        {"sid": sup['reference']}
                    ).fetchone()
                    
                    if result:
                        s2_evidence.append(dict(result._mapping))
        
        return {
            **node,
            'supports': supports,
            'children': children,
            's2_evidence': s2_evidence
        }
    
    # ==================== JSON Import/Export ====================
    
    def import_from_json(self, json_path: Path, project_id: str) -> str:
        """Import claim_graph.json into database."""
        data = json.loads(json_path.read_text())
        
        # Create graph
        graph_id = self.create_graph(project_id)
        
        # Map claim_id to node_id
        node_map = {}
        
        # First pass: create all nodes
        for item in data:
            claim_id = item['claim_id']
            
            # Create claim if doesn't exist
            with self.engine.connect() as conn:
                existing = conn.execute(
                    text("SELECT claim_id FROM claim WHERE claim_id = :cid"),
                    {"cid": claim_id}
                ).fetchone()
                
                if not existing:
                    conn.execute(
                        text("""
                            INSERT INTO claim (claim_id, project_id, statement, status)
                            VALUES (:cid, :pid, :stmt, :status)
                        """),
                        {
                            "cid": claim_id,
                            "pid": project_id,
                            "stmt": item['claim_text'],
                            "status": item.get('status', 'proposed')
                        }
                    )
                    conn.commit()
            
            # Create node
            node_id = self.add_node(claim_id)
            node_map[claim_id] = node_id
        
        # Second pass: set parent relationships and add supports
        for item in data:
            node_id = node_map[item['claim_id']]
            parent_claim_id = item.get('parent_id')
            
            if parent_claim_id and parent_claim_id in node_map:
                with self.engine.connect() as conn:
                    conn.execute(
                        text("""
                            UPDATE claim_graph_node 
                            SET parent_node_id = :parent 
                            WHERE node_id = :nid
                        """),
                        {"parent": node_map[parent_claim_id], "nid": node_id}
                    )
                    conn.commit()
            
            # Add supports
            for sup_ref in item.get('support', []):
                # Determine support type
                if sup_ref.startswith('10.'):  # DOI
                    sup_type = 'citation'
                elif sup_ref.endswith('.json') or sup_ref.endswith('.csv'):
                    sup_type = 'artifact'
                else:
                    sup_type = 'citation'  # Default
                
                self.add_support(node_id, sup_type, sup_ref, item.get('notes'))
        
        # Set thesis node
        thesis_items = [item for item in data if item['claim_id'].lower() == 'thesis']
        if thesis_items:
            thesis_node_id = node_map[thesis_items[0]['claim_id']]
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE claim_graph_meta 
                        SET thesis_node_id = :tid 
                        WHERE graph_id = :gid
                    """),
                    {"tid": thesis_node_id, "gid": graph_id}
                )
                conn.commit()
        
        return graph_id
    
    def export_to_json(self, project_id: str) -> List[Dict[str, Any]]:
        """Export graph to claim_graph.json format."""
        with self.engine.connect() as conn:
            # Get all nodes for project
            results = conn.execute(
                text("""
                    SELECT n.*, c.statement as claim_text, c.status
                    FROM claim_graph_node n
                    JOIN claim c ON n.claim_id = c.claim_id
                    WHERE c.project_id = :pid
                    ORDER BY n.position
                """),
                {"pid": project_id}
            ).fetchall()
            
            output = []
            for row in results:
                node_id = row.node_id
                supports = self.get_supports(node_id)
                
                # Find parent claim_id
                parent_claim_id = None
                if row.parent_node_id:
                    parent_node = conn.execute(
                        text("SELECT claim_id FROM claim_graph_node WHERE node_id = :nid"),
                        {"nid": row.parent_node_id}
                    ).fetchone()
                    if parent_node:
                        parent_claim_id = parent_node.claim_id
                
                output.append({
                    "claim_id": row.claim_id,
                    "claim_text": row.claim_text,
                    "parent_id": parent_claim_id,
                    "support": [s['reference'] for s in supports],
                    "status": row.status or "unlinked",
                    "notes": supports[0]['notes'] if supports and supports[0]['notes'] else ""
                })
            
            return output
