"""
Automated Evidence Gathering - Link S2 Pipeline with Claim Graph

This module provides functions to automatically gather evidence for claims
in the claim graph using the S2 pipeline.
"""

import sys
sys.path.insert(0, '/Users/maxa/AI-Scientist-v2')

from sqlalchemy import create_engine
from ai_scientist.database.claim_graph_service import ClaimGraphService
from ai_scientist.evidence.evidence_grounder_s2 import EvidenceGrounderS2
from ai_scientist.evidence_service import EvidenceService
from ai_scientist.model.evidence import S2SearchRoundRequest

def gather_evidence_for_unsupported_claims(
    project_id: str,
    min_support: int = 2,
    max_claims: int = 5
):
    """
    Find unsupported claims in the graph and automatically run S2 pipeline.
    
    Args:
        project_id: Project ID to search
        min_support: Minimum number of supports required
        max_claims: Maximum number of claims to process
    """
    engine = create_engine("sqlite:///ai_scientist.sqlite")
    claim_service = ClaimGraphService(engine)
    evidence_service = EvidenceService(engine)
    grounder = EvidenceGrounderS2(evidence_service)
    
    # Find unsupported claims
    unsupported = claim_service.find_unsupported_claims(project_id, min_support)
    
    print(f"Found {len(unsupported)} unsupported claims")
    
    # Process up to max_claims
    for i, claim_data in enumerate(unsupported[:max_claims]):
        claim_id = claim_data['claim_id']
        claim_text = claim_data['claim_text']
        
        print(f"\n{'='*80}")
        print(f"Processing claim {i+1}/{min(len(unsupported), max_claims)}")
        print(f"Claim ID: {claim_id}")
        print(f"Statement: {claim_text}")
        print(f"Current supports: {claim_data['support_count']}")
        print(f"{'='*80}\n")
        
        # Run S2 pipeline
        req = S2SearchRoundRequest(
            claim_id=claim_id,
            search_round_index=1,
            current_query=None,  # Will use claim text
            pubmed_retmax=10
        )
        
        try:
            result = grounder.orchestrate_round(req)
            
            print(f"\nS2 Pipeline Results:")
            print(f"  Next Action: {result.next_action}")
            print(f"  Supports Found: {len(result.new_support_ids)}")
            
            # Link S2 supports to claim graph
            if result.new_support_ids:
                node_id = claim_data['node_id']
                for support_id in result.new_support_ids:
                    claim_service.add_support(
                        node_id,
                        'claim_support',  # S2 evidence
                        support_id,
                        notes=f"Auto-gathered by S2 pipeline"
                    )
                print(f"  ✓ Linked {len(result.new_support_ids)} S2 supports to claim graph")
            
        except Exception as e:
            print(f"  ✗ Error running S2 pipeline: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Evidence gathering complete!")
    print(f"{'='*80}")

def link_existing_s2_evidence_to_graph(project_id: str):
    """
    Link existing S2 evidence (claim_support) to claim graph nodes.
    """
    engine = create_engine("sqlite:///ai_scientist.sqlite")
    claim_service = ClaimGraphService(engine)
    
    from sqlalchemy import text
    with engine.connect() as conn:
        # Find all claim_support records for this project
        results = conn.execute(
            text("""
                SELECT cs.support_id, cs.claim_id 
                FROM claim_support cs
                JOIN claim c ON cs.claim_id = c.claim_id
                WHERE c.project_id = :pid
            """),
            {"pid": project_id}
        ).fetchall()
        
        linked_count = 0
        for row in results:
            support_id = row.support_id
            claim_id = row.claim_id
            
            # Get node for this claim
            nodes = claim_service.get_nodes_for_claim(claim_id)
            if not nodes:
                print(f"Warning: No graph node for claim {claim_id}, skipping")
                continue
            
            node_id = nodes[0]['node_id']
            
            # Check if already linked
            existing = conn.execute(
                text("""
                    SELECT support_id FROM claim_graph_support 
                    WHERE node_id = :nid AND reference = :ref
                """),
                {"nid": node_id, "ref": support_id}
            ).fetchone()
            
            if not existing:
                claim_service.add_support(
                    node_id,
                    'claim_support',
                    support_id,
                    notes="Existing S2 evidence"
                )
                linked_count += 1
        
        print(f"✓ Linked {linked_count} existing S2 evidence records to claim graph")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated evidence gathering for claim graph")
    parser.add_argument("--project-id", required=True, help="Project ID")
    parser.add_argument("--min-support", type=int, default=2, help="Minimum supports required")
    parser.add_argument("--max-claims", type=int, default=5, help="Max claims to process")
    parser.add_argument("--link-existing", action="store_true", help="Link existing S2 evidence")
    
    args = parser.parse_args()
    
    if args.link_existing:
        link_existing_s2_evidence_to_graph(args.project_id)
    else:
        gather_evidence_for_unsupported_claims(
            args.project_id,
            args.min_support,
            args.max_claims
        )
