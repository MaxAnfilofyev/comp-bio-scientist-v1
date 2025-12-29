"""
Test to verify next_action logic when eligible_n = 0
"""
import sys
sys.path.append(".")

from ai_scientist.evidence.evidence_grounder_s2 import EvidenceGrounderS2
from ai_scientist.model.evidence import OrchestrateRoundRequest
from ai_scientist.evidence.s2_client import S2SearchResponse, S2PaperHit

def main():
    grounder = EvidenceGrounderS2()
    svc = grounder.evidence_service
    
    # Mock S2 to return papers WITHOUT abstracts (eligible_n will be 0)
    def mock_no_abstracts(req):
        return S2SearchResponse(
            total=5, offset=0, hits=[
                S2PaperHit(
                    paper_id=f"NO_ABSTRACT_{i}",
                    rank=i,
                    title=f"Paper {i} with no abstract",
                    abstract=None,  # No abstract!
                    external_ids={"DOI": f"10.1234/noabs{i}"},
                    year=2020,
                    venue="Test",
                    tldr=None,
                    publication_types=["JournalArticle"],
                    citation_count=10
                )
                for i in range(3)
            ],
            compiled_query=req.query
        )
    
    grounder.s2.search_relevance = mock_no_abstracts
    
    # Setup claim
    import uuid
    from sqlalchemy import text
    claim_id = f"TEST_ZERO_ELIGIBLE_{uuid.uuid4().hex[:8]}"
    
    with svc.engine.begin() as conn:
        conn.execute(text("INSERT OR IGNORE INTO policy_snapshot (policy_version, created_at, policy_yaml, policy_hash) VALUES ('test', CURRENT_TIMESTAMP, 'dummy', 'hash')"))
        conn.execute(text("""
            INSERT INTO claim (claim_id, project_id, statement, claim_type, module, created_by, policy_version, created_at, updated_at)
            VALUES (:cid, 'TEST', 'Test claim', 'fact', 'test', 'test', 'test', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """), {"cid": claim_id})
    
    # Run rounds
    print("Testing next_action logic when eligible_n = 0")
    print("=" * 80)
    
    current_query = None  # Track query across rounds
    
    for round_idx in range(1, 4):
        print(f"\nROUND {round_idx}")
        print("-" * 80)
        print(f"Query: {current_query or 'Test claim'}")
        
        req = OrchestrateRoundRequest(
            project_id="TEST",
            claim_id=claim_id,
            search_round_index=round_idx,
            policy_id="test",
            pubmed_retmax=5,
            max_rounds=5,
            current_query=current_query
        )
        
        resp = grounder.orchestrate_round(req)
        
        print(f"Ingested: {resp.summary_json['ingested_n']}")
        print(f"Eligible: {resp.summary_json['eligible_n']}")
        print(f"Next Action: {resp.next_action}")
        if resp.next_query:
            print(f"Next Query: {resp.next_query[:80]}...")
        
        if round_idx == 1:
            assert resp.next_action == "REWRITE_QUERY_BROADEN", f"Round 1 should BROADEN when eligible=0, got {resp.next_action}"
            print("✓ Correctly chose BROADEN for round 1")
        elif round_idx == 2:
            assert resp.next_action == "REWRITE_QUERY_BROADEN", f"Round 2 should BROADEN when eligible=0, got {resp.next_action}"
            print("✓ Correctly chose BROADEN for round 2")
        elif round_idx == 3:
            assert resp.next_action == "STOP_OPEN_GAP", f"Round 3 should STOP when eligible=0, got {resp.next_action}"
            print("✓ Correctly chose STOP_OPEN_GAP for round 3")
            break
        
        # Update query for next round
        if resp.next_query:
            current_query = resp.next_query
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
