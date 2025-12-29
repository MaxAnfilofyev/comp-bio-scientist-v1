import sys
import os
sys.path.append(os.getcwd())

from ai_scientist.evidence.evidence_grounder_s2 import EvidenceGrounderS2
from ai_scientist.model.evidence import OrchestrateRoundRequest
from ai_scientist.evidence.s2_client import S2SearchResponse, S2PaperHit
from unittest.mock import MagicMock

def main():
    print("Testing Query Rewrite with Mocked S2")
    print("=" * 80)
    
    grounder = EvidenceGrounderS2()
    svc = grounder.evidence_service
    
    # Mock S2 to return different off-topic papers each round
    round_counter = [0]  # Use list to allow mutation in closure
    
    def mock_search(req):
        round_counter[0] += 1
        return S2SearchResponse(
            total=1, offset=0, hits=[
                S2PaperHit(
                    paper_id=f"MOCK_PAPER_{round_counter[0]}",
                    rank=0,
                    title=f"Review of cellular transport mechanisms (Round {round_counter[0]})",
                    abstract=f"General review of transport in cells. Round {round_counter[0]} result.",
                    external_ids={"DOI": f"10.1234/mock.round{round_counter[0]}"},
                    year=2020 + round_counter[0],
                    venue="Mock Reviews",
                    tldr=f"Transport review {round_counter[0]}",
                    publication_types=["Review"],
                    citation_count=50 + round_counter[0] * 10
                )
            ],
            compiled_query=req.query  # Use the actual query sent
        )
    
    grounder.s2.search_relevance = mock_search
    
    # Setup claim
    import uuid
    claim_id = f"TEST_{uuid.uuid4().hex[:8]}"
    
    with svc.engine.begin() as conn:
        from sqlalchemy import text
        conn.execute(text("INSERT OR IGNORE INTO policy_snapshot (policy_version, created_at, policy_yaml, policy_hash) VALUES ('test_pol', CURRENT_TIMESTAMP, 'dummy', 'hash')"))
        conn.execute(text("""
            INSERT INTO claim (claim_id, project_id, statement, claim_type, module, created_by, policy_version, created_at, updated_at)
            VALUES (:cid, 'TEST_PROJ', 'Active transport of ATP along axons involves diffusion mechanisms', 'mechanism', 'test', 'test', 'test_pol', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """), {"cid": claim_id})
    
    # Run 3 rounds
    current_query = None
    for round_idx in range(1, 4):
        print(f"\n{'='*80}")
        print(f"ROUND {round_idx}")
        print(f"{'='*80}")
        print(f"Query: {current_query or 'Active transport of ATP along axons involves diffusion mechanisms'}")
        
        req = OrchestrateRoundRequest(
            project_id="TEST_PROJ",
            claim_id=claim_id,
            search_round_index=round_idx,
            policy_id="test_pol",
            pubmed_retmax=5,
            max_rounds=3,
            current_query=current_query
        )
        
        resp = grounder.orchestrate_round(req)
        
        print(f"\nNext Action: {resp.next_action}")
        print(f"Next Query: {resp.next_query}")
        print(f"Metrics: T={resp.summary_json.get('T', 0):.2f}")
        
        if resp.next_query and resp.next_query != current_query:
            print(f"\n✓ QUERY CHANGED!")
            print(f"  Old: {current_query or 'None'}")
            print(f"  New: {resp.next_query[:150]}...")
            current_query = resp.next_query
        else:
            print(f"\n✗ Query did not change")
            if resp.next_action == "REWRITE_QUERY_DISAMBIGUATE":
                print("  ERROR: Rewrite requested but query unchanged!")
                break
        
        if resp.next_action in ["STOP_MAX_ROUNDS", "STOP_OPEN_GAP"]:
            break
    
    print(f"\n{'='*80}")
    print("Test Complete")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
