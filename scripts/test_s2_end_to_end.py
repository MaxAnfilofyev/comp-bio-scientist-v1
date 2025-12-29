import sys
import os
import json
import logging

# Ensure path includes project root
sys.path.append(os.getcwd())

from ai_scientist.evidence.evidence_grounder_s2 import EvidenceGrounderS2
from ai_scientist.model.evidence import OrchestrateRoundRequest
from ai_scientist.evidence_service import EvidenceService

# Setup Logging
logging.basicConfig(level=logging.INFO)

def main():
    print("Starting S2 End-to-End Test...")
    
    # 1. Initialize
    try:
        grounder = EvidenceGrounderS2()
        svc = grounder.evidence_service
        
        # MOCK S2 for stability and to force a Positive Result for Promotion testing
        from unittest.mock import MagicMock
        from ai_scientist.evidence.s2_client import S2SearchResponse, S2PaperHit
        
        grounder.s2.search_relevance = MagicMock(return_value=S2SearchResponse(
            total=1, offset=0, hits=[
                S2PaperHit(
                    paper_id="TEST_PAPER_1",
                    rank=0,
                    title="SNAREs mediate fusion",
                    abstract="We confirm that VAMP727 (an R-SNARE) mediates membrane fusion with the vacuole.",
                    external_ids={"DOI": "10.1234/test.paper"},
                    year=2024,
                    venue="Test Journal",
                    is_open_access=True,
                    open_access_pdf_url="http://example.com/pdf",
                    tldr="SNAREs fuse membranes.",
                    publication_types=["JournalArticle"],
                    citation_count=100
                )
            ],
            compiled_query="SNARE fusion"
        ))
        
    except Exception as e:
        print(f"Failed to init grounder: {e}")
        return

    # 2. Setup Data (Need a Claim)
    # We will insert a dummy claim into DB directly to test against
    import uuid
    claim_id = f"TEST_CLAIM_{uuid.uuid4().hex[:8]}"
    project_id = "TEST_PROJ"
    
    print(f"Creating Test Claim: {claim_id}")
    with svc.engine.begin() as conn:
        from sqlalchemy import text
    print(f"Creating Test Claim: {claim_id}")
    with svc.engine.begin() as conn:
        from sqlalchemy import text
        # 1. Insert Policy Snapshot to satisfy FK
        conn.execute(
            text("INSERT OR IGNORE INTO policy_snapshot (policy_version, created_at, policy_yaml, policy_hash) VALUES (:pver, CURRENT_TIMESTAMP, 'dummy', 'dummy_hash')"),
            {"pver": "s2_strict_v1.0"}
        )
        
        # 2. Insert Claim
        conn.execute(
            text("""
                INSERT INTO claim (
                    claim_id, project_id, statement, claim_type, module, 
                    created_by, policy_version, created_at, updated_at
                ) VALUES (
                    :cid, :pid, :stmt, 'fact', 'test_module',
                    'test_script', 's2_strict_v1.0', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
            """),
            {"cid": claim_id, "pid": project_id, "stmt": "SNARE proteins mediate membrane fusion."}
        )

    # 3. Create Request
    req = OrchestrateRoundRequest(
        project_id=project_id,
        claim_id=claim_id,
        search_round_index=1,
        policy_id="s2_strict_v1.0",
        pubmed_retmax=3, # Small batch for testing
        ingest_candidates_cap=3
    )
    
    # 4. Run Round
    print("Running Orchestrate Round...")
    try:
        resp = grounder.orchestrate_round(req)
        print("Round Complete!")
        print(f"Round ID: {resp.round_id}")
        print(f"Done: {resp.done}")
        print(f"Supports Created: {resp.supports_created}")
        print(f"Next Action: {resp.next_action}")
        print("Summary JSON:", json.dumps(resp.summary_json, indent=2))
        
        if resp.candidate_results:
            print("\nCandidate Results:")
            for c in resp.candidate_results:
                print(f" - {c.candidate_id} ({c.doi}): Triage={c.topic_triage_passed}, Decision={c.decision_outcome}")
                
    except Exception as e:
        print(f"Orchestration Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
