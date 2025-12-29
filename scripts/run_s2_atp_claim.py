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
    print("=" * 80)
    print("S2 Evidence Pipeline - ATP Transport Claim")
    print("=" * 80)
    
    # 1. Initialize
    try:
        grounder = EvidenceGrounderS2()
        svc = grounder.evidence_service
    except Exception as e:
        print(f"Failed to init grounder: {e}")
        return

    # 2. Setup Claim
    import uuid
    claim_id = f"ATP_CLAIM_{uuid.uuid4().hex[:8]}"
    project_id = "ATP_PROJECT"
    claim_statement = "Active transport of ATP along axons involves diffusion mechanisms"
    
    print(f"\nClaim ID: {claim_id}")
    print(f"Statement: {claim_statement}")
    print(f"Project: {project_id}\n")
    
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
                    :cid, :pid, :stmt, 'mechanism', 'atp_transport',
                    'run_s2_pipeline', 's2_strict_v1.0', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
            """),
            {"cid": claim_id, "pid": project_id, "stmt": claim_statement}
        )

    # 3. Create Request
    req = OrchestrateRoundRequest(
        project_id=project_id,
        claim_id=claim_id,
        search_round_index=1,
        policy_id="s2_strict_v1.0",
        pubmed_retmax=10,  # Get 10 candidates
        ingest_candidates_cap=10
    )
    
    # 4. Run Round
    print("Running S2 Search & Evaluation...")
    print("-" * 80)
    try:
        resp = grounder.orchestrate_round(req)
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Round ID: {resp.round_id}")
        print(f"Search Run ID: {resp.search_run_id}")
        print(f"Done: {resp.done}")
        print(f"Next Action: {resp.next_action}")
        print(f"\nSupports Found: {resp.supports_found}")
        print(f"Supports Created: {len(resp.supports_created)}")
        
        if resp.supports_created:
            print("\nSupport IDs:")
            for sid in resp.supports_created:
                print(f"  - {sid}")
        
        print("\n" + "-" * 80)
        print("METRICS")
        print("-" * 80)
        print(json.dumps(resp.summary_json, indent=2))
        
        if resp.candidate_results:
            print("\n" + "-" * 80)
            print(f"CANDIDATE RESULTS ({len(resp.candidate_results)} total)")
            print("-" * 80)
            for i, c in enumerate(resp.candidate_results, 1):
                print(f"\n{i}. Candidate: {c.candidate_id}")
                print(f"   DOI: {c.doi}")
                print(f"   Hard Gate: {c.hard_gate_passed}")
                print(f"   Topic Triage: {c.topic_triage_passed}")
                print(f"   Decision: {c.decision_outcome}")
                if c.created_support_id:
                    print(f"   ✓ PROMOTED to Support: {c.created_support_id}")
                if c.mismatch_codes:
                    print(f"   Mismatch Codes: {', '.join(c.mismatch_codes)}")
        
        print("\n" + "=" * 80)
        print("Pipeline execution complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nOrchestration Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
