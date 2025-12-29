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
    print("S2 Evidence Pipeline - SNARE Proteins Claim")
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
    claim_id = f"METAB_CLAIM_{uuid.uuid4().hex[:8]}"
    project_id = "METAB_PROJECT"
    claim_statement = "Regions with reduced local ATP supply are more vulnerable under metabolic stress."
    
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
                    :cid, :pid, :stmt, 'mechanism', 'snare_fusion',
                    'run_s2_snare', 's2_strict_v1.0', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
            """),
            {"cid": claim_id, "pid": project_id, "stmt": claim_statement}
        )

    # 3. Multi-Round Control Loop
    max_rounds = 5
    current_query = None  # Start with None to use claim statement
    all_supports = []
    
    for round_idx in range(1, max_rounds + 1):
        print("\n" + "=" * 80)
        print(f"ROUND {round_idx}")
        print("=" * 80)
        
        # Create Request
        req = OrchestrateRoundRequest(
            project_id=project_id,
            claim_id=claim_id,
            search_round_index=round_idx,
            policy_id="s2_strict_v1.0",
            pubmed_retmax=10,
            ingest_candidates_cap=10,
            max_rounds=max_rounds,
            current_query=current_query
        )
        
        print(f"Query: {current_query or claim_statement}")
        print("-" * 80)
        
        # Execute Round
        try:
            resp = grounder.orchestrate_round(req)
            
            # Display Results
            print(f"\nRound {round_idx} Complete:")
            print(f"  Search Run ID: {resp.search_run_id}")
            print(f"  Next Action: {resp.next_action}")
            print(f"  Supports Found: {resp.supports_found}")
            
            # Collect supports
            if resp.supports_created:
                all_supports.extend(resp.supports_created)
                print(f"  ✓ New Supports: {len(resp.supports_created)}")
            
            # Show metrics
            metrics = resp.summary_json
            print(f"\n  Metrics:")
            print(f"    Ingested: {metrics.get('ingested_n', 0)}")
            print(f"    Eligible: {metrics.get('eligible_n', 0)}")
            print(f"    Topic Pass: {metrics.get('topic_pass_n', 0)} (T={metrics.get('T', 0):.2f})")
            print(f"    Entailment Pass: {metrics.get('entailment_pass_n', 0)} (Y={metrics.get('Y', 0):.2f})")
            
            if metrics.get('mismatch_code_counts'):
                print(f"    Mismatch Codes: {metrics['mismatch_code_counts']}")
            
            # Show candidate summary
            if resp.candidate_results:
                rejected = sum(1 for c in resp.candidate_results if c.decision_outcome == "REJECTED")
                promoted = sum(1 for c in resp.candidate_results if c.created_support_id)
                print(f"\n  Candidates: {len(resp.candidate_results)} total ({rejected} rejected, {promoted} promoted)")
            
            # Check termination conditions
            terminal_actions = ["STOP_OPEN_GAP", "STOP_MAX_ROUNDS"]
            
            if resp.done or resp.next_action in terminal_actions:
                print(f"\n  → Pipeline terminating: {resp.next_action}")
                break
            
            # Handle continuation actions
            if resp.next_action in ["REWRITE_QUERY_DISAMBIGUATE", "REWRITE_QUERY_BROADEN"]:
                if resp.next_query:
                    print(f"\n  → Query will be rewritten for next round")
                    print(f"     New query: {resp.next_query[:100]}...")
                    current_query = resp.next_query
                else:
                    print(f"\n  → Query rewrite requested but no new query provided, stopping")
                    break
                    
            elif resp.next_action == "CONTINUE_PAGINATION":
                print(f"\n  → Continuing with pagination")
                # Keep same query, just continue
                
            else:
                print(f"\n  → Continuing with action: {resp.next_action}")
                if resp.next_query:
                    current_query = resp.next_query
                    
        except Exception as e:
            print(f"\nRound {round_idx} Failed: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Rounds Executed: {round_idx}")
    print(f"Total Supports Created: {len(all_supports)}")
    
    if all_supports:
        print(f"\nSupport IDs:")
        for sid in all_supports:
            print(f"  - {sid}")
    else:
        print("\nNo evidence found supporting the claim.")
    
    print(f"\nResults stored in database under claim: {claim_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()
