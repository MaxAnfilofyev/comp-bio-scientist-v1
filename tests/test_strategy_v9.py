from ai_scientist.evidence_control_loop import calculate_ratios, calculate_failure_profile, decide_next_action_from_profile
from ai_scientist.model.evidence import RunFailureProfile

def test_binding_failure():
    # Case: E_pmc Low, High Gate Failures for Linking
    summary = {
        "ingested_n": 100, "pmcid_eligible_n": 5, "total_hits": 1000, 
        "retstart": 0, "n_results": 100,
        "gate_failure_counts": {
            "PMID_TO_PMCID_LINKED": 95, # 95 failures out of 100
            "IN_PMC_FULLTEXT": 95
        }
    }
    query = "foo"
    profile = calculate_failure_profile(summary, query)
    
    print(f"DEBUG: Mode={profile.failure_mode}, Desc={profile.description}")
    assert profile.failure_mode == "BINDING_FAILURE_PMC_LINKING"
    
    # Action check: Binding Failure -> Switch
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "SWITCH_RETRIEVAL_STRATEGY_S2_FIRST"
    print("test_binding_failure PASSED")

if __name__ == "__main__":
    test_binding_failure()
