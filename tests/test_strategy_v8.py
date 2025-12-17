from ai_scientist.evidence_control_loop import calculate_ratios, calculate_failure_profile, decide_next_action_from_profile
from ai_scientist.model.evidence import RunFailureProfile

def test_pagination_exhaustion():
    # Start = Total Hits -> Exhausted
    summary = {
        "ingested_n": 0, "total_hits": 50, "n_results": 0, "retstart": 50
    }
    profile = calculate_failure_profile(summary, "query")
    assert profile.failure_mode == "EXHAUSTED_PAGINATION"
    
    # Check Logic: If T is high, it's a gap. If low, relax?
    # Case A: Low T (Unlikely if exhausted, but possible if prior pages were drift)
    profile.ratios = {"E_pmc": 0.0, "T": 0.0, "Y": 0.0}
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_RELAX_CONSTRAINTS"
    
    # Case B: High T (We saw good stuff but no entitlement)
    # Actually, if exhausted, profile ratios likely 0 unless we preserve them.
    # The profile calculcator re-calculates ratios from summary.
    # If n_results=0, summary might allow accumulating previous stats?
    # No, summary is PER ROUND usually.
    # But if we pretend we have high T from accumulation (orchestrator handles this usually)
    profile.ratios["T"] = 0.8
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "STOP_OPEN_GAP"
    print("test_pagination_exhaustion PASSED")

def test_starvation_split():
    # Case 1: Low E_pmc, Query has subset -> STARVATION_PMC_SUBSET
    summary = {
        "ingested_n": 100, "pmcid_eligible_n": 5, "total_hits": 1000, 
        "retstart": 0, "n_results": 100
    }
    query = '("atp"[MeSH]) AND "pubmed pmc"[sb]'
    profile = calculate_failure_profile(summary, query)
    print(f"DEBUG: Mode={profile.failure_mode}, Ratios={profile.ratios}")
    assert profile.failure_mode == "STARVATION_PMC_SUBSET"
    
    # Action check: Deep Starvation -> Switch
    summary["retstart"] = 1200 
    summary["total_hits"] = 2000 # > 1000 triggers switch
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "SWITCH_RETRIEVAL_STRATEGY_S2_FIRST"
    print("test_starvation_split PASSED")

def test_acquisition_phase():
    # E_pmc OK (0.2), but T Low (0.05) -> Precision Phase
    summary = {
        "ingested_n": 100, "pmcid_eligible_n": 20, "topic_pass_n": 1,
        "total_hits": 1000, "n_results": 100
    }
    profile = calculate_failure_profile(summary, "query")
    assert profile.failure_mode == "OFF_TOPIC_DRIFT"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_DISAMBIGUATE"
    print("test_acquisition_phase PASSED")

if __name__ == "__main__":
    test_pagination_exhaustion()
    test_starvation_split()
    test_acquisition_phase()
