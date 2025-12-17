from ai_scientist.evidence_control_loop import calculate_ratios, calculate_failure_profile, decide_next_action_from_profile
from ai_scientist.model.evidence import RunFailureProfile

def test_split_eligibility_metrics():
    # Scenario: 100 ingested, 20 passed PMC gate, 50 passed DOI gate.
    # pmcid_eligible_n = 20 (assuming intersection)
    # But wait, logic says E_pmc = pmcid_eligible_n / ingested
    summary = {
        "ingested_n": 100, 
        "pmcid_eligible_n": 20, # This is the "survived everything" number
        "topic_pass_n": 10,
        "supports_found_n": 0,
        "gate_failure_counts": {"IN_PMC_FULLTEXT": 80} 
    }
    r = calculate_ratios(summary)
    assert r["E_pmc"] == 0.2
    assert r["T"] == 0.5 # 10/20
    print("test_split_eligibility_metrics PASSED")

def test_strategy_switch_dispatch():
    # E_pmc low (0.05), Total Hits High (>1000)
    summary = {
        "ingested_n": 100, "pmcid_eligible_n": 5, "total_hits": 2000, 
        "retstart": 0
    }
    profile = calculate_failure_profile(summary, "query")
    assert profile.failure_mode == "STARVATION"
    assert profile.pmcid_rate == 0.05
    
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    # Start -> Page first
    assert action == "CONTINUE_PAGINATION"
    
    # If deep page -> Change Strategy
    summary["retstart"] = 450
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "CHANGE_RETRIEVAL_STRATEGY"
    print("test_strategy_switch_dispatch PASSED")

def test_precision_stage():
    # E_pmc OK (0.5), T Low (0.05)
    summary = {
        "ingested_n": 100, "pmcid_eligible_n": 50, "topic_pass_n": 2,
        "total_hits": 100
    }
    profile = calculate_failure_profile(summary, "query")
    assert profile.failure_mode == "OFF_TOPIC_DRIFT"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_DISAMBIGUATE" # Prompt should force positive anchors
    print("test_precision_stage PASSED")

if __name__ == "__main__":
    test_split_eligibility_metrics()
    test_strategy_switch_dispatch()
    test_precision_stage()
