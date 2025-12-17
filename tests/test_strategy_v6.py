from ai_scientist.evidence_control_loop import calculate_ratios, calculate_failure_profile, decide_next_action_from_profile
from ai_scientist.model.evidence import RunFailureProfile

def test_ratios():
    summary = {
        "ingested_n": 100, 
        "pmcid_eligible_n": 20, 
        "topic_pass_n": 10, 
        "supports_found_n": 1,
        "entailment_judged_n": 5
    }
    r = calculate_ratios(summary)
    assert r["E"] == 0.2
    assert r["T"] == 0.5
    assert r["Y"] == 0.2
    print("test_ratios PASSED")

def test_anomaly():
    # Query has "pubmed pmc", hits > 10, but E < 0.1
    summary = {
        "total_hits": 100, "ingested_n": 50, "pmcid_eligible_n": 0
    }
    query = "foo AND \"pubmed pmc\"[sb]"
    profile = calculate_failure_profile(summary, query)
    assert profile.eligibility_anomaly == True
    assert profile.failure_mode == "STARVATION" # or Anomaly flag handled in dispatch
    
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    # Page 1 anomaly -> Pagination for Page 2
    assert action == "CONTINUE_PAGINATION" 
    print("test_anomaly Page 1 PASSED")

def test_drift_dispatch():
    # E ok (0.5), T low (0.0), Y (0.0)
    summary = {
        "ingested_n": 50, "pmcid_eligible_n": 25, "topic_pass_n": 0, "total_hits": 100
    }
    profile = calculate_failure_profile(summary, "query")
    assert profile.failure_mode == "OFF_TOPIC_DRIFT"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_DISAMBIGUATE"
    print("test_drift_dispatch PASSED")

def test_gap_dispatch():
    # E ok, T ok, Y low (0.0)
    summary = {
        "ingested_n": 50, "pmcid_eligible_n": 25, "topic_pass_n": 25, 
        "entailment_judged_n": 10, "supports_found_n": 0,
        "total_hits": 1000, "retstart": 0
    }
    profile = calculate_failure_profile(summary, "query")
    assert profile.failure_mode == "EVIDENCE_GAP"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "CONTINUE_PAGINATION" # Recall first
    print("test_gap_dispatch PASSED")

if __name__ == "__main__":
    test_ratios()
    test_anomaly()
    test_drift_dispatch()
    test_gap_dispatch()
