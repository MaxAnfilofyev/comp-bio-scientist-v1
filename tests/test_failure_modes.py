from ai_scientist.evidence_control_loop import calculate_failure_profile, decide_next_action_from_profile
from ai_scientist.model.evidence import RunFailureProfile

def test_over_constrained():
    summary = {"total_hits": 0, "supports_found_n": 0}
    profile = calculate_failure_profile(summary)
    assert profile.failure_mode == "OVER_CONSTRAINED"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_RELAX_CONSTRAINTS"
    assert mode == "RELAX"
    print("test_over_constrained PASSED")

def test_starvation():
    summary = {
        "total_hits": 100, "pmcid_eligible_n": 0, 
        "supports_found_n": 0, "retstart": 0
    }
    profile = calculate_failure_profile(summary)
    assert profile.failure_mode == "STARVATION"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "CONTINUE_PAGINATION" # shallow depth
    assert mode == "PAGINATE"
    
    # Deep depth -> Relax
    summary["retstart"] = 400
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_RELAX_CONSTRAINTS"
    assert mode == "RELAX"
    print("test_starvation PASSED")

def test_off_topic():
    summary = {
        "total_hits": 50, "pmcid_eligible_n": 10, 
        "topic_pass_n": 0, "supports_found_n": 0,
        "mismatch_code_counts": {"WRONG_ENTITY": 5}
    }
    profile = calculate_failure_profile(summary)
    assert profile.failure_mode == "OFF_TOPIC_DRIFT"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_DISAMBIGUATE"
    assert mode == "DISAMBIGUATE"
    print("test_off_topic PASSED")

def test_evidence_gap():
    summary = {
        "total_hits": 50, "pmcid_eligible_n": 10,
        "topic_pass_n": 5, "supports_found_n": 0,
        "retstart": 400 # exhausted page
    }
    profile = calculate_failure_profile(summary)
    assert profile.failure_mode == "EVIDENCE_GAP"
    action, _, mode = decide_next_action_from_profile(profile, summary, 0, 5)
    assert action == "REWRITE_QUERY_TIGHTEN"
    assert mode == "TIGHTEN"
    print("test_evidence_gap PASSED")

if __name__ == "__main__":
    test_over_constrained()
    test_starvation()
    test_off_topic()
    test_evidence_gap()
