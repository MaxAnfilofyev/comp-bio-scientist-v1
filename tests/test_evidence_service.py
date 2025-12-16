
import os
import re
import json
import pytest
from datetime import datetime
from sqlalchemy import create_engine, text
from ai_scientist.evidence_service import (
    EvidenceService,
    CreateSearchRunRequest, Provider, PolicyRef,
    UpsertWorkBatchRequest, WorkUpsert,
    IngestCandidatesRequest, CandidateIngestItem,
    RecordQualityCheckRequest, CheckType, Verdict, ActorRef,
    RecordDecisionRequest, DecisionOutcome,
    GetEffectiveStatusRequest, PromoteCandidateRequest
)

# Path to migration file to extract DDL
MIGRATION_FILE = "ai_scientist/database/migrations/versions/27707676918b_initial_schema.py"

def get_ddl():
    with open(MIGRATION_FILE, "r") as f:
        content = f.read()
    
    # Simple extraction of the sql string inside upgrade()
    # Looking for sql = """ ... """
    match = re.search(r'sql = """(.*?)"""', content, re.DOTALL)
    if not match:
        raise ValueError("Could not extract SQL from migration file")
    return match.group(1)

@pytest.fixture
def db_url(tmp_path):
    db_file = tmp_path / "test_evidence.db"
    return f"sqlite:///{db_file}"

@pytest.fixture
def service(db_url):
    # Setup DB
    engine = create_engine(db_url)
    ddl = get_ddl()
    
    # Split DDL? No, executescript handles multiple statements if sqlite3 driver is used.
    # SQLAlchemy execute with text might trigger multiple statements issue if not properly handled
    # but the migration file says `raw_conn.executescript(sql)`.
    # We should do the same.
    
    with engine.connect() as conn:
        # Get raw connection
        raw_conn = conn.connection
        raw_conn.executescript(ddl)
        
    svc = EvidenceService(db_url=db_url)
    return svc

def test_evidence_service_flow(service):
    # 1. Create Policy Snapshot (FK requirement)
    policy_id = "test_policy_v1"
    with service.engine.begin() as conn:
        conn.execute(
            text("INSERT INTO policy_snapshot (policy_version, created_at, policy_yaml, policy_hash) VALUES (:v, :c, :y, :h)"),
            {"v": policy_id, "c": "now", "y": "yaml", "h": "hash"}
        )
        # Create Project (FK requirement)
        conn.execute(
            text("INSERT INTO project (project_id, title, status, created_at, updated_at, policy_version) VALUES (:pid, :t, :s, :c, :u, :pv)"),
            {"pid": "proj1", "t": "Test Project", "s": "active", "c": "now", "u": "now", "pv": policy_id}
        )

    # 2. Create Search Run
    req_run = CreateSearchRunRequest(
        project_id="proj1",
        provider=Provider.PMC,
        query_template_id="q1",
        query_text="test query",
        policy=PolicyRef(policy_id=policy_id),
        result_count_total=100
    )
    resp_run = service.create_search_run(req_run)
    assert resp_run.search_run_id.startswith("SR_")

    # 3. Upsert Work
    doi = "10.1234/test"
    req_work = UpsertWorkBatchRequest(
        works=[WorkUpsert(doi=doi, title="Test Work", year=2024)]
    )
    resp_works = service.upsert_work_batch(req_work)
    assert resp_works.normalized_dois == [doi]

    # 4. Ingest Candidates
    req_ingest = IngestCandidatesRequest(
        search_run_id=resp_run.search_run_id,
        candidates=[
            CandidateIngestItem(doi=doi, rank_in_results=1, retrieval_score_raw=0.9)
        ]
    )
    resp_ingest = service.ingest_candidates(req_ingest)
    assert len(resp_ingest.candidate_ids) == 1
    candidate_id = resp_ingest.candidate_ids[0]

    # 5. Record Quality Check
    actor = ActorRef(agent_id="test_agent", role="tester")
    req_check = RecordQualityCheckRequest(
        candidate_id=candidate_id,
        check_type=CheckType.has_doi,
        verdict=Verdict.PASS,
        policy=PolicyRef(policy_id=policy_id),
        executed_by=actor
    )
    resp_check = service.record_quality_check(req_check)
    assert resp_check.check_id.startswith("QC_")

    # 6. Record Decision
    req_dec = RecordDecisionRequest(
        candidate_id=candidate_id,
        outcome=DecisionOutcome.PROMOTED,
        policy=PolicyRef(policy_id=policy_id),
        decided_by=actor
    )
    resp_dec = service.record_decision(req_dec)
    assert resp_dec.decision_id.startswith("DEC_")

    # 7. Get Effective Status
    resp_status = service.get_candidate_effective_status(
        GetEffectiveStatusRequest(candidate_id=candidate_id)
    )
    assert resp_status.status.candidate_id == candidate_id
    assert resp_status.status.latest_decision["outcome"] == DecisionOutcome.PROMOTED.value
    assert resp_status.status.latest_checks[CheckType.has_doi]["verdict"] == Verdict.PASS.value

    # 8. Promote to Support (Claim)
    # create claim first
    claim_id = "clm1"
    with service.engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO claim (
                    claim_id, project_id, module, statement, claim_type,
                    created_by, created_at, updated_at, policy_version
                ) VALUES (
                    :cid, :pid, :mod, :stmt, :ctype,
                    :cby, :now, :now, :pv
                )
            """),
            {
                "cid": claim_id, "pid": "proj1", "mod": "mod1", "stmt": "stmt1", "ctype": "fact",
                "cby": "me", "now": "now", "pv": policy_id
            }
        )
    
    req_prom = PromoteCandidateRequest(
        claim_id=claim_id,
        candidate_id=candidate_id,
        support_type="citation",
        promotion_reason="looks good",
        created_by="me",
        policy_version=policy_id
    )
    resp_prom = service.promote_candidate_to_support(req_prom)
    assert resp_prom.support_id.startswith("SUP_")

if __name__ == "__main__":
    # Allow running directly
    # Need to setup temp DB manually if running main
    pass
