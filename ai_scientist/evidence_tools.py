
import json
from typing import Optional
from agents import function_tool
from ai_scientist.evidence_service import (
    EvidenceService,
    CreateSearchRunRequest, Provider, PolicyRef,
    UpsertWorkBatchRequest, WorkUpsert,
    IngestCandidatesRequest, CandidateIngestItem,
    RecordQualityCheckRequest, CheckType, Verdict, ActorRef,
    RecordDecisionRequest, DecisionOutcome,
    GetEffectiveStatusRequest, PromoteCandidateRequest
)

# Global service instance
_SERVICE = EvidenceService()

@function_tool
def create_search_run(
    project_id: str,
    provider: str,
    query_template_id: str,
    query_text: str,
    policy_id: str,
    filters_json: str = "{}",
    result_count_total: int = 0,
    top_k_stored: int = 0,
    provider_cursor_json: str = "{}",
    notes: Optional[str] = None,
    policy_hash: Optional[str] = None
) -> str:
    """
    Records a new search run.
    """
    req = CreateSearchRunRequest(
        project_id=project_id,
        provider=Provider(provider),
        query_template_id=query_template_id,
        query_text=query_text,
        filters_json=json.loads(filters_json),
        policy=PolicyRef(policy_id=policy_id, policy_hash=policy_hash),
        result_count_total=result_count_total,
        top_k_stored=top_k_stored,
        provider_cursor_json=json.loads(provider_cursor_json),
        notes=notes
    )
    resp = _SERVICE.create_search_run(req)
    return resp.json()

@function_tool
def upsert_work_batch(works_json: str) -> str:
    """
    Upserts a batch of works.
    works_json must be a JSON list of WorkUpsert objects.
    """
    works_data = json.loads(works_json)
    works = [WorkUpsert(**w) for w in works_data]
    req = UpsertWorkBatchRequest(works=works)
    resp = _SERVICE.upsert_work_batch(req)
    return resp.json()

@function_tool
def ingest_candidates(
    search_run_id: str,
    candidates_json: str,
    enforce_max_n: int = 500
) -> str:
    """
    Ingests candidates for a search run.
    candidates_json must be a JSON list of CandidateIngestItem objects.
    """
    cands_data = json.loads(candidates_json)
    candidates = [CandidateIngestItem(**c) for c in cands_data]
    req = IngestCandidatesRequest(
        search_run_id=search_run_id,
        candidates=candidates,
        enforce_max_n=enforce_max_n
    )
    resp = _SERVICE.ingest_candidates(req)
    return resp.json()

@function_tool
def record_quality_check(
    candidate_id: str,
    check_type: str,
    verdict: str,
    policy_id: str,
    executed_by_agent_id: str,
    claim_id: Optional[str] = None,
    details_json: str = "{}",
    policy_hash: Optional[str] = None,
    executed_by_role: Optional[str] = None
) -> str:
    """
    Records a quality check for a candidate.
    """
    req = RecordQualityCheckRequest(
        candidate_id=candidate_id,
        claim_id=claim_id,
        check_type=CheckType(check_type),
        verdict=Verdict(verdict),
        details_json=json.loads(details_json),
        policy=PolicyRef(policy_id=policy_id, policy_hash=policy_hash),
        executed_by=ActorRef(agent_id=executed_by_agent_id, role=executed_by_role)
    )
    resp = _SERVICE.record_quality_check(req)
    return resp.json()

@function_tool
def record_decision(
    candidate_id: str,
    outcome: str,
    policy_id: str,
    decided_by_agent_id: str,
    claim_id: Optional[str] = None,
    basis_json: str = "{}",
    policy_hash: Optional[str] = None,
    decided_by_role: Optional[str] = None
) -> str:
    """
    Records a decision for a candidate.
    """
    req = RecordDecisionRequest(
        candidate_id=candidate_id,
        claim_id=claim_id,
        outcome=DecisionOutcome(outcome),
        basis_json=json.loads(basis_json),
        policy=PolicyRef(policy_id=policy_id, policy_hash=policy_hash),
        decided_by=ActorRef(agent_id=decided_by_agent_id, role=decided_by_role)
    )
    resp = _SERVICE.record_decision(req)
    return resp.json()

@function_tool
def get_candidate_effective_status(candidate_id: str, claim_id: Optional[str] = None) -> str:
    """
    Gets the effective status of a candidate.
    """
    req = GetEffectiveStatusRequest(candidate_id=candidate_id, claim_id=claim_id)
    resp = _SERVICE.get_candidate_effective_status(req)
    return resp.json()

@function_tool
def promote_candidate_to_support(
    claim_id: str,
    candidate_id: str,
    support_type: str,
    promotion_reason: str,
    created_by: str,
    policy_version: str
) -> str:
    """
    Promotes a candidate to claim support.
    """
    req = PromoteCandidateRequest(
        claim_id=claim_id,
        candidate_id=candidate_id,
        support_type=support_type,
        promotion_reason=promotion_reason,
        created_by=created_by,
        policy_version=policy_version
    )
    resp = _SERVICE.promote_candidate_to_support(req)
    return resp.json()
