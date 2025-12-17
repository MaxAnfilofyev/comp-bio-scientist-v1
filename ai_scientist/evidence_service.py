from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, constr
from sqlalchemy import create_engine, text

from ai_scientist.database.config import DATABASE_URL


# ---------- Enums ----------

class Provider(str, Enum):
    PUBMED = "PUBMED"
    PMC = "PMC"
    CORE = "CORE"
    SEMANTIC_SCHOLAR = "SEMANTIC_SCHOLAR"
    CROSSREF = "CROSSREF"


class CheckType(str, Enum):
    has_doi = "has_doi"
    in_pmc_fulltext = "in_pmc_fulltext"
    preprint_policy = "preprint_policy"
    journal_whitelist = "journal_whitelist"
    RETRACTION_CHECK = "retraction_check"
    EOC_CHECK = "eoc_check"
    TRUSTED_TYPE_CHECK = "trusted_type_check"
    MIN_CITATIONS_CHECK = "min_citations_check"
    SUPPORTS_CLAIM_CHECK = "supports_claim_check"
    ANCHOR_EXTRACTION = "anchor_extraction"
    CLAIM_ENTAILMENT_LLM = "claim_entailment_llm"
    MANUAL_REVIEW = "manual_review"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NA = "NA"


class DecisionOutcome(str, Enum):
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    HOLD = "HOLD"
    ELIGIBLE_SUPPORT = "ELIGIBLE_SUPPORT"
    SELECTED_AS_SUPPORT = "SELECTED_AS_SUPPORT"


# ---------- Common ----------


class PolicyRef(BaseModel):
    policy_id: str = Field(..., examples=["pmc_strict_v1.2.0"])
    policy_hash: Optional[str] = Field(None, description="sha256 of canonical policy text")


class ActorRef(BaseModel):
    agent_id: str = Field(..., examples=["archivist", "archivist_v2", "pi"])
    role: Optional[str] = Field(None, examples=["archivist", "reviewer", "pi"])


# ---------- Search run ----------

class CreateSearchRunRequest(BaseModel):
    project_id: str
    provider: Provider
    query_template_id: str
    query_text: str
    filters_json: dict[str, Any] = Field(default_factory=dict)
    policy: PolicyRef
    result_count_total: int = 0
    top_k_stored: int = 0
    provider_cursor_json: dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class CreateSearchRunResponse(BaseModel):
    search_run_id: str
    created_at: datetime


# ---------- Work ----------

class WorkUpsert(BaseModel):
    doi: constr(min_length=3, max_length=255)
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    publisher: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    indexing_json: dict[str, Any] = Field(default_factory=dict)


class UpsertWorkBatchRequest(BaseModel):
    works: list[WorkUpsert]


class UpsertWorkBatchResponse(BaseModel):
    normalized_dois: list[str]


# ---------- Candidate ingestion ----------

class CandidateIngestItem(BaseModel):
    doi: constr(min_length=3, max_length=255)
    rank_in_results: int
    retrieval_score_raw: Optional[float] = None
    features_json: dict[str, Any] = Field(default_factory=dict)
    composite_score: Optional[float] = None
    scoring_version: str = "v1"


class IngestCandidatesRequest(BaseModel):
    search_run_id: str
    candidates: list[CandidateIngestItem]
    enforce_max_n: int = 500  # safety guardrail


class IngestCandidatesResponse(BaseModel):
    candidate_ids: list[str]


# Note: `Base` for `WorkFulltextCache` is assumed to be defined elsewhere, e.g., `from sqlalchemy.ext.declarative import declarative_base; Base = declarative_base()`
# ---------- FullText Cache ----------

class StoreFulltextRequest(BaseModel):
    doi: str
    pmcid: Optional[str] = None
    source: str = "PMC"
    format: str = "JATS_XML"
    content: str
    license: Optional[str] = None

class StoreFulltextResponse(BaseModel):
    doi: str
    content_hash: str
    stored_at: str


# ---------- Checks ----------

class RecordQualityCheckRequest(BaseModel):
    candidate_id: str
    claim_id: Optional[str] = None
    check_type: CheckType
    verdict: Verdict
    details_json: dict[str, Any] = Field(default_factory=dict)
    policy: PolicyRef
    executed_by: ActorRef


class RecordQualityCheckResponse(BaseModel):
    check_id: str
    executed_at: datetime


# ---------- Decisions ----------

class RecordDecisionRequest(BaseModel):
    candidate_id: str
    claim_id: Optional[str] = None
    outcome: DecisionOutcome
    basis_json: dict[str, Any] = Field(
        default_factory=dict,
        description="Pointers to checks + short rationale"
    )
    policy: PolicyRef
    decided_by: ActorRef


class RecordDecisionResponse(BaseModel):
    decision_id: str
    decided_at: datetime


# ---------- Effective status ----------

class CandidateEffectiveStatus(BaseModel):
    candidate_id: str
    claim_id: Optional[str] = None
    latest_decision: Optional[dict[str, Any]] = None
    latest_checks: dict[CheckType, dict[str, Any]] = Field(default_factory=dict)


class GetEffectiveStatusRequest(BaseModel):
    candidate_id: str
    claim_id: Optional[str] = None


class GetEffectiveStatusResponse(BaseModel):
    status: CandidateEffectiveStatus


# ---------- FullText Cache ----------

class StoreFulltextRequest(BaseModel):
    doi: str
    pmcid: Optional[str] = None
    source: str = "PMC"
    format: str = "JATS_XML"
    content: str
    license: Optional[str] = None

class StoreFulltextResponse(BaseModel):
    doi: str
    content_hash: str
    stored_at: str


# ---------- Promotion ----------

class PromoteCandidateRequest(BaseModel):
    claim_id: str
    candidate_id: str
    support_type: str = Field(..., examples=["citation", "primary_measurement"])
    promotion_reason: str
    created_by: str
    policy_version: str
    anchor_excerpt: Optional[str] = None
    anchor_location_json: Optional[dict[str, Any]] = None
    verification_status: str = "unverified"


class PromoteCandidateResponse(BaseModel):
    support_id: str


# ---------- Service Implementation ----------

class EvidenceService:
    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(db_url)

    def _get_utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _normalize_doi(self, doi: str) -> str:
        # Lowercase, strip https://doi.org/, trim spaces
        d = doi.lower().strip()
        if d.startswith("https://doi.org/"):
            d = d[len("https://doi.org/"):]
        return d

    def create_search_run(self, request: CreateSearchRunRequest) -> CreateSearchRunResponse:
        """
        Creates a new search run record.
        Idempotency: If a similar run (same project, query, provider, time-window?) exists?
        Caller should supply deterministic ID/key if needed, but here we generate a new ID based on spec.
        We will generate a unique ID if not provided (though spec says caller supplies ID or key).
        We'll generate one here for simplicity if not passed, but spec implies we create it.
        We'll use a random UUID or timestamp based ID.
        """
        # Generate ID: SR_{YYYY_MM_DD}_{Random}
        import uuid
        now_str = self._get_utc_now()
        search_run_id = f"SR_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO search_run (
                        search_run_id, project_id, provider, query_text, created_at,
                        policy_version, filters_json, result_count_total, top_k_stored, notes
                    ) VALUES (
                        :sid, :pid, :prov, :q, :cat, :pver, :filt, :cnt, :topk, :notes
                    )
                """),
                {
                    "sid": search_run_id,
                    "pid": request.project_id,
                    "prov": request.provider.value,
                    "q": request.query_text,
                    "cat": now_str,
                    "pver": request.policy.policy_id,  # Using policy_id as policy_version fk reference
                    "filt": json.dumps(request.filters_json),
                    "cnt": request.result_count_total,
                    "topk": request.top_k_stored,
                    "notes": request.notes
                }
            )

        return CreateSearchRunResponse(
            search_run_id=search_run_id,
            created_at=datetime.fromisoformat(now_str)
        )

    def upsert_work_batch(self, request: UpsertWorkBatchRequest) -> UpsertWorkBatchResponse:
        """
        Upserts works by DOI.
        """
        normalized_dois = []
        now_str = self._get_utc_now()

        with self.engine.begin() as conn:
            for w in request.works:
                norm_doi = self._normalize_doi(w.doi)
                normalized_dois.append(norm_doi)
                
                # Check if exists to preserve created_at or other immutable fields if needed,
                # but "Upsert" usually implies overwrite of mutable fields.
                # SQLite INSERT OR REPLACE replaces *everything*, losing created_at if we don't be careful.
                # Better to use INSERT OR IGNORE then UPDATE, or standard UPSERT syntax (sqlite >= 3.24)
                
                # Using SQLite UPSERT
                conn.execute(
                    text("""
                        INSERT INTO work (
                            doi, title, year, venue, publisher, pmid, pmcid, indexing_json,
                            created_at, updated_at
                        ) VALUES (
                            :doi, :title, :year, :venue, :pub, :pmid, :pmcid, :idx,
                            :now, :now
                        )
                        ON CONFLICT(doi) DO UPDATE SET
                            title=excluded.title,
                            year=excluded.year,
                            venue=excluded.venue,
                            publisher=excluded.publisher,
                            pmid=excluded.pmid,
                            pmcid=excluded.pmcid,
                            indexing_json=excluded.indexing_json,
                            updated_at=excluded.updated_at
                    """),
                    {
                        "doi": norm_doi,
                        "title": w.title,
                        "year": w.year,
                        "venue": w.venue,
                        "pub": w.publisher,
                        "pmid": w.pmid,
                        "pmcid": w.pmcid,
                        "idx": json.dumps(w.indexing_json),
                        "now": now_str
                    }
                )
        
        return UpsertWorkBatchResponse(normalized_dois=normalized_dois)

    def ingest_candidates(self, request: IngestCandidatesRequest) -> IngestCandidatesResponse:
        """
        Ingests candidates for a search run.
        """
        if request.enforce_max_n > 0 and len(request.candidates) > request.enforce_max_n:
             raise ValueError(f"Candidate count {len(request.candidates)} exceeds limit {request.enforce_max_n}")

        candidate_ids = []
        now_str = self._get_utc_now()
        
        # Need to get policy_version from search_run to start with?
        # Or candidates have their own policy_version in schema?
        # Schema says: candidate.policy_version REFERENCES policy_snapshot.
        # But input request doesn't have policy. The search_run has a policy. 
        # We should look up the search_run's policy_version or use the one from search_run.
        
        with self.engine.connect() as conn:
            # Look up policy_version from search_run
            res = conn.execute(
                text("SELECT policy_version FROM search_run WHERE search_run_id = :sid"),
                {"sid": request.search_run_id}
            ).fetchone()
            if not res:
                raise ValueError(f"Search run {request.search_run_id} not found")
            policy_version = res[0]

        with self.engine.begin() as conn:
            for c in request.candidates:
                norm_doi = self._normalize_doi(c.doi)
                # Ensure work exists? 
                # Spec says: block if no DOI. We validated DOI presence in Pydantic (DoiStr).
                # But we need Foreign Key integrity. 
                # Ideally upsert_work_batch was called first. 
                # If not, we might fail FK constraint.
                # We will assume works are upserted. If not, this will bubble up FK error.
                
                # Generate candidate_id: CAND_{search_run_hash}_{doi_hash} ?? or just random
                # Spec doesn't strictly specify ID format.
                import uuid
                # Use a deterministic ID based on run+doi to ensure idempotency if we want, or just random.
                # "Store bounded candidate set".
                # If we ingest the same candidate twice for same run, schema has UNIQUE(search_run_id, doi).
                # So we should use INSERT OR IGNORE or handle it.
                # Attempt to retrieve existing?
                # We'll use insert.
                
                candidate_id = f"CAND_{uuid.uuid4().hex}"
                
                try:
                    conn.execute(
                        text("""
                            INSERT INTO candidate (
                                candidate_id, search_run_id, doi, rank_in_results,
                                retrieval_score_raw, features_json, composite_score,
                                scoring_version, policy_version, created_at
                            ) VALUES (
                                :cid, :sid, :doi, :rank, :raw, :feat, :comp, :sver, :pver, :now
                            )
                        """),
                        {
                            "cid": candidate_id,
                            "sid": request.search_run_id,
                            "doi": norm_doi,
                            "rank": c.rank_in_results,
                            "raw": c.retrieval_score_raw,
                            "feat": json.dumps(c.features_json),
                            "comp": c.composite_score,
                            "sver": c.scoring_version,
                            "pver": policy_version,
                            "now": now_str
                        }
                    )
                    candidate_ids.append(candidate_id)
                except Exception as e:
                    # If UNIQUE constraint fails (already exists), assume we return ID of existing?
                    # Or fail. Spec says "unique (search_run_id, doi)".
                    # We will fail for now or could lookup.
                    # Let's try to lookup if insert fails?
                    # For bulk ingest, maybe better to fail?
                    # The prompt says "Persist only a bounded set."
                    # I'll just let it raise for now if duplicates.
                    raise e

        return IngestCandidatesResponse(candidate_ids=candidate_ids)

    def record_quality_check(self, request: RecordQualityCheckRequest) -> RecordQualityCheckResponse:
        import uuid
        check_id = f"QC_{uuid.uuid4().hex}"
        now_str = self._get_utc_now()
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO candidate_quality_check (
                        check_id, candidate_id, claim_id, check_type, verdict,
                        policy_id, policy_hash, details_json, executed_by, executed_at
                    ) VALUES (
                        :cid, :cand_id, :claim_id, :ctype, :verdict,
                        :pid, :phash, :dets, :exec_by, :now
                    )
                """),
                {
                    "cid": check_id,
                    "cand_id": request.candidate_id,
                    "claim_id": request.claim_id,
                    "ctype": request.check_type.value,
                    "verdict": request.verdict.value,
                    "pid": request.policy.policy_id,
                    "phash": request.policy.policy_hash,
                    "dets": json.dumps(request.details_json),
                    "exec_by": request.executed_by.agent_id,
                    "now": now_str
                }
            )
            
        return RecordQualityCheckResponse(check_id=check_id, executed_at=datetime.fromisoformat(now_str))

    def record_decision(self, request: RecordDecisionRequest) -> RecordDecisionResponse:
        import uuid
        decision_id = f"DEC_{uuid.uuid4().hex}"
        now_str = self._get_utc_now()
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO candidate_decision (
                        decision_id, candidate_id, claim_id, outcome,
                        basis_json, policy_id, policy_hash, decided_by, decided_at
                    ) VALUES (
                        :did, :cand_id, :claim_id, :outcome,
                        :basis, :pid, :phash, :dby, :now
                    )
                """),
                {
                    "did": decision_id,
                    "cand_id": request.candidate_id,
                    "claim_id": request.claim_id,
                    "outcome": request.outcome.value,
                    "basis": json.dumps(request.basis_json),
                    "pid": request.policy.policy_id,
                    "phash": request.policy.policy_hash,
                    "dby": request.decided_by.agent_id,
                    "now": now_str
                }
            )
            
        return RecordDecisionResponse(decision_id=decision_id, decided_at=datetime.fromisoformat(now_str))

    def get_candidate_effective_status(self, request: GetEffectiveStatusRequest) -> GetEffectiveStatusResponse:
        with self.engine.connect() as conn:
            # Latest decision
            dec_row = conn.execute(
                text("""
                    SELECT * FROM vw_candidate_latest_decision
                    WHERE candidate_id = :cand_id
                      AND (claim_id = :claim_id OR claim_id IS NULL)
                    ORDER BY decided_at DESC
                    LIMIT 1
                """),
                {"cand_id": request.candidate_id, "claim_id": request.claim_id}
            ).fetchone()
            
            latest_decision = None
            if dec_row:
                latest_decision = dict(dec_row._mapping)

            # Latest checks
            # Note: The view vw_candidate_latest_checks partitions by check_type
            # We want all check types for this candidate/claim combination
            # AND assume GLOBAL checks apply to specific claim? 
            # View partitions by (candidate_id, COALESCE(claim_id, 'GLOBAL'), check_type)
            # If we ask for claim_id='X', we probably accept checks where claim_id='X' OR claim_id IS NULL?
            # The view separates them. 
            # If I want effective status for Claim X, do I look at Global checks? Yes usually.
            # But the view is keyed. 
            # Let's fetch both and merge? Or simple query.
            # "Return 'current truth' derived from latest checks".
            
            checks_cursor = conn.execute(
                text("""
                    SELECT * FROM vw_candidate_latest_checks
                    WHERE candidate_id = :cand_id
                      AND (claim_id = :claim_id OR claim_id IS NULL)
                """),
                {"cand_id": request.candidate_id, "claim_id": request.claim_id}
            )
            
            latest_checks = {}
            for row in checks_cursor:
                # If we have duplicate check_type (one null claim_id, one specific), which wins?
                # Probably specific one wins?
                # View partitions include claim_id.
                # So we might get check_type='has_doi' (claim=NULL) and check_type='has_doi' (claim='X').
                # We should prefer claim='X'.
                ct = CheckType(row.check_type)
                row_dict = dict(row._mapping)
                
                if ct not in latest_checks:
                    latest_checks[ct] = row_dict
                else:
                    # If existing is global (claim_id is None) and new is specific, replace.
                    if latest_checks[ct]['claim_id'] is None and row_dict['claim_id'] is not None:
                         latest_checks[ct] = row_dict
            
            return GetEffectiveStatusResponse(
                status=CandidateEffectiveStatus(
                    candidate_id=request.candidate_id,
                    claim_id=request.claim_id,
                    latest_decision=latest_decision,
                    latest_checks=latest_checks
                )
            )

    def store_work_fulltext(self, request: StoreFulltextRequest) -> StoreFulltextResponse:
        import hashlib
        content_hash = hashlib.sha256(request.content.encode("utf-8")).hexdigest()
        now_str = self._get_utc_now()
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO work_fulltext_cache (
                        doi, pmcid, source, format, content, content_hash, retrieved_at, license
                    ) VALUES (
                        :doi, :pmcid, :src, :fmt, :content, :hash, :now, :lic
                    )
                    ON CONFLICT(doi, source, format) DO UPDATE SET
                        content = excluded.content,
                        content_hash = excluded.content_hash,
                        retrieved_at = excluded.retrieved_at,
                        license = excluded.license
                """),
                {
                    "doi": request.doi,
                    "pmcid": request.pmcid,
                    "src": request.source,
                    "fmt": request.format,
                    "content": request.content,
                    "hash": content_hash,
                    "now": now_str,
                    "lic": request.license
                }
            )
            
        return StoreFulltextResponse(doi=request.doi, content_hash=content_hash, stored_at=now_str)

    def promote_candidate_to_support(self, request: PromoteCandidateRequest) -> PromoteCandidateResponse:
        """
        Promotes a candidate to claim_support.
        """
        import uuid
        support_id = f"SUP_{uuid.uuid4().hex}"
        now_str = self._get_utc_now()
        
        # We need to lookup project_id and doi from candidate/search_run?
        # Support table needs project_id, claim_id, doi.
        # Candidate has doi + search_run_id. Search_run has project_id.
        
        with self.engine.begin() as conn:
            # 1. Fetch info
            info = conn.execute(
                text("""
                    SELECT c.doi, sr.project_id
                    FROM candidate c
                    JOIN search_run sr ON c.search_run_id = sr.search_run_id
                    WHERE c.candidate_id = :cid
                """),
                {"cid": request.candidate_id}
            ).fetchone()
            
            if not info:
                raise ValueError(f"Candidate {request.candidate_id} not found")
            
            doi, project_id = info
            
            # 2. Insert into claim_support
            conn.execute(
                text("""
                    INSERT INTO claim_support (
                        support_id, project_id, claim_id, doi, support_type,
                        verification_status, promoted_from_candidate_id,
                        promotion_reason, created_by, created_at, policy_version_at_promotion,
                        anchor_excerpt, anchor_location_json
                    ) VALUES (
                        :sid, :pid, :clid, :doi, :stype,
                        :vstat, :cid,
                        :reason, :cby, :now, :pver,
                        :anchor, :loc_json
                    )
                """),
                {
                    "sid": support_id,
                    "pid": project_id,
                    "clid": request.claim_id,
                    "doi": doi,
                    "stype": request.support_type,
                    "vstat": request.verification_status,
                    "cid": request.candidate_id,
                    "reason": request.promotion_reason,
                    "cby": request.created_by,
                    "now": now_str,
                    "pver": request.policy_version,
                    "anchor": request.anchor_excerpt,
                    "loc_json": json.dumps(request.anchor_location_json) if request.anchor_location_json else None
                }
            )
            
        return PromoteCandidateResponse(support_id=support_id)
