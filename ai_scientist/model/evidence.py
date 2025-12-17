from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, List, Dict
from pydantic import BaseModel, Field, conint, confloat

# -------------------------
# Enums
# -------------------------

CheckVerdict = Literal["PASS", "FAIL", "UNKNOWN", "NA", "AMBIGUOUS"]

CheckType = Literal[
    "IN_PMC_FULLTEXT",  # Legacy / Composite
    "PMID_TO_PMCID_LINKED",
    "PMC_FULLTEXT_FETCHED",
    "PMC_FULLTEXT_PARSED",
    "HAS_DOI",
    "PREPRINT_POLICY",
    "RETRACTION_POLICY",
    "TOPIC_TRIAGE_LLM",
    "SUPPORTS_CLAIM_CHECK",
    "ANCHOR_EXTRACTION"
]

MismatchCode = Literal[
    "WRONG_DOMAIN",
    "WRONG_ENTITY",
    "WRONG_PROCESS",
    "NON_AXONAL",
    "NO_MEASUREMENT",
    "REVIEW_ONLY",
    "WRONG_PROCESS",
    "WRONG_ENTITY",
    "NON_AXONAL",
    "NO_MEASUREMENT",
    "QUERY_SYNTAX_ERROR",
    "METHOD_MISMATCH",
    "AMBIGUOUS",
]

EntailmentVerdict = Literal[
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "NOT_SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_SPECIFICITY",
]

DecisionOutcome = Literal[
    "REJECTED",
    "HOLD",
    "ELIGIBLE_SUPPORT",
    "SELECTED_AS_SUPPORT",
]

NextAction = Literal[
    "CONTINUE_SAME_QUERY",
    "CONTINUE_PAGINATION",
    "REWRITE_QUERY_BROADEN",
    "REWRITE_QUERY_NARROW",
    "REWRITE_QUERY_DISAMBIGUATE",
    "REWRITE_QUERY_RELAX_CONSTRAINTS",
    "REWRITE_QUERY_TIGHTEN",
    "REWRITE_QUERY_TIGHTEN_WITH_MODALITY",
    "CHANGE_RETRIEVAL_STRATEGY",
    "SWITCH_RETRIEVAL_STRATEGY_S2_FIRST",
    "STOP_OPEN_GAP",
    "STOP_MAX_ROUNDS"
]

RewriteMode = Literal[
    "DISAMBIGUATE",
    "RELAX",
    "TIGHTEN",
    "PAGINATE",
    "REPAIR",
    "INITIAL"
]

DriftClass = Literal[
    "WRONG_ENTITY",
    "WRONG_PROCESS",
    "WRONG_CONTEXT",
    "WRONG_MODALITY",
    "WRONG_DOMAIN",
    "AMBIGUOUS",
    "UNKNOWN"
]

FailureMode = Literal[
    "NONE",
    "OVER_CONSTRAINED",       # 0 hits
    "STARVATION_PMC_SUBSET",  # Subset enforced, low hits
    "STARVATION_POLICY_MISMATCH", # Subset NOT enforced, gates fail
    "BINDING_FAILURE_PMC_LINKING", # High link failure rate
    "OFF_TOPIC_DRIFT", 
    "EVIDENCE_GAP",
    "EXHAUSTED_PAGINATION",
    "EXHAUSTED"
]

GateName = Literal[
    "HAS_PMID",
    "HAS_DOI",
    "IN_PMC_FULLTEXT",
    "LANGUAGE_ALLOWED",
    "RETRACTION_OK",
    "EOC_OK"
]

# -------------------------
# Tool I/O Models
# -------------------------

class PubMedSearchRequest(BaseModel):
    project_id: str
    claim_id: Optional[str] = None
    query: str
    sort: Literal["best_match", "pub_date"] = "best_match"
    retmax: conint(ge=1, le=500) = 50
    retstart: conint(ge=0) = 0
    use_history: bool = True

class PubMedSearchResponse(BaseModel):
    search_run_id: str
    provider: Literal["PUBMED"]
    query: str
    total_hits: int
    pmids: List[str]
    webenv: Optional[str] = None
    query_key: Optional[str] = None
    created_at: datetime

class PubMedArticle(BaseModel):
    pmid: str
    title: Optional[str]
    abstract: Optional[str]
    journal: Optional[str]
    year: Optional[int]
    publication_types: List[str] = []
    mesh_terms: List[str] = []
    doi: Optional[str]

class PubMedFetchMetadataResponse(BaseModel):
    articles: List[PubMedArticle]

class PMIDToPMCID(BaseModel):
    pmid: str
    pmcid: Optional[str]
    link_status: Literal["FOUND", "NOT_FOUND", "ERROR"]

class PubMedLinkToPMCResponse(BaseModel):
    mappings: List[PMIDToPMCID]

class PMCFetchFulltextResponse(BaseModel):
    pmcid: str
    content_format: Literal["JATS_XML"] = "JATS_XML"
    content: str  # JATS XML
    content_hash: str
    content_size: int
    retrieved_at: datetime
    source: Literal["PMC"] = "PMC"

class CandidateSpan(BaseModel):
    span_id: str
    xpath: str
    text: str
    keyword_score: Optional[float] = None

class SpanExtractResponse(BaseModel):
    spans: List[CandidateSpan]

class LLMTopicTriageResponse(BaseModel):
    topic_match: Literal["PASS", "FAIL", "AMBIGUOUS"]
    evidence_likelihood: confloat(ge=0, le=1)
    mismatch_codes: List[MismatchCode] = []
    positive_anchors: List[str] = []
    negative_anchors: List[str] = []
    drift_concepts: List[str] = []
    drift_class: Optional[DriftClass] = None # Primary drift class
    
    # Suggestions for Rewriter
    suggested_entity_synonyms: List[str] = []
    suggested_process_synonyms: List[str] = []
    suggested_context_synonyms: List[str] = []
    suggested_modalities: List[str] = []
    
    query_hints_positive: List[str] = []
    query_hints_negative: List[str] = []
    note: Optional[str] = None

class LLMEntailmentJudgeResponse(BaseModel):
    verdict: EntailmentVerdict
    confidence: confloat(ge=0, le=1)
    anchor_quote: Optional[str] = None
    anchor_xpath: Optional[str] = None
    weakness_notes: List[str] = []
    mismatch_codes: List[MismatchCode] = []

class QueryBlocks(BaseModel):
    entity: str
    context: str
    process: str
    modality: str
    exclusion: Optional[str] = None # Negative clauses

class LLMRewriteQueryResponse(BaseModel):
    query_blocks: Optional[QueryBlocks] = None # Structured
    query: str # Compiled string (for fallback/logging)
    note: Optional[str] = None

# -------------------------
# Orchestrator Core Models
# -------------------------

class OrchestrateRoundRequest(BaseModel):
    project_id: str
    claim_id: str
    search_round_index: conint(ge=0)
    max_rounds: conint(ge=1) = 5
    policy_id: str
    policy_hash: Optional[str] = None

    # If None, orchestrator derives from last round, or constructs initial query.
    current_query: Optional[str] = None

    # Safety caps
    pubmed_retmax: conint(ge=1, le=500) = 50
    ingest_candidates_cap: conint(ge=1, le=100) = 20
    topic_triage_cap: conint(ge=1, le=100) = 20
    entailment_cap: conint(ge=1, le=100) = 10

class CandidateEvalResult(BaseModel):
    candidate_id: str
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None

    hard_gate_passed: bool
    topic_triage_passed: Optional[bool] = None
    entailment_verdict: Optional[EntailmentVerdict] = None
    entailment_confidence: Optional[confloat(ge=0, le=1)] = None

    created_support_id: Optional[str] = None
    decision_outcome: Optional[DecisionOutcome] = None

    mismatch_codes: List[MismatchCode] = Field(default_factory=list)

class RunFailureProfile(BaseModel):
    failure_mode: FailureMode
    bottlenecks: Dict[str, int] = {} # GateName -> count
    query_health: bool
    description: str
    # Granular Eligibility
    pmcid_rate: float = 0.0
    doi_rate: float = 0.0
    policy_pass_rate: float = 0.0
    ratios: Dict[str, float] = {} # E, T, Y

class OrchestrateRoundResponse(BaseModel):
    round_id: str
    search_run_id: str
    claim_id: str
    query: str
    supports_found: int
    supports_created: List[str]
    next_action: NextAction
    next_query: Optional[str] = None
    reason: str
    done: bool
    summary_json: dict
    failure_profile: Optional[RunFailureProfile] = None
    claim_id: str
    query: str

    supports_found: int
    supports_created: List[str] = Field(default_factory=list)

    candidate_results: List[CandidateEvalResult] = Field(default_factory=list)

    next_action: NextAction
    next_query: Optional[str] = None
    reason: str

    done: bool = False
    summary_json: Dict[str, Any] = Field(default_factory=dict)
