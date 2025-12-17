import sys
import os
import json
import uuid
import sqlalchemy as sa
from sqlalchemy.orm import Session

# Ensure path
sys.path.append(os.getcwd())

from ai_scientist.evidence_control_loop import (
    orchestrate_claim_support_round,
    LedgerRepo,
    OrchestrateRoundRequest,
    RealLLMClient,
    NextAction,
    CandidateSpan
)
from ai_scientist.model.evidence import (
    PubMedSearchResponse, PubMedArticle, PMCFetchFulltextResponse
)

# Mock Clients
class MockPubMed:
    def search(self, **kwargs):
        # Return 2 hits
        return PubMedSearchResponse(
            search_run_id="mock_sr",
            provider="PUBMED",
            query=kwargs.get("query"),
            total_hits=2,
            pmids=["1111", "2222"],
            created_at="2025-01-01T00:00:00Z"
        )
    def fetch_metadata(self, **kwargs):
        return [
            PubMedArticle(pmid="1111", title="Relevant Paper", abstract="ATP diffusion in logic.", journal="J Biol", year=2021, doi="10.1000/1", publication_types=[]),
            PubMedArticle(pmid="2222", title="Irrelevant Paper", abstract="Something else.", journal="Nature", year=2022, doi="10.1000/2", publication_types=[])
        ]
    def link_to_pmc(self, **kwargs):
        return [
            {"pmid": "1111", "pmcid": "PMC1111", "link_status": "FOUND"},
            {"pmid": "2222", "pmcid": None, "link_status": "NOT_FOUND"}
        ]

class MockPMC:
    def fetch_jats(self, **kwargs):
        return PMCFetchFulltextResponse(
            pmcid="PMC1111",
            content="<article><body><p>This paragraph contains the word ATP and diffusion, supporting the claim properly.</p></body></article>",
            content_hash="abc",
            content_size=100,
            retrieved_at="2025-01-01T00:00:00Z"
        )

class MockCrossref:
    def retraction_check(self, **kwargs): return {"is_retracted": False}
    def eoc_check(self, **kwargs): return {"has_eoc": False}

class MockLLM(RealLLMClient):
    def topic_triage(self, claim, candidate, policy_id):
        # Pass relevant paper, fail irrelevant
        if "Relevant" in candidate.get("title", ""):
            return {"topic_match": "PASS", "evidence_likelihood": 0.9}
        return {"topic_match": "FAIL", "mismatch_codes": ["WRONG_DOMAIN"]}
        
    def entailment(self, claim, span, policy_id):
        # Support if correct span
        if "supporting" in span.text:
            return {"verdict": "SUPPORTED", "confidence": 0.95, "anchor_quote": "This paragraph contains...", "reasoning": "Good."}
        return {"verdict": "NOT_SUPPORTED", "confidence": 0.1}

# Setup DB
db_url = "sqlite:///test_control_loop.sqlite"
if os.path.exists("test_control_loop.sqlite"):
    os.remove("test_control_loop.sqlite")

engine = sa.create_engine(db_url)

# Re-run migration logic manually or just basic tables?
# We'll rely on the existing migration script if we ran it, or just create tables via raw SQL for speed in test.
# Actually, let's use the repo definitions to infer needed tables or just run the raw SQL from `initial_schema.py`.
# Easier: Just use the imported schema sql if I had it, but I don't.
# I will define minimal schema here for the test.

schema_sql = """
CREATE TABLE project (project_id TEXT PRIMARY KEY, title TEXT, status TEXT DEFAULT 'active', created_at TEXT, updated_at TEXT, policy_version TEXT);
CREATE TABLE policy_snapshot (policy_version TEXT PRIMARY KEY, created_at TEXT, policy_yaml TEXT, policy_hash TEXT);
CREATE TABLE claim (claim_id TEXT PRIMARY KEY, project_id TEXT, module TEXT, statement TEXT, claim_type TEXT, status TEXT DEFAULT 'proposed', created_by TEXT, created_at TEXT, updated_at TEXT, policy_version TEXT, strength_target TEXT DEFAULT 'medium', evidence_required TEXT DEFAULT 'citation', priority TEXT DEFAULT 'P1', disposition TEXT DEFAULT 'undecided', canonical_for_manuscript INTEGER DEFAULT 0, etag INTEGER DEFAULT 1);
CREATE TABLE search_run (search_run_id TEXT PRIMARY KEY, project_id TEXT, provider TEXT, query_text TEXT, created_at TEXT, policy_version TEXT, filters_json TEXT, result_count_total INTEGER DEFAULT 0, top_k_stored INTEGER DEFAULT 0, notes TEXT);
CREATE TABLE work (doi TEXT PRIMARY KEY, title TEXT, year INTEGER, venue TEXT, pmid TEXT, pmcid TEXT, pubmed_url TEXT, pmc_url TEXT, created_at TEXT, updated_at TEXT, full_text_available INTEGER DEFAULT 0, retraction_status TEXT DEFAULT 'unknown', eoc_status TEXT DEFAULT 'unknown');
CREATE TABLE candidate (candidate_id TEXT PRIMARY KEY, search_run_id TEXT, doi TEXT, rank_in_results INTEGER, composite_score REAL, scoring_version TEXT DEFAULT 'v1', policy_version TEXT, created_at TEXT, retrieval_score_raw REAL, features_json TEXT);
CREATE TABLE candidate_quality_check (check_id TEXT PRIMARY KEY, candidate_id TEXT, claim_id TEXT, check_type TEXT, verdict TEXT, policy_id TEXT, policy_hash TEXT, details_json TEXT, executed_by TEXT, executed_at TEXT);
CREATE TABLE candidate_decision (decision_id TEXT PRIMARY KEY, candidate_id TEXT, claim_id TEXT, outcome TEXT, basis_json TEXT, policy_id TEXT, policy_hash TEXT, decided_by TEXT, decided_at TEXT);
CREATE TABLE claim_support (support_id TEXT PRIMARY KEY, project_id TEXT, claim_id TEXT, doi TEXT, support_type TEXT, verification_status TEXT, anchor_excerpt TEXT, anchor_location_json TEXT, promotion_reason TEXT, created_by TEXT, created_at TEXT, policy_version_at_promotion TEXT, promoted_from_candidate_id TEXT);
CREATE TABLE claim_search_round (round_id TEXT PRIMARY KEY, project_id TEXT, claim_id TEXT, search_run_id TEXT, round_index INTEGER, summary_json TEXT, next_action TEXT, created_at TEXT);
CREATE TABLE claim_gap (gap_id TEXT PRIMARY KEY, project_id TEXT, claim_id TEXT, gap_type TEXT, recommendation TEXT, resolved INTEGER, created_by TEXT, created_at TEXT, resolved_at TEXT, resolution_note TEXT, assigned_to TEXT);
"""

with engine.begin() as conn:
    for stmt in schema_sql.split(';'):
        if stmt.strip():
            conn.execute(sa.text(stmt))
            
    # Seeds
    conn.execute(sa.text("INSERT INTO policy_snapshot VALUES ('v1', 'now', 'yaml', 'hash')"))
    conn.execute(sa.text("INSERT INTO project VALUES ('p1', 'Test Proj', 'active', 'now', 'now', 'v1')"))
    conn.execute(sa.text("INSERT INTO claim (claim_id, project_id, module, statement, claim_type, policy_version, created_by, created_at, updated_at) VALUES ('c1', 'p1', 'mod', 'ATP diffusion occurs', 'mech', 'v1', 'me', 'now', 'now')"))

# Run Test
repo = LedgerRepo(engine)
req = OrchestrateRoundRequest(
    project_id="p1",
    claim_id="c1",
    round_index=0,
    policy_id="v1",
    current_query="ATP diffusion",
    pubmed_retmax=5
)

print("Starting Orchestrator Round...")
resp = orchestrate_claim_support_round(
    req,
    pubmed=MockPubMed(),
    pmc=MockPMC(),
    crossref=MockCrossref(),
    llm=MockLLM(),
    repo=repo
)

print(f"Round ID: {resp.round_id}")
print(f"Stats: {resp.summary_json}")
print(f"Done: {resp.done}")
print(f"Next: {resp.next_action}")

assert resp.supports_found == 1, f"Expected 1 support, got {resp.supports_found}"
assert resp.done is True
assert resp.next_action == "CONTINUE_SAME_QUERY" # Or whatever logic dictates when support is found. Logic says: IF supports_found > 0: "CONTINUE_SAME_QUERY", "Support found..." (wait, logic text said stop searching? Yes, but Action enum is CONTINUE_SAME_QUERY with a 'Done' flag set to True in response, or action is 'STOP'? Logic code: return ('CONTINUE_SAME_QUERY', 'Support found...', None). Orchestrator sets done=True.)

print("Verification SUCCESS")
