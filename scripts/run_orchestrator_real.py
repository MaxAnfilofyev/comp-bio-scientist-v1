import sys
import os
import json
import sqlalchemy as sa
from datetime import datetime, timezone

# Add path
sys.path.append(os.getcwd())

from ai_scientist.evidence_control_loop import (
    orchestrate_claim_support_round,
    LedgerRepo,
    OrchestrateRoundRequest,
    RealLLMClient,
)
from ai_scientist.model.evidence import (
    PubMedSearchResponse, PubMedArticle, PMCFetchFulltextResponse,
    PMIDToPMCID
)

from ai_scientist.medline_gate import (
    MedlinePmcEvidenceGate,
    PubMedSearchRequest as GateSearchRequest,
    PubMedFetchMetadataRequest as GateMetaRequest,
    PubMedLinkToPmcRequest as GateLinkRequest,
    PubMedFetchFullTextRequest as GateFullTextRequest
)
from ai_scientist.evidence_service import EvidenceService

# ------------------------------------------------------------------------------
# ADAPTERS
# ------------------------------------------------------------------------------

class PubMedAdapter:
    def __init__(self, gate: MedlinePmcEvidenceGate):
        self.gate = gate

    def search(self, project_id, claim_id, query, retmax, retstart=0, policy_id=None, search_run_id=None):
        # Convert Orchestrator call -> Gate Request
        req = GateSearchRequest(
            project_id=project_id,
            query=query,
            policy_query_template="({query}) AND \"pubmed pmc\"[sb] AND english[la]", # STRICT PMC Policy per Strategy v8
            retmax=retmax,
            retstart=retstart,
            policy_id=policy_id,
            search_run_id=search_run_id
        )
        resp = self.gate.pubmed_search(req)
        
        # Convert Gate Response -> Orchestrator Response (Model)
        # Assuming model/evidence.py matches close enough or we map fields
        return PubMedSearchResponse(
            search_run_id=resp.search_run_id,
            provider="PUBMED",
            query=resp.query,
            total_hits=resp.count_total,
            pmids=resp.pmids, # list[str]
            webenv=resp.webenv,
            query_key=resp.query_key,
            created_at=datetime.now(timezone.utc)
        )

    def fetch_metadata(self, pmids, include_abstract=True):
        req = GateMetaRequest(
            project_id="global_adapter", # dummy
            pmids=pmids,
            retmax=len(pmids)
        )
        resp = self.gate.pubmed_fetch_metadata(req)
        
        # Map Gate WorkMetadata -> Orchestrator PubMedArticle
        articles = []
        for w in resp.works:
            articles.append(PubMedArticle(
                pmid=w.pmid,
                title=w.title,
                abstract=None, # MedlineGate metadata fetch doesn't populate abstract yet in existing impl? 
                               # Wait, `medline_gate.py` has `has_abstract` bool, but check `has_abs` logic.
                               # It parses <Abstract> tag existence but didn't store text in `works` list in `PubMedWorkMetadata`.
                               # `PubMedWorkMetadata` schema in `medline_gate.py` does NOT have `abstract` string field!
                               # Ah, I need abstract for Triage.
                               # I checked `medline_gate.py` code: `class PubMedWorkMetadata` has no abstract string.
                               # I MUST update `MedlinePmcEvidenceGate` or implement a custom fetch here.
                               # I will patch it locally in Adapter by calling EFetch myself or just assume `fetch_metadata` in gate needs update.
                               # I CANNOT modify MedlineGate easily right now without disrupting previous steps trace.
                               # Adapter Strategy: I will do a raw EFetch if Gate doesn't give me abstract.
                               # Actually, let's just do a specific EFetch here in adapter.
                               
                journal=w.journal,
                year=w.year,
                publication_types=w.publication_types,
                doi=w.doi
            ))
        
        # Abstract Backfill for Triage
        # Since MedlineGate didn't give abstract text, I need to fetch it.
        # But wait, `llm_topic_triage` needs abstract.
        # I will implement a quick batch fetch for abstracts using the gate's internal `_requests_post` if possible, 
        # or just requests.
        from ai_scientist.medline_gate import NCBI_BASE_URL
        import requests
        import xml.etree.ElementTree as ET
        
        if pmids:
            try:
                efetch_url = f"{NCBI_BASE_URL}/efetch.fcgi"
                r = requests.post(efetch_url, data={"db":"pubmed", "id":",".join(pmids), "retmode":"xml"})
                if r.status_code == 200:
                    root = ET.fromstring(r.text)
                    for art in articles:
                        # Find corresponding node
                        # Iterate whole tree is slow but O(N).
                        # Optimization: map pmid to abstract text
                        pass
                    
                    # Better: Parse once to a dict
                    pmid_map = {}
                    for medline_cit in root.findall(".//MedlineCitation"):
                        p = medline_cit.findtext("PMID")
                        ab_node = medline_cit.find(".//Abstract/AbstractText")
                        if p is not None and ab_node is not None:
                             pmid_map[p] = "".join(ab_node.itertext())
                    
                    for art in articles:
                        art.abstract = pmid_map.get(art.pmid)
            except Exception as e:
                print(f"Abstract Fetch Warning: {e}")

        return articles

    def link_to_pmc(self, pmids):
        req = GateLinkRequest(project_id="adapter", pmids=pmids)
        resp = self.gate.pubmed_link_to_pmc(req)
        
        # Map
        results = []
        for link in resp.links:
            pmid = link.pmid
            pmcids = link.pmcids
            first = pmcids[0] if pmcids else None
            results.append(PMIDToPMCID(
                pmid=pmid,
                pmcid=first,
                link_status="FOUND" if first else "NOT_FOUND"
            ))
        return results

class PMCAdapter:
    def __init__(self, gate: MedlinePmcEvidenceGate):
        self.gate = gate

    def fetch_jats(self, pmcid, cache_policy="USE_CACHE"):
        req = GateFullTextRequest(project_id="adapter", pmcid=pmcid)
        resp = self.gate.pubmed_fetch_fulltext(req)
        
        return PMCFetchFulltextResponse(
            pmcid=pmcid,
            content=resp.xml_content or "",
            content_hash="hash_placeholder",
            content_size=len(resp.xml_content or ""),
            retrieved_at=datetime.utcnow()
        )

class MockCrossref:
    def retraction_check(self, **kwargs): return {"is_retracted": False}
    def eoc_check(self, **kwargs): return {"has_eoc": False}

# ------------------------------------------------------------------------------
# RUNNER
# ------------------------------------------------------------------------------

def main():
    # Setup DB
    db_url = "sqlite:///ai_scientist_manual_search_1765844287.sqlite" # Use existing DB to keep history?
    # Or create new for clean trace?
    # User said "run ... and produce a trace".
    # I'll use a new DB-lite for the trace run to avoid polluting the big one or getting unique constraint errors if not handled.
    db_url = "sqlite:///orchestrator_trace.sqlite"
    if os.path.exists("orchestrator_trace.sqlite"):
        os.remove("orchestrator_trace.sqlite")
        
    engine = sa.create_engine(db_url)
    
    # Init Schema (Minimal or Full?)
    # I'll rely on Repo to fail if tables missing, so I must create them.
    # Reuse valid SQL from `initial_schema.py` or just the subset control loop needs.
    # I'll carry over the subset I used in verify_control_loop.py which works.
    
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
        # Prepare Policy
        import hashlib
        policy_yaml = "id: v1\ndescription: Strict PMC Policy\nrules:\n  - check: in_pmc_fulltext\n    verdict: PASS"
        policy_hash = hashlib.sha256(policy_yaml.encode("utf-8")).hexdigest()
        now_ts = datetime.now(timezone.utc).isoformat()
        
        conn.execute(sa.text("INSERT INTO policy_snapshot (policy_version, created_at, policy_yaml, policy_hash) VALUES (:ver, :at, :yaml, :hash)"), 
                     {"ver": "v1", "at": now_ts, "yaml": policy_yaml, "hash": policy_hash})
                     
        conn.execute(sa.text("INSERT INTO project (project_id, title, status, created_at, updated_at, policy_version) VALUES (:pid, :title, 'active', :at, :at, :pol)"),
                     {"pid": "p1", "title": "Trace Proj", "at": now_ts, "pol": "v1"})
                     
        conn.execute(sa.text("INSERT INTO claim (claim_id, project_id, module, statement, claim_type, policy_version, created_by, created_at, updated_at) VALUES (:cid, :pid, 'mech', :stmt, 'mechanism', :pol, 'user', :at, :at)"),
                     {"cid": "c1", "pid": "p1", "stmt": "Active transport of ATP along axons involves diffusion mechanisms", "pol": "v1", "at": now_ts})

    # Service & Adapters
    # EvidenceService needs an engine too.
    ev_service = EvidenceService(db_url=db_url)
    gate = MedlinePmcEvidenceGate(evidence_service=ev_service)
    
    pubmed_client = PubMedAdapter(gate)
    pmc_client = PMCAdapter(gate)
    crossref_client = MockCrossref()
    llm_client = RealLLMClient() # connects to OpenAI
    repo = LedgerRepo(engine)
    
    req = OrchestrateRoundRequest(
        project_id="p1",
        claim_id="c1",
        search_round_index=0,
        policy_id="v1",
        current_query="ATP diffusion axon", # Clean starter query
        pubmed_retmax=50
    )
    
    print(">>> Starting Real Orchestrator Trace (Multi-Step)...")
    
    round_limit = 5
    current_round = 0
    current_req = req
    
    while current_round < round_limit:
        print(f"\n--- Round {current_round} [Query: '{current_req.current_query}'] ---")
        
        current_req.search_round_index = current_round
        
        resp = orchestrate_claim_support_round(
            current_req,
            pubmed=pubmed_client,
            pmc=pmc_client,
            crossref=crossref_client,
            llm=llm_client,
            repo=repo
        )
        
        print(f"Round {current_round} Result: Next={resp.next_action}, Support={resp.supports_found}")
        print(f"Metrics: {json.dumps(resp.summary_json, indent=2)}")
        
        if resp.done:
            print(">>> Orchestrator reports DONE.")
            break
            
        # Prepare next round
        current_round += 1
        
        # Determine next query
        next_q = resp.next_query  # Use rewrite if provided
        if not next_q:
             # If no rewrite, check if we should continue same query (Orchestrator handles pagination internally via DB state lookup now)
             # But we must pass the *same* query string so logic detects continuity.
             next_q = resp.query 
             
        current_req = OrchestrateRoundRequest(
            project_id=req.project_id,
            claim_id=req.claim_id,
            search_round_index=current_round,
            policy_id=req.policy_id,
            current_query=next_q,
            pubmed_retmax=req.pubmed_retmax,
            max_rounds=round_limit
        )
    
    print(">>> Trace Complete")

if __name__ == "__main__":
    main()
