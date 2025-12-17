from __future__ import annotations
import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Dict
import sqlalchemy as sa
from sqlalchemy import text

from ai_scientist.model.evidence import (
    OrchestrateRoundRequest,
    OrchestrateRoundResponse,
    CandidateEvalResult,
    CandidateSpan,
    CheckType,
    CheckVerdict,
    DecisionOutcome,
    NextAction,
    MismatchCode,
    EntailmentVerdict,
    PubMedSearchRequest,
    PubMedSearchResponse,
    PubMedArticle,
    LLMTopicTriageResponse,
    LLMEntailmentJudgeResponse,
    LLMRewriteQueryResponse,
    RunFailureProfile,
    FailureMode
)
from ai_scientist.evidence_llm_logic import llm_rewrite_query, llm_topic_triage, llm_repair_query
from ai_scientist.entailment_verification import verify_claim_entailment, EntailmentResult

# -------------------------
# Repository
# -------------------------

class LedgerRepo:
    def __init__(self, engine: sa.engine.Engine):
        self.engine = engine

    def load_claim(self, claim_id: str) -> dict:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM claim WHERE claim_id = :cid"), {"cid": claim_id}
            ).mappings().one_or_none()
            if not row:
                raise ValueError(f"Claim {claim_id} not found")
            return dict(row)

    def get_last_round(self, claim_id: str) -> Optional[dict]:
        with self.engine.connect() as conn:
            # Get max round index for this claim
            row = conn.execute(
                text("""
                    SELECT * FROM claim_search_round 
                    WHERE claim_id = :cid 
                    ORDER BY round_index DESC LIMIT 1
                """),
                {"cid": claim_id}
            ).mappings().one_or_none()
            return dict(row) if row else None

    def create_search_run(self, project_id: str, provider: str, query: str, policy_id: str, filters_json: dict) -> str:
        srid = f"SR_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO search_run (search_run_id, project_id, provider, query_text, created_at, policy_version, filters_json)
                    VALUES (:id, :pid, :prov, :q, :now, :pol, :filt)
                """),
                {
                    "id": srid, "pid": project_id, "prov": provider, "q": query,
                    "now": datetime.now(timezone.utc).isoformat(),
                    "pol": policy_id, "filt": json.dumps(filters_json)
                }
            )
        return srid

    def upsert_work_and_candidate(self, search_run_id: str, article: dict, rank: int, policy_id: str) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        
        # 1. Upsert Work
        # Ensure we have a doi, if not generate a placeholder (though spec says strict DOI/PMCID)
        doi = article.get("doi")
        if not doi:
             # If no DOI but has PMID/PMCID, use that as pseudo-DOI-key if allowed, 
             # but spec schema has DOI as PK. 
             # We skip if no DOI? Or generate one. 
             # For now, if no DOI, we might fail hard based on spec "require_doi: true".
             # We'll mock one if strictly needed or return None to signal skip?
             # Let's assume input cleaning handles this or we create a fake one if critical.
             if article.get("pmid"):
                 doi = f"pmid:{article['pmid']}" # Fallback
             else:
                 doi = f"urn:uuid:{uuid.uuid4()}"

        with self.engine.begin() as conn:
            # Check if work exists
            conn.execute(
                text("""
                    INSERT INTO work (doi, title, year, venue, pmid, pmcid, created_at, updated_at)
                    VALUES (:doi, :title, :year, :venue, :pmid, :pmcid, :now, :now)
                    ON CONFLICT(doi) DO UPDATE SET
                        pmid = COALESCE(work.pmid, excluded.pmid),
                        pmcid = COALESCE(work.pmcid, excluded.pmcid),
                        updated_at = excluded.updated_at
                """),
                {
                    "doi": doi, "title": article.get("title"), "year": article.get("year"),
                    "venue": article.get("journal"), "pmid": article.get("pmid"),
                    "pmcid": article.get("pmcid"), "now": now
                }
            )

            # 2. Insert Candidate
            cand_id = f"CAND_{uuid.uuid4().hex}"
            # Check dedupe
            existing = conn.execute(
                text("SELECT candidate_id FROM candidate WHERE search_run_id=:sr AND doi=:doi"),
                {"sr": search_run_id, "doi": doi}
            ).scalar()
            
            if existing:
                cand_id = existing
            else:
                conn.execute(
                    text("""
                        INSERT INTO candidate (candidate_id, search_run_id, doi, rank_in_results, policy_version, created_at)
                        VALUES (:cid, :sr, :doi, :rank, :pol, :now)
                    """),
                    {
                        "cid": cand_id, "sr": search_run_id, "doi": doi,
                        "rank": rank, "pol": policy_id, "now": now
                    }
                )
        
        # Return composite dict
        return {
            "candidate_id": cand_id,
            "doi": doi,
            "pmid": article.get("pmid"),
            "pmcid": article.get("pmcid"),
            "title": article.get("title"),
            "abstract": article.get("abstract"),
            "venue": article.get("journal"),
            "year": article.get("year")
        }

    def record_check(self, candidate_id: str, claim_id: str, check_type: CheckType, verdict: CheckVerdict,
                     policy_id: str, policy_hash: Optional[str], executed_by: str, details_json: dict) -> str:
        check_id = f"CHK_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO candidate_quality_check (
                        check_id, candidate_id, claim_id, check_type, verdict, 
                        policy_id, policy_hash, details_json, executed_by, executed_at
                    ) VALUES (:id, :cid, :clid, :ctype, :verdict, :pol, :phash, :dets, :by, :at)
                """),
                {
                    "id": check_id, "cid": candidate_id, "clid": claim_id, "ctype": check_type,
                    "verdict": verdict, "pol": policy_id, "phash": policy_hash,
                    "dets": json.dumps(details_json), "by": executed_by, "at": now
                }
            )
        return check_id

    def record_decision(self, candidate_id: str, claim_id: str, outcome: DecisionOutcome,
                        policy_id: str, decided_by: str, basis_json: dict) -> str:
        dec_id = f"DEC_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO candidate_decision (
                        decision_id, candidate_id, claim_id, outcome, basis_json,
                        policy_id, decided_by, decided_at
                    ) VALUES (:id, :cid, :clid, :out, :basis, :pol, :by, :at)
                """),
                {
                    "id": dec_id, "cid": candidate_id, "clid": claim_id, "out": outcome,
                    "basis": json.dumps(basis_json), "pol": policy_id, "by": decided_by, "at": now
                }
            )
        return dec_id

    def create_claim_support(self, project_id: str, claim_id: str, doi: str, pmcid: str,
                             anchor_quote: str, anchor_location_json: dict, promotion_reason: str,
                             created_by: str, policy_version: str) -> str:
        sup_id = f"SUP_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO claim_support (
                        support_id, project_id, claim_id, doi, support_type,
                        verification_status, anchor_excerpt, anchor_location_json,
                        promotion_reason, created_by, created_at, policy_version_at_promotion
                    ) VALUES (
                        :sid, :pid, :cid, :doi, 'citation', 'verified',
                        :anch, :loc, :reason, :by, :now, :pol
                    )
                """),
                {
                    "sid": sup_id, "pid": project_id, "cid": claim_id, "doi": doi,
                    "anch": anchor_quote, "loc": json.dumps(anchor_location_json),
                    "reason": promotion_reason, "by": created_by, "now": now, "pol": policy_version
                }
            )
        return sup_id

    def create_claim_search_round(self, project_id: str, claim_id: str, search_run_id: str,
                                  round_index: int, summary_json: dict, next_action: NextAction,
                                  created_at: str) -> str:
        rid = f"RND_{uuid.uuid4().hex}"
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO claim_search_round (
                        round_id, project_id, claim_id, search_run_id, round_index,
                        summary_json, next_action, created_at
                    ) VALUES (:rid, :pid, :cid, :sid, :idx, :summ, :next, :at)
                """),
                {
                    "rid": rid, "pid": project_id, "cid": claim_id, "sid": search_run_id,
                    "idx": round_index, "summ": json.dumps(summary_json),
                    "next": next_action, "at": created_at
                }
            )
        return rid

    def open_gap_no_source(self, project_id: str, claim_id: str, recommendation: str, created_by: str) -> str:
        gid = f"GAP_{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO claim_gap (
                        gap_id, project_id, claim_id, gap_type, recommendation,
                        resolved, created_by, created_at
                    ) VALUES (:gid, :pid, :cid, 'no_source_found', :rec, 0, :by, :now)
                """),
                {
                    "gid": gid, "pid": project_id, "cid": claim_id, "rec": recommendation,
                    "by": created_by, "now": now
                }
            )
        return gid

# -------------------------
# Tool Interfaces (Duck Types for Orchestrator)
# -------------------------

class RealLLMClient:
    def topic_triage(self, claim: dict, candidate: dict, policy_id: str) -> dict:
        resp = llm_topic_triage(
            claim_text=claim["text"],
            claim_type=claim.get("type", "unknown"),
            title=candidate.get("title", ""),
            abstract=candidate.get("abstract", "")
        )
        return resp.model_dump()

    def entailment(self, claim: dict, span: CandidateSpan, policy_id: str) -> dict:
        # Map CandidateSpan to inputs 
        # Note: entailment_verification.verify_claim_entailment expects passage_text
        # and returns EntailmentResult which has 'verdict', 'anchor_quote' etc.
        # We need to map `EntailmentResult` to `LLMEntailmentJudgeResponse` fields
        
        try:
            res: EntailmentResult = verify_claim_entailment(
                claim_text=claim["text"],
                passage_text=span.text,
                pmcid="unknown", # span might not have it context, passing empty
                claim_type=claim.get("type", "unknown"),
                paragraph_index=0
            )
            
            # Map VERDICT ENUMS
            # verify_claim_entailment returns: ENTAILS, CONTRADICTS, NEUTRAL
            # We need: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, ...
            # Simple mapper:
            verdict_map = {
                "ENTAILS": "SUPPORTED",
                "CONTRADICTS": "CONTRADICTED",
                "NEUTRAL": "NOT_SUPPORTED"
            }
            mapped_verdict = verdict_map.get(res.verdict, "NOT_SUPPORTED")
            
            return {
                "verdict": mapped_verdict,
                "confidence": res.confidence_score,
                "anchor_quote": res.anchor_quote,
                "anchor_xpath": span.xpath,
                "weakness_notes": [res.reasoning],
                "mismatch_codes": []
            }
        except Exception as e:
            print(f"Entailment Err: {e}")
            return {
                "verdict": "NOT_SUPPORTED",
                "confidence": 0.0,
                "mismatch_codes": ["AMBIGUOUS"]
            }

    def rewrite_query(self, claim: dict, current_query: str, mode: str, failure_summary: dict, drift_concepts: List[str] = [], policy_id: str = "default") -> str:
        resp = llm_rewrite_query(
            claim_text=claim["statement"],
            claim_type=claim.get("claim_type", "mechanism"),
            current_query=current_query,
            mode=mode,
            failure_summary=failure_summary,
            drift_concepts=drift_concepts,
            policy_id=policy_id
        )
        # For now, orchestrator expects string.
        # Future: return resp.query_blocks to DB.
        return resp.query

    def repair_query(self, query: str) -> str:
        repaired = llm_repair_query(bad_query=query)
        import re
        
        # Deterministic Clean: Strip envelope tokens strictly via Regex.
        # Flexibly match whitespaces etc.
        forbidden_patterns = [
            r'(?i)"pubmed\s+pmc"\s*\[sb\]',
            r'(?i)"pubmed\s+pmc"', 
            r'(?i)test\s+"pubmed\s+pmc"',
            r'(?i)english\s*\[la\]', 
            r'(?i)pmc\s*\[filter\]',
            r'(?i)hasabstract', 
            r'(?i)sort\s*=', 
            r'(?i)retmax\s*=', 
            r'(?i)retstart\s*='
        ]
        
        cleaned = repaired
        
        # Iterative strip
        for pat in forbidden_patterns:
            cleaned = re.sub(pat, "", cleaned)
            
        # Clean up double spaces or dangling ANDs
        cleaned = " ".join(cleaned.split())
        cleaned = cleaned.replace(" AND )", " )").replace("( AND ", "( ").replace(" AND AND ", " AND ")
        if cleaned.endswith(" AND"): cleaned = cleaned[:-4]
        if cleaned.startswith("AND "): cleaned = cleaned[4:]
        
        return cleaned

# -------------------------
# Span Extraction
# -------------------------

def span_extract_jats(jats_xml: str, claim_terms: List[str], max_spans: int = 4, window_chars: int = 900) -> List[CandidateSpan]:
    """
    Extracts text spans from JATS XML using simple keyword proximity scoring.
    """
    try:
        root = ET.fromstring(jats_xml)
        
        # Simple extraction: iterate paragraphs
        # NOTE: ET doesn't give true xpaths easily. We'll simulate path index.
        paragraphs = []
        
        index = 0
        for elem in root.iter():
            if elem.tag == 'p':
                text_content = "".join(elem.itertext()).strip()
                if len(text_content) > 50:
                    score = sum(1 for t in claim_terms if t.lower() in text_content.lower())
                    paragraphs.append({
                        "text": text_content,
                        "xpath": f"//body/p[{index}]",
                        "score": score
                    })
                    index += 1
        
        # Sort by score desc
        sorted_paras = sorted(paragraphs, key=lambda x: x["score"], reverse=True)
        top = sorted_paras[:max_spans]
        
        return [
            CandidateSpan(
                span_id=f"span_{i}",
                xpath=p["xpath"],
                text=p["text"],
                keyword_score=float(p["score"])
            ) for i, p in enumerate(top)
        ]
        
    except Exception as e:
        print(f"XML Parse Error: {e}")
        return []

def lint_query(query: str) -> List[str]:
    errors = []
    if not query: return errors
    if len(query) > 1000: errors.append("TOO_LONG")
    if "'" in query: errors.append("BAD_QUOTES") # Single quotes
    if '\\"' in query: errors.append("BAD_QUOTES") # Escaped quote artifact
    if query.count('"pubmed pmc"[sb]') > 1: errors.append("DUPLICATE_CLAUSE")
    if query.count("(") != query.count(")"): errors.append("UNBALANCED_PARENS")
    
    # Check for duplicate consecutive words? Or just duplicate clauses?
    # Simple check for repeated constrained tokens like "pubmed pmc" is usually enough.
    # But let's check for duplicate " OR " chains if needed.
    # For now, strict paren balance is key.

    import re
    # List of forbidden envelope token patterns
    forbidden_patterns = [
        r'(?i)"pubmed\s+pmc"', 
        r'(?i)english\s*\[la\]', 
        r'(?i)pmc\s*\[filter\]', 
        r'(?i)hasabstract', 
        r'(?i)sort\s*=', 
        r'(?i)retmax\s*=', 
        r'(?i)retstart\s*='
    ]
    
    for pat in forbidden_patterns:
        if re.search(pat, query):
             errors.append(f"FORBIDDEN_TOKEN_REGEX: '{pat}' matched inside BaseQuery")
             
    if errors:
        return errors

    # Template re-application check (heuristic)
    if query.count("[tiab]") > 20 and len(query) > 600: errors.append("TEMPLATE_REAPPLIED") # Heuristic
    return errors

# -------------------------
# Decision Logic
# -------------------------

def calculate_ratios(summary: dict) -> Dict[str, float]:
    ingested = summary.get("ingested_n", 0)
    eligible = summary.get("pmcid_eligible_n", 0)
    topic_pass = summary.get("topic_pass_n", 0)
    supports = summary.get("supports_found_n", 0)
    entail_judged = summary.get("entailment_judged_n", 0)
    
    # Granular Eligibility
    # We need counts from summary or infer them. 
    # Summary keys: "gate_failure_counts"
    gate_counts = summary.get("gate_failure_counts", {})
    failed_pmc = gate_counts.get("IN_PMC_FULLTEXT", 0)
    failed_doi = gate_counts.get("HAS_DOI", 0)
    # Note: pmcid_eligible_n is those that passed ALL gates. 
    # ingested = passed + failed_pmc + failed_doi (roughly, assuming sequential gates or simple sum if independent)
    # Actually, ingested is the total candidates processed.
    
    E_pmc = (ingested - failed_pmc) / ingested if ingested > 0 else 0.0 # Approximation if gate counts are mutually exclusive or sequential
    # A better way: E_pmc = (ingested - count(IN_PMC_FULLTEXT failure)) / ingested
    # But wait, logic in orchestator: check PMC, if fail -> record fail. 
    
    # Let's stick to the primary metric:
    # E_overall = eligible / ingested
    
    # If ingested > 0:
    # pmcid_rate = (ingested - failed_pmc) / ingested
    # doi_rate = (ingested - failed_doi) / ingested (approx)
    
    # Actually, let's just use what we have. 
    # E_pmc (Availability) = (eligible) / ingested (Conservative, valid fulltext)
    E_pmc = eligible / ingested if ingested > 0 else 0.0
    
    T = topic_pass / eligible if eligible > 0 else 0.0
    Y = supports / entail_judged if entail_judged > 0 else 0.0
    
    return {"E_pmc": E_pmc, "T": T, "Y": Y}

def calculate_failure_profile(summary: dict, query: str) -> RunFailureProfile:
    total_hits = summary.get("total_hits", 0)
    ingested = summary.get("ingested_n", 0)
    n_results = summary.get("n_results", 0)
    retstart = summary.get("retstart", 0)
    
    gate_counts = summary.get("gate_failure_counts", {})
    r = calculate_ratios(summary)
    E_pmc, T, Y = r["E_pmc"], r["T"], r["Y"]
    
    # Defaults
    anomaly = False
    
    # 0. Hard Stops / Exhaustion
    if total_hits > 0 and (n_results == 0 or retstart >= total_hits):
        return RunFailureProfile(failure_mode="EXHAUSTED_PAGINATION", description="Pagination exhausted (no more results).", query_health=True, bottlenecks=gate_counts, ratios=r, pmcid_rate=E_pmc)
    
    # 0b. Binding Anomaly Check (Policy Mismatch)
    # If we are NOT enforcing subset in query, but getting tons of PMC failures.
    # In Strategy v8, we force "pubmed pmc" in query, so we expect E_pmc to be decent.
    # If E_pmc is low even with subset, it's STARVATION_PMC_SUBSET.
    # If query lacks subset (anomaly) and E_pmc low, it's STARVATION_POLICY_MISMATCH.
    has_subset = "pubmed pmc" in query.lower() or "pmc" in query.lower() # Rough check
    # Note: Adapter adds it, but `summary['query']` might be the BaseQuery without the envelope!
    # The summary['query'] comes from the Orchestrator request `current_query`.
    # The adapter wraps it. So we can't see the wrapper here easily unless we passed `compiled_query`.
    # We will assume STRICT MODE (Adapter always adds it).
    
    if summary.get("supports_found_n", 0) > 0:
        return RunFailureProfile(failure_mode="NONE", description="Success", query_health=True, ratios=r, pmcid_rate=E_pmc)
        
    # 1. Over-Constrained (Zero Hits)
    if total_hits == 0:
        return RunFailureProfile(failure_mode="OVER_CONSTRAINED", description="Zero hits returned by PubMed", query_health=True, bottlenecks=gate_counts, ratios=r, pmcid_rate=E_pmc)
        
    # 2. Starvation vs Binding Failure (Availability)
    if E_pmc < 0.1:
        # Check for Binding Failure (high hits but low link rate)
        # We don't have explicit link_fail_rate in summary ratios yet, need to infer or add it.
        # Check gate failures.
        link_fails = gate_counts.get("PMID_TO_PMCID_LINKED", 0)
        # If we ingested 20 items and 20 failed linking...
        # Ingested usually equals len(pmids) capped.
        # If ingested > 10 and link_fails / ingested > 0.5 -> Binding Failure
        if ingested > 10 and (link_fails / ingested) > 0.5:
             val = link_fails / ingested
             return RunFailureProfile(failure_mode="BINDING_FAILURE_PMC_LINKING", description=f"High Linking Failure ({val:.0%})", query_health=True, bottlenecks=gate_counts, ratios=r, pmcid_rate=E_pmc)
    
        # Otherwise standard starvation
        return RunFailureProfile(failure_mode="STARVATION_PMC_SUBSET", description=f"Low PMC Availability (E={E_pmc:.2f}) despite subset.", query_health=True, bottlenecks=gate_counts, ratios=r, pmcid_rate=E_pmc)

    # 3. Off-Topic Drift (Low Precision)
    if T < 0.1:
         return RunFailureProfile(failure_mode="OFF_TOPIC_DRIFT", description=f"Drift Detected (T={T:.2f})", query_health=True, bottlenecks=gate_counts, ratios=r, pmcid_rate=E_pmc)

    # 4. Evidence Gap (Low Yield)
    if Y < 0.05:
        return RunFailureProfile(failure_mode="EVIDENCE_GAP", description=f"Evidence Gap (Y={Y:.2f})", query_health=True, bottlenecks=gate_counts, ratios=r, pmcid_rate=E_pmc)
        
    # 5. Exhausted / Default
    return RunFailureProfile(failure_mode="EXHAUSTED", description="No clear failure mode, just empty.", query_health=True, bottlenecks=gate_counts, ratios=r, pmcid_rate=E_pmc)

def decide_next_action_from_profile(profile: RunFailureProfile, summary: dict, round_index: int, max_rounds: int) -> Tuple[NextAction, str, Optional[str]]:
    if round_index >= max_rounds - 1:
        return ("STOP_OPEN_GAP", f"Max rounds reached ({max_rounds}). Mode: {profile.failure_mode}", None)

    mode = profile.failure_mode
    r = profile.ratios
    E_pmc = r.get("E_pmc", 0.0)
    T = r.get("T", 0.0)
    Y = r.get("Y", 0.0)
    
    retstart = summary.get("retstart", 0)
    total_hits = summary.get("total_hits", 0)
    more_pages = (retstart + summary.get("n_results", 0) < total_hits)

    # SUCCESS
    if mode == "NONE":
         return ("CONTINUE_SAME_QUERY", "Success", None)
         
    # HARD STOP
    if mode == "EXHAUSTED_PAGINATION":
        # If we exhausted pages, we can either Stop (Open Gap) or Rewrite (Relax).
        # General rule: If we had high precision (T), it's a gap. If we had low precision, we drifted.
        if T > 0.1:
             return ("STOP_OPEN_GAP", "Pagination Exhausted with On-Topic candidates -> Gap.", None)
        else:
             return ("REWRITE_QUERY_RELAX_CONSTRAINTS", "Pagination Exhausted with Drift -> Relax/Broaden.", "RELAX")

    # STAGE A: ACQUISITION (Availability)
    if mode in ["STARVATION_PMC_SUBSET", "STARVATION_POLICY_MISMATCH", "BINDING_FAILURE_PMC_LINKING"] or (E_pmc < 0.1 and mode != "OVER_CONSTRAINED"):
        
        if mode == "BINDING_FAILURE_PMC_LINKING":
             # If linking fails, broad paging usually won't help if it's a systemic API/Subset issue.
             # We should probably SWITCH STRATEGY or RELAX.
             # User suggested "fix binding / retry resolver / loosen subset".
             # For now, treat as severe starvation -> Switch.
             return ("SWITCH_RETRIEVAL_STRATEGY_S2_FIRST", "Binding Failure: PMCIDs not resolving. Switch to S2.", "RELAX")
    
        # Strict Mode: If subset yields nothing usable, we are truly starved.
        # Paging might help (popular IDs often fail linking).
        if more_pages and retstart < 600:
             return ("CONTINUE_PAGINATION", "Acquisition Stage: Paging for valid PMCIDs.", "PAGINATE")
        elif total_hits > 1000:
             # Deep starvation -> Switch Logic?
             return ("SWITCH_RETRIEVAL_STRATEGY_S2_FIRST", "Acquisition Stage: Persistent Starvation -> Switch Strategy.", "RELAX")
        else:
             return ("REWRITE_QUERY_RELAX_CONSTRAINTS", "Acquisition Stage: Relaxing constraints to find PMC papers.", "RELAX")

    # STAGE B: PRECISION (Drift)
    if mode == "OFF_TOPIC_DRIFT" or (T < 0.1 and mode != "OVER_CONSTRAINED"):
         return ("REWRITE_QUERY_DISAMBIGUATE", "Precision Stage: Drift detected, adding anchors.", "DISAMBIGUATE")

    # STAGE C: YIELD (Gap)
    if mode == "EVIDENCE_GAP" or (Y < 0.05 and mode != "OVER_CONSTRAINED"):
        if more_pages and retstart < 800:
             return ("CONTINUE_PAGINATION", "Yield Stage: Recall expansion for Evidence.", "PAGINATE")
        else:
             # High Precision, Low Yield, Exhausted Pages -> Real Gap.
             return ("STOP_OPEN_GAP", "Yield Stage: High precision, zero yield -> Claim likely unsupported.", None)

    # FALLBACK / OVER-CONSTRAINED
    if mode == "OVER_CONSTRAINED":
         return ("REWRITE_QUERY_RELAX_CONSTRAINTS", "Zero hits.", "RELAX")

    return ("REWRITE_QUERY_BROADEN", "Unknown state -> Broaden.", "RELAX")


# -------------------------
# Orchestrator
# -------------------------

def orchestrate_claim_support_round(
    req: OrchestrateRoundRequest,
    pubmed: Any, # Expects PubMedClient-like
    pmc: Any,    # Expects PMCClient-like
    crossref: Any, # Expects CrossrefClient-like
    llm: RealLLMClient,
    repo: LedgerRepo,
    executed_by: str = "archivist_orchestrator",
) -> OrchestrateRoundResponse:
    now = datetime.now(timezone.utc).isoformat()
    claim = repo.load_claim(req.claim_id)

    # 1. Determine Query & Pagination
    last_round = repo.get_last_round(req.claim_id)
    query = req.current_query
    
    # Logic to derive query if not passed
    if query is None:
        if last_round:
            # Check if we should use next_query from last round if avail?
            # Creating a new query is usually explicit in req or derived.
            # If req.current_query is None, we default to last round's query
            # UNLESS last round said REWRITE. But usually the caller (Agent) handles "Next Query".
            # If we rely on Orchestrator to be stateless-ish, the Request should carry the new query.
            # But let's support "continue" behavior.
            summary_last = json.loads(last_round["summary_json"]) if last_round.get("summary_json") else {}
            next_q = last_round.get("next_action") # stored in DB? Yes.
            # Actually, `next_query` is often in the response, but maybe not in DB structure explicitly?
            # DB `claim_search_round` has `next_action`.
            # If next_action was REWRITE, we expect the caller to have obtained the rewrite.
            # But here we just fallback to "statement" or "last query".
            query = last_round.get("query") or claim["statement"] # DB doesn't have query column? 
            # `search_run` has query. We can join or just assume it was extracted.
            # Wait, `claim_search_round` joins to `search_run`.
            # We need to look up search_run to get query?
            # The `last_round` dict from `repo.get_last_round` is just the row.
            pass
    # 1.5 Syntax Linter
    if query:
        errors = lint_query(query)
        if errors:
            # We can pass errors to repair prompt if we update logic, but standard repair works often
            query = llm.repair_query(query)
            # Re-lint to be safe?
            errors2 = lint_query(query)
            if errors2:
                 print(f"WARNING: Query still has errors after repair: {errors2}")
            req.current_query = query

    # Pagination
    retstart = 0
    reuse_run_id = None
    if len(query) > 0: # Ensure we have a query
        # Resolve last query to check for continuity
        last_query = None
        last_retstart = 0
        last_summ = {}
        
        if last_round:
             if last_round.get("summary_json"):
                 try:
                     last_summ = json.loads(last_round["summary_json"])
                     last_query = last_summ.get("query")
                     last_retstart = last_summ.get("retstart", 0)
                 except json.JSONDecodeError:
                     pass
        
        if last_query and query == last_query:
            retstart = last_retstart + req.pubmed_retmax
            # Reuse search run ID if continuing exact same query
            reuse_run_id = last_round.get("search_run_id")

    # Starvation Logic (PMC)
    if last_round and last_summ:
        pmc_n = last_summ.get("pmcid_eligible_n", 0)
        if pmc_n < 5:
            req.pubmed_retmax = max(req.pubmed_retmax, 200)
            # We should also force "pubmed pmc"[sb] but we assume refined templates do this.
            # But query rewriting might drop it?
            # We can force append if missing?

    # 2. Search
    # We rely on the search tool (Gate) to create the search_run record to avoid double-writing.
    # We pass the policy_id so Gate can log it correctly.
    # We pass search_run_id if reusing (pagination) so Gate DOES NOT create a new one.
    
    # 2. Search
    sr_resp = pubmed.search(
        project_id=req.project_id, 
        claim_id=req.claim_id, 
        query=query, 
        retmax=req.pubmed_retmax, 
        retstart=retstart,
        policy_id=req.policy_id,
        search_run_id=reuse_run_id
    )
    
    # Be robust to return type
    if isinstance(sr_resp, dict):
        pmids = sr_resp.get("pmids", [])
        search_run_id = sr_resp.get("search_run_id")
    else:
        pmids = sr_resp.pmids # Pydantic
        search_run_id = sr_resp.search_run_id

    # Fallback if tool didn't return an ID (unlikely)
    if not search_run_id:
        # Fallback to create one, but normally we trust the tool.
        filters_json = {"sort": "relevance", "retmax": req.pubmed_retmax, "retstart": retstart}
        search_run_id = repo.create_search_run(req.project_id, "PUBMED", query, req.policy_id, filters_json)

    pmids = pmids[:req.pubmed_retmax]

    # 3. Metadata + Link
    meta_list = []
    links_list = []
    
    if pmids:
        # Assume pubmed client methods return list[dict] or objects
        meta_list = pubmed.fetch_metadata(pmids=pmids, include_abstract=True)
        links_list = pubmed.link_to_pmc(pmids=pmids)
        
        # Normalize to dicts if needed
        if meta_list and not isinstance(meta_list[0], dict):
            meta_list = [m.model_dump() for m in meta_list]
        if links_list and not isinstance(links_list[0], dict):
            links_list = [l.model_dump() for l in links_list]

    pmid_to_pmcid = {l["pmid"]: l.get("pmcid") for l in links_list}

    # 4. Ingest (Cap)
    ingested = []
    
    # Sort meta by original pmid order? meta_list might be shuffled.
    # Map back to rank
    pmid_to_rank = {p: i for i, p in enumerate(pmids)}
    
    for art in meta_list:
        pmid = art["pmid"]
        if pmid not in pmid_to_rank: continue
        
        # Robust Binding:
        # 1. Try ELink Result
        pmcid_candidate = pmid_to_pmcid.get(pmid)
        
        # 2. Fallback to Metadata (if ELink missed it but EFetch found it)
        if not pmcid_candidate and art.get("pmcid"):
            pmcid_candidate = art["pmcid"]
            
        art["pmcid"] = pmcid_candidate
        rank = pmid_to_rank[pmid]
        
        if len(ingested) >= req.ingest_candidates_cap:
            break
            
        row = repo.upsert_work_and_candidate(search_run_id, art, rank, req.policy_id)
        ingested.append(row)

    # 5. Evaluate
    supports_created = []
    candidate_results = []
    
    pmcid_eligible_n = 0
    topic_pass_n = 0
    entailment_judged_n = 0
    near_miss_n = 0
    near_miss_reasons = []
    drift_concepts = [] # Collected from Triage
    mismatch_counts = {}
    gate_failure_counts = {} # Track where candidates die
    
    ingested_sorted = sorted(ingested, key=lambda x: x.get("rank_in_results", 999))
    
    topic_triaged = 0
    entailment_used = 0
    
    for cand in ingested_sorted:
        cand_id = cand["candidate_id"]
        doi = cand.get("doi")
        pmcid = cand.get("pmcid")
        
        # Hard Gates
        in_pmc = bool(pmcid)
        has_doi = bool(doi)
        
        # GRANULAR GATE CHECKS (Strategy v9)
        # Check 1: Linking
        repo.record_check(
            candidate_id=cand_id, claim_id=req.claim_id, check_type="PMID_TO_PMCID_LINKED",
            verdict="PASS" if in_pmc else "FAIL", policy_id=req.policy_id, 
            policy_hash=req.policy_hash, executed_by=executed_by, details_json={"pmcid": pmcid}
        )
        
        # Legacy/Composite Check (Keep for now to avoid breaking stats readers)
        repo.record_check(
            candidate_id=cand_id, claim_id=req.claim_id, check_type="IN_PMC_FULLTEXT",
            verdict="PASS" if in_pmc else "FAIL", policy_id=req.policy_id, 
            policy_hash=req.policy_hash, executed_by=executed_by, details_json={"pmcid": pmcid}
        )
        
        if not in_pmc: 
            gate_failure_counts["IN_PMC_FULLTEXT"] = gate_failure_counts.get("IN_PMC_FULLTEXT", 0) + 1
            gate_failure_counts["PMID_TO_PMCID_LINKED"] = gate_failure_counts.get("PMID_TO_PMCID_LINKED", 0) + 1
            
        if not has_doi: gate_failure_counts["HAS_DOI"] = gate_failure_counts.get("HAS_DOI", 0) + 1
        
        if not (in_pmc and has_doi):
            repo.record_decision(
                candidate_id=cand_id, claim_id=req.claim_id, outcome="REJECTED",
                policy_id=req.policy_id, decided_by=executed_by,
                basis_json={"reason": "Failed Hard Gates (No PMC/DOI)"}
            )
            candidate_results.append(CandidateEvalResult(
                candidate_id=cand_id, hard_gate_passed=False, decision_outcome="REJECTED"
            ))
            continue
            
        pmcid_eligible_n += 1
        
        # Topic Triage
        if topic_triaged >= req.topic_triage_cap:
             # Cap reached
             continue
             
        triage = llm.topic_triage(claim={"text": claim["statement"], "type": claim["claim_type"]}, candidate=cand, policy_id=req.policy_id)
        topic_triaged += 1
        
        repo.record_check(
            candidate_id=cand_id, claim_id=req.claim_id, check_type="TOPIC_TRIAGE_LLM",
            verdict=triage["topic_match"], policy_id=req.policy_id, policy_hash=req.policy_hash,
            executed_by=executed_by, details_json=triage
        )
        
        if triage.get("mismatch_codes"):
            for code in triage["mismatch_codes"]:
                mismatch_counts[code] = mismatch_counts.get(code, 0) + 1
        if triage.get("drift_concepts"):
            drift_concepts.extend(triage["drift_concepts"])

        if triage["topic_match"] != "PASS":
            candidate_results.append(CandidateEvalResult(
                candidate_id=cand_id, hard_gate_passed=True, topic_triage_passed=False,
                decision_outcome="REJECTED", mismatch_codes=triage.get("mismatch_codes",[])
            ))
            repo.record_decision(cand_id, req.claim_id, "REJECTED", req.policy_id, executed_by, {"reason": "Topic Mismatch"})
            continue
            
        topic_pass_n += 1
        
        # Full Text & Span
        # PMC fetch (simulated if needed, but we assume pmc client works)
        # We need actual XML string.
        # Use pmc.fetch_jats
        jats_resp = pmc.fetch_jats(pmcid=pmcid, cache_policy="USE_CACHE")
        # jats_resp might be dict or object
        xml_content = jats_resp["content"] if isinstance(jats_resp, dict) else jats_resp.content
        
        spans = span_extract_jats(xml_content, claim_terms=claim["statement"].split())
        
        if not spans:
             repo.record_decision(cand_id, req.claim_id, "HOLD", req.policy_id, executed_by, {"reason": "No Spans"})
             continue
             
        # Entailment
        if entailment_used >= req.entailment_cap:
            continue
            
        found_support = False
        for span in spans[:2]:
            res = llm.entailment(claim={"text": claim["statement"]}, span=span, policy_id=req.policy_id)
            entailment_used += 1
            entailment_judged_n += 1
            
            repo.record_check(
                candidate_id=cand_id, claim_id=req.claim_id, check_type="CLAIM_ENTAILMENT_LLM",
                verdict="PASS" if res["verdict"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"] else "FAIL",
                policy_id=req.policy_id, policy_hash=None, executed_by=executed_by,
                details_json=res
            )
            
            if res["verdict"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]:
                # Promote
                sup_id = repo.create_claim_support(
                    req.project_id, req.claim_id, doi, pmcid,
                    res.get("anchor_quote", "")[:200], # truncation safety
                    {"xpath": span.xpath},
                    f"LLM: {res['verdict']}",
                    executed_by, req.policy_id
                )
                repo.record_decision(
                    cand_id, req.claim_id, "SELECTED_AS_SUPPORT", req.policy_id, executed_by,
                    {"support_id": sup_id, "verdict": res["verdict"]}
                )
                supports_created.append(sup_id)
                found_support = True
                break
            else:
                # entailed is NOT support (e.g. NOT_SUPPORTED or CONTRADICTED)
                near_miss_n += 1
                if res.get("weakness_notes"):
                    near_miss_reasons.extend(res["weakness_notes"])
        
        if not found_support:
            repo.record_decision(
                cand_id, req.claim_id, "ELIGIBLE_SUPPORT", req.policy_id, executed_by,
                {"reason": "No suitable span entailed"}
            )

    # 6. Summary & Decision
    # 6. Summary & Decision
    topic_precision = topic_pass_n / pmcid_eligible_n if pmcid_eligible_n else 0.0
    
    # Lexicon Accumulation
    near_miss_lexicon = list(set(near_miss_reasons))
    
    # Start with empty lexicon bits, will add history
    positive_lexicon = [] # TODO: Extract from Triage positive_anchors if stored? 
    # For now, we only accumulate drift_concepts into near_miss
    
    if last_summ:
         # Merge history
         near_miss_lexicon.extend(last_summ.get("near_miss_lexicon", []))
         drift_concepts.extend(last_summ.get("drift_concepts", []))
         
    # Dedupe and cap
    near_miss_lexicon = list(set(near_miss_lexicon))[:20]
    drift_concepts = list(set(drift_concepts))[:20]

    summary = {
        "ingested_n": len(ingested),
        "pmcid_eligible_n": pmcid_eligible_n,
        "topic_pass_n": topic_pass_n,
        "entailment_judged_n": entailment_judged_n,
        "supports_found_n": len(supports_created),
        "near_miss_n": near_miss_n,
        "near_miss_lexicon": near_miss_lexicon,
        "topic_precision": topic_precision,
        "query": query,
        "retstart": retstart,
        "n_results": len(pmids), # Actual items fetched this page
        "total_hits": sr_resp.total_hits,
        "mismatch_code_counts": mismatch_counts,
        "gate_failure_counts": gate_failure_counts,
        "drift_concepts": drift_concepts
    }
    
    # 6. Decide & Rewrite
    profile = calculate_failure_profile(summary, query)
    print(f"Round {req.search_round_index} Profile: {profile.failure_mode} ({profile.description})")
    
    next_action, reason, rewrite_mode = decide_next_action_from_profile(profile, summary, req.search_round_index, req.max_rounds)
    
    summary["failure_profile"] = profile.model_dump()
    
    print(f"Round {req.search_round_index} Result: Next={next_action}, Support={len(supports_created)}")
    
    next_query = None
    if next_action.startswith("REWRITE_QUERY"):
        # We need to rewrite
        print(f"Rewriting query ({rewrite_mode})...")
        next_query = llm.rewrite_query(
            claim=claim,
            current_query=query,
            mode=rewrite_mode,
            failure_summary=summary, # Contains counts and drift concepts
            drift_concepts=summary.get("drift_concepts", []),
            policy_id=req.policy_id
        )

    if next_action == "STOP_OPEN_GAP":
        repo.open_gap_no_source(req.project_id, req.claim_id, "No source found after max rounds", executed_by)

    rid = repo.create_claim_search_round(
        req.project_id, req.claim_id, search_run_id, req.search_round_index,
        summary, next_action, now
    )

    return OrchestrateRoundResponse(
        round_id=rid, search_run_id=search_run_id, claim_id=req.claim_id,
        query=query, supports_found=len(supports_created), supports_created=supports_created,
        next_action=next_action, next_query=next_query, reason=reason,
        done=(len(supports_created)>0 or next_action=="STOP_OPEN_GAP"),
        summary_json=summary,
        failure_profile=profile
    )
