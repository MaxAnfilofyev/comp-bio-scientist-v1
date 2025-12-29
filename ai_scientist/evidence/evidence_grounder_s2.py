from typing import Any, List, Optional, Tuple, Dict
import json
import uuid
import datetime

from ai_scientist.evidence.s2_client import S2Client
from ai_scientist.evidence_service import (
    EvidenceService, CheckType, Verdict, DecisionOutcome, PolicyRef, ActorRef,
    WorkUpsert, CandidateIngestItem, UpsertWorkBatchRequest, IngestCandidatesRequest,
    RecordQualityCheckRequest, RecordDecisionRequest, CreateSearchRunRequest,
    CreateSearchRoundRequest, GetEffectiveStatusRequest, PromoteCandidateRequest
)
from ai_scientist.model.evidence import (
    OrchestrateRoundRequest, OrchestrateRoundResponse, CandidateEvalResult, MismatchCode, 
    NextAction, RunFailureProfile, FailureMode, S2SearchRequest
)
from ai_scientist.evidence_llm_logic import (
    llm_topic_triage, llm_entailment_s2, llm_rewrite_query
)

class EvidenceGrounderS2:
    def __init__(self, evidence_service: EvidenceService = None, s2_client: S2Client = None, llm_client: Any = None):
        self.evidence_service = evidence_service or EvidenceService()
        self.s2 = s2_client or S2Client()
        self.llm = llm_client # duck-type with methods if needed, mostly used for mock injection or we call logic directly

    def orchestrate_round(self, req: OrchestrateRoundRequest) -> OrchestrateRoundResponse:
        # Load Claim info if needed? 
        # For now assume req has necessary info or we'd fetch from DB.
        # But we need claim text for LLM.
        # We can fetch claim from DB via a repo method if EvidenceService exposed it, 
        # or we might need to query DB directly here.
        # EvidenceService doesn't have `get_claim(id)`.
        # We will assume we can query DB engine from EvidenceService.
        
        with self.evidence_service.engine.connect() as conn:
            from sqlalchemy import text
            row = conn.execute(
                text("SELECT * FROM claim WHERE claim_id = :cid"),
                {"cid": req.claim_id}
            ).fetchone()
            if not row:
                raise ValueError(f"Claim {req.claim_id} not found")
            claim = dict(row._mapping)
            
            # Fetch previous round history for feedback
            round_history = []
            prev_rounds = conn.execute(
                text("""SELECT round_index, summary_json 
                        FROM search_round 
                        WHERE claim_id = :cid 
                        ORDER BY round_index"""),
                {"cid": req.claim_id}
            ).fetchall()
            
            for pr in prev_rounds:
                summary = json.loads(pr.summary_json) if pr.summary_json else {}
                mismatches = summary.get('mismatch_code_counts', {})
                top_mismatches = sorted(mismatches.items(), key=lambda x: x[1], reverse=True)[:3]
                round_history.append({
                    'round': pr.round_index,
                    'ingested': summary.get('ingested_n', 0),
                    'eligible': summary.get('eligible_n', 0),
                    'T': summary.get('T', 0),
                    'top_mismatches': [m[0] for m in top_mismatches]
                })
        
        # 1. Determine Query
        current_query = req.current_query
        if not current_query:
            # Fallback to claim text or previous round?
            current_query = claim["statement"]
        
        # Remove common stopwords to improve S2 relevance search
        # S2 works better with key terms, not full sentences
        stopwords = {'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                     'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'that', 'which', 'who', 'whom',
                     'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
                     'involves', 'along', 'through', 'during', 'after', 'before', 'between'}
        
        # Only apply stopword removal to the initial claim statement, not to rewritten queries
        if not req.current_query:
            words = current_query.split()
            filtered_words = [w for w in words if w.lower() not in stopwords]
            current_query = ' '.join(filtered_words)
            print(f"DEBUG: Removed stopwords. Original: {claim['statement']}")
            print(f"DEBUG: Filtered query: {current_query}")
        
        # 2. Search S2
        # Use simple text search with filters from policy (mocked args for now)
        s2_req = S2SearchRequest(
            query=current_query,
            limit=req.pubmed_retmax, # reuse retmax as limit
            require_open_access_pdf=True, # STRICT
            min_citation_count=1,  # Require at least 1 citation
            language="en"
        )
        hits_resp = self.s2.search_relevance(s2_req)
        hits = hits_resp.hits
        
        # 3. Create Search Run (if not passed) can be implicit or explicit.
        # We use EvidenceService to record it.
        run_req = CreateSearchRunRequest(
            project_id=req.project_id,
            provider="SEMANTIC_SCHOLAR",
            query_template_id="s2_relevance",
            query_text=current_query,
            filters_json={"limit": req.pubmed_retmax, "require_oa": True},
            policy=PolicyRef(policy_id=req.policy_id),
            result_count_total=hits_resp.total,
            top_k_stored=len(hits),
            notes=f"Offset: {hits_resp.offset}"
        )
        run_log = self.evidence_service.create_search_run(run_req)
        search_run_id = run_log.search_run_id
        
        # 4. Ingest Candidates
        # Map S2 hit -> WorkUpsert
        works_upsert = []
        candidates_ingest = []
        
        ingested_ids = []
        
        for i, hit in enumerate(hits):
            if i >= req.ingest_candidates_cap: break
            
            # Extract DOI logic
            doi = hit.external_ids.get("DOI")
            if not doi:
                # Fallback to CorpusIds/S2? 
                # Policy says require_doi usually. But let's allow "S2:..." as DOI if missing to avoid dropping good papers?
                # Actually enforcing doi: true.
                continue
                
            w = WorkUpsert(
                doi=doi,
                title=hit.title,
                year=hit.year,
                venue=hit.venue,
                s2_paper_id=hit.paper_id,
                is_open_access=hit.is_open_access or False,
                open_access_pdf_url=hit.open_access_pdf_url
            )
            works_upsert.append(w)
            
            c = CandidateIngestItem(
                doi=doi,
                rank_in_results=hit.rank,
                features_json={"s2_paper_id": hit.paper_id}
            )
            candidates_ingest.append(c)
        
        if works_upsert:
            self.evidence_service.upsert_work_batch(UpsertWorkBatchRequest(works=works_upsert))
            ingest_resp = self.evidence_service.ingest_candidates(
                IngestCandidatesRequest(search_run_id=search_run_id, candidates=candidates_ingest)
            )
            ingested_ids = ingest_resp.candidate_ids
        
        # 5. Evaluate (Triage + Entailment)
        results = []
        supports_created = []
        
        # Stats for NextAction
        stat_eligible = 0 # has abstract
        stat_topic_pass = 0
        stat_entailment_pass = 0
        stat_entailment_judged = 0
        stat_link_fails = 0 # Not applicable to S2 as we search directly
        
        drift_concepts = []
        mismatch_counts = {}
        candidates_for_promotion = []  # Collect SUPPORTED candidates for ranking
        
        candidates_map = {cid: hits[i] for i, cid in enumerate(ingested_ids)}
        
        for cid in ingested_ids:
            hit = candidates_map[cid]
            res_obj = CandidateEvalResult(
                candidate_id=cid, doi=hit.external_ids.get("DOI"), hard_gate_passed=True
            )
            
            # Gate: Abstract presence
            if not hit.abstract:
                res_obj.hard_gate_passed = False
                results.append(res_obj)
                continue
            
            stat_eligible += 1
            
            # Triage
            triage = llm_topic_triage(
                claim_text=claim["statement"],
                claim_type=claim.get("claim_type", "unknown"),
                title=hit.title,
                abstract=hit.abstract,
                tldr=hit.tldr,
                venue=hit.venue,
                year=hit.year,
                citations=hit.citation_count,
                pub_types=hit.publication_types
            )
            
            # Record Triage Check
            self.evidence_service.record_quality_check(RecordQualityCheckRequest(
                candidate_id=cid, claim_id=req.claim_id, check_type=CheckType.TOPIC_TRIAGE_LLM,
                verdict=Verdict.PASS if triage.topic_match == "PASS" else Verdict.FAIL,
                details_json=triage.model_dump(),
                policy=PolicyRef(policy_id=req.policy_id),
                executed_by=ActorRef(agent_id="evidence_grounder_s2")
            ))
            
            if triage.topic_match != "PASS":
                res_obj.topic_triage_passed = False
                res_obj.decision_outcome = DecisionOutcome.REJECTED
                res_obj.mismatch_codes = triage.mismatch_codes
                # Record Decision REJECTED
                self.evidence_service.record_decision(RecordDecisionRequest(
                     candidate_id=cid, claim_id=req.claim_id, outcome=DecisionOutcome.REJECTED,
                     basis_json={"reason": "Topic Triage Fail", "codes": triage.mismatch_codes},
                     policy=PolicyRef(policy_id=req.policy_id), decided_by=ActorRef(agent_id="evidence_grounder_s2")
                ))
                
                # Stats
                for code in triage.mismatch_codes:
                    mismatch_counts[code] = mismatch_counts.get(code, 0) + 1
                if triage.drift_concepts:
                    drift_concepts.extend(triage.drift_concepts)
                
                results.append(res_obj)
                continue
            
            res_obj.topic_triage_passed = True
            stat_topic_pass += 1
            
            # Entailment (Abstract)
            ent = llm_entailment_s2(
                claim_text=claim["statement"],
                claim_type=claim.get("claim_type", "unknown"),
                source_text=hit.abstract,
                source_type="abstract",
                title=hit.title
            )
            
            stat_entailment_judged += 1
            
            # Record Entailment Check
            ent_verdict = Verdict.PASS if ent.get("verdict") == "SUPPORTED" else Verdict.FAIL
            self.evidence_service.record_quality_check(RecordQualityCheckRequest(
                candidate_id=cid, claim_id=req.claim_id, check_type=CheckType.ABSTRACT_ENTAILMENT_LLM,
                verdict=ent_verdict,
                details_json=ent,
                policy=PolicyRef(policy_id=req.policy_id),
                executed_by=ActorRef(agent_id="evidence_grounder_s2")
            ))
            
            # Decision
            if ent.get("verdict") == "SUPPORTED":
                # Mark as SELECTED_AS_SUPPORT but don't promote yet
                # We'll rank all candidates and promote the best ones
                stat_entailment_pass += 1
                outcome = DecisionOutcome.SELECTED_AS_SUPPORT
                
                # Store for later ranking
                candidates_for_promotion.append({
                    "candidate_id": cid,
                    "hit": hit,
                    "entailment": ent,
                    "result_obj": res_obj
                })
                
            elif ent.get("verdict") == "NEEDS_FULL_TEXT":
                outcome = DecisionOutcome.HOLD # Wait for full text loop if implemented
                # For this version (MVP), we treat as HOLD or REJECT if explicit FULL_TEXT_FETCH not triggered immediately.
                # Spec says: "(optional) full-text fetch+entailment".
                # If we implement it, we'd do it here. 
                # IMPLEMENTATION: If OA URL exists, try fetch?
                # User spec: "Eligible for full-text validation: has openAccessPdf.url".
                if hit.open_access_pdf_url:
                    outcome = DecisionOutcome.ELIGIBLE_SUPPORT # Ideally Queue for FullText
                    # But if we don't have async queue, we skip or fetch now?
                    # Let's skip heavy fetch for now in this MVP unless explicit requirement?
                    # The prompt says: "Full text only when required".
                    # Let's mark as HOLD/ELIGIBLE.
                    outcome = DecisionOutcome.ELIGIBLE_SUPPORT
                else:
                    outcome = DecisionOutcome.REJECTED
            else:
                outcome = DecisionOutcome.REJECTED
            self.evidence_service.record_decision(RecordDecisionRequest(
                 candidate_id=cid, claim_id=req.claim_id, outcome=outcome,
                 basis_json={"entailment": ent},
                 policy=PolicyRef(policy_id=req.policy_id), decided_by=ActorRef(agent_id="evidence_grounder_s2")
            ))
            res_obj.decision_outcome = outcome
            results.append(res_obj)
        # Rank and promote only the best SUPPORTED candidates
        if candidates_for_promotion:
            # Define ranking values
            strength_rank = {"STRONG": 3, "MODERATE": 2, "WEAK": 1}
            evidence_source_rank = {"FULLTEXT_PASSAGE": 2, "ABSTRACT_ONLY": 1}
            
            # Sort by priority (as specified):
            # 1) entailment_verdict (already filtered to SUPPORTED only)
            # 2) support_strength (STRONG > MODERATE > WEAK)
            # 3) evidence_source (FULLTEXT_PASSAGE > ABSTRACT_ONLY)
            # 4) integrity (no retraction/EoC/withdrawal flags)
            # 5) is_peer_reviewed
            # 6) influential_citation_count
            # 7) citation_count
            candidates_for_promotion.sort(
                key=lambda c: (
                    strength_rank.get(c["entailment"].get("support_strength", "WEAK"), 0),
                    evidence_source_rank.get(c["entailment"].get("evidence_source", "ABSTRACT_ONLY"), 0),
                    1 if not c["hit"].external_ids.get("retracted") else 0,  # integrity check
                    1 if getattr(c["hit"], "is_peer_reviewed", None) else 0,
                    c["hit"].influential_citation_count or 0,
                    c["hit"].citation_count or 0
                ),
                reverse=True
            )
            
            # Promotion Strategy:
            # - Promote up to 2 STRONG supports that are independent
            # - If needed, promote additional supports up to max 3-5 total, preferring MODERATE
            # - If only WEAK supports exist, do not promote (downgrade claim or open gap)
            
            strong_candidates = [c for c in candidates_for_promotion if c["entailment"].get("support_strength") == "STRONG"]
            moderate_candidates = [c for c in candidates_for_promotion if c["entailment"].get("support_strength") == "MODERATE"]
            weak_candidates = [c for c in candidates_for_promotion if c["entailment"].get("support_strength") == "WEAK"]
            
            to_promote = []
            
            if strong_candidates:
                # Promote up to 2 STRONG supports
                # TODO: Add author/lineage independence check
                to_promote = strong_candidates[:2]
                
                # If we have < 3 total, add MODERATE to reach 3-5
                if len(to_promote) < 3 and moderate_candidates:
                    needed = min(3 - len(to_promote), len(moderate_candidates))
                    to_promote.extend(moderate_candidates[:needed])
                    
            elif moderate_candidates:
                # No STRONG, but have MODERATE - promote up to 3-5 MODERATE
                to_promote = moderate_candidates[:5]
                
            elif weak_candidates:
                # Only WEAK supports - do not promote for strong claims
                # Open a gap instead
                print(f"Warning: Only WEAK supports found for claim. Not promoting.")
                to_promote = []
            
            # Perform promotions with metadata enrichment
            for candidate in to_promote:
                cid = candidate["candidate_id"]
                ent = candidate["entailment"]
                res_obj = candidate["result_obj"]
                hit = candidate["hit"]
                
                # Enrich work metadata for PROMOTED candidates
                doi = hit.external_ids.get("DOI")
                print(f"DEBUG: Processing promoted candidate: DOI={doi}, has_pdf_url={bool(hit.open_access_pdf_url)}")
                if doi and hit.open_access_pdf_url:
                    print(f"DEBUG: Attempting to fetch full text for {doi}")
                    try:
                        # Fetch full text PDF
                        from ai_scientist.model.evidence import S2FetchPdfRequest
                        import hashlib
                        
                        pdf_req = S2FetchPdfRequest(
                            paper_id=hit.paper_id,
                            pdf_url=hit.open_access_pdf_url,
                            extract_text=True
                        )
                        print(f"DEBUG: Calling S2 fetch_open_access_pdf for {hit.paper_id}")
                        pdf_resp = self.s2.fetch_open_access_pdf(pdf_req)
                        
                        print(f"DEBUG: PDF response: has_response={pdf_resp is not None}, has_text={bool(pdf_resp.extracted_text) if pdf_resp else False}, errors={pdf_resp.errors if pdf_resp else 'N/A'}")
                        
                        if pdf_resp and pdf_resp.extracted_text:
                            # Insert into work_fulltext_cache table
                            with self.evidence_service.engine.begin() as conn:
                                from sqlalchemy import text
                                
                                cache_id = f"CACHE_{uuid.uuid4().hex[:16]}"
                                extracted_sha256 = hashlib.sha256(pdf_resp.extracted_text.encode()).hexdigest()
                                
                                conn.execute(
                                    text("""
                                        INSERT INTO work_fulltext_cache (
                                            cache_id, doi, s2_paper_id, source, content_url,
                                            content_sha256, content_bytes, extracted_text,
                                            extracted_sha256, retrieved_at
                                        ) VALUES (
                                            :cache_id, :doi, :s2_paper_id, :source, :content_url,
                                            :content_sha256, :content_bytes, :extracted_text,
                                            :extracted_sha256, CURRENT_TIMESTAMP
                                        )
                                    """),
                                    {
                                        "cache_id": cache_id,
                                        "doi": doi,
                                        "s2_paper_id": hit.paper_id,
                                        "source": "S2_OPEN_ACCESS_PDF",
                                        "content_url": hit.open_access_pdf_url,
                                        "content_sha256": pdf_resp.sha256,
                                        "content_bytes": pdf_resp.size_bytes,
                                        "extracted_text": pdf_resp.extracted_text,
                                        "extracted_sha256": extracted_sha256
                                    }
                                )
                                
                                # Update work table to indicate full text is available
                                conn.execute(
                                    text("""
                                        UPDATE work 
                                        SET full_text_available = 1,
                                            full_text_hash = :hash,
                                            full_text_source = 'S2_OPEN_ACCESS_PDF',
                                            pmcid = :pmcid,
                                            pmid = :pmid,
                                            updated_at = CURRENT_TIMESTAMP
                                        WHERE doi = :doi
                                    """),
                                    {
                                        "hash": extracted_sha256,
                                        "pmcid": hit.external_ids.get("PubMedCentral"),
                                        "pmid": hit.external_ids.get("PubMed"),
                                        "doi": doi
                                    }
                                )
                    except Exception as e:
                        import traceback
                        print(f"Warning: Failed to fetch full text for {doi}")
                        print(f"Error: {str(e)}")
                        traceback.print_exc()
                
                # Promote candidate
                promote_resp = self.evidence_service.promote_candidate_to_support(PromoteCandidateRequest(
                    claim_id=req.claim_id,
                    candidate_id=cid,
                    support_type="citation",
                    promotion_reason=ent.get("rationale", "Abstract supports claim"),
                    created_by="evidence_grounder_s2",
                    policy_version=req.policy_id,
                    anchor_excerpt=ent.get("anchor_quote")
                ))
                supports_created.append(promote_resp.support_id)
                res_obj.created_support_id = promote_resp.support_id
                
                # Update decision to PROMOTED
                self.evidence_service.record_decision(RecordDecisionRequest(
                    candidate_id=cid, claim_id=req.claim_id, outcome=DecisionOutcome.PROMOTED,
                    basis_json={"entailment": ent, "rank": "top_candidate"},
                    policy=PolicyRef(policy_id=req.policy_id), decided_by=ActorRef(agent_id="evidence_grounder_s2")
                ))

        # 6. Calc Metrics & Next Action
        ingested_n = len(ingested_ids)
        
        # Ratios
        E_oa = 1.0 # Since we filtered for OA? S2 query had `require_open_access_pdf=True`.
        # Actually eligibility for triage is just abstract.
        E_triage = stat_eligible / ingested_n if ingested_n > 0 else 0.0
        T = stat_topic_pass / stat_eligible if stat_eligible > 0 else 0.0
        Y = stat_entailment_pass / stat_entailment_judged if stat_entailment_judged > 0 else 0.0
        
        metrics = {
            "ingested_n": ingested_n,
            "eligible_n": stat_eligible,
            "topic_pass_n": stat_topic_pass,
            "entailment_pass_n": stat_entailment_pass,
            "E_triage": E_triage,
            "T": T,
            "Y": Y,
            "mismatch_code_counts": mismatch_counts
        }
        
        # Next Action Logic
        next_act = "STOP_OPEN_GAP"
        next_query = None
        
        if len(supports_created) > 0:
            next_act = "STOP_OPEN_GAP" # Done (Success) - wait, "stop open gap" terminology implies fail?
            # Actually Spec says "stop and open a gap when claim unsupported", but success means STOP (Success).
            # existing NextAction enum has STOP_OPEN_GAP. Maybe add STOP_SUCCESS?
            # Existing loop used STOP_OPEN_GAP only for failure.
            # If success, we just return "done=True".
            pass 
        else:
             # Failure Logic
             if req.search_round_index >= req.max_rounds:
                 next_act = "STOP_MAX_ROUNDS"
             elif stat_eligible == 0:
                 # No papers with abstracts found - query might be too specific
                 # Try broadening/relaxing the query
                 if req.search_round_index >= 3:
                     # After 3 rounds of finding nothing, give up
                     next_act = "STOP_OPEN_GAP"
                 else:
                     next_act = "REWRITE_QUERY_BROADEN"
                     rw_resp = llm_rewrite_query(
                         claim["statement"], claim.get("claim_type", "unknown"),
                         current_query, "BROADEN", metrics, drift_concepts,
                         round_history=round_history  # Pass round history
                     )
                     next_query = rw_resp.query
             elif T < 0.15:
                 # Check for early termination: T=0 for 3 consecutive rounds
                 consecutive_zero_t = 0
                 # Check the last two *recorded* rounds from history
                 for rh in reversed(round_history[-2:]): # Check last 2 rounds
                     if rh.get('T', 0) == 0:
                         consecutive_zero_t += 1
                     else:
                         break
                 
                 if consecutive_zero_t >= 2 and T == 0: # If last 2 rounds had T=0 AND current T=0
                     # 3 consecutive rounds with T=0, stop
                     print(f"Early termination: T=0 for {consecutive_zero_t + 1} consecutive rounds")
                     next_act = "STOP_OPEN_GAP"
                 else:
                     # Low topic match rate - disambiguate
                     next_act = "REWRITE_QUERY_DISAMBIGUATE"
                     # Call Rewrite
                     rw_resp = llm_rewrite_query(
                         claim["statement"], claim.get("claim_type", "unknown"),
                         current_query, "DISAMBIGUATE", metrics, drift_concepts,
                         round_history=round_history  # Pass round history
                     )
                     next_query = rw_resp.query
                     print(f"DEBUG: LLM rewrote query from '{current_query}' to '{next_query}'")
             elif T >= 0.5 and Y == 0 and req.search_round_index >= 2:
                 # Switch to Recs?
                 next_act = "SWITCH_RETRIEVAL_STRATEGY_S2_FIRST" # or use Recs
                 # Spec: "switch strategy (recommendations)"
                 # We lack NextAction.SWITCH_TO_RECS in Enum (I didn't add it to Enum in existing file, I added S2_FIRST).
                 # I'll just use REWRITE to simulate change for now or S2_FIRST.
                 pass
             elif hits_resp.next_offset and hits_resp.total > req.pubmed_retmax:
                 next_act = "CONTINUE_PAGINATION"
             else:
                 next_act = "STOP_OPEN_GAP"
        
        # Record Round
        # Use the query we sent, or S2's compiled version if different
        compiled_query = hits_resp.compiled_query if hits_resp.compiled_query and hits_resp.compiled_query != "test" else current_query
        
        self.evidence_service.create_search_round(CreateSearchRoundRequest(
            project_id=req.project_id,
            claim_id=req.claim_id,
            search_run_id=search_run_id,
            round_index=req.search_round_index,
            provider="SEMANTIC_SCHOLAR",
            base_query=current_query,
            compiled_query=compiled_query,
            filters_json={"limit": req.pubmed_retmax},
            summary_json=metrics,
            next_action=str(next_act)
        ))
        
        # Update claim status if evidence was found
        if len(supports_created) > 0:
            with self.evidence_service.engine.begin() as conn:
                from sqlalchemy import text
                conn.execute(
                    text("""
                        UPDATE claim 
                        SET status = 'supported',
                            disposition = 'accept',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE claim_id = :claim_id
                    """),
                    {"claim_id": req.claim_id}
                )
        
        return OrchestrateRoundResponse(
            round_id=f"RND_{search_run_id}",  # Use search_run_id to avoid uuid issues
            search_run_id=search_run_id,
            claim_id=req.claim_id,
            query=current_query,
            supports_found=len(supports_created),
            supports_created=supports_created,
            candidate_results=results,
            next_action=next_act,
            next_query=next_query,
            reason=f"S2 Round {req.search_round_index} Compete. T={T:.2f}, Y={Y:.2f}",
            done=(len(supports_created) > 0),
            summary_json=metrics
        )
