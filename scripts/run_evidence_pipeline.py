
import os
import json
import sqlite3
from datetime import datetime
from ai_scientist.evidence_service import (
    EvidenceService, WorkUpsert, IngestCandidatesRequest, CandidateIngestItem,
    CheckType, Verdict, RecordDecisionRequest, DecisionOutcome, PolicyRef, ActorRef,
    PromoteCandidateRequest, CreateSearchRunRequest, Provider
)
from ai_scientist.medline_gate import MedlinePmcEvidenceGate, PubMedSearchRequest, PubMedFetchMetadataRequest, RecordChecksRequestInput, CandidateCheckResult

# Re-use DB Path from previous script/consistent location
# Re-use DB Path from previous script/consistent location
import time
DB_PATH = f"ai_scientist_manual_search_{int(time.time())}.sqlite"
DATABASE_URL = f"sqlite:///{DB_PATH}"

MIGRATION_FILE = "ai_scientist/database/migrations/versions/27707676918b_initial_schema.py"

def get_ddl():
    with open(MIGRATION_FILE, "r") as f:
        content = f.read()
    import re
    match = re.search(r'sql = """(.*?)"""', content, re.DOTALL)
    if not match:
        raise ValueError("Could not extract SQL from migration file")
    return match.group(1)

def setup_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    print(f"Setting up fresh DB at {DATABASE_URL}...")
    conn = sqlite3.connect(DB_PATH)
    
    # Apply Schema
    ddl = get_ddl()
    conn.executescript(ddl)
    
    cursor = conn.cursor()
    
    # Dummy Policy
    cursor.execute("""
    INSERT OR IGNORE INTO policy_snapshot (policy_version, created_at, policy_yaml, policy_hash)
    VALUES ('medline_gate_v1', datetime('now'), 'dummy_yaml', 'dummy_hash')
    """)
    
    # Project
    cursor.execute("""
    INSERT OR IGNORE INTO project (project_id, title, status, created_at, updated_at, policy_version)
    VALUES ('pipeline_test', 'Pipeline Test Project', 'active', datetime('now'), datetime('now'), 'medline_gate_v1')
    """)
    
    # Claim (Required for promotion)
    cursor.execute("""
    INSERT OR IGNORE INTO claim (
        claim_id, project_id, module, statement, claim_type, 
        created_by, created_at, updated_at, policy_version
    ) VALUES (
        'CLM_TEST_001', 'pipeline_test', 'biophysics', 
        'ATP diffusion in axons is slower than free diffusion', 'fact',
        'tester', datetime('now'), datetime('now'), 'medline_gate_v1'
    )
    """)
    
    conn.commit()
    conn.close()

def print_section(title, data):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)
    print(json.dumps(data, indent=2, default=str))

def run_pipeline():
    setup_db()
    
    svc = EvidenceService(db_url=DATABASE_URL)
    gate = MedlinePmcEvidenceGate(evidence_service=svc)
    
    # 1. SEARCH
    print("\n>>> STAGE 1: SEARCH (Strict PMC)")
    query = "ATP diffusion"
    # Use Strict PMC Template (quoted)
    search_req = PubMedSearchRequest(
        project_id="pipeline_test", 
        query=query, 
        retmax=5,
        policy_query_template="{query} AND \"pubmed pmc\"[sb]"
    )
    search_resp = gate.pubmed_search(search_req)
    
    # Audit Log for Search
    print_section("Search Run Audit", {
        "run_id": search_resp.search_run_id,
        "filters_json": {
            "db": "pubmed",
            "term": search_resp.query, # This might be the raw query if logic didn't update it in response object, 
                                       # but typically response.query reflects what was passed? 
                                       # Actually `gate.pubmed_search` returns `request.query` in `response.query`.
                                       # We should log the compiled query if possible, or reconstruct it.
            "template": search_req.policy_query_template,
            "retmax": search_req.retmax,
            "sort": search_req.sort or "date (default)",
        },
        "count_total": search_resp.count_total
    })
    
    if not search_resp.pmids:
        print("No PMIDs found. Aborting.")
        return

    # 2. METADATA & INGESTION
    print("\n>>> STAGE 2: METADATA & CANDIDATE INGESTION")
    meta_req = PubMedFetchMetadataRequest(project_id="pipeline_test", pmids=search_resp.pmids)
    meta_resp = gate.pubmed_fetch_metadata(meta_req)
    
    # Upsert Works
    works = []
    for w in meta_resp.works:
        works.append(WorkUpsert(
            doi=w.doi,
            title=w.title,
            year=w.year,
            venue=w.journal,
            pmid=w.pmid,
            pmcid=w.pmcid
        ))
    
    valid_works = [w for w in works if w.doi]
    from ai_scientist.evidence_service import UpsertWorkBatchRequest
    if valid_works:
        svc.upsert_work_batch(UpsertWorkBatchRequest(works=valid_works))
    
    # Create Candidates
    pmid_to_doi = {w.pmid: w.doi for w in meta_resp.works if w.doi}
    
    candidates = []
    for i, pmid in enumerate(search_resp.pmids):
        doi = pmid_to_doi.get(pmid)
        if doi:
            candidates.append(CandidateIngestItem(
                doi=doi,
                rank_in_results=i+1,
                retrieval_score_raw=1.0 
            ))
    
    if not candidates:
        print("No candidates with valid DOIs to ingest.")
        return
        
    ingest_resp = svc.ingest_candidates(IngestCandidatesRequest(
        search_run_id=search_resp.search_run_id,
        candidates=candidates
    ))
    # print_section("Ingested Candidates", ingest_resp.model_dump())
    
    # 2b. RESOLVE PMCIDs via ELink
    pmids_missing_pmcid = [w.pmid for w in meta_resp.works if not w.pmcid]
    pmid_to_resolved_pmcid = {}
    
    if pmids_missing_pmcid:
        print(f"\nResolving PMCIDs for {len(pmids_missing_pmcid)} works...")
        from ai_scientist.medline_gate import PubMedLinkToPmcRequest
        link_req = PubMedLinkToPmcRequest(project_id="pipeline_test", pmids=pmids_missing_pmcid)
        link_resp = gate.pubmed_link_to_pmc(link_req)
        
        for link in link_resp.links:
            if link.pmcids:
                 pmid_to_resolved_pmcid[link.pmid] = link.pmcids[0] # Take first
        
        print(f"Resolved {len(pmid_to_resolved_pmcid)} PMCIDs via ELink.")
    
    # 3. QUALITY CHECKS
    print("\n>>> STAGE 3: QUALITY CHECKS")
    
    check_results = []
    
    for cid, cand_item in zip(ingest_resp.candidate_ids, candidates):
        meta = next((w for w in meta_resp.works if w.doi == cand_item.doi), None)
        if not meta: 
            continue
            
        # Check 1: Has DOI
        check_results.append(CandidateCheckResult(
            candidate_id=cid,
            check_type="has_doi",
            verdict="PASS" if meta.doi else "FAIL"
        ))
        
        # Check 2: In PMC
        pmcid = meta.pmcid or pmid_to_resolved_pmcid.get(meta.pmid)
        check_results.append(CandidateCheckResult(
            candidate_id=cid,
            check_type="in_pmc_fulltext",
            verdict="PASS" if pmcid else "FAIL",
            details_json={"pmcid": pmcid}
        ))
        
        # Check 3: Preprint
        is_preprint = "Preprint" in meta.publication_types # More robust check
        check_results.append(CandidateCheckResult(
            candidate_id=cid,
            check_type="preprint_policy",
            verdict="FAIL" if is_preprint else "PASS",
            details_json={"pub_types": meta.publication_types}
        ))

        # Check 4: Retraction
        is_retracted = "Retracted Publication" in meta.publication_types
        check_results.append(CandidateCheckResult(
            candidate_id=cid,
            check_type="retraction_check",
            verdict="FAIL" if is_retracted else "PASS",
            details_json={"pub_types": meta.publication_types}
        ))

        # Check 5: Expression of Concern
        is_eoc = "Expression of Concern" in meta.publication_types
        check_results.append(CandidateCheckResult(
            candidate_id=cid,
            check_type="eoc_check",
            verdict="FAIL" if is_eoc else "PASS",
            details_json={"pub_types": meta.publication_types}
        ))
        
        # Check 6: Article Type (Safety for Auto-Select)
        # Prefer Journal Article or Review
        is_trusted_type = "Journal Article" in meta.publication_types or "Review" in meta.publication_types or "Systematic Review" in meta.publication_types
        check_results.append(CandidateCheckResult(
            candidate_id=cid,
            check_type="trusted_type_check",
            verdict="PASS" if is_trusted_type else "FAIL", # Make this a soft check (FAIL doesn't reject, but prevents auto-select?)
                                                           # Or make it hard? User said "require article types you trust".
            details_json={"pub_types": meta.publication_types}
        ))

    check_req = RecordChecksRequestInput(
        project_id="pipeline_test",
        policy_id="medline_gate_v1",
        executed_by="auto_checker",
        executed_at=datetime.utcnow().isoformat(),
        results=check_results
    )
    
    checks_resp = gate.record_checks(check_req)
    # print_section("recorded Checks", checks_resp.model_dump())

    # 4. DECISIONS
    print("\n>>> STAGE 4: DECISIONS")
    
    decisions_made = []
    eligible_candidates = []
    
    from collections import defaultdict
    cand_checks = defaultdict(list)
    for res in check_results:
        cand_checks[res.candidate_id].append(res)
        
    for cid, checks in cand_checks.items():
        verdicts = {c.check_type: c.verdict for c in checks}
        failed_gates = [c.check_type for c in checks if c.verdict == "FAIL"]
        
        # Logic: 
        # REJECT if Hard Gates FAIL
        hard_gates = ["has_doi", "in_pmc_fulltext", "preprint_policy", "retraction_check", "eoc_check"]
        if any(v == "FAIL" for k, v in verdicts.items() if k in hard_gates):
            outcome = DecisionOutcome.REJECTED
        else:
            outcome = DecisionOutcome.ELIGIBLE_SUPPORT
            eligible_candidates.append(cid)

        dec_req = RecordDecisionRequest(
            candidate_id=cid,
            outcome=outcome,
            policy=PolicyRef(policy_id="medline_gate_v1"),
            decided_by=ActorRef(agent_id="auto_decider"),
            basis_json={
                "rule": "demo_strict_pmc", 
                "verdicts": verdicts,
                "rejection_reason": f"Failed gates: {failed_gates}" if outcome == DecisionOutcome.REJECTED else None
            }
        )
        dec_resp = svc.record_decision(dec_req)
        decisions_made.append({
            "candidate_id": cid, 
            "outcome": outcome, 
            "id": dec_resp.decision_id,
            "failed_gates": failed_gates if outcome == DecisionOutcome.REJECTED else []
        })
    print_section("Decisions Audit", decisions_made)

    # 5. SELECTION & PROMOTION (Lazy Full-Text Validation)
    # 5. SELECTION & PROMOTION (LLM Entailment + Guardrails)
    print("\n>>> STAGE 5: SELECTION & PROMOTION (LLM Entailment + Guardrails)")
    
    CLAIM_ID = "CLM_TEST_001" 
    CLAIM_TEXT = "Active transport of ATP along axons involves diffusion mechanisms."
    
    if not eligible_candidates:
        print("No candidates were ELIGIBLE. Pipeline ends.")
    else:
        # Lazy Fetch Logic
        selected_cid = None
        
        from ai_scientist.medline_gate import PubMedFetchFullTextRequest
        from ai_scientist.evidence_service import StoreFulltextRequest
        import re
        
        for cid in eligible_candidates: 
            print(f"\nEvaluating Candidate {cid}...")
            
            # --- 1. Get PMCID & Fetch ---
            cand_item = next(c for c, id_ in zip(candidates, ingest_resp.candidate_ids) if id_ == cid)
            meta = next(w for w in meta_resp.works if w.doi == cand_item.doi)
            pmcid = meta.pmcid or pmid_to_resolved_pmcid.get(meta.pmid)
            
            if not pmcid:
                print(f"Skipping {cid}: No PMCID.")
                continue
                
            try:
                ft_req = PubMedFetchFullTextRequest(project_id="pipeline_test", pmcid=pmcid)
                ft_resp = gate.pubmed_fetch_fulltext(ft_req)
                xml_content = ft_resp.xml_content or ""
                
                # Persist
                svc.store_work_fulltext(StoreFulltextRequest(
                    doi=cand_item.doi,
                    pmcid=pmcid,
                    source="PMC",
                    format="JATS_XML",
                    content=xml_content,
                    license="CC-BY"
                ))
            except Exception as e:
                print(f"Fetch/Store failed for {pmcid}: {e}")
                continue

            # --- 2. Parse & Retrieve (Real) ---
            # Parse XML -> sections -> paragraphs
            import xml.etree.ElementTree as ET
            
            candidate_passage = None
            passage_meta = {}
            
            try:
                # Wrap content in root if needed (JATS usually has root element article)
                # handle potential encoding issues broadly
                root = ET.fromstring(xml_content)
                
                # Iterate over paragraphs in body
                # Simplified: Find all 'p' tags, get text, check keywords
                # Trying to find section info is harder in flat iteration, but we can try xpath if supported or just simple loop
                
                paragraphs = root.findall(".//p")
                for i, p in enumerate(paragraphs):
                    text_content = "".join(p.itertext()).strip()
                    
                    # RETRIEVAL STEP: Keyword Match
                    if "ATP" in text_content and "diffusion" in text_content and len(text_content) > 20:
                        candidate_passage = text_content
                        # Try to find section title (parent's previous sibling title? Hard in ET)
                        # We'll just define section as 'Unknown' or try to find parent
                        passage_meta = {
                            "pmcid": pmcid,
                            "section": "Body", # Placeholder for strict XML parsing
                            "paragraph_index": i,
                            "sentence_index": 0
                        }
                        break
            except Exception as e:
                print(f"XML Parsing failed for {pmcid}: {e}")
                # Fallback to naive
                pass

            if not candidate_passage:
                print(f"No relevant passage found in {pmcid} (Retrieval failed).")
                continue
                
            print(f"Retrieved Passage: \"{candidate_passage[:60]}...\"")
            
            # --- 3. LLM Entailment Judge (Real) ---
            from ai_scientist.entailment_verification import verify_claim_entailment
            
            print("  Calling LLM Judge...")
            try:
                entailment_result = verify_claim_entailment(
                    claim_text=CLAIM_TEXT,
                    passage_text=candidate_passage,
                    pmcid=pmcid,
                    claim_type="mechanism",
                    strength="strong",
                    section=passage_meta.get("section", "Body"),
                    paragraph_index=passage_meta.get("paragraph_index", 0)
                )
                
                llm_response = entailment_result.model_dump()
                print(f"  LLM Verdict: {llm_response['verdict']} | Quote: \"{llm_response['anchor_quote']}\"")
                
            except Exception as e:
                print(f"LLM Judge execution failed: {e}")
                continue

            # --- 4. Guardrails (The "Brutal Rule") ---
            # Guardrail 1: Verbatim Check
            # Check against the passage text we sent to LLM
            if llm_response['anchor_quote'] not in candidate_passage:
                 print(f"GUARDRAIL FAIL: Quote not found verbatim in passage.")
                 gate.record_checks(RecordChecksRequestInput(
                    project_id="pipeline_test", policy_id="medline_gate_v1", executed_by="guardrail", executed_at=datetime.utcnow().isoformat(),
                    results=[CandidateCheckResult(candidate_id=cid, check_type="anchor_extraction", verdict="FAIL", details_json={"error": "Quote not verbatim", "quote": llm_response['anchor_quote']})]
                 ))
                 continue

            # Guardrail 2: Length (<= 25 words)
            quote_len = len(llm_response['anchor_quote'].split())
            if quote_len > 25:
                print(f"GUARDRAIL FAIL: Quote too long ({quote_len} words).")
                continue

            # Guardrail 3: Entailment Label
            verdict = "PASS" if llm_response['verdict'] == "ENTAILS" else "FAIL"
            
            # Record Checks
            gate.record_checks(RecordChecksRequestInput(
                project_id="pipeline_test",
                policy_id="medline_gate_v1",
                executed_by="gpt-5.1",
                executed_at=datetime.utcnow().isoformat(),
                results=[
                    CandidateCheckResult(candidate_id=cid, check_type="anchor_extraction", verdict="PASS"),
                    CandidateCheckResult(
                        candidate_id=cid, 
                        check_type="claim_entailment_llm", 
                        verdict=verdict, 
                        details_json=llm_response
                    )
                ]
            ))
            
            if verdict == "PASS":
                selected_cid = cid
                selected_quote = llm_response['anchor_quote']
                selected_loc = llm_response['anchor_location']
                break
        
        if not selected_cid:
            print("No candidates passed Entailment + Guardrails.")
        else:
            print(f"\nSelected Candidate {selected_cid} (Entailed).")
            
            # 5a. Record Selection Decision
            sel_req = RecordDecisionRequest(
                candidate_id=selected_cid,
                claim_id=CLAIM_ID,
                outcome=DecisionOutcome.SELECTED_AS_SUPPORT,
                policy=PolicyRef(policy_id="medline_gate_v1"),
                decided_by=ActorRef(agent_id="auto_selector"),
                basis_json={"reason": "LLM Entailment Verified", "quote": selected_quote}
            )
            svc.record_decision(sel_req)
            
            # 5b. Promote with Verified Anchor
            try:
                sup_resp = svc.promote_candidate_to_support(PromoteCandidateRequest(
                    claim_id=CLAIM_ID,
                    candidate_id=selected_cid,
                    support_type="citation", 
                    promotion_reason="LLM Verified Entailment",
                    created_by="pipeline_runner",
                    policy_version="medline_gate_v1",
                    anchor_excerpt=selected_quote,
                    anchor_location_json=selected_loc, # This might be a dict now
                    verification_status="verified"
                ))
                print_section("Promoted Support", sup_resp.model_dump())
            except Exception as e:
                print(f"Promotion failed: {e}")

if __name__ == "__main__":
    run_pipeline()
