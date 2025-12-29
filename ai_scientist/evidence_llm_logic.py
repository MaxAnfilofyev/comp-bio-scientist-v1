from typing import Any, List, Optional
import json
from ai_scientist.model.evidence import (
    LLMTopicTriageResponse,
    LLMRewriteQueryResponse,
    QueryBlocks,
    MismatchCode
)
from ai_scientist.llm import (
    create_client,
    get_response_from_llm,
    extract_json_between_markers
)

DEFAULT_MODEL = "gpt-5.2"

# -------------------------------------------------------------------------
# Prompt Templates (from Spec)
# -------------------------------------------------------------------------

TOPIC_TRIAGE_SYSTEM_PROMPT = """You are a biomedical evidence triage assistant. Output STRICT JSON only.

Task:
1. Decide if the candidate is on-topic for supporting the claim.
2. Estimate evidence_likelihood (0-1): probability the full text contains direct evidence relevant to the claim.
3. If FAIL, provide mismatch_codes from this enum:
   {MISMATCH_ENUM}
4. Provide structured suggestions for QUERY BUILDING. If this paper is RELEVANT, what terms should be in the blocks?
   - suggested_entity_synonyms: specific entity names found in text (e.g. "KIF5A", "microtubule").
   - suggested_process_synonyms: specific mechanisms (e.g. "anterograde transport").
   - suggested_context_synonyms: compartments/tissues (e.g. "sciatic nerve", "axoplasm").
   - suggested_modalities: specific methods used in this paper (e.g. "kymograph", "TIRF").
5. Provide query_hints_positive (terms to add), query_hints_negative (terms to exclude), drift_concepts (terms to negate), and anchors.
   - drift_concepts: List specific terms the paper is About that match the Mismatch Code (e.g. "active transport", "gap junction"). REQUIRED.
   - positive_anchors: Terms that would force the intended topic (e.g. "FRAP", "diffusion coefficient").
   - negative_anchors: Top confounders to exclude.

Return JSON:
{{
   "topic_match": "PASS|FAIL",
   "evidence_likelihood": 0.0-1.0,
   "mismatch_codes": [],
   "drift_concepts": ["concept1"], 
   "drift_class": "WRONG_ENTITY|WRONG_PROCESS|WRONG_CONTEXT|WRONG_MODALITY|WRONG_DIRECTION|AMBIGUOUS",
   "suggested_entity_synonyms": ["syn1"],
   "suggested_process_synonyms": ["syn1"],
   "suggested_context_synonyms": ["syn1"],
   "suggested_modalities": ["method1"],
   "positive_anchors": [],
   "negative_anchors": [],
   "query_hints_positive": [],
   "query_hints_negative": [],
   "note": "optional"
}}
"""

TOPIC_TRIAGE_USER_TEMPLATE = """Claim:
- claim_text: "{CLAIM_TEXT}"
- claim_type: "{CLAIM_TYPE}"

Candidate:
- title: "{TITLE}"
- abstract: "{ABSTRACT}"
- tldr: "{TLDR}"
- venue: "{VENUE}"
- year: "{YEAR}"
- citations: "{CITATIONS}"
- pub_types: "{PUB_TYPES}"
"""

UNIVERSAL_ENTAILMENT_SYSTEM_PROMPT = """You are an evidence judge. Your job is to decide whether the provided SOURCE TEXT entails the CLAIM.

Rules:
- Be strict. Do not guess.
- Treat TLDR as non-authoritative; use it only as context, never as the evidence basis.
- If the claim is mechanistic or quantitative, require explicit mechanistic or quantitative statements.
- Return STRICT JSON only.

Labels:
- "SUPPORTED": source clearly supports claim.
- "NOT_SUPPORTED": source does not establish the claim.
- "CONTRADICTED": source states the opposite or incompatible result.
- "NEEDS_FULL_TEXT": source is suggestive but too vague/underspecified; likely requires full text.

Also return:
- support_strength: {STRONG|MODERATE|WEAK}
- anchor_quote: <=25 words copied from SOURCE TEXT (empty if not supported)
- rationale: 1-2 sentences, concrete.
- required_next: {NONE|FETCH_FULLTEXT|FIND_DIFFERENT_PAPER|REFINE_CLAIM}

JSON Format:
{{
  "verdict": "SUPPORTED", 
  "support_strength": "STRONG",
  "anchor_quote": "text...",
  "rationale": "reasoning...",
  "required_next": "NONE"
}}
"""

UNIVERSAL_ENTAILMENT_USER_TEMPLATE = """CLAIM:
- claim_text: "{CLAIM_TEXT}"
- claim_type: "{CLAIM_TYPE}"

SOURCE:
- source_type: "{SOURCE_TYPE}"
- title: "{TITLE}"
- venue: "{VENUE}"
- year: "{YEAR}"
- text: \"\"\"{SOURCE_TEXT}\"\"\"
"""


REWRITE_QUERY_SYSTEM_PROMPT = """You are a biomedical expert query optimizer for Semantic Scholar.

IMPORTANT: Semantic Scholar uses RELEVANCE-BASED search, NOT strict Boolean logic.
- Generate simple keyword queries with the most important terms
- Avoid complex Boolean operators (AND, OR, NOT)
- Use 3-7 key terms that best represent the concept
- Put exact phrases in quotes if critical
- Keep queries simple and focused

FAILURES DIAGNOSED:
{failure_reason}

DRIFT CONCEPTS (What we found instead):
{drift_concepts}

STRATEGY: {mode}
RULES:
1. DISAMBIGUATE:
   - Add 1-2 specific terms (methods, measurements, or context) to narrow focus
   - Example: "ATP axon" → "ATP axon diffusion FRAP"
2. BROADEN:
   - Remove specific method terms
   - Use more general synonyms
   - Example: "ATP axon diffusion FRAP" → "ATP axon transport"
3. TIGHTEN:
   - Add very specific measurement methods or context
   - Example: "ATP axon" → "ATP axon diffusion coefficient fluorescence"

OUTPUT FORMAT - Return STRICT JSON:
{{
  "keywords": ["term1", "term2", "term3", ...],
  "exact_phrases": ["exact phrase 1", "exact phrase 2", ...],
  "query": "simple keyword query (will be auto-generated)",
  "note": "brief explanation of changes"
}}

EXAMPLES:
- Good: "SNARE membrane fusion synaptic vesicle"
- Good: "ATP diffusion axon FRAP"
- Bad: "(ATP) AND (axon OR neurite) AND (diffusion OR FRAP)"
- Bad: Complex nested Boolean expressions
"""

REWRITE_QUERY_USER_TEMPLATE = """Claim: {CLAIM_TEXT}
Current query: {CURRENT_QUERY}
Mode: {MODE}
Mismatches found: {COUNTS}
Drift Concepts (what we found instead): {DRIFT_CONCEPTS}

{ROUND_HISTORY}

Your task: Generate a NEW keyword query that will find better papers.

Instructions for {MODE} mode:
- **DISAMBIGUATE**: Add 1-2 specific terms to narrow focus (e.g., add "FRAP" or "diffusion coefficient" for measurement methods)
- **BROADEN**: Remove specific method terms, use more general synonyms
- **TIGHTEN**: Add very specific measurement methods or experimental context

IMPORTANT:
- Generate 3-7 key terms
- Use simple keywords, NOT Boolean operators
- Put critical exact phrases in quotes if needed
- Focus on the most discriminative terms
- TARGET: Aim for 5-15 relevant papers (not 2, not 100)

Example transformations:
- DISAMBIGUATE: "ATP axon transport" → "ATP axon diffusion FRAP fluorescence"
- BROADEN: "ATP axon diffusion FRAP" → "ATP axon transport"
"""

REPAIR_QUERY_SYSTEM_PROMPT = """Fix this PubMed query to be syntactically valid and compliant.
Rules: 
1. Replace single quotes with double quotes.
2. Remove duplicate clauses.
3. Remove unknown filters.
4. STRIP POLICY ENVELOPE TOKENS: Remove "pubmed pmc"[sb], pmc[filter], english[la] if present. We only want the BaseQuery.
Return only the repaired query.
"""

REPAIR_QUERY_USER_TEMPLATE = "Query: {bad_query}"

# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------

def llm_topic_triage(
    claim_text: str,
    claim_type: str,
    title: str,
    abstract: str,
    tldr: str = None,
    venue: str = None,
    year: Any = None,
    citations: int = 0,
    pub_types: List[str] = [],
    policy_id: str = "default",
    model: str = DEFAULT_MODEL
) -> LLMTopicTriageResponse:
    """
    Evaluates title/abstract relevance to the claim.
    """
    client, model_name = create_client(model)
    
    # Construct prompts
    # Inject pure enum values into system prompt for clarity
    mismatch_enum_str = ", ".join(MismatchCode.__args__)
    system_prompt = TOPIC_TRIAGE_SYSTEM_PROMPT.format(MISMATCH_ENUM=mismatch_enum_str)
    
    user_prompt = TOPIC_TRIAGE_USER_TEMPLATE.format(
        CLAIM_TEXT=claim_text,
        CLAIM_TYPE=claim_type,
        TITLE=title or "No Title",
        ABSTRACT=abstract or "No Abstract",
        TLDR=tldr or "N/A",
        VENUE=venue or "N/A",
        YEAR=str(year) if year else "N/A",
        CITATIONS=str(citations),
        PUB_TYPES=", ".join(pub_types) if pub_types else "N/A"
    )

    try:
        response_text, _ = get_response_from_llm(
            prompt=user_prompt,
            client=client,
            model=model_name,
            system_message=system_prompt,
            temperature=0.0  # structured data extraction
        )

        parsed = extract_json_between_markers(response_text)
        if not parsed:
             # Try parsing raw json if markers missing
             parsed = json.loads(response_text)
             
        # Validate via Pydantic
        data = parsed
        return LLMTopicTriageResponse(
            topic_match=data.get("topic_match"),
            evidence_likelihood=data.get("evidence_likelihood"),
            mismatch_codes=data.get("mismatch_codes", []),
            drift_concepts=data.get("drift_concepts", ["UNKNOWN_DRIFT"]) if not data.get("drift_concepts") else data.get("drift_concepts"),
            drift_class=data.get("drift_class"),
            suggested_entity_synonyms=data.get("suggested_entity_synonyms", []),
            suggested_process_synonyms=data.get("suggested_process_synonyms", []),
            suggested_context_synonyms=data.get("suggested_context_synonyms", []),
            suggested_modalities=data.get("suggested_modalities", []),
            positive_anchors=data.get("positive_anchors", []),
            negative_anchors=data.get("negative_anchors", []),
            query_hints_positive=data.get("query_hints_positive", []),
            query_hints_negative=data.get("query_hints_negative", []),
            note=data.get("note")
        )

    except Exception as e:
        print(f"LLM Topic Triage Failed: {e}")
        # Fail safe
        return LLMTopicTriageResponse(
            topic_match="FAIL",
            evidence_likelihood=0.0,
            mismatch_codes=["AMBIGUOUS"],
            note=f"Error: {str(e)}"
        )


def llm_rewrite_query(
    claim_text: str,
    claim_type: str,
    current_query: str,
    mode: str, # TIGHTEN, BROADEN, DISAMBIGUATE, RELAX
    failure_summary: dict,
    drift_concepts: List[str] = [],
    round_history: List[dict] = [],  # New parameter for round feedback
    policy_id: str = "default",
    model: str = DEFAULT_MODEL
) -> LLMRewriteQueryResponse:
    """
    Rewrites the PubMed query based on failure analysis.
    """
    client, model_name = create_client(model)
    
    counts = failure_summary.get("mismatch_code_counts", {})
    
    # Format round history for prompt
    history_text = ""
    if round_history:
        history_text = "Previous rounds:\n"
        for rh in round_history[-3:]:  # Only show last 3 rounds
            history_text += f"- Round {rh.get('round', '?')}: {rh.get('ingested', 0)} papers, "
            history_text += f"{rh.get('eligible', 0)} eligible, T={rh.get('T', 0):.2f}\n"
            if rh.get('top_mismatches'):
                history_text += f"  Top mismatches: {', '.join(rh['top_mismatches'][:3])}\n"
    
    user_prompt = REWRITE_QUERY_USER_TEMPLATE.format(
        CLAIM_TEXT=claim_text,
        CLAIM_TYPE=claim_type,
        CURRENT_QUERY=current_query,
        MODE=mode,
        COUNTS=json.dumps(counts, indent=2),
        DRIFT_CONCEPTS=", ".join(drift_concepts) if drift_concepts else "None",
        ROUND_HISTORY=history_text if history_text else "No previous rounds."
    )

    try:
        response_text, _ = get_response_from_llm(
            prompt=user_prompt,
            client=client,
            model=model_name,
            system_message=REWRITE_QUERY_SYSTEM_PROMPT, # Uses new block prompt
            temperature=0.3
        )
        
        parsed = extract_json_between_markers(response_text)
        if not parsed:
             parsed = json.loads(response_text)
             
        print(f"DEBUG LLM REWRITE: Raw response = {json.dumps(parsed, indent=2)}")
        
        data = parsed
        blocks_data = None  # Initialize to avoid UnboundLocalError
        
        # New format: keywords and exact_phrases
        if "keywords" in data:
            keywords = data.get("keywords", [])
            exact_phrases = data.get("exact_phrases", [])
            
            # Build simple query: keywords + "exact phrases"
            query_parts = []
            query_parts.extend(keywords)
            query_parts.extend([f'"{phrase}"' for phrase in exact_phrases])
            
            base_query = " ".join(query_parts)
            print(f"DEBUG: Compiled keywords {keywords} + phrases {exact_phrases} -> '{base_query}'")
        else:
            # Fallback to old format for backwards compatibility
            blocks_data = data.get("query_blocks", {})
            
            # Compile Blocks into simple keyword query (not Boolean)
            parts = []
            if blocks_data.get("entity"): parts.append(blocks_data['entity'].strip('() '))
            if blocks_data.get("context"): parts.append(blocks_data['context'].strip('() '))
            if blocks_data.get("process"): parts.append(blocks_data['process'].strip('() '))
            if blocks_data.get("modality"): parts.append(blocks_data['modality'].strip('() '))
            
            base_query = " ".join(parts)
        
        # Fallback: if blocks are empty but LLM provided a direct query string
        if not base_query and data.get("query"):
            base_query = data["query"]
        
        # Final fallback: if still empty, use current query
        if not base_query:
            print(f"Warning: LLM rewrite produced empty query, using current query as fallback")
            base_query = current_query
            
        return LLMRewriteQueryResponse(
            query=base_query,
            query_blocks=QueryBlocks(**blocks_data) if blocks_data else None,
            note=data.get("note")
        )

    except Exception as e:
        print(f"LLM Rewrite Failed: {e}")
        return LLMRewriteQueryResponse(query=current_query, note=f"Failed: {e}") # Fallback

def llm_repair_query(
    bad_query: str,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Fixes syntax errors in PubMed query.
    """
    client, model_name = create_client(model)
    user_prompt = REPAIR_QUERY_USER_TEMPLATE.format(bad_query=bad_query)
    
    try:
        response_text, _ = get_response_from_llm(
            prompt=user_prompt,
            client=client,
            model=model_name,
            system_message=REPAIR_QUERY_SYSTEM_PROMPT,
            temperature=0.0
        )
        return response_text.strip().strip('"')
    except Exception as e:
        print(f"Query Repair Failed: {e}")
        return bad_query

def llm_entailment_s2(
    claim_text: str,
    claim_type: str,
    source_text: str,
    source_type: str = "abstract",
    title: str = "",
    venue: str = "",
    year: Any = "",
    model: str = DEFAULT_MODEL
) -> dict:
    """
    Universal entailment judge.
    Returns dict compatible with LLMEntailmentJudgeResponse construction but generic dict for flexibility first.
    """
    client, model_name = create_client(model)
    
    user_prompt = UNIVERSAL_ENTAILMENT_USER_TEMPLATE.format(
        CLAIM_TEXT=claim_text,
        CLAIM_TYPE=claim_type,
        SOURCE_TYPE=source_type,
        TITLE=title,
        VENUE=venue,
        YEAR=str(year),
        SOURCE_TEXT=source_text
    )
    
    try:
        response_text, _ = get_response_from_llm(
            prompt=user_prompt,
            client=client,
            model=model_name,
            system_message=UNIVERSAL_ENTAILMENT_SYSTEM_PROMPT,
            temperature=0.0
        )
        parsed = extract_json_between_markers(response_text)
        if not parsed:
            parsed = json.loads(response_text)
            
        return parsed
    except Exception as e:
        print(f"Entailment Failed: {e}")
        return {
            "verdict": "NOT_SUPPORTED",
            "support_strength": "WEAK",
            "rationale": f"Error: {e}",
            "required_next": "NONE"
        }
