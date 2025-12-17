from typing import Any, List, Optional
import json
from ai_scientist.model.evidence import (
    LLMTopicTriageResponse,
    LLMRewriteQueryResponse,
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
"""


REWRITE_QUERY_SYSTEM_PROMPT = """You are a biomedical expert query optimizer.
Your goal is to rewrite the search query to better find papers supporting the claim.

FAILURES DIAGNOSED:
{failure_reason}

DRIFT CONCEPTS (What we found instead):
{drift_concepts}

STRATEGY: {mode}
RULES:
1. DISAMBIGUATE:
   - You MUST add 1-2 positive anchors (specific methods, tissues, or measurements) that confirm the topic.
   - You MAY add negative keywords (NOT ...) but max 4 terms.
   - Do NOT remove the core entity.
2. RELAX:
   - Remove "modality" block first.
   - Remove negative exclusions.
   - Simplify context.
3. TIGHTEN:
   - Add a modality block or precise timestamp.
   - Add negative exclusions for drift concepts.

OUTPUT FORMAT (JSON Blocks):
{{
  "entity": "core protein/gene",
  "process": "mechanism or action",
  "context": "tissue or cell type",
  "modality": "measurement method (optional)",
  "exclusion": "NOT (term1 OR term2)"
}}
"""

REWRITE_QUERY_USER_TEMPLATE = """Claim: {CLAIM_TEXT}
Current query (compiled): {CURRENT_QUERY}
Mode: {MODE}
Mismatches: {COUNTS}
Near miss notes: {NEAR_MISS_NOTES}
Drift Concepts (from triage): {DRIFT_CONCEPTS}

Instructions:
- **DISAMBIGUATE (Phase A)**:
  - Add 1 STRONG positive anchor to "modality" or "process" (e.g. "diffusion coefficient", "FRAP").
  - Add up to 2 NEGATIVE anchors to "exclusion" derived from Drift Concepts.
- **RELAX / BROADEN (Phase B)**:
  - Remove "modality" block entirely.
  - Simplify "process" block to core terms only.
  - Drop "context" specificity.
- **TIGHTEN / NARROW (Phase A)**:
  - Reinstate "modality" block.
  - Add specific "context" terms.
  - Add "exclusion" for known drift.
- **PAGINATE (Phase B)**:
  - Keep query identical to semantic intent, maybe minor synonym expansion.

Constraint: NEVER add broad verbs like "transport", "movement", "trafficking", "signaling", "regulation", "function", "role", "effect", "impact" unless the claim is specifically about that general concept.
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
        ABSTRACT=abstract or "No Abstract"
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
    policy_id: str = "default",
    model: str = DEFAULT_MODEL
) -> LLMRewriteQueryResponse:
    """
    Rewrites the PubMed query based on failure analysis.
    """
    client, model_name = create_client(model)
    
    counts = failure_summary.get("mismatch_code_counts", {})
    # Extract near miss notes if available, or just generic info
    # In a real impl, we'd pass specific notes from the summary
    
    user_prompt = REWRITE_QUERY_USER_TEMPLATE.format(
        CLAIM_TEXT=claim_text,
        CLAIM_TYPE=claim_type,
        CURRENT_QUERY=current_query,
        MODE=mode,
        COUNTS=json.dumps(counts, indent=2),
        NEAR_MISS_NOTES="(See mismatch codes)",
        DRIFT_CONCEPTS=", ".join(drift_concepts) if drift_concepts else "None"
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
             
        data = parsed
        blocks_data = data.get("query_blocks", {})
        
        # Compile Blocks into String
        # (ENTITY) AND (CONTEXT) ...
        parts = []
        if blocks_data.get("entity"): parts.append(f"({blocks_data['entity'].strip('() ')})")
        if blocks_data.get("context"): parts.append(f"({blocks_data['context'].strip('() ')})")
        if blocks_data.get("process"): parts.append(f"({blocks_data['process'].strip('() ')})")
        if blocks_data.get("modality"): parts.append(f"({blocks_data['modality'].strip('() ')})")
        
        base_query = " AND ".join(parts)
        
        # Add Exclusions
        if blocks_data.get("exclusion"):
            excl = blocks_data["exclusion"]
            if not excl.startswith("NOT "): excl = f"NOT ({excl})"
            base_query += f" {excl}"
            
        return LLMRewriteQueryResponse(
            query=base_query,
            query_blocks=blocks_data,
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
