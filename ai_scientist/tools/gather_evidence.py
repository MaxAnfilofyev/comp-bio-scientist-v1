from typing import Dict, Any
from ai_scientist.tools.base_tool import BaseTool
from ai_scientist.database.claim_graph_service import ClaimGraphService
from ai_scientist.evidence.evidence_grounder_s2 import EvidenceGrounderS2
from ai_scientist.evidence_service import EvidenceService
from ai_scientist.model.evidence import OrchestrateRoundRequest
from sqlalchemy import create_engine

class GatherEvidenceTool(BaseTool):
    """
    Automatically gather S2 evidence for unsupported claims in the claim graph.
    """
    def __init__(self):
        super().__init__(
            name="gather_evidence",
            description="Automatically search Semantic Scholar for evidence to support claims in the claim graph that currently lack sufficient support.",
            parameters=[
                {"name": "project_id", "type": "str", "description": "Project ID (optional, defaults to current)."},
                {"name": "min_support", "type": "int", "description": "Minimum number of supports required (default: 2)."},
                {"name": "max_claims", "type": "int", "description": "Maximum number of claims to process in this batch (default: 5)."},
            ]
        )

    def use_tool(self, project_id: str = "DEFAULT_PROJECT", min_support: int = 2, max_claims: int = 5) -> Dict[str, Any]:
        engine = create_engine("sqlite:///ai_scientist.sqlite")
        claim_service = ClaimGraphService(engine)
        evidence_service = EvidenceService(engine)
        grounder = EvidenceGrounderS2(evidence_service)

        unsupported = claim_service.find_unsupported_claims(project_id, min_support)
        processed_count = 0
        results_summary = []

        for claim_data in unsupported[:max_claims]:
            claim_id = claim_data['claim_id']
            print(f"Gathering evidence for: {claim_id}")
            
            req = OrchestrateRoundRequest(
                project_id=project_id,
                claim_id=claim_id,
                search_round_index=1,
                policy_id="s2_strict_v1.0", # Default policy for automated gathering
                current_query=None,
                pubmed_retmax=10
            )

            try:
                res = grounder.orchestrate_round(req)
                new_supports = len(res.supports_created)
                
                # Link found supports back to graph
                if res.supports_created:
                    node_id = claim_data['node_id']
                    for sid in res.supports_created:
                        claim_service.add_support(
                            node_id, 
                            "claim_support", 
                            sid, 
                            notes="Auto-gathered by gather_evidence"
                        )
                
                results_summary.append({
                    "claim_id": claim_id,
                    "supports_found": new_supports,
                    "action": res.next_action
                })
                processed_count += 1
                
            except Exception as e:
                results_summary.append({
                    "claim_id": claim_id,
                    "error": str(e)
                })

        return {
            "processed_claims": processed_count,
            "details": results_summary,
            "remaining_unsupported": len(unsupported) - processed_count
        }
