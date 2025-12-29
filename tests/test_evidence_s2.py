import unittest
from unittest.mock import MagicMock, patch
import json
import uuid
from datetime import datetime

# Local imports
from ai_scientist.evidence_service import EvidenceService, CheckType, Verdict, DecisionOutcome
from ai_scientist.evidence.s2_client import S2Client, S2SearchResponse, S2PaperHit
from ai_scientist.evidence.evidence_grounder_s2 import EvidenceGrounderS2
from ai_scientist.model.evidence import OrchestrateRoundRequest, NextAction, S2SearchRequest

class TestEvidenceGrounderS2(unittest.TestCase):

    def setUp(self):
        # Mock Dependencies
        self.mock_ev_service = MagicMock() # Removed spec to allow dynamic engine attr
        self.mock_s2_client = MagicMock()
        self.mock_llm_logic = MagicMock()
        
        # Engine mock specifically for raw connection
        self.mock_conn = MagicMock()
        self.mock_ev_service.engine = MagicMock() # Explicitly create engine mock
        self.mock_ev_service.engine.connect.return_value.__enter__.return_value = self.mock_conn
        
        # Setup Grounder
        self.grounder = EvidenceGrounderS2(
            evidence_service=self.mock_ev_service,
            s2_client=self.mock_s2_client,
            llm_client=self.mock_llm_logic
        )

    @patch("ai_scientist.evidence.evidence_grounder_s2.llm_topic_triage")
    @patch("ai_scientist.evidence.evidence_grounder_s2.llm_entailment_s2")
    def test_orchestrate_round_success(self, mock_entailment, mock_triage):
        # 1. Setup Data
        claim_id = "CLAIM_123"
        project_id = "PROJ_ABC"
        
        # Mock DB Claim Lookup
        mock_row = MagicMock()
        mock_row._mapping = {"claim_id": claim_id, "statement": "Kinesin-1 transports mitochondria.", "claim_type": "mechanism"}
        self.mock_conn.execute.return_value.fetchone.return_value = mock_row
        
        # Mock S2 Search
        s2_hits = [
            S2PaperHit(
                paper_id="S2_P1", rank=0, title="Kinesin transport", abstract="It moves things.",
                external_ids={"DOI": "10.1038/nature123"}, year=2020, venue="Nature",
                is_open_access=True, open_access_pdf_url="http://pdf",
                tldr="Moves things", publication_types=["JournalArticle"],
                citation_count=10
            )
        ]
        self.mock_s2_client.search_relevance.return_value = S2SearchResponse(
            total=100, offset=0, hits=s2_hits, compiled_query="Kinesin"
        )
        
        # Mock Search Run Creation
        run_mock = MagicMock()
        run_mock.search_run_id = "SR_TEST"
        self.mock_ev_service.create_search_run.return_value = run_mock
        
        # Mock Ingest
        ingest_mock = MagicMock()
        ingest_mock.candidate_ids = ["CAND_1"]
        self.mock_ev_service.ingest_candidates.return_value = ingest_mock
        
        # Mock LLM Triage -> PASS
        mock_triage_resp = MagicMock()
        mock_triage_resp.topic_match = "PASS"
        mock_triage_resp.mismatch_codes = []
        mock_triage_resp.model_dump.return_value = {}
        mock_triage.return_value = mock_triage_resp
        
        # Mock Entailment -> SUPPORTED
        mock_entailment.return_value = {
            "verdict": "SUPPORTED",
            "rationale": "Direct evidence.",
            "anchor_quote": "It moves things."
        }
        
        # Mock Promotion
        promote_resp = MagicMock()
        promote_resp.support_id = "SUP_1"
        self.mock_ev_service.promote_candidate_to_support.return_value = promote_resp
        
        # 2. Execute
        req = OrchestrateRoundRequest(
            project_id=project_id, claim_id=claim_id, search_round_index=1,
            policy_id="test_pol", pubmed_retmax=5
        )
        resp = self.grounder.orchestrate_round(req)
        
        # 3. Verify
        self.assertTrue(resp.done)
        self.assertEqual(len(resp.supports_created), 1)
        self.assertEqual(resp.supports_created[0], "SUP_1")
        self.assertEqual(resp.next_action, "STOP_OPEN_GAP") # Actually success maps to STOP_OPEN_GAP in current enum?
        
        # Check calls
        self.mock_s2_client.search_relevance.assert_called_once()
        self.mock_ev_service.record_quality_check.assert_called() # Triage + Entailment
        self.mock_ev_service.promote_candidate_to_support.assert_called_once()
        self.mock_ev_service.create_search_round.assert_called_once()

    @patch("ai_scientist.evidence.evidence_grounder_s2.llm_topic_triage")
    def test_orchestrate_round_triage_fail(self, mock_triage):
        # 1. Setup Data same as above
        mock_row = MagicMock()
        mock_row._mapping = {"claim_id": "C1", "statement": "Text", "claim_type": "fact"}
        self.mock_conn.execute.return_value.fetchone.return_value = mock_row
        
        s2_hits = [
             S2PaperHit(
                paper_id="S2_P2", rank=0, title="Off topic", abstract="Physics stuff.",
                external_ids={"DOI": "10.111/phys"}, year=2020,
                tldr="Physics", venue="Phys Rev", publication_types=["JournalArticle"],
                citation_count=5
            )
        ]
        self.mock_s2_client.search_relevance.return_value = S2SearchResponse(total=10, offset=0, hits=s2_hits, compiled_query="Q")
        
        run_mock = MagicMock()
        run_mock.search_run_id = "SR_FAIL"
        self.mock_ev_service.create_search_run.return_value = run_mock
        
        ingest_mock = MagicMock()
        ingest_mock.candidate_ids = ["CAND_FAIL"]
        self.mock_ev_service.ingest_candidates.return_value = ingest_mock
        
        # FAIL Triage
        mock_triage_resp = MagicMock()
        mock_triage_resp.topic_match = "FAIL"
        mock_triage_resp.mismatch_codes = ["WRONG_DOMAIN"]
        mock_triage_resp.drift_concepts = ["Quantum"]
        mock_triage_resp.model_dump.return_value = {}
        mock_triage.return_value = mock_triage_resp
        
        # 2. Execute
        req = OrchestrateRoundRequest(project_id="P", claim_id="C1", search_round_index=1, policy_id="pol")
        resp = self.grounder.orchestrate_round(req)
        
        # 3. Verify
        self.assertFalse(resp.done)
        self.assertEqual(len(resp.supports_created), 0)
        self.assertEqual(resp.candidate_results[0].decision_outcome, DecisionOutcome.REJECTED)
        
        # Check Next Action (Low Triage -> Rewrite)
        # T = 0.0 < 0.15 -> DISAMBIGUATE
        self.assertEqual(resp.next_action, "REWRITE_QUERY_DISAMBIGUATE")

if __name__ == '__main__':
    unittest.main()
