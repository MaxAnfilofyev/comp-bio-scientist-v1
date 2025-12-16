
import pytest
import json
from unittest.mock import MagicMock, patch
from ai_scientist.medline_gate import (
    MedlinePmcEvidenceGate,
    PubMedSearchRequest, PubMedFetchMetadataRequest, PubMedLinkToPmcRequest,
    RecordChecksRequestInput, CandidateCheckResult
)
from ai_scientist.evidence_service import EvidenceService

@pytest.fixture
def mock_evidence_service():
    return MagicMock(spec=EvidenceService)

@pytest.fixture
def gate(mock_evidence_service):
    return MedlinePmcEvidenceGate(evidence_service=mock_evidence_service)

def test_pubmed_search(gate):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "esearchresult": {
            "count": "100",
            "retmax": "20",
            "retstart": "0",
            "idlist": ["123", "456"],
            "webenv": "WEBENV123",
            "querykey": "1"
        }
    }
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        req = PubMedSearchRequest(project_id="p1", query="cancer")
        
        # We need to mock create_search_run return value
        mock_run = MagicMock()
        mock_run.search_run_id = "SR_TEST"
        gate.evidence_service.create_search_run.return_value = mock_run

        resp = gate.pubmed_search(req)
        
        assert resp.search_run_id == "SR_TEST"
        assert resp.count_total == 100
        assert resp.pmids == ["123", "456"]
        assert resp.webenv == "WEBENV123"
        
        mock_get.assert_called_once()
        gate.evidence_service.create_search_run.assert_called_once()

def test_pubmed_fetch_metadata(gate):
    xml_content = """
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation Status="MEDLINE">
                <PMID>12345</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                    <Journal>
                        <Title>Journal of Testing</Title>
                        <JournalIssue>
                            <PubDate>
                                <Year>2023</Year>
                            </PubDate>
                        </JournalIssue>
                    </Journal>
                    <ELocationID EIdType="doi">10.1000/12345</ELocationID>
                    <Abstract>
                        <AbstractText>Some abstract.</AbstractText>
                    </Abstract>
                    <PublicationTypeList>
                        <PublicationType>Journal Article</PublicationType>
                    </PublicationTypeList>
                </Article>
            </MedlineCitation>
            <PubmedData>
                <ArticleIdList>
                    <ArticleId IdType="doi">10.1000/12345</ArticleId>
                    <ArticleId IdType="pmc">PMC99999</ArticleId>
                </ArticleIdList>
            </PubmedData>
        </PubmedArticle>
    </PubmedArticleSet>
    """
    mock_resp = MagicMock()
    mock_resp.content = xml_content.encode('utf-8')
    
    with patch("requests.post", return_value=mock_resp):
        req = PubMedFetchMetadataRequest(project_id="p1", pmids=["12345"])
        resp = gate.pubmed_fetch_metadata(req)
        
        assert len(resp.works) == 1
        w = resp.works[0]
        assert w.pmid == "12345"
        assert w.doi == "10.1000/12345"
        assert w.pmcid == "PMC99999"
        assert w.year == 2023
        assert w.title == "Test Article"
        assert "Journal Article" in w.publication_types
        assert w.has_abstract is True

def test_pubmed_link_to_pmc(gate):
    xml_content = """
    <eLinkResult>
        <LinkSet>
            <IdList>
                <Id>12345</Id>
            </IdList>
            <LinkSetDb>
                <Link>
                    <Id>99999</Id>
                </Link>
            </LinkSetDb>
        </LinkSet>
    </eLinkResult>
    """
    mock_resp = MagicMock()
    mock_resp.content = xml_content.encode('utf-8')
    
    with patch("requests.post", return_value=mock_resp):
        req = PubMedLinkToPmcRequest(project_id="p1", pmids=["12345"])
        resp = gate.pubmed_link_to_pmc(req)
        
        assert len(resp.links) == 1
        l = resp.links[0]
        assert l.pmid == "12345"
        assert l.pmcids == ["PMC99999"] # logic adds PMC prefix? yes I verified code

def test_record_checks(gate):
    mock_check_resp = MagicMock()
    mock_check_resp.check_id = "QC_1"
    gate.evidence_service.record_quality_check.return_value = mock_check_resp
    
    req = RecordChecksRequestInput(
        project_id="p1",
        policy_id="pol1",
        executed_by="agent1",
        executed_at="now",
        results=[
            CandidateCheckResult(
                candidate_id="c1",
                check_type="has_doi",
                verdict="PASS"
            )
        ]
    )
    
    resp = gate.record_checks(req)
    assert resp.inserted_check_ids == ["QC_1"]
    assert resp.counts_by_verdict["PASS"] == 1
    gate.evidence_service.record_quality_check.assert_called_once()

if __name__ == "__main__":
    pass
