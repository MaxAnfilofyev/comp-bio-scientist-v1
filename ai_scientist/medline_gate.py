from __future__ import annotations

import requests
import xml.etree.ElementTree as ET
from typing import Optional, Literal

from pydantic import BaseModel, Field
from datetime import datetime, timezone

from ai_scientist.evidence_service import (
    EvidenceService,
    RecordQualityCheckRequest,
    CheckType,
    Verdict,
    PolicyRef,
    ActorRef,
    CreateSearchRunRequest,
    Provider
)

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Request Schemas

class PubMedSearchRequest(BaseModel):
    project_id: str
    query: str = Field(..., description="Raw PubMed query (user intent).")
    policy_query_template: Optional[str] = Field(None, description="Template to enforce policy, e.g. '{query} AND pmc[sb]'")
    use_history: bool = True
    retmax: int = Field(200, ge=1, le=100000)
    retstart: int = Field(0, ge=0)
    sort: Optional[Literal["relevance", "pub+date", "date"]] = "relevance"
    mindate: Optional[str] = None  # YYYY/MM/DD or YYYY
    maxdate: Optional[str] = None  # YYYY/MM/DD or YYYY
    datetype: Optional[Literal["pdat", "edat", "mdat"]] = "pdat"
    api_key: Optional[str] = None  # NCBI API key (optional)
    policy_id: Optional[str] = None # Added for consistency
    search_run_id: Optional[str] = None # Added for pagination reuse


class PubMedSearchResponse(BaseModel):
    search_run_id: str
    provider: Literal["PUBMED"] = "PUBMED"
    query: str
    count_total: int
    retmax: int
    retstart: int
    pmids: list[str]
    webenv: Optional[str] = None
    query_key: Optional[str] = None
    warnings: list[str] = []


class PubMedFetchMetadataRequest(BaseModel):
    project_id: str
    pmids: Optional[list[str]] = None
    webenv: Optional[str] = None
    query_key: Optional[str] = None
    retmax: int = Field(200, ge=1, le=10000)
    retstart: int = Field(0, ge=0)
    api_key: Optional[str] = None


class PubMedWorkMetadata(BaseModel):
    pmid: str
    doi: Optional[str] = None
    pmcid: Optional[str] = None

    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None

    publication_types: list[str] = []
    medline_indexing_status: Optional[str] = None  # if present in record
    is_preprint_like: Optional[bool] = None        # derived from pub types
    has_abstract: Optional[bool] = None

    # Provenance
    fetched_at: str  # ISO8601


class PubMedFetchMetadataResponse(BaseModel):
    works: list[PubMedWorkMetadata]
    not_found_pmids: list[str] = []
    warnings: list[str] = []


class PubMedLinkToPmcRequest(BaseModel):
    project_id: str
    pmids: list[str] = Field(..., min_length=1, max_length=5000)
    linkname: str = "pubmed_pmc"
    api_key: Optional[str] = None


class PmidToPmcids(BaseModel):
    pmid: str
    pmcids: list[str] = []


class PubMedLinkToPmcResponse(BaseModel):
    links: list[PmidToPmcids]
    warnings: list[str] = []


class PubMedFetchFullTextRequest(BaseModel):
    project_id: str
    pmcid: str = Field(..., description="PMCID to fetch (e.g. PMC123456).")

class PubMedFetchFullTextResponse(BaseModel):
    pmcid: str
    xml_content: Optional[str] = None
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# Check/Decision Schemas

# Redeclare CheckType if needed or import. Imported.
# Redeclare Verdict if needed or import. Imported.

class CandidateCheckTarget(BaseModel):
    candidate_id: str
    claim_id: Optional[str] = None
    doi: str
    pmid: Optional[str] = None
    pmcid: Optional[str] = None


class CandidateCheckResult(BaseModel):
    candidate_id: str
    claim_id: Optional[str] = None
    check_type: str
    verdict: str  # Pydantic validation to Enum happens in RecordChecksRequest or logic? 
                  # Spec uses Literal["PASS"...], logic uses Enum. 
    details_json: dict = {}


class RecordChecksRequestInput(BaseModel):
    # This is the input to the tool function
    project_id: str
    policy_id: str
    policy_hash: Optional[str] = None
    executed_by: str
    executed_at: str  # ISO8601
    results: list[CandidateCheckResult]


class RecordChecksResponse(BaseModel):
    inserted_check_ids: list[str]
    counts_by_verdict: dict[str, int]


# --- Gate Implementation ---

class MedlinePmcEvidenceGate:
    def __init__(self, evidence_service: Optional[EvidenceService] = None):
        self.evidence_service = evidence_service or EvidenceService()

    def _get_utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _requests_post(self, url: str, data: dict, api_key: Optional[str]) -> requests.Response:
        params = {}
        if api_key:
            params["api_key"] = api_key
        # NCBI requires tool/email
        data["tool"] = "ai_scientist_archivist"
        data["email"] = "bot@example.com"
        
        import time
        for i in range(3):
            resp = requests.post(url, data=data, params=params)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                time.sleep((i + 1) * 2) # 2s, 4s, 6s
                continue
            resp.raise_for_status()
            return resp
        # Final attempt
        resp.raise_for_status()
        return resp

    def _requests_get(self, url: str, params: dict, api_key: Optional[str]) -> requests.Response:
        if api_key:
            params["api_key"] = api_key
        params["tool"] = "ai_scientist_archivist"
        params["email"] = "bot@example.com"
        
        import time
        for i in range(3):
            resp = requests.get(url, params=params)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                time.sleep((i + 1) * 2)
                continue
            resp.raise_for_status()
            return resp
        # Final attempt
        resp.raise_for_status()
        return resp

    def pubmed_search(self, request: PubMedSearchRequest) -> PubMedSearchResponse:
        """
        Executes ESearch against PubMed.
        Records a 'search_run' in EvidenceService unless search_run_id is provided (pagination).
        """
        url = f"{NCBI_BASE_URL}/esearch.fcgi"
        
        # Compile query if template provided
        final_query = request.query
        if request.policy_query_template:
            final_query = request.policy_query_template.format(query=request.query)
        
        print(f"DEBUG: final_query='{final_query}'")

        params = {
            "db": "pubmed",
            "term": final_query,
            "usehistory": "y" if request.use_history else "n",
            "retmax": request.retmax,
            "retstart": request.retstart,
            "retmode": "json"
        }
        if request.sort:
            params["sort"] = request.sort
        if request.mindate:
            params["mindate"] = request.mindate
        if request.maxdate:
            params["maxdate"] = request.maxdate
        if request.datetype:
            params["datetype"] = request.datetype
        
        resp = self._requests_get(url, params, request.api_key)
        data = resp.json()
        
        esearchresult = data.get("esearchresult", {})
        count = int(esearchresult.get("count", 0))
        ids = esearchresult.get("idlist", [])
        webenv = esearchresult.get("webenv")
        query_key = esearchresult.get("querykey")
        
        # Logic: New Run vs Reuse
        if request.search_run_id:
            final_run_id = request.search_run_id
            # Reuse existing run, no DB insert.
        else:
            # filters_json should capture the parameters used
            filters = {
                "retmax": request.retmax,
                "retstart": request.retstart,
                "sort": request.sort,
                "mindate": request.mindate,
                "maxdate": request.maxdate,
                "policy_template": request.policy_query_template
            }
            
            run_req = CreateSearchRunRequest(
                project_id=request.project_id,
                provider=Provider.PUBMED,
                query_template_id="custom_pubmed" if not request.policy_query_template else "policy_template",
                query_text=final_query, 
                policy=PolicyRef(policy_id=request.policy_id or "medline_gate_unknown", policy_hash=None), 
                result_count_total=count,
                top_k_stored=len(ids), 
                filters_json=filters,
                notes=f"Raw Query: {request.query}", 
                provider_cursor_json={"webenv": webenv, "query_key": query_key} if webenv else {}
            )
            
            real_run = self.evidence_service.create_search_run(run_req)
            final_run_id = real_run.search_run_id

        return PubMedSearchResponse(
            search_run_id=final_run_id,
            provider="PUBMED",
            query=request.query,
            count_total=count,
            retmax=len(ids),
            retstart=request.retstart,
            pmids=ids,
            webenv=webenv,
            query_key=query_key
        )

    def pubmed_fetch_metadata(self, request: PubMedFetchMetadataRequest) -> PubMedFetchMetadataResponse:
        url = f"{NCBI_BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "retmax": request.retmax,
            "retstart": request.retstart
        }
        if request.pmids:
            params["id"] = ",".join(request.pmids)
        elif request.webenv and request.query_key:
            params["WebEnv"] = request.webenv
            params["query_key"] = request.query_key
        else:
             return PubMedFetchMetadataResponse(works=[])

        resp = self._requests_post(url, params, request.api_key) # EFetch often better as POST for many IDs
        # Parse XML
        root = ET.fromstring(resp.content)
        
        works = []
        for article in root.findall(".//PubmedArticle"):
            medline = article.find("MedlineCitation")
            if medline is None:
                continue
                
            pmid = medline.findtext("PMID")
            if pmid is None:
                continue
            
            article_data = medline.find("Article")
            title = article_data.findtext("ArticleTitle")
            
            journal = article_data.find("Journal")
            journal_title = journal.findtext("Title") if journal is not None else None
            
            # Year
            year_str = None
            if journal:
                journal_issue = journal.find("JournalIssue")
                if journal_issue is not None:
                    pub_date = journal_issue.find("PubDate")
                    if pub_date is not None:
                        year_str = pub_date.findtext("Year")
            if not year_str: # fallback to ArticleDate
                art_date = article_data.find("ArticleDate")
                if art_date is not None:
                    year_str = art_date.findtext("Year")
            
            year = int(year_str) if year_str and year_str.isdigit() else None
            
            # DOI / PMCID
            doi = None
            pmcid = None
            
            # IDs in PubmedData
            pubmed_data = article.find("PubmedData")
            if pubmed_data is not None:
                article_ids = pubmed_data.find("ArticleIdList")
                if article_ids is not None:
                    for aid in article_ids.findall("ArticleId"):
                        id_type = aid.get("IdType")
                        val = aid.text
                        if id_type == "doi":
                            doi = val
                        elif id_type == "pmc":
                            pmcid = val
            
            # ElocationID DOI fallback
            if not doi:
                for eloc in article_data.findall("ELocationID"):
                    if eloc.get("EIdType") == "doi":
                        doi = eloc.text
                        break
            
            # Pub types
            pub_types = []
            pt_list = article_data.find("PublicationTypeList")
            if pt_list is not None:
                for pt in pt_list.findall("PublicationType"):
                    if pt.text:
                        pub_types.append(pt.text)
            
            is_preprint = "Preprint" in pub_types
            
            has_abs = article_data.find("Abstract") is not None
            
            works.append(PubMedWorkMetadata(
                pmid=pmid,
                doi=doi,
                pmcid=pmcid,
                title=title,
                journal=journal_title,
                year=year,
                publication_types=pub_types,
                medline_indexing_status=medline.get("Status"),
                is_preprint_like=is_preprint,
                has_abstract=has_abs,
                fetched_at=self._get_utc_now()
            ))
            
        return PubMedFetchMetadataResponse(works=works)

    def pubmed_link_to_pmc(self, request: PubMedLinkToPmcRequest) -> PubMedLinkToPmcResponse:
        url = f"{NCBI_BASE_URL}/elink.fcgi"
        
        # Batching might be needed if len(pmids) > large number, but tool spec says up to 5000.
        # ELink POST handles many.
        
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "linkname": request.linkname,
            "retmode": "xml",
            "id": ",".join(request.pmids)
        }
        
        resp = self._requests_post(url, params, request.api_key)
        root = ET.fromstring(resp.content)
        
        links = []
        for linkset in root.findall("LinkSet"):
            # Input ID
            id_list = linkset.find("IdList")
            if id_list is None:
                continue
            src_id = id_list.findtext("IdList/Id") # Actually typical structure is just <Id>
            if not src_id:
                src_id = id_list.findtext("Id")
            if src_id is None:
                continue 
            
            # Linked IDs
            ls_db = linkset.find("LinkSetDb")
            pmcids = []
            if ls_db is not None:
                for link in ls_db.findall("Link"):
                    lid = link.findtext("Id")
                    if lid:
                        pmcids.append(f"PMC{lid}") # ELink returns numeric ID for PMC usually? 
                        # Actually ELink db=pmc returns numeric ID. PMCID usually PMC+number.
                        # Spec usually expects PMCID string. I will normalize to PMC prefix if missing?
                        # Wait, checking standard. PMC numeric ID is what eutils uses. EFetch db=pmc id=...
                        # But standard string is PMCxxxx.
                        # I'll append PMC prefix if it looks numeric.
            
            # ELink structure can be complex if multiple inputs.
            # Actually with multiple IDs, ELink returns multiple LinkSets, one per IdList? No.
            # It returns one LinkSet usually if submitted together? 
            # Or multiple LinkSet if one-to-one?
            # ELink documentation: If multiple IDs provided, check correspondence=yes/no (default no).
            # "If id is a list of UIDs... ELink returns a separate LinkSet for each UID" if cmd=neighbor (default).
            pass # The loop above handles multiple LinkSets
            
            links.append(PmidToPmcids(
                pmid=src_id,
                pmcids=pmcids
            ))
            
        return PubMedLinkToPmcResponse(links=links)

    def pubmed_fetch_fulltext(self, request: PubMedFetchFullTextRequest) -> PubMedFetchFullTextResponse:
        url = f"{NCBI_BASE_URL}/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": request.pmcid,
            "retmode": "xml"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return PubMedFetchFullTextResponse(
            pmcid=request.pmcid,
            xml_content=response.text
        )

    def record_checks(self, request: RecordChecksRequestInput) -> RecordChecksResponse:
        inserted_ids = []
        counts = {"PASS": 0, "FAIL": 0, "UNKNOWN": 0, "NA": 0}
        
        for res in request.results:
            # Map verdict string to Enum
            try:
                verdict_enum = Verdict(res.verdict)
            except ValueError:
                verdict_enum = Verdict.UNKNOWN
            
            req = RecordQualityCheckRequest(
                candidate_id=res.candidate_id,
                claim_id=res.claim_id,
                check_type=CheckType(res.check_type),
                verdict=verdict_enum,
                details_json=res.details_json,
                policy=PolicyRef(policy_id=request.policy_id, policy_hash=request.policy_hash),
                executed_by=ActorRef(agent_id=request.executed_by, role="archivist")
            )
            
            chk_resp = self.evidence_service.record_quality_check(req)
            inserted_ids.append(chk_resp.check_id)
            if res.verdict in counts:
                counts[res.verdict] += 1
            
        return RecordChecksResponse(inserted_check_ids=inserted_ids, counts_by_verdict=counts)
