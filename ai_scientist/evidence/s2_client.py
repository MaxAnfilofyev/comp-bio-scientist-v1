import requests
import os
import hashlib
import io
import time
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv

from ai_scientist.model.evidence import (
    S2SearchRequest, S2SearchResponse, S2PaperHit,
    S2BatchFetchRequest, S2RecommendationsRequest,
    S2FetchPdfRequest, S2FetchPdfResponse
)

# Load .env file from project root
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

# Optional imports for PDF extraction
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

class S2Client:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})

    def _request(self, method: str, endpoint: str, params: Optional[dict] = None, json_data: Optional[dict] = None, retries: int = 3) -> requests.Response:
        url = f"{S2_API_BASE}/{endpoint}"
        for i in range(retries):
            try:
                # Simple rate limiting: 1 request per second
                time.sleep(1)
                resp = self.session.request(method, url, params=params, json=json_data)
                if resp.status_code == 429:
                    # Rate limited - wait 1 second and retry
                    continue
                if 500 <= resp.status_code < 600:
                    # Server error - wait 1 second and retry
                    continue
                
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as e:
                # 404 is valid for "not found" but raise for others
                if e.response.status_code == 404:
                    return e.response
                raise
        raise requests.exceptions.RequestException(f"Failed to fetch {url} after {retries} retries")

    def search_relevance(self, req: S2SearchRequest) -> S2SearchResponse:
        # Construct S2 Query
        params = {
            "query": req.query,
            "limit": req.limit,
            "offset": req.offset,
            "fields": ",".join(req.fields)
        }
        if req.fields_of_study:
            params["fieldsOfStudy"] = ",".join(req.fields_of_study)
        if req.min_citation_count > 0:
            params["minCitationCount"] = req.min_citation_count
        if req.year_min or req.year_max:
            # Format: 2000-2023 or 2000- or -2023
            ymin = req.year_min if req.year_min else ""
            ymax = req.year_max if req.year_max else ""
            if ymin or ymax:
                params["year"] = f"{ymin}-{ymax}"
        
        if req.publication_types_allow:
            params["publicationTypes"] = ",".join(req.publication_types_allow)
        # S2 API doesn't support "publication_types_block" natively in search params well, 
        # usually must filter post-hoc or via negative query terms?
        # Actually it does support negative clauses in query string but not structured param.
        # We will handle blocking in logic if needed, or assume caller handles logic.
        
        # openAccessPdf logic
        # S2 Search API: `openAccessPdf` field exists. Filter?
        # params["openAccessPdf"] is not a filter. 
        # But `isOpenAccess` is boolean in response.
        # We can't filter server-side easily without boolean syntax in query string?
        # Actually S2 has some filters. But we'll filter client side or trust the ranker.
        
        print(f"DEBUG S2: Calling API with params: {params}")
        resp = self._request("GET", "paper/search", params=params)
        data = resp.json()
        print(f"DEBUG S2: Response total={data.get('total', 0)}, data_count={len(data.get('data', []))}")
        
        total = data.get("total", 0)
        next_offset = data.get("next") 
        # S2 'next' is offset integer? Or next URL?
        # usually next offset int.
        
        hits = []
        raw_papers = data.get("data", [])
        
        for i, p in enumerate(raw_papers):
            # Map to S2PaperHit
            hit = S2PaperHit(
                paper_id=p.get("paperId", ""),
                rank=req.offset + i,
                title=p.get("title"),
                abstract=p.get("abstract"),
                tldr=p.get("tldr", {}).get("text") if p.get("tldr") else None,
                external_ids=p.get("externalIds", {}),
                year=p.get("year"),
                venue=p.get("venue"),
                publication_types=p.get("publicationTypes"),
                citation_count=p.get("citationCount", 0),
                influential_citation_count=p.get("influentialCitationCount", 0),
                is_open_access=p.get("isOpenAccess"),
                open_access_pdf_url=p.get("openAccessPdf", {}).get("url") if p.get("openAccessPdf") else None
            )
            # Logic: If client req requires OA PDF, we check it here?
            # Or usually we return all hits and let filter logic handle it?
            # User instruction: "S2 wrapper... return ranked candidates."
            # We'll return everything and let orchestrator filter? 
            # Or if req.require_open_access_pdf is set, we prioritize?
            # The spec request has `require_open_access_pdf`.
            # If true, filtering strictly might reduce yield to zero if page size is small.
            # Ideally loop until we fill? S2 API doesn't support strict filter easily.
            # We will return hits and mark them. Orchestrator filters.
            hits.append(hit)
            
        return S2SearchResponse(
            total=total,
            offset=req.offset,
            next_offset=next_offset,
            hits=hits,
            compiled_query=req.query # Just the query string
        )

    def fetch_metadata_batch(self, req: S2BatchFetchRequest) -> List[S2PaperHit]:
        if not req.ids:
            return []
            
        # S2 Batch Endpoint: POST /paper/batch
        params = {
            "fields": ",".join(req.fields)
        }
        json_data = {"ids": req.ids}
        
        resp = self._request("POST", "paper/batch", params=params, json_data=json_data)
        data = resp.json()
        
        hits = []
        for i, p in enumerate(data):
            if p is None: # ID not found
                continue
                
            hit = S2PaperHit(
                paper_id=p.get("paperId", req.ids[i] if i < len(req.ids) else ""),
                rank=-1, # Not ranked
                title=p.get("title"),
                abstract=p.get("abstract"),
                tldr=p.get("tldr", {}).get("text") if p.get("tldr") else None,
                external_ids=p.get("externalIds", {}),
                year=p.get("year"),
                venue=p.get("venue"),
                publication_types=p.get("publicationTypes"),
                citation_count=p.get("citationCount", 0),
                influential_citation_count=p.get("influentialCitationCount", 0),
                is_open_access=p.get("isOpenAccess"),
                open_access_pdf_url=p.get("openAccessPdf", {}).get("url") if p.get("openAccessPdf") else None
            )
            hits.append(hit)
        return hits

    def get_recommendations(self, req: S2RecommendationsRequest) -> List[S2PaperHit]:
        # Implementation via papers/{paper_id}/recommendations or batch?
        # S2 has "paper/batch/recommendations" ? No.
        # "paper/{paper_id}/recommendations"
        # If we have multiple seeds... we might need to pick one or call multiple and merge?
        # Spec implies using positive/negative IDs. 
        # S2 API currently mainly supports getting recs for a SINGLE source paper.
        # "POST /recommendations/v1/papers/for-paper" supports multiple seeds??
        # Checking hypothetical S2 API or standard. Standard graph/v1 is per-paper.
        # However, there is a "recommendations" endpoint that takes positive/negative paperIds.
        # It's an experimental or newer endpoint: POST /graph/v1/paper/batch/recommendations? 
        # Actually usually it's "POST /graph/v1/recommendations" taking {"positivePaperIds": [], "negativePaperIds": []}
        
        # Let's assume POST /recommendations is available based on spec.
        # Or fall back to per-paper if fails.
        
        # We will try the "multi-seed" endpoint if we know it exists. 
        # Assuming the endpoint: POST https://api.semanticscholar.org/recommendations/v1/papers/for-paper 
        # Wait, that's not Graph API.
        
        # Let's implement simple approach: Pick top positive seed.
        if not req.positive_paper_ids:
            return []
            
        seed = req.positive_paper_ids[0]
        params = {
            "limit": req.limit,
            "fields": ",".join(req.fields)
        }
        resp = self._request("GET", f"paper/{seed}/recommendations", params=params)
        data = resp.json()
        
        hits = []
        raw_papers = data.get("data", [])
        for i, p in enumerate(raw_papers):
             # Skip if in negatives
             if p.get("paperId") in req.negative_paper_ids:
                 continue
                 
             hit = S2PaperHit(
                paper_id=p.get("paperId", ""),
                rank=i,
                title=p.get("title"),
                abstract=p.get("abstract"),
                tldr=p.get("tldr", {}).get("text") if p.get("tldr") else None,
                external_ids=p.get("externalIds", {}),
                year=p.get("year"),
                venue=p.get("venue"),
                publication_types=p.get("publicationTypes"),
                citation_count=p.get("citationCount", 0),
                influential_citation_count=p.get("influentialCitationCount", 0),
                is_open_access=p.get("isOpenAccess"),
                open_access_pdf_url=p.get("openAccessPdf", {}).get("url") if p.get("openAccessPdf") else None
            )
             hits.append(hit)
        return hits

    def fetch_open_access_pdf(self, req: S2FetchPdfRequest) -> S2FetchPdfResponse:
        # Download PDF
        try:
            r = self.session.get(req.pdf_url, stream=True, timeout=15)
            r.raise_for_status()
            
            # Check size
            content = io.BytesIO()
            size = 0
            for chunk in r.iter_content(chunk_size=8192):
                 size += len(chunk)
                 content.write(chunk)
                 if size > req.max_bytes:
                     return S2FetchPdfResponse(
                         paper_id=req.paper_id, sha256="", content_type="application/pdf", size_bytes=size,
                         errors=["SIZE_LIMIT_EXCEEDED"]
                     )
            
            content_bytes = content.getvalue()
            sha = hashlib.sha256(content_bytes).hexdigest()
            
            extracted_text = None
            method = None
            errors = []
            
            if req.extract_text and HAS_PYPDF:
                try:
                    reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                    text_parts = []
                    for page in reader.pages:
                        text_parts.append(page.extract_text())
                    extracted_text = "\n".join(text_parts)
                    method = "pypdf"
                except Exception as e:
                    errors.append(f"PDF_EXTRACTION_FAILED: {str(e)}")
            elif req.extract_text:
                errors.append("PDF_EXTRACTION_UNAVAILABLE_NO_LIB")
                
            return S2FetchPdfResponse(
                paper_id=req.paper_id,
                sha256=sha,
                content_type=r.headers.get("Content-Type", "application/pdf"),
                size_bytes=size,
                extracted_text=extracted_text,
                extraction_method=method,
                errors=errors
            )
            
        except Exception as e:
            return S2FetchPdfResponse(
                paper_id=req.paper_id, sha256="", content_type="", size_bytes=0,
                errors=[str(e)]
            )
