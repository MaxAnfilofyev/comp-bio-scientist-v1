import os
import requests
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import backoff

from ai_scientist.tools.base_tool import BaseTool


def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class SemanticScholarSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchSemanticScholar",
        description: str = (
            "Search for relevant literature using Semantic Scholar. "
            "Provide a search query to find relevant papers."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query to find relevant papers.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.S2_API_KEY = os.getenv("S2_API_KEY")
        if not self.S2_API_KEY:
            warnings.warn(
                "No Semantic Scholar API key found. Requests will be subject to stricter rate limits. "
                "Set the S2_API_KEY environment variable for higher limits."
            )

    def use_tool(self, **kwargs: Any) -> Optional[str]:
        query = kwargs.get("query")
        if not query:
            return "Error: Query not provided."
        papers = self.search_for_papers(query)
        if papers:
            return self.format_papers(papers)
        else:
            return "No papers found."

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
    )
    def search_for_papers(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None
        
        headers = {}
        if self.S2_API_KEY:
            headers["X-API-KEY"] = self.S2_API_KEY
        
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": self.max_results,
                # 'doi' is returned inside externalIds; requesting it directly triggers 400 on search endpoint.
                "fields": "title,authors,venue,year,abstract,citationCount,externalIds",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}")
        rsp.raise_for_status()
        results = rsp.json()
        total = results.get("total", 0)
        if total == 0:
            return None

        papers = results.get("data", [])
        # Sort papers by citationCount in descending order
        papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
        return papers

    def format_papers(self, papers: List[Dict]) -> str:
        paper_strings = []
        for i, paper in enumerate(papers):
            authors = ", ".join(
                [author.get("name", "Unknown") for author in paper.get("authors", [])]
            )
            paper_strings.append(
                f"""{i + 1}: {paper.get("title", "Unknown Title")}. {authors}. {paper.get("venue", "Unknown Venue")}, {paper.get("year", "Unknown Year")}.
Number of citations: {paper.get("citationCount", "N/A")}
Abstract: {paper.get("abstract", "No abstract available.")}"""
            )
        return "\n\n".join(paper_strings)


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    S2_API_KEY = os.getenv("S2_API_KEY")
    headers = {}
    if not S2_API_KEY:
        warnings.warn(
            "No Semantic Scholar API key found. Requests will be subject to stricter rate limits."
        )
    else:
        headers["X-API-KEY"] = S2_API_KEY
    
    if not query:
        return None
    
    rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": query,
                "limit": result_limit,
                # 'doi' is nested under externalIds; requesting it directly causes API 400.
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount,externalIds",
            },
        )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


class SemanticScholarRecommendationsTool(BaseTool):
    def __init__(
        self,
        name: str = "GetPaperRecommendations",
        description: str = (
            "Get recommended papers based on a list of positive paper IDs (S2 PaperIds). "
            "Useful for discovering relevant literature similar to what you have found."
        ),
        max_results: int = 10,
    ):
        parameters = [
            {
                "name": "positive_paper_ids",
                "type": "list[str]",
                "description": "List of Semantic Scholar Paper IDs to use as positive examples.",
            },
            {
                "name": "negative_paper_ids",
                "type": "list[str]",
                "description": "Optional list of Semantic Scholar Paper IDs to use as negative examples.",
            },
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.S2_API_KEY = os.getenv("S2_API_KEY")

    def use_tool(self, **kwargs: Any) -> List[Dict]:
        positive_paper_ids = kwargs.get("positive_paper_ids", [])
        negative_paper_ids = kwargs.get("negative_paper_ids")
        return get_recommendations(positive_paper_ids, negative_paper_ids, limit=self.max_results, api_key=self.S2_API_KEY)


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def get_recommendations(
    positive_paper_ids: List[str],
    negative_paper_ids: Optional[List[str]] = None,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> List[Dict]:
    if not positive_paper_ids:
        return []

    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key
    
    payload = {
        "positivePaperIds": positive_paper_ids,
        "negativePaperIds": negative_paper_ids or [],
    }

    url = "https://api.semanticscholar.org/recommendations/v1/papers/"
    # For getting specific fields back (optional, but good for consistency)
    params = {
        "limit": limit,
        "fields": "title,authors,venue,year,abstract,citationCount,externalIds"
    }

    rsp = requests.post(url, headers=headers, json=payload, params=params)
    
    if rsp.status_code == 404:
        # One or more paper IDs not found
        print("[warn] Recommendations 404: One or more input IDs not found in S2 graph.")
        return []

    rsp.raise_for_status()
    results = rsp.json()
    return results.get("recommendedPapers", [])
