import os
import requests
import time
from typing import Any, Dict, List, Optional
import backoff

from ai_scientist.tools.base_tool import BaseTool


def on_backoff(details: Any) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


class CrossRefSearchTool(BaseTool):
    def __init__(
        self,
        name: str = "SearchCrossRef",
        description: str = (
            "Search for relevant literature and DOIs using CrossRef. "
            "Useful for finding DOIs missing from other sources."
        ),
        max_results: int = 5,
    ):
        parameters = [
            {
                "name": "query",
                "type": "str",
                "description": "The search query (e.g. paper title) to find works.",
            }
        ]
        super().__init__(name, description, parameters)
        self.max_results = max_results
        self.email = os.getenv("CROSSREF_EMAIL", "ai-scientist@example.com")

    def use_tool(self, **kwargs: Any) -> Optional[List[Dict]]:
        query = kwargs.get("query")
        if not query:
            return None
        return self.search_works(query)

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
        on_backoff=on_backoff,
        max_tries=5,
    )
    def search_works(self, query: str) -> Optional[List[Dict]]:
        if not query:
            return None
        
        headers = {
            "User-Agent": f"AI-Scientist/1.0 (mailto:{self.email})"
        }
        
        # CrossRef API: /works?query.title=...
        rsp = requests.get(
            "https://api.crossref.org/works",
            headers=headers,
            params={
                "query": query,
                "rows": self.max_results,
                "select": "title,author,created,DOI,container-title",
            },
            timeout=10,
        )
        
        if rsp.status_code == 429:
            # Explicitly raise for backoff if 429
            rsp.raise_for_status()
            
        if rsp.status_code != 200:
            print(f"CrossRef error {rsp.status_code}: {rsp.text[:200]}")
            return None

        data = rsp.json()
        items = data.get("message", {}).get("items", [])
        
        results = []
        for item in items:
            # Extract and normalize fields to match our internal schema similar to S2
            
            # Title is a list in CrossRef usually
            title_list = item.get("title", [])
            title = title_list[0] if title_list else "Unknown Title"
            
            # Authors
            authors_list = item.get("author", [])
            authors = []
            for a in authors_list:
                given = a.get("given", "")
                family = a.get("family", "")
                full = f"{given} {family}".strip()
                if full:
                    authors.append(full)
            
            # Year
            # date-parts is [[2004, 9, 23]]
            created = item.get("created", {})
            date_parts = created.get("date-parts", [[]])
            year = date_parts[0][0] if (date_parts and date_parts[0]) else None
            
            # Venue
            venue_list = item.get("container-title", [])
            venue = venue_list[0] if venue_list else None

            results.append({
                "title": title,
                "authors": authors,
                "year": year,
                "doi": item.get("DOI"),
                "venue": venue,
                "source": "crossref"
            })
            
        return results
