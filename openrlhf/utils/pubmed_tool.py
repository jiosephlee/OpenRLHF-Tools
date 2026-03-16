"""PubMed search tool for multi-turn agent tool calling.

Uses NCBI E-utilities (ESearch + EFetch) — free, no API key required
for moderate usage (<3 req/sec without key, 10 req/sec with).

Set NCBI_API_KEY env var to increase rate limits (optional).
"""

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_API_KEY = os.environ.get("NCBI_API_KEY", "")
_MAX_RESULTS = 5  # keep responses concise for RL context windows


def _api_params() -> str:
    return f"&api_key={_API_KEY}" if _API_KEY else ""


def _esearch(query: str, max_results: int) -> List[str]:
    """Return list of PMIDs matching *query*."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    })
    url = f"{_BASE}/esearch.fcgi?{params}{_api_params()}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: List[str]) -> List[Dict[str, str]]:
    """Fetch title + abstract for each PMID."""
    if not pmids:
        return []
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    })
    url = f"{_BASE}/efetch.fcgi?{params}{_api_params()}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        tree = ET.parse(resp)

    results = []
    for article in tree.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//Abstract")

        pmid = pmid_el.text if pmid_el is not None else "?"
        title = title_el.text if title_el is not None else "No title"

        # Abstract may have multiple <AbstractText> sections
        abstract_parts = []
        if abstract_el is not None:
            for part in abstract_el.findall("AbstractText"):
                label = part.get("Label", "")
                text = "".join(part.itertext()).strip()
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."

        results.append({"pmid": pmid, "title": title, "abstract": abstract})
    return results


def pubmed_search(query: str, max_results: int = _MAX_RESULTS) -> str:
    """Search PubMed and return titles + abstracts for top results.

    Args:
        query: Search query (supports PubMed syntax, e.g. "aspirin AND toxicity").
        max_results: Number of results to return (default 5, max 10).
    """
    max_results = min(int(max_results), 10)
    try:
        pmids = _esearch(query, max_results)
        if not pmids:
            return f"No PubMed results found for query: {query!r}"
        papers = _efetch(pmids)
    except Exception as e:
        return f"PubMed search failed: {e}"

    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(
            f"[{i}] PMID {p['pmid']}: {p['title']}\n"
            f"    Abstract: {p['abstract']}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI tool schema
# ---------------------------------------------------------------------------
PUBMED_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "pubmed_search",
        "description": (
            "Search PubMed for biomedical literature. Returns titles and "
            "abstracts of the most relevant papers. Supports PubMed query "
            "syntax (e.g. 'aspirin AND hepatotoxicity', 'CYP3A4 inhibitor')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PubMed search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

PUBMED_OPENAI_TOOLS: List[Dict[str, Any]] = [PUBMED_SEARCH_TOOL]

PUBMED_CALLABLES: Dict[str, Any] = {
    "pubmed_search": pubmed_search,
}
