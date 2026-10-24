# agents/retriever.py
# ============================================================
# AI Blog Writer — Retrieval Node
# Uses Tavily to search the web, then deduplicates evidence.
# Keeps outputs GROUNDED — no hallucinated statistics.
# ============================================================

import os
from dotenv import load_dotenv
from utils.logger import log_stage, log_info, log_warning, log_success
from schemas.models import RetrievalResult, AIBlogWriterState

load_dotenv()


import asyncio

def _get_tavily_client():
    """Returns a Tavily client if API key is available."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=api_key)
    except ImportError:
        return None


def _deduplicate(snippets: list[str]) -> list[str]:
    """
    Simple deduplication: remove near-duplicate snippets.
    AI Blog Writer's retriever keeps evidence clean and non-redundant.
    """
    seen = set()
    unique = []
    for s in snippets:
        # Use first 80 chars as fingerprint
        fingerprint = s[:80].lower().strip()
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(s)
    return unique[:6]  # Cap at 6 snippets for context efficiency


async def retriever_node(state: AIBlogWriterState) -> dict:
    """
    Retriever Node — only runs when router says needs_retrieval=True.
    Searches Tavily, deduplicates results, returns grounded evidence.
    """
    topic = state["topic"]
    log_stage("RETRIEVER", f"Searching for: '{topic}'")

    client = _get_tavily_client()

    if client is None:
        log_warning("Tavily key not set — skipping retrieval, proceeding with generation.")
        result = RetrievalResult(
            query=topic,
            snippets=[],
            sources=[],
            is_grounded=False
        )
        return {"retrieval_results": result, "errors": []}

    try:
        # Use to_thread for the blocking network call
        response = await asyncio.to_thread(
            client.search,
            query=topic,
            search_depth="advanced",
            max_results=8,
            include_answer=True
        )

        raw_snippets = []
        sources = []

        for r in response.get("results", []):
            content = r.get("content", "").strip()
            url = r.get("url", "")
            if content:
                raw_snippets.append(content)
                sources.append(url)

        # Add Tavily's synthesized answer if available
        if response.get("answer"):
            raw_snippets.insert(0, response["answer"])

        # Deduplicate
        clean_snippets = _deduplicate(raw_snippets)

        log_success(f"Retrieved {len(clean_snippets)} unique evidence snippets")
        for i, s in enumerate(clean_snippets[:3]):
            log_info(f"  [{i+1}] {s[:100]}...")

        result = RetrievalResult(
            query=topic,
            snippets=clean_snippets,
            sources=list(dict.fromkeys(sources))[:6],
            is_grounded=True
        )
        return {"retrieval_results": result, "errors": []}

    except Exception as e:
        log_warning(f"Retrieval failed: {e} — continuing without grounding")
        result = RetrievalResult(
            query=topic,
            snippets=[],
            sources=[],
            is_grounded=False
        )
        return {"retrieval_results": result, "errors": [f"Retrieval error: {str(e)}"]}
