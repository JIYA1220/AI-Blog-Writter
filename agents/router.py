# agents/router.py
# ============================================================
# AI Blog Writer — Adaptive Router Node
# Decides: does this topic need web retrieval, or pure generation?
# This is a CONDITIONAL EDGE in the LangGraph DAG.
# ============================================================

import os
from utils.llm import get_llm
from utils.json_utils import parse_llm_json
from utils.logger import log_stage, log_info
from schemas.models import AIBlogWriterState


ROUTER_PROMPT = """You are a routing agent for a blog writing system.

Given a blog topic, decide whether the writer needs to search the web for:
- Recent statistics, news, or data
- Factual claims that need grounding
- Current events or evolving topics

OR whether the topic can be written from general knowledge alone:
- Timeless concepts (e.g., "How to be productive")
- Opinion / thought leadership pieces
- Well-established technical explanations

Topic: {topic}
Target Audience: {audience}

Respond ONLY with valid JSON:
{{
  "needs_retrieval": true or false,
  "reason": "one sentence explanation"
}}
"""


async def router_node(state: AIBlogWriterState) -> dict:
    """
    Router Node — first node in the DAG.
    Sets state['needs_retrieval'] = True or False.
    LangGraph uses this to conditionally route to retriever or planner.
    """
    log_stage("ROUTER", f"Analyzing topic: '{state['topic']}'")

    # Check if search is globally disabled
    enable_search = os.getenv("ENABLE_SEARCH", "true").lower() == "true"
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not enable_search or not tavily_key:
        log_info("Web Search is disabled via config or missing API key — forcing GENERATION mode.")
        return {"needs_retrieval": False, "errors": []}

    llm = get_llm(temperature=0.1)  # Low temp for deterministic routing

    prompt = ROUTER_PROMPT.format(
        topic=state["topic"],
        audience=state.get("target_audience", "general readers")
    )

    try:
        response = await llm.ainvoke(prompt)
        decision = parse_llm_json(response.content)
        needs_retrieval = bool(decision.get("needs_retrieval", False))
        reason = decision.get("reason", "No reason provided")

        log_info(f"Decision: {'RETRIEVAL' if needs_retrieval else 'GENERATION'} — {reason}")

        return {"needs_retrieval": needs_retrieval, "errors": []}

    except Exception as e:
        log_stage("ROUTER", f"Fallback to generation (parse error: {e})", style="bold yellow")
        return {"needs_retrieval": False, "errors": [f"Router error: {str(e)}"]}


def route_decision(state: AIBlogWriterState) -> str:
    """
    Conditional edge function.
    Returns the name of the NEXT NODE based on router decision.
    """
    if state.get("needs_retrieval", False):
        return "retriever"
    return "planner"
