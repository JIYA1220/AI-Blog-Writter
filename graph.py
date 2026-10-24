# graph.py
# ============================================================
# AI Blog Writer — LangGraph DAG Definition
# This is the ENTIRE system wired together as a stateful graph.
#
# Flow:
#   START
#     → router          (decides: retrieval or generation?)
#       → retriever     (IF needs_retrieval=True)
#       → planner       (always runs after router/retriever)
#         → dispatcher  (fans out N parallel writers via Send)
#           → [section_writer_0, section_writer_1, ...N]  ← PARALLEL
#         → reducer     (merges all sections deterministically)
#           → evaluator (LLM-as-judge: score 1-10, pass if >= 7)
#   END
# ============================================================

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from schemas.models import AIBlogWriterState
from agents.router import router_node, route_decision
from agents.retriever import retriever_node
from agents.planner import planner_node
from agents.writer import dispatch_sections, section_writer_node
from agents.reducer import reducer_node
from agents.evaluator import evaluator_node

load_dotenv()


def build_graph() -> StateGraph:
    """
    Builds and compiles the AI Blog Writer LangGraph DAG.
    Returns a compiled, executable graph.
    """
    # ── Initialize the graph with shared state ──
    graph = StateGraph(AIBlogWriterState)

    # ── Add all nodes ──
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("planner", planner_node)
    graph.add_node("section_writer", section_writer_node)  # parallel workers
    graph.add_node("reducer", reducer_node)
    graph.add_node("evaluator", evaluator_node)

    # ── Wire the edges ──

    # 1. Start → Router (always)
    graph.add_edge(START, "router")

    # 2. Router → conditional branch
    graph.add_conditional_edges(
        "router",
        route_decision,  # returns "retriever" or "planner"
        {
            "retriever": "retriever",
            "planner": "planner",
        }
    )

    # 3. Retriever → Planner (retrieval always feeds into planning)
    graph.add_edge("retriever", "planner")

    # 4. Planner → Section Writer (via Send API — parallel fan-out)
    #    dispatch_sections returns List[Send("section_writer", state)]
    #    LangGraph handles parallel execution automatically
    graph.add_conditional_edges(
        "planner",
        dispatch_sections,
        ["section_writer"]
    )

    # 5. section_writer → Reducer (all parallel workers fan-in here)
    graph.add_edge("section_writer", "reducer")

    # 6. Reducer → Evaluator
    graph.add_edge("reducer", "evaluator")

    # 7. Evaluator → END
    graph.add_edge("evaluator", END)

    # ── Compile the graph ──
    compiled = graph.compile()

    return compiled


# Singleton — import this in main.py
ai_blog_writer_graph = build_graph()
