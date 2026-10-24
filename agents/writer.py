# agents/writer.py
# ============================================================
# AI Blog Writer — Parallel Section Writers
# Uses LangGraph's Send API to fan out one worker per section.
# Workers run SIMULTANEOUSLY — not one after another.
# Each worker writes exactly one section based on the plan.
# ============================================================

from typing import List
from langgraph.types import Send
from utils.llm import get_llm
from utils.json_utils import message_text
from utils.logger import log_stage, log_info, log_success, log_warning
from schemas.models import SectionPlan, WrittenSection, AIBlogWriterState


# ─────────────────────────────────────────────────────────────
# Section worker state — minimal state passed to each worker
# ─────────────────────────────────────────────────────────────
from typing import TypedDict
from typing import Optional


class SectionWorkerState(TypedDict):
    section: SectionPlan
    topic: str
    tone: str
    research_context: str


# ─────────────────────────────────────────────────────────────
# Writer prompt — structured and constrained
# ─────────────────────────────────────────────────────────────
WRITER_PROMPT = """You are an expert blog writer. Write ONE section of a blog post.

Blog Topic: {topic}
Tone: {tone}

Section to Write:
  Title: {title}
  Goal: {goal}
  Constraints:
{constraints}
  Word Budget: {word_budget} words (stay within ±50 words)

Research Evidence (use if relevant, cite naturally, don't make up stats):
{research_context}

Write ONLY the section content. Start directly with the section text.
Do NOT include the section title in your output — it will be added separately.
Do NOT write any other sections.
Stay within the word budget.
"""


# ─────────────────────────────────────────────────────────────
# Section dispatcher — creates Send() for each section
# ─────────────────────────────────────────────────────────────
def dispatch_sections(state: AIBlogWriterState) -> List[Send]:
    """
    This function fans out one Send() per section.
    LangGraph executes all workers in PARALLEL (same superstep).
    """
    blog_plan = state.get("blog_plan")
    if not blog_plan or not blog_plan.sections:
        log_warning("No blog plan found — skipping parallel writing")
        return []

    # Build shared research context
    research_context = ""
    retrieval = state.get("retrieval_results")
    if retrieval and retrieval.snippets:
        research_context = "\n".join([f"• {s[:250]}" for s in retrieval.snippets[:5]])
    else:
        research_context = "No specific research — use well-established knowledge."

    log_stage("DISPATCHER", f"Fanning out {len(blog_plan.sections)} parallel writers")

    sends = []
    for section in blog_plan.sections:
        worker_state = SectionWorkerState(
            section=section,
            topic=blog_plan.topic,
            tone=blog_plan.tone,
            research_context=research_context
        )
        sends.append(Send("section_writer", worker_state))

    return sends


# ─────────────────────────────────────────────────────────────
# Section writer node — one instance per section
# ─────────────────────────────────────────────────────────────
async def section_writer_node(state: SectionWorkerState) -> dict:
    """
    Individual section writer. One of these runs per section, in parallel.
    Writes to 'written_sections' which uses operator.add as reducer.
    """
    section: SectionPlan = state["section"]
    topic = state["topic"]
    tone = state["tone"]
    research_context = state["research_context"]

    log_stage("WRITER", f"Writing section [{section.section_id}]: {section.title}", style="bold magenta")

    constraints_text = "\n".join([f"  - {c}" for c in section.constraints]) if section.constraints else "  - Write clearly and concisely"

    llm = get_llm(temperature=0.7)
    prompt = WRITER_PROMPT.format(
        topic=topic,
        tone=tone,
        title=section.title,
        goal=section.goal,
        constraints=constraints_text,
        word_budget=section.word_budget,
        research_context=research_context
    )

    try:
        response = await llm.ainvoke(prompt)
        content = message_text(response.content).strip()
        word_count = len(content.split())

        written = WrittenSection(
            section_id=section.section_id,
            title=section.title,
            content=content,
            word_count=word_count,
            is_valid=True
        )

        log_success(f"Section [{section.section_id}] done — {word_count} words")
        return {"written_sections": [written]}

    except Exception as e:
        log_warning(f"Section [{section.section_id}] failed: {e}")
        fallback = WrittenSection(
            section_id=section.section_id,
            title=section.title,
            content=f"[Section generation failed: {str(e)}]",
            word_count=0,
            is_valid=False
        )
        return {"written_sections": [fallback], "errors": [f"Writer error section {section.section_id}: {str(e)}"]}
