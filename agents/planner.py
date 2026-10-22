# agents/planner.py
# ============================================================
# AI Blog Writer — Structured Planner Node
# Creates a detailed BlogPlan: section goals, constraints,
# word budgets. This is what makes sections deterministic —
# workers don't improvise, they execute a structured spec.
# ============================================================

import os
from utils.llm import get_llm
from utils.json_utils import parse_llm_json
from utils.logger import log_stage, log_info, log_success
from schemas.models import BlogPlan, SectionPlan, AIBlogWriterState

WORD_BUDGET = int(os.getenv("SECTION_WORD_BUDGET", 300))

PLANNER_PROMPT = """You are a senior content strategist. Create a detailed blog plan.

Topic: {topic}
Target Audience: {audience}
Tone: {tone}
Total Word Budget: {total_budget} words
Per-Section Budget: {section_budget} words

Research Context (use if relevant, ignore if empty):
{research_context}

Create a blog plan with 4-6 sections. Each section must have:
- A clear title (heading)
- A specific goal (what must this section achieve?)
- 2-3 constraints (rules the writer must follow)
- Whether it needs real facts/data (requires_retrieval: true/false)

Return ONLY valid JSON in this exact format:
{{
  "topic": "{topic}",
  "target_audience": "{audience}",
  "tone": "{tone}",
  "total_word_budget": {total_budget},
  "sections": [
    {{
      "section_id": 0,
      "title": "Introduction: Hook the Reader",
      "goal": "Capture attention and explain why this topic matters",
      "constraints": ["Start with a surprising fact or question", "No more than 2 paragraphs", "End with a clear thesis"],
      "word_budget": {section_budget},
      "requires_retrieval": false
    }},
    ...more sections...
  ]
}}
"""


async def planner_node(state: AIBlogWriterState) -> dict:
    """
    Planner Node — creates the structured BlogPlan.
    This plan is the contract each parallel worker follows.
    """
    topic = state["topic"]
    audience = state.get("target_audience", "general readers")
    tone = state.get("tone", "informative and engaging")

    log_stage("PLANNER", f"Creating structured plan for '{topic}'")

    # Build research context from retrieval results
    research_context = ""
    retrieval = state.get("retrieval_results")
    if retrieval and retrieval.snippets:
        snippets = retrieval.snippets[:4]
        research_context = "\n".join([f"• {s[:200]}" for s in snippets])
        log_info(f"Using {len(snippets)} evidence snippets in plan")
    else:
        research_context = "No retrieval data — use general knowledge."

    total_budget = 1500
    num_sections = 5
    section_budget = total_budget // num_sections

    llm = get_llm(temperature=0.4)
    prompt = PLANNER_PROMPT.format(
        topic=topic,
        audience=audience,
        tone=tone,
        total_budget=total_budget,
        section_budget=section_budget,
        research_context=research_context
    )

    try:
        response = await llm.ainvoke(prompt)
        data = parse_llm_json(response.content)

        # Build validated BlogPlan
        sections = []
        for i, s in enumerate(data.get("sections", [])):
            section = SectionPlan(
                section_id=s.get("section_id", i),
                title=s.get("title", f"Section {i+1}"),
                goal=s.get("goal", "Cover this topic clearly"),
                constraints=s.get("constraints", []),
                word_budget=s.get("word_budget", WORD_BUDGET),
                requires_retrieval=s.get("requires_retrieval", False)
            )
            sections.append(section)

        blog_plan = BlogPlan(
            topic=data.get("topic", topic),
            target_audience=data.get("target_audience", audience),
            tone=data.get("tone", tone),
            sections=sections,
            total_word_budget=data.get("total_word_budget", total_budget)
        )

        log_success(f"Plan created: {len(sections)} sections, {blog_plan.total_word_budget} words total")
        for s in sections:
            log_info(f"  [{s.section_id}] {s.title} ({s.word_budget}w)")

        return {"blog_plan": blog_plan, "errors": []}

    except Exception as e:
        log_stage("PLANNER", f"JSON parse failed — using fallback plan: {e}", style="bold yellow")
        # Fallback plan if LLM returns bad JSON
        fallback_sections = [
            SectionPlan(section_id=0, title="Introduction", goal="Hook the reader", word_budget=WORD_BUDGET),
            SectionPlan(section_id=1, title="Background", goal="Provide context", word_budget=WORD_BUDGET),
            SectionPlan(section_id=2, title="Core Concepts", goal="Explain the main ideas", word_budget=WORD_BUDGET),
            SectionPlan(section_id=3, title="Practical Applications", goal="Show real-world use", word_budget=WORD_BUDGET),
            SectionPlan(section_id=4, title="Conclusion", goal="Summarise and call to action", word_budget=WORD_BUDGET),
        ]
        blog_plan = BlogPlan(topic=topic, target_audience=audience, tone=tone, sections=fallback_sections)
        return {"blog_plan": blog_plan, "errors": [f"Planner parse error: {str(e)}"]}
