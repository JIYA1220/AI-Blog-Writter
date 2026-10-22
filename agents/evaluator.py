# agents/evaluator.py
# ============================================================
# AI Blog Writer: Evaluator Node (LLM-as-Judge)
# Pointwise judge: scores the final blog 1-10 (pass if >= 7).
#
# Known limitation: only the first EVAL_MAX_CHARS characters of the blog are
# shown to the judge, so text after that point cannot affect the score.
# ============================================================

from utils.llm import get_llm
from utils.json_utils import parse_llm_json
from utils.logger import log_stage, log_success
from schemas.models import BlogEvaluation, AIBlogWriterState

EVAL_MAX_CHARS = 4000

EVALUATOR_PROMPT = """You are a senior editor. Evaluate the following blog post.

Topic: {topic}
Target Audience: {audience}
Requested Tone: {tone}

Blog Content:
{content}

Evaluate based on:
1. Relevance to topic and audience.
2. Adherence to the requested tone.
3. Logical flow and structure.
4. Depth of information and factual grounding (if applicable).

Respond ONLY with valid JSON:
{{
  "score": (int 1-10),
  "reasoning": "detailed explanation of the score",
  "suggestions": ["suggestion 1", "suggestion 2"],
  "is_pass": true/false (true if score >= 7)
}}
"""


def build_evaluator_prompt(topic: str, audience: str, tone: str, content: str,
                           max_chars: int = EVAL_MAX_CHARS) -> str:
    """Fill the judge prompt. Content is cut to max_chars (None means no cut)."""
    shown = content if max_chars is None else content[:max_chars]
    return EVALUATOR_PROMPT.format(topic=topic, audience=audience, tone=tone, content=shown)


def parse_evaluation(raw: str) -> BlogEvaluation:
    """Parse a judge reply into a BlogEvaluation. Raises on invalid JSON or schema."""
    data = parse_llm_json(raw)
    return BlogEvaluation(
        score=data.get("score", 0),
        reasoning=data.get("reasoning", "No reasoning provided"),
        suggestions=data.get("suggestions", []),
        is_pass=data.get("is_pass", False),
    )


async def evaluator_node(state: AIBlogWriterState) -> dict:
    """
    Evaluator Node (LLM-as-judge).
    Critiques the final blog and provides a score.
    """
    final_blog = state.get("final_blog")
    if not final_blog or not final_blog.full_content:
        return {"evaluation": None}

    log_stage("EVALUATOR", "Evaluating final blog quality...")

    llm = get_llm(temperature=0.1)
    prompt = build_evaluator_prompt(
        topic=state["topic"],
        audience=state.get("target_audience", "general readers"),
        tone=state.get("tone", "informative"),
        content=final_blog.full_content,
    )

    try:
        response = await llm.ainvoke(prompt)
        evaluation = parse_evaluation(response.content)

        if evaluation.is_pass:
            log_success(f"Evaluation PASSED (Score: {evaluation.score}/10)")
        else:
            log_stage("EVALUATOR", f"Evaluation FAILED (Score: {evaluation.score}/10)", style="bold yellow")

        return {"evaluation": evaluation}

    except Exception as e:
        log_stage("EVALUATOR", f"Evaluation failed: {e}", style="bold red")
        return {"evaluation": None, "errors": [f"Evaluator error: {str(e)}"]}
