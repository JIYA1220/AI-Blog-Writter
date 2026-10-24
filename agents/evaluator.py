# agents/evaluator.py
# ============================================================
# AI Blog Writer: Evaluator Node (LLM-as-Judge)
# Pointwise judge: scores the final blog 1-10 (pass if >= 7).
#
# The judge reads the whole post up to EVALUATOR_MAX_CHARS characters
# (default 20000, enough for posts of about 2,500 words). If a post is longer
# than the cap, it is cut from the end and a warning is logged, never silently.
# Lower the cap for a model whose context window cannot hold a full post.
#
# History: the first version cut every post to 4000 characters. The judge audit
# (evaluation/) measured that this changed the pass/fail verdict for most long
# posts, which is why the default was raised. LEGACY_EVAL_MAX_CHARS keeps the
# old value so the audit can still reproduce the "before" condition.
# ============================================================

import logging
import os

from utils.llm import get_llm
from utils.json_utils import parse_llm_json
from utils.logger import log_stage, log_success, log_warning
from schemas.models import BlogEvaluation, AIBlogWriterState

logger = logging.getLogger("ai_blog_writer.evaluator")

DEFAULT_EVALUATOR_MAX_CHARS = 20000
LEGACY_EVAL_MAX_CHARS = 4000        # the old hard-coded cap; used by evaluation/ for the "before" view

_USE_CONFIG = object()

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


def evaluator_max_chars() -> int:
    """Character cap for the text shown to the judge, from EVALUATOR_MAX_CHARS (read at call time)."""
    raw = os.getenv("EVALUATOR_MAX_CHARS", "").strip()
    if not raw:
        return DEFAULT_EVALUATOR_MAX_CHARS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("EVALUATOR_MAX_CHARS=%r is not a positive integer; using the default %d",
                       raw, DEFAULT_EVALUATOR_MAX_CHARS)
        return DEFAULT_EVALUATOR_MAX_CHARS
    return value


def build_evaluator_prompt(topic: str, audience: str, tone: str, content: str,
                           max_chars=_USE_CONFIG, warn: bool = True) -> str:
    """
    Fill the judge prompt. By default the content is cut to EVALUATOR_MAX_CHARS and a warning is logged
    when that removes text. Pass max_chars=None to show everything, or an int to set the cap explicitly.
    """
    cap = evaluator_max_chars() if max_chars is _USE_CONFIG else max_chars
    shown = content
    if cap is not None and len(content) > cap:
        shown = content[:cap]
        if warn:
            message = (f"Post is {len(content)} characters, over the evaluator cap of {cap} "
                       f"(EVALUATOR_MAX_CHARS); the last {len(content) - cap} characters are not shown to the judge.")
            logger.warning(message)
            log_warning(message)
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
