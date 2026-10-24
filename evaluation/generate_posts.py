"""
Generate the 60 posts (30 topics x 2 variants) with the real pipeline.

Variants (no Tavily key, so retrieval cannot be the contrast):
  A: SECTION_WORD_BUDGET=150  (short posts)
  B: SECTION_WORD_BUDGET=400  (long posts)
Retrieval is disabled for both (ENABLE_SEARCH=false).

Every LLM call is logged verbatim to evaluation/runs/<run>/calls.jsonl.
Existing posts are skipped, so the script can be stopped and resumed.

Usage:
  python -m evaluation.generate_posts --model ollama/qwen2.5:3b [--limit 2] [--topics t01,t02]
"""

import argparse
import asyncio
import datetime
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _stream in (sys.stdout, sys.stderr):                   # the Rich logger prints non-ASCII symbols
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from evaluation.llm_log import LoggedLLM, make_run_dir  # noqa: E402
from utils.llm import get_llm                            # noqa: E402

EVAL_DIR = os.path.join(ROOT, "evaluation")
POSTS_DIR = os.path.join(EVAL_DIR, "posts")
VARIANTS = {"A": 150, "B": 400}


def load_topics():
    with open(os.path.join(EVAL_DIR, "topics.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["topics"], cfg["audience"], cfg["tone"]


def install_logged_llm(model: str, log_path: str):
    """Make every pipeline node use a logging wrapper around get_llm(model_name=model)."""
    def factory(temperature=0.7):
        return LoggedLLM(get_llm(temperature=temperature, model_name=model), log_path,
                         model=model, temperature=temperature, tag="generate")
    import agents.evaluator, agents.planner, agents.router, agents.writer  # noqa: E401
    for module in (agents.router, agents.planner, agents.writer, agents.evaluator):
        module.get_llm = factory
    import agents.reducer
    agents.reducer._save_blog = lambda blog, topic: None   # keep output/ clean

    async def no_evaluator(state):                         # the study judges posts separately
        return {"evaluation": None}
    agents.evaluator.evaluator_node = no_evaluator          # must happen before `import graph`


async def generate_one(graph, topic: dict, audience: str, tone: str, variant: str, model: str):
    os.environ["SECTION_WORD_BUDGET"] = str(VARIANTS[variant])
    os.environ["ENABLE_SEARCH"] = "false"
    inputs = {"topic": topic["topic"], "target_audience": audience, "tone": tone,
              "needs_retrieval": False, "retrieval_results": None, "blog_plan": None,
              "written_sections": [], "final_blog": None, "evaluation": None, "errors": []}
    state = await graph.ainvoke(inputs, config={"recursion_limit": 50, "max_concurrency": 2})
    blog = state.get("final_blog")
    meta = {
        "post_id": f"{topic['id']}_{variant}", "topic_id": topic["id"], "variant": variant,
        "topic": topic["topic"], "audience": audience, "tone": tone,
        "section_word_budget": VARIANTS[variant], "retrieval": False,
        "model": model, "date": datetime.date.today().isoformat(),
        "words": blog.total_words if blog else 0,
        "chars": len(blog.full_content) if blog else 0,
        "sections": blog.sections_count if blog else 0,
        "production_ready": bool(blog and blog.is_production_ready),
        "pipeline_errors": state.get("errors", []),
    }
    return (blog.full_content if blog else ""), meta


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama/qwen2.5:3b")
    ap.add_argument("--limit", type=int, default=None, help="only the first N topics")
    ap.add_argument("--topics", default=None, help="comma-separated topic ids")
    args = ap.parse_args()

    topics, audience, tone = load_topics()
    if args.topics:
        wanted = set(args.topics.split(","))
        topics = [t for t in topics if t["id"] in wanted]
    if args.limit:
        topics = topics[:args.limit]

    todo = [(t, v) for t in topics for v in VARIANTS
            if not os.path.exists(os.path.join(POSTS_DIR, f"{t['id']}_{v}.md"))]
    calls_per_post = 1 + 5      # planner + about 5 writers (router and evaluator are skipped)
    print(f"{len(todo)} posts to generate, about {len(todo) * calls_per_post} LLM calls, model {args.model}")

    run_dir = make_run_dir(os.path.join(EVAL_DIR, "runs"), "generate")
    log_path = os.path.join(run_dir, "calls.jsonl")
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({"model": args.model, "variants": VARIANTS, "retrieval": False,
                   "temperatures": {"planner": 0.4, "writer": 0.7},
                   "date": datetime.date.today().isoformat(), "n_posts_planned": len(todo)}, f, indent=2)
    install_logged_llm(args.model, log_path)
    from graph import ai_blog_writer_graph

    os.makedirs(POSTS_DIR, exist_ok=True)
    for i, (topic, variant) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {topic['id']} {variant}: {topic['topic']}", flush=True)
        content, meta = await generate_one(ai_blog_writer_graph, topic, audience, tone, variant, args.model)
        if meta["sections"] == 0:
            print("   FAILED: no sections produced, not saved", flush=True)
            continue
        base = os.path.join(POSTS_DIR, meta["post_id"])
        with open(base + ".md", "w", encoding="utf-8") as f:
            f.write(content)
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({**meta, "run_dir": os.path.relpath(run_dir, ROOT)}, f, indent=2)
        print(f"   saved: {meta['words']} words, {meta['sections']} sections, errors={len(meta['pipeline_errors'])}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
