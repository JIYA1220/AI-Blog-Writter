# AI Blog Writer (Agentic Blog Generation System)

[![tests](https://github.com/JIYA1220/Ai-Blog-Writter/actions/workflows/tests.yml/badge.svg)](https://github.com/JIYA1220/Ai-Blog-Writter/actions/workflows/tests.yml)

<div align="center">

<!-- BADGES / STACK (TOP) -->
<img src="https://img.shields.io/badge/LangGraph-Async%20DAG-7C3AED?style=for-the-badge" />
<img src="https://img.shields.io/badge/LangChain-LLM%20Framework-0EA5E9?style=for-the-badge" />
<img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge" />
<img src="https://img.shields.io/badge/FastAPI-API-059669?style=for-the-badge" />
<img src="https://img.shields.io/badge/Tavily-Web%20Search-111827?style=for-the-badge" />
<img src="https://img.shields.io/badge/Ollama-Local%20LLMs-1F2937?style=for-the-badge" />
<img src="https://img.shields.io/badge/OpenRouter-Cloud%20LLMs-F59E0B?style=for-the-badge" />
<img src="https://img.shields.io/badge/Groq-Cloud%20LLMs-F55036?style=for-the-badge" />
<img src="https://img.shields.io/badge/Gemini-Google%20GenAI-2563EB?style=for-the-badge" />
<img src="https://img.shields.io/badge/Pydantic-Types%20%26%20Validation-16A34A?style=for-the-badge" />
<img src="https://img.shields.io/badge/Rich-CLI%20UI-22C55E?style=for-the-badge" />
<img src="https://img.shields.io/badge/Uvicorn-ASGI%20Server-0EA5E9?style=for-the-badge" />
<img src="https://img.shields.io/badge/pytest-Testing-14B8A6?style=for-the-badge" />

<br/><br/>

<b>An agentic blog writing system</b> built as a <b>stateful, async LangGraph DAG</b> with <b>parallel section writing</b>, optional <b>web grounding</b>, and an <b>LLM-as-a-judge evaluator</b>.

<br/>

</div>

Give it a topic, an audience and a tone. It plans the post, writes the sections in parallel, merges them into one markdown file, and has an LLM editor score the result. Run it from the **CLI**, a **Streamlit UI**, or a **FastAPI server**. The repo also contains a small audit of its own LLM judge (see [Judge audit](#judge-audit)).

## Quick start

```bash
git clone https://github.com/JIYA1220/Ai-Blog-Writter.git
cd Ai-Blog-Writter
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # default model: local Ollama (ollama pull qwen2.5:7b)
streamlit run app.py
```

Run the tests (no network and no API keys needed): `pytest -q`

---

## Demo video

**Watch or download:** [demo.mp4](assets/demo.mp4)

---

## What it does

1. **Router** decides whether the topic needs web retrieval. With no Tavily key it always chooses generation.
2. **Retriever** searches the web through Tavily when the router asks for it, and removes duplicate snippets.
3. **Planner** creates a structured multi-section plan (titles, goals, constraints, word budgets).
4. **Writers** write the sections in parallel, one worker per section.
5. **Reducer** merges the sections in a fixed order into one markdown post and saves it under `output/`.
6. **Evaluator** scores the post from 1 to 10 with an LLM judge; 7 or more counts as a pass.

---

## Architecture (LangGraph DAG)

```text
START
  └─► ROUTER (Async)    : decide retrieval or generation
        ├─► RETRIEVER   : Tavily search + deduplication (optional)
        └─► PLANNER     : structured BlogPlan (section goals)
              └─► DISPATCHER : parallel fan-out (Send API)
                    ├─► WRITER [Section 0] ┐
                    ├─► WRITER [Section 1] ├─ parallel async
                    └─► WRITER [Section N] ┘
                          └─► REDUCER     : merge in order, save markdown
                                └─► EVALUATOR : LLM judge (score + feedback)
                                      └─► END
```

Stack: LangGraph and LangChain for orchestration, Pydantic v2 for typed state and outputs, Streamlit and FastAPI with Uvicorn for the interfaces, Rich for CLI logging, LangSmith for optional tracing, and pytest with httpx for tests.

---

## Configuration

Copy `.env.example` to `.env` and fill in only what you use. `.env` is git-ignored; never commit it.

### Choosing a model provider

`MODEL_NAME` selects the provider (default: `ollama/qwen2.5:7b`):

| `MODEL_NAME` | Provider | Needs |
|---|---|---|
| `ollama/<model>` | local Ollama | Ollama running, no key |
| `openrouter/<model>` | OpenRouter (OpenAI-compatible, `https://openrouter.ai/api/v1`) | `OPENROUTER_API_KEY` |
| `groq/<model>` | Groq (OpenAI-compatible, `https://api.groq.com/openai/v1`) | `GROQ_API_KEY` |
| any other name | Google Gemini | `GOOGLE_API_KEY` |

A bare `vendor/model` name is also sent to OpenRouter when only `OPENROUTER_API_KEY` is set, but the explicit `openrouter/` prefix is preferred. A missing key raises an error that names the variable. For Gemini, use a model name that is currently available in Google's docs; no Gemini default is set.

### Other variables

| Variable | Meaning | Default |
|---|---|---|
| `TAVILY_API_KEY` | enables web retrieval; without it the router always chooses generation | unset |
| `ENABLE_SEARCH` | set to `false` to disable retrieval even with a key | `true` |
| `SECTION_WORD_BUDGET` | words per section | `300` |
| `EVALUATOR_MAX_CHARS` | characters of a post the judge reads; a longer post is cut from the end and a warning is logged | `20000` (about 2,500 words) |
| `MAX_CONCURRENCY` | parallel section writers in the CLI | `3` |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | enable LangSmith tracing | unset |

Lower `EVALUATOR_MAX_CHARS` for a model whose context window cannot hold a full post.

---

## How to run

### CLI
```bash
python main.py
```

### Streamlit UI (recommended)
```bash
streamlit run app.py
```
It has a form for topic, audience and tone, a sidebar to browse earlier posts in `output/`, and a download button.

### FastAPI server
```bash
python server.py        # http://localhost:8000
```

```bash
curl http://localhost:8000/

curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The Rise of Agentic AI Systems in 2025",
    "target_audience": "tech-curious professionals",
    "tone": "informative and engaging"
  }'
```

The response has the post `title` and markdown `content`, the evaluation `score` and `evaluation` text (both `null` if the judge reply could not be parsed), and `word_count`.

---

## Testing

```bash
pytest -q
```

The offline suite has 120 tests and needs no network and no API keys. It uses a scripted fake LLM to cover: Pydantic schemas, both router branches, the dispatcher (one `Send` per section), deterministic reducer ordering, JSON parsing (plain, fenced, invalid), provider selection in `get_llm`, the full graph with and without retrieval, the evaluator reading a full-length post, the FastAPI `/generate` endpoint, and the statistics used by the judge audit. GitHub Actions runs it on every push.

---

## Judge audit

The LLM judge was audited on 60 generated posts (30 topics, each written as a short and a long version) with a 120B judge. The audit measures consistency and robustness. **Agreement with a human reader was not measured, so it does not show that the judge is accurate.**

| Finding | Result |
|---|---|
| The old evaluator showed the judge only the first 4,000 characters. On the 56 posts longer than that, the pass rate was | 25.0% (14/56) on the truncated view and 80.4% (45/56) on the full text; the pass/fail verdict changed on 33 of 56 posts |
| Fix | the evaluator now reads the whole post up to `EVALUATOR_MAX_CHARS` and logs a warning if it must cut one |
| Pairwise comparison (short post against long post, both orders) | the same post won in both orders on 76.7% (23/30) of topics |
| Position effect | in 6 of the 7 flipped topics the first-shown post won both orders, but an exact binomial test gives p = 0.125, so it is not significant |
| Repeatability | mean score difference between runs of 0.20 points |

The before/after of the fix compares the same posts and the same judge using replies that already existed; the live pipeline was not re-run. The long posts are about twice the length of the short ones, so length and variant are confounded in the pairwise results. A local 3B judge was also tried and gave almost every post an 8, so it was not used for any conclusion.

Read more: [`evaluation/STUDY_SUMMARY.md`](evaluation/STUDY_SUMMARY.md) (summary, including how the Groq daily token limit was handled) and [`evaluation/REPORT.md`](evaluation/REPORT.md) (tables, exact prompts and the commands that were run).

---

## Project structure

```text
.
├── main.py              # CLI entry point (async)
├── server.py            # FastAPI server
├── app.py               # Streamlit UI
├── graph.py             # LangGraph DAG definition
├── list_models.py       # lists Gemini models for your key (needs the extra package google-generativeai)
├── requirements.txt     # Python dependencies (not version-pinned)
├── .env.example         # example environment variables
│
├── agents/              # graph nodes: router, retriever, planner, writer, reducer, evaluator
├── schemas/             # Pydantic models (state + structured outputs)
├── utils/               # LLM client factory, JSON parsing, logging helpers
├── tests/               # offline test suite (no network, no API keys)
├── .github/workflows/   # CI: runs pytest
├── evaluation/          # judge audit: scripts, posts, results, raw call logs, report
└── output/              # generated blogs saved as .md files
```

---

## Limitations

- The judge is a single-pass, pointwise LLM score. It reads up to `EVALUATOR_MAX_CHARS` characters (default 20000); a longer post is cut from the end with a logged warning. Before the fix it saw only the first 4,000 characters.
- By default the judge uses the same `MODEL_NAME` as the writers, so it can favour its own style. Its agreement with human judgement was not measured.
- The `is_pass` flag comes from the model's own reply, not from the score.
- Retrieval only helps if a Tavily key is set. The grounding is snippets pasted into the writer prompt, with no claim-level verification.
- Section writers run in parallel and do not see each other's text, so sections can overlap or repeat.
- `requirements.txt` is not version-pinned.

---

## Troubleshooting

**".env file not found"**: create one with `cp .env.example .env`.

**Generation fails or the output is empty.** Common causes:
- the API key for your provider is missing (`GOOGLE_API_KEY`, `OPENROUTER_API_KEY` or `GROQ_API_KEY`)
- `MODEL_NAME` is wrong or has no provider prefix
- the provider is not reachable (network issue)
- Ollama is not running (if you use an `ollama/...` model)

**Ollama (local).** Install Ollama, make sure it is running, then pull your model:
```bash
ollama pull qwen2.5:7b
```

---

## Author

Built by **@JIYA1220**.
