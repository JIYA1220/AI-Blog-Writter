# AI Blog Writer (Agentic Blog Generation System)

<div align="center">

<!-- BADGES / STACK (TOP) -->
<img src="https://img.shields.io/badge/LangGraph-Async%20DAG-7C3AED?style=for-the-badge" />
<img src="https://img.shields.io/badge/LangChain-LLM%20Framework-0EA5E9?style=for-the-badge" />
<img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge" />
<img src="https://img.shields.io/badge/FastAPI-API-059669?style=for-the-badge" />
<img src="https://img.shields.io/badge/Tavily-Web%20Search-111827?style=for-the-badge" />
<img src="https://img.shields.io/badge/Ollama-Local%20LLMs-1F2937?style=for-the-badge" />
<img src="https://img.shields.io/badge/OpenRouter-Cloud%20LLMs-F59E0B?style=for-the-badge" />
<img src="https://img.shields.io/badge/Gemini-Google%20GenAI-2563EB?style=for-the-badge" />
<img src="https://img.shields.io/badge/Pydantic-Types%20%26%20Validation-16A34A?style=for-the-badge" />
<img src="https://img.shields.io/badge/Rich-CLI%20UI-22C55E?style=for-the-badge" />
<img src="https://img.shields.io/badge/Uvicorn-ASGI%20Server-0EA5E9?style=for-the-badge" />
<img src="https://img.shields.io/badge/pytest-Testing-14B8A6?style=for-the-badge" />

<br/><br/>

<b>An agentic blog writing system</b> built as a <b>stateful, async LangGraph DAG</b> with <b>parallel section writing</b>, optional <b>web grounding</b>, and an <b>LLM-as-a-judge evaluator</b>.

<br/>

</div>

---  
Run it via **CLI**, **Streamlit UI**, or a **FastAPI server**.

---

##  Tools / Tech Stack (Top)

**Core Orchestration**
- **LangGraph** — stateful DAG orchestration (async) + parallel fan-out
- **LangChain** — LLM tooling & integrations

**LLM Providers (choose one)**
- **Ollama (local)** — run models locally (no cloud key needed)
- **OpenRouter (cloud)** — access multiple hosted models
- **Google Gemini (cloud)** — Google GenAI models

**Grounding / Search**
- **Tavily** — web search for retrieval (used only when `TAVILY_API_KEY` is set)

**UI / API**
- **Streamlit** — interactive app (generate + browse past blogs)
- **FastAPI** + **Uvicorn** — REST API for production use

**Observability**
- **LangSmith** — tracing, enabled when `LANGSMITH_API_KEY` is set

**Developer Experience**
- **Pydantic v2** — typed schemas + validation
- **Rich** — clean CLI UI/logging
- **pytest** + **httpx** — offline test suite, run in GitHub Actions

---

##  Demo Video

 **Watch / Download:** [demo.mp4](assets/demo.mp4)


---

##  What it does

Given:
- a **topic**
- a **target audience**
- a **tone**

…this system:
1. decides whether the topic needs retrieval (**router**; always chooses generation when no Tavily key is set)
2. retrieves web evidence via Tavily when the router asks for it (**retriever**)
3. creates a structured multi-section plan (**planner**)
4. writes sections **in parallel** (**writer fan-out**)
5. merges sections into a final markdown blog (**reducer**)
6. scores the result 1 to 10 with an LLM judge, pass if the score is 7 or higher (**evaluator**)

Generated blogs are saved as `.md` files under `output/`.

---

##  Architecture (LangGraph DAG)

```text
START
  └─► ROUTER (Async)    ── Decide: Retrieval vs Generation
        ├─► RETRIEVER   ── Tavily Search + Deduplication (optional)
        └─► PLANNER     ── Structured BlogPlan (Section Goals)
              └─► DISPATCHER ── Parallel Fan-Out (Send API)
                    ├─► WRITER [Section 0] ┐
                    ├─► WRITER [Section 1] ├─ Parallel Async
                    └─► WRITER [Section N] ┘
                          └─► REDUCER     ── Merge & Save Markdown
                                └─► EVALUATOR ── LLM-as-Judge (Score + Feedback)
                                      └─► END
```

---

##  Installation

### 1) Clone the repo
```bash
git clone https://github.com/JIYA1220/Ai-Blog-Writter.git
cd Ai-Blog-Writter
```

### 2) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

##  Environment Variables

Copy `.env.example` → `.env`:

```bash
cp .env.example .env
```

### Choosing a model provider
`MODEL_NAME` selects the provider (default: `ollama/qwen2.5:7b`):

| `MODEL_NAME` | Provider | Needs |
|---|---|---|
| `ollama/<model>` | local Ollama | Ollama running, no key |
| `openrouter/<model>` | OpenRouter (OpenAI-compatible client, `https://openrouter.ai/api/v1`) | `OPENROUTER_API_KEY` |
| any other name | Google Gemini | `GOOGLE_API_KEY` |

A bare `vendor/model` name is also sent to OpenRouter when only `OPENROUTER_API_KEY` is set. Use the explicit `openrouter/` prefix. A missing key raises a clear error that names the variable. Use a Gemini model name that is currently available in Google's docs; no Gemini default is set.

### Other variables
- `TAVILY_API_KEY`: enables web retrieval. Without it the router always chooses generation.
- `ENABLE_SEARCH`: set to `false` to disable retrieval even with a key.
- `SECTION_WORD_BUDGET`: words per section (default 300).
- `MAX_CONCURRENCY`: parallel section writers in the CLI (default 3).
- `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`: enable LangSmith tracing.

Example `.env`:
```env
MODEL_NAME=ollama/qwen2.5:7b
# MODEL_NAME=openrouter/meta-llama/llama-3.1-8b-instruct:free
# OPENROUTER_API_KEY=...
# TAVILY_API_KEY=...

SECTION_WORD_BUDGET=300
MAX_CONCURRENCY=3
```

---

##  How to Run

You can run the system in **three modes**:

### 1) CLI Mode
```bash
python main.py
```

### 2) Streamlit UI (recommended)
```bash
streamlit run app.py
```

What you get:
- a blog generation form (topic / audience / tone)
- a sidebar to browse previously generated markdown files in `output/`
- download button for generated blogs

### 3) FastAPI Server (REST API)
```bash
python server.py
```

Server runs at:
- `http://localhost:8000`

---

##  API Usage (FastAPI)

### Health check
```bash
curl http://localhost:8000/
```

### Generate a blog
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "The Rise of Agentic AI Systems in 2025",
    "target_audience": "tech-curious professionals",
    "tone": "informative and engaging"
  }'
```

Response includes:
- blog `title` + full markdown `content`
- evaluation `score` and `evaluation` text (both `null` if the judge reply could not be parsed)
- `word_count`

---

##  Project Structure

```text
.
├── main.py              # CLI entry point (async)
├── server.py            # FastAPI server
├── app.py               # Streamlit UI
├── graph.py             # LangGraph DAG definition
├── list_models.py       # Lists Gemini models for your key (needs the extra package google-generativeai)
├── requirements.txt     # Python dependencies
├── .env.example         # Example env vars
│
├── pytest.ini
├── tests/               # Offline test suite (no network, no API keys)
├── .github/workflows/   # CI: runs pytest
│
├── agents/              # Graph nodes (router/retriever/planner/writer/reducer/evaluator)
├── schemas/             # Pydantic models (state + structured outputs)
├── utils/               # LLM client config, logging helpers, etc.
└── output/              # Generated blogs saved as .md files
```

---

##  Testing

```bash
pytest -q
```

The suite runs offline with a scripted fake LLM: Pydantic schemas, the router branches, the dispatcher (one `Send` per section), deterministic reducer ordering, JSON parsing (plain, fenced, invalid), provider selection in `get_llm`, the full graph with and without retrieval, and the FastAPI `/generate` endpoint. CI runs it on every push.

##  Limitations

- The judge is a single-pass, pointwise LLM score. It only sees the first 4000 characters of the blog (`EVAL_MAX_CHARS` in `agents/evaluator.py`), so later sections cannot change the score.
- The judge uses the same `MODEL_NAME` as the writers, so it can favour its own style. Its agreement with human judgement has not been measured yet.
- The `is_pass` flag comes from the model's own reply, not from the score.
- Retrieval only helps if a Tavily key is set; the grounding is snippets pasted into the writer prompt, with no claim-level verification.
- Section writers run in parallel and do not see each other's text, so sections can overlap or repeat.
- `requirements.txt` is not version-pinned.

---

##  Troubleshooting

### “.env file not found!”
Create one:
```bash
cp .env.example .env
```

### Generation fails / empty output
Common causes:
- missing API key (Gemini/OpenRouter)
- wrong `MODEL_NAME`
- provider not reachable (network issue)
- Ollama not running locally (if using `ollama/...` model)

### Ollama tips (local)
Install Ollama and ensure it’s running, then pull your model:
```bash
ollama pull qwen2.5:7b
```

## 👤 Author

Built by **@JIYA1220**.
