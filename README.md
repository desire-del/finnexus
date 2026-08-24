<div align="center">

# FinNexus

**Evidence-grounded financial intelligence over SEC filings.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/vector%20store-pgvector-336791)](https://github.com/pgvector/pgvector)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io/)
[![status](https://img.shields.io/badge/status-active%20development-f59e0b)](https://github.com/desire-del/finnexus)

[Repository](https://github.com/desire-del/finnexus) · [Quick start](#quick-start) · [Configuration](#configuration)

</div>

FinNexus is a local Retrieval-Augmented Generation application for ingesting SEC filings and asking financial questions against their contents. It combines dense and lexical retrieval, produces structured answers with citations, and exposes latency, throughput, token usage, and traces.

> [!IMPORTANT]
> FinNexus is under active development. It is a research and engineering project, not financial advice or a production-ready financial service.

## What works today

- Discover and download the latest SEC filing for a ticker or CIK with EdgarTools.
- Extract sections, normalize text, create token-aware chunks, and persist their provenance.
- Switch embedding backends between Hugging Face, OpenAI, and Ollama through `.env`.
- Isolate processing versions by embedding provider, model, dimension, and processing fingerprint.
- Retry incomplete or failed processing versions without mixing chunks from separate attempts.
- Retrieve evidence with pgvector, PostgreSQL full-text search, and Reciprocal Rank Fusion.
- Generate structured, cited answers with configurable OpenAI, Hugging Face, or Ollama chat models.
- Inspect answers, source excerpts, SEC links, latency, throughput, and token usage in Streamlit.
- Export OpenTelemetry/OpenInference traces to Phoenix.
- Run ingestion from either the Streamlit interface or a dedicated CLI.

FinNexus currently handles one company and one filing type per ingestion request, targeting the latest matching filing. Queries are stateless and filter active chunks by ticker and filing type.

There is **no FastAPI, REST, or other network API layer yet**. The supported interfaces are Streamlit, the ingestion CLI, and the local example in `main.py`.

## Technology

| Area | Current implementation |
|---|---|
| Runtime | Python 3.12, `uv` |
| Database | PostgreSQL 16, pgvector, SQLAlchemy async, asyncpg |
| SEC source | EdgarTools |
| RAG | LangChain, langchain-postgres, tiktoken |
| Embeddings | Hugging Face, OpenAI, Ollama |
| Chat models | OpenAI, Hugging Face, Ollama |
| UI | Streamlit |
| Observability | Phoenix, OpenTelemetry, OpenInference |
| Logging | structlog |
| Local services | Docker Compose |

## Quick start

### Requirements

- Git with SSH access to GitHub
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Docker with Compose

### 1. Clone and install

```bash
git clone git@github.com:desire-del/finnexus.git
cd finnexus
uv sync --dev
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Set a valid SEC identity before ingestion:

```env
EDGAR_IDENTITY="Your Name your.email@example.com"
```

Then configure the credentials required by the selected embedding and LLM providers. The default profile uses local Hugging Face embeddings and an OpenAI chat model, so `LLM_API_KEY` must be set for the default configuration.

### 3. Start PostgreSQL and Phoenix

```bash
docker compose up -d postgres phoenix
docker compose ps
```

| Service | Local address |
|---|---|
| PostgreSQL | `localhost:5433` |
| Phoenix UI | [http://localhost:6006](http://localhost:6006) |
| Phoenix OTLP/HTTP | `http://localhost:6006/v1/traces` |
| Phoenix OTLP/gRPC | `localhost:4317` |

### 4. Ingest a filing

The default form type is `10-K`:

```bash
uv run rag-sec-ingest AAPL
```

Useful variants:

```bash
# Select another form
uv run rag-sec-ingest AAPL --form-type 10-Q

# Use a SEC CIK
uv run rag-sec-ingest 320193 --form-type 10-K

# Customize chunking
uv run rag-sec-ingest AAPL --chunk-size 1000 --chunk-overlap 150

# Emit a machine-readable result
uv run rag-sec-ingest AAPL --json
```

Run `uv run rag-sec-ingest --help` for the complete CLI reference.

### 5. Launch the application

```bash
uv run streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). The application provides:

- **Assistant** — question answering, cited evidence, SEC passage links, token usage, and a latency breakdown;
- **Filings** — ingestion and an inventory of filings compatible with the active embedding profile.

On a remote or headless machine, prevent Streamlit from trying to open a browser:

```bash
uv run streamlit run streamlit_app.py --server.headless true
```

Use an SSH tunnel when the application runs on another host:

```bash
ssh -L 8501:localhost:8501 user@server
```

## Configuration

All runtime configuration is loaded from `.env` with `pydantic-settings`.

### Embeddings

Change `EMBEDDING_PROVIDER` to switch profiles. Each provider has its own model and expected dimension:

```env
# huggingface | openai | ollama
EMBEDDING_PROVIDER=huggingface

EMBEDDING_HUGGINGFACE_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_HUGGINGFACE_DIMENSION=384

EMBEDDING_OPENAI_MODEL_NAME=text-embedding-3-small
EMBEDDING_OPENAI_DIMENSION=1536

EMBEDDING_OLLAMA_MODEL_NAME=nomic-embed-text
EMBEDDING_OLLAMA_DIMENSION=768
```

> [!NOTE]
> Processing versions are tied to their embedding provider, model, and dimension. After changing profiles, ingest the filing again. The retriever intentionally excludes chunks created with an incompatible profile.

For OpenAI-compatible embedding endpoints, set `EMBEDDING_API_KEY` and optionally `EMBEDDING_BASE_URL`. For Ollama, `EMBEDDING_BASE_URL` can override the local endpoint.

### Chat model

```env
# openai | huggingface | ollama
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
LLM_API_KEY=your_api_key
LLM_BASE_URL=
LLM_TEMPERATURE=0.0
```

The exact model name, credentials, and optional base URL must be valid for the selected LangChain provider integration.

### Observability

Phoenix tracing is enabled by default:

```env
# phoenix | none
OBSERVABILITY_PROVIDER=phoenix
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_PROJECT_NAME=finexus
OBSERVABILITY_CAPTURE_CONTENT=false
```

`OBSERVABILITY_CAPTURE_CONTENT=false` redacts query and response content while retaining operational metadata. Set `OBSERVABILITY_PROVIDER=none` to disable tracing.

## Repository layout

```text
.
├── scripts/
│   ├── prepare_financebench_corpus.py  # Resolve and ingest evaluation corpus
│   └── evaluate_financebench.py        # Run the FinanceBench baseline
├── src/rag_sec/
│   ├── application/     # Runtime and query/filing workflows
│   ├── cli/             # Ingestion command
│   ├── database/        # Async database lifecycle and repositories
│   ├── generation/      # Context construction and cited generation
│   ├── ingestion/       # SEC discovery-to-activation pipeline
│   ├── models/          # SQLAlchemy persistence models
│   ├── prompts/         # Markdown prompt templates and resource loader
│   ├── providers/       # Embedding and chat-model factories
│   ├── retrieval/       # Hybrid pgvector/PostgreSQL retrieval
│   ├── schemas/         # Pydantic contracts and enums
│   ├── ui/              # Sync bridge used by Streamlit
│   ├── config.py
│   ├── logging.py
│   └── observability.py
├── docker-compose.yml
├── main.py              # Minimal hard-coded query example
├── streamlit_app.py     # Streamlit entrypoint
├── pyproject.toml
└── uv.lock
```

Local drafts, notebooks, and tests are intentionally not tracked at this stage.

## Data and processing model

The main persisted entities are:

- `Company` — SEC identity and ticker metadata;
- `Filing` — filing identity, dates, URLs, and fetch state;
- `ProcessingVersion` — fingerprinted processing and embedding profile;
- `Chunk` — searchable text, metadata, provenance, and embedding;
- `IngestionRun` / `IngestionError` — execution state and failures.

Ingestion is idempotent for the same filing and processing fingerprint. An `ACTIVE` version is skipped. A `FAILED` or incomplete `BUILDING` version is reset atomically: partial chunks are deleted, the version is attached to the new run, and processing restarts from a clean `BUILDING` state.

## Current limitations

- No HTTP API or authentication layer.
- No conversation persistence or multi-turn retrieval context.
- No historical multi-filing or cross-company reasoning workflow.
- No reranker or query analyzer.
- No deterministic numerical calculation engine.
- No schema migration framework; initialization currently creates and adjusts database objects directly.
- Recovery treats an existing `BUILDING` version as retryable; there is no worker lease or heartbeat yet.
- Evaluation, security hardening, CI/CD, and production deployment remain future work.

## Roadmap

The next priority is a reproducible evaluation baseline for retrieval, grounding, citations, answer quality, latency, and cost. Later milestones include reranking, query intelligence, temporal and cross-company analysis, numerical QA, security controls, an API layer, automated quality gates, and deployment.

## Contributing

Issues, technical discussions, and focused pull requests are welcome. When proposing a RAG change, include the expected effect on retrieval quality, grounding, latency, or cost and explain how it can be measured.

---

<div align="center">

**FinNexus** · From SEC filings to measurable financial intelligence.

</div>
