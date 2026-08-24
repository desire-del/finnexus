# FinNexus — Current Architecture and Evaluation Context

> Source-of-truth snapshot of the repository as it exists on 2026-08-24.
> This document describes the current implementation; it does not prescribe a
> replacement architecture or a roadmap.

## 1. Project scope and entry points

FinNexus is a Python RAG application for ingesting SEC filings, indexing
section-aware chunks in PostgreSQL/pgvector, retrieving filing evidence, and
generating cited answers. The currently implemented user surfaces are:

- `main.py`: a hard-coded asynchronous query example.
- `streamlit_app.py`: the interactive assistant, filing browser, and ingestion
  UI.
- `rag-sec-ingest` (`src/rag_sec/cli/ingest.py`): the installed ingestion CLI.
- `scripts/*.py`: FinanceBench corpus preparation and evaluation compatibility
  entry points.

There is no HTTP API or FastAPI layer in the repository.

The project uses Python `>=3.12`, `uv`, SQLAlchemy async, PostgreSQL, pgvector,
EdgarTools, LangChain, configurable embedding/chat providers, Streamlit,
Phoenix/OpenTelemetry/OpenInference, and local JSON evaluation artifacts.

## 2. Current repository structure

```text
.
├── main.py
├── streamlit_app.py
├── pyproject.toml
├── docker-compose.yml
├── scripts/
│   ├── prepare_financebench_corpus.py
│   ├── evaluate_financebench.py
│   ├── evaluate_financebench_retriever_ablation.py
│   ├── evaluate_financebench_bm25_ablation.py
│   ├── evaluate_financebench_bm25_fusion.py
│   └── evaluate_financebench_reranker.py
├── src/rag_sec/
│   ├── application/
│   │   ├── runtime.py
│   │   ├── query.py
│   │   └── filings.py
│   ├── cli/ingest.py
│   ├── config.py
│   ├── database/
│   │   ├── manager.py
│   │   ├── repositories.py
│   │   ├── company_repository.py
│   │   ├── filing_repository.py
│   │   ├── ingestion_repository.py
│   │   └── processing_repository.py
│   ├── evaluation/
│   │   ├── artifacts.py
│   │   ├── cli.py
│   │   ├── models.py
│   │   ├── retrieval.py
│   │   ├── reranker.py
│   │   ├── runner.py
│   │   ├── studies.py
│   │   ├── suite.py
│   │   ├── corpus/financebench.py
│   │   ├── datasets/financebench.py
│   │   └── evaluators/
│   │       ├── matching.py
│   │       └── retrieval.py
│   ├── generation/
│   │   ├── context.py
│   │   ├── generator.py
│   │   ├── models.py
│   │   ├── sources.py
│   │   └── usage.py
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── edgar_client.py
│   │   ├── pipeline.py
│   │   ├── processing.py
│   │   └── run_tracker.py
│   ├── models/
│   ├── prompts/generation/
│   ├── providers/
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── bm25_store.py
│   │   └── bm25.py
│   ├── schemas/
│   ├── ui/service.py
│   ├── logging.py
│   └── observability.py
├── data/evaluation/financebench/       # ignored by Git
└── notebooks/evaluation/               # ignored by Git
```

### 2.1 Application orchestration

| File | Current responsibility | Main dependants |
|---|---|---|
| `application/runtime.py` | `RAGRuntime` owns lazy/cached database, embedding model, `Retriever`, and `Generator`; implements retrieval-only and full warmup plus shutdown. | `main.py`, UI service, evaluation |
| `application/query.py` | `embed_query()`, complete `execute_query()`, latency/throughput construction, trace input/output, and simplified `answer_query()`. | CLI example, Streamlit service, generation evaluation runner |
| `application/filings.py` | Queries active filings compatible with the selected embedding profile and returns `AvailableFiling`. | Streamlit filing browser |

`RAGRuntime` uses `cached_property`, while `get_runtime()` is process-cached with
`lru_cache`. `warmup_retrieval()` initializes the database/view, warms the
embedding provider, and creates the LangChain `PGVectorStore`. `warmup()` adds
lazy construction of `Generator` and therefore the chat model.

### 2.2 Database and persistence

`database/manager.py` creates the async SQLAlchemy engine/session factory. Its
`initialize()` method:

1. ensures the pgvector extension exists;
2. creates ORM tables;
3. recreates the `active_chunks` SQL view;
4. exposes only chunks whose `ProcessingVersion.status` is `active` and whose
   embedding is non-null.

The repositories are separated by aggregate:

- `CompanyRepository`: lookup/create/get-or-create.
- `FilingRepository`: filing lookup, creation, fetch/failure state.
- `ProcessingRepository`: processing version identity, chunk persistence,
  retry reset, validation/activation, and active-version filtering.
- `IngestionRepository`: ingestion run counters, stages, failures, and
  completion.
- `database/repositories.py`: compatibility re-export only; new application
  code imports the dedicated modules.

The ORM models under `models/` are `Company`, `Filing`, `ProcessingVersion`,
`Chunk`, `IngestionRun`, and `IngestionError`. Pydantic command/read models and
enums live separately under `schemas/`.

### 2.3 Ingestion

| File | Responsibility |
|---|---|
| `ingestion/edgar_client.py` | EdgarTools discovery by ticker/CIK/accession, conversion to local schemas, filing content retrieval, hashes, and section extraction. |
| `ingestion/chunker.py` | `SectionChunker`, based on `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, creates stable chunk IDs and section locators. |
| `ingestion/processing.py` | `ProcessingProfile` owns version constants, deterministic fingerprinting, conservative normalization, processing schema creation, and embedding-dimension validation. |
| `ingestion/run_tracker.py` | `IngestionRunTracker` persists stages, skipped results, and failure state. |
| `ingestion/pipeline.py` | `IngestionPipeline` remains the end-to-end coordinator for discovery through activation. |

The ingestion sequence is:

```text
latest ticker/CIK or exact accession
  -> create IngestionRun
  -> EdgarTools discovery
  -> register Company and Filing
  -> fast ACTIVE/fingerprint idempotency check
  -> fetch source and persist content hash
  -> compute ProcessingProfile fingerprint
  -> reuse/reset FAILED or BUILDING ProcessingVersion, or create a new one
  -> extract and normalize SEC sections
  -> chunk
  -> embed all chunks
  -> persist chunks
  -> validate non-zero chunk count
  -> activate version and supersede compatible prior active version
  -> complete IngestionRun
```

`ProcessingRepository.reset_version()` locks the retryable version, deletes its
partial chunks, changes its `ingestion_run_id`, resets status/count/timestamps,
and does so inside the caller's transaction.

### 2.4 Generation

- `generation/context.py`: `ContextBuilder` deduplicates retrieved chunks and
  assigns source identifiers `S1`, `S2`, ... while building the SEC context.
- `generation/generator.py`: `Generator` invokes the configured chat model with
  structured output `GeneratedAnswer`, validates returned source identifiers,
  records observability attributes, and returns `RAGAnswer`.
- `generation/models.py`: `GeneratedAnswer`, `SourceInfo`, `TokenUsage`,
  `QueryMetrics`, and `RAGAnswer`.
- `generation/sources.py`: cited-source metadata, readable excerpts, and SEC
  text-fragment deep links.
- `generation/usage.py`: provider token usage extraction with a character-based
  fallback estimate.
- `prompts/generation/system.md` and `user.md`: actual generation prompts loaded
  by `prompts/loader.py`.

### 2.5 Interfaces

`ui/service.py` bridges synchronous Streamlit reruns to one background asyncio
event loop. `RAGService` performs runtime warmup, query execution, ingestion,
filing listing, and shutdown. `streamlit_app.py` contains presentation,
conversation state, metrics, source links, formula normalization, assistant and
filing pages. It is currently a single large UI module.

The ingestion CLI supports latest-filing ingestion by ticker/CIK and exact
accession ingestion. The scripts under `scripts/` are compatibility wrappers;
evaluation logic now lives in `rag_sec.evaluation`.

## 3. Current query/RAG pipeline

The production query flow is:

```text
main.py or Streamlit RAGService.answer()
  -> RAGRuntime.warmup()
  -> application.answer_query()
  -> application.execute_query()
      -> embed_query(question, runtime.embedding_model)
      -> Retriever.search(
           query_embedding,
           ticker,
           form_type,
           mode omitted => "hybrid"
         )
      -> Generator.generate(question, documents)
          -> ContextBuilder.build(documents)
          -> generation prompts
          -> structured chat-model invocation
          -> citation validation/source metadata
      -> attach QueryMetrics
  -> RAGAnswer(answer, sources, usage, metrics)
```

`execute_query()` requires `ticker` and `form_type`; it does not accept an
accession number, retrieval mode, or explicit top-k. Production therefore uses
`Retriever.search()`'s default `mode="hybrid"` and the runtime's default
retrieval limits.

`embed_query()` is outside `Retriever`. It strips/validates the question,
invokes `aembed_query`, checks the configured dimension, and creates the
`query.embedding` trace. BM25-only evaluation passes no embedding; every mode
with a dense branch requires one.

### Query result models

- `QueryExecution`: dataclass containing both `RAGAnswer` and ordered retrieved
  LangChain `Document` instances; evaluation uses this richer result.
- `RAGAnswer`: answer text, validated `SourceInfo` list, `TokenUsage`, and
  `QueryMetrics`.
- `QueryMetrics`: total/embedding/retrieval/generation latencies, generation
  tokens per second, retrieval documents per second, retrieved-document count,
  and cited-source count.

### Query observability

The explicit span hierarchy is:

```text
rag.query
  ├── query.embedding
  ├── retrieval.search
  └── generation.answer
       └── generation.build_context
```

`observability.py` provides `@track`, span attributes/input/output, and source
tags. Phoenix is the only implemented exporter; `none` disables tracing.
OpenAI and optionally LangChain instrumentation add lower-level spans. Content
capture is controlled by `OBSERVABILITY_CAPTURE_CONTENT`.

## 4. Retriever implementation

### 4.1 Classes and mode selection

There is one production `Retriever` class in
`src/rag_sec/retrieval/retriever.py`. It exposes five literal modes through the
`mode` argument of `search()`:

- `dense`
- `lexical`
- `hybrid`
- `bm25`
- `bm25_hybrid`

It owns two storage/search mechanisms:

- LangChain `PGVectorStore` over the `active_chunks` PostgreSQL view.
- `BM25Store`, which reads filtered ORM chunks and passes them to the local
  Python BM25 implementation.

All modes return `list[langchain_core.documents.Document]`. Filters always
include the current embedding provider/model/dimension profile; optional
filters are ticker, form type, and accession number.

### 4.2 Dense mode

**Input**

- normalized query string;
- externally computed query vector;
- optional filing filters;
- result `top_k`.

**Process**

- `_build_filters()` builds LangChain metadata filters.
- `PGVectorStore.asimilarity_search_by_vector()` queries `active_chunks`.
- No hybrid configuration is supplied.

**Output**

- top `result_limit` documents from pgvector.

The processing metadata records `DistanceMetric.COSINE`, and no alternative
distance is exposed in retrieval configuration. `dense_top_k` is used as the
dense branch depth in hybrid modes; a direct dense request uses the requested
`k` passed to the vector store.

### 4.3 PostgreSQL lexical mode

**Input**

- query string plus a query embedding (required by the current vector-store
  call even when the custom fusion discards dense results);
- active embedding profile and optional filing filters.

**Process**

- constructs LangChain PostgreSQL `HybridSearchConfig`;
- uses `tsv_lang="pg_catalog.english"` and `fts_query=query`;
- sets `primary_top_k=dense_top_k` and
  `secondary_top_k=lexical_top_k`;
- supplies `lexical_only_ranking()`, which discards primary/dense results and
  sorts the secondary mappings by their `distance` field descending.

**Output**

- truncated PostgreSQL FTS results.

This is PostgreSQL full-text search as implemented by `langchain-postgres`; it
is not BM25. Tsquery construction and the underlying rank expression are owned
by that library, not by FinNexus code.

### 4.4 LangChain hybrid mode

**Input**

- query vector, raw query text, filters, and result limit.

**Process**

- one `PGVectorStore` call with `HybridSearchConfig`;
- dense depth `dense_top_k`;
- PostgreSQL FTS depth `lexical_top_k`;
- `langchain_postgres...reciprocal_rank_fusion`;
- fixed `rrf_k=60`.

**Output**

- fused documents truncated by the vector-store call to `result_limit`.

There are no weights in this production mode. Deduplication/fusion details are
inside `langchain-postgres`.

### 4.5 BM25 mode

**Input**

- raw query and optional ticker/form/accession filters;
- `top_k`; no query embedding is required.

**Process**

1. `BM25Store.search()` queries every active, embedded chunk matching the
   embedding profile and filing filters using SQLAlchemy.
2. Each row becomes a LangChain `Document`.
3. `rank_bm25()` tokenizes with a local regex retaining lower-case words,
   numbers, periods, and hyphens.
4. It computes Okapi BM25 in Python with defaults `k1=1.5`, `b=0.75`.

**Output**

- documents with positive BM25 score, sorted descending, truncated to top-k.

No PostgreSQL BM25 extension or persistent lexical index is used. The filtered
corpus is loaded and tokenized for every search. The BM25 score is held in the
temporary `BM25Result`; it is not copied into returned `Document.metadata`.

### 4.6 BM25 hybrid mode

**Input**

- query text, query embedding, filters, and final result limit.

**Process**

- obtains `lexical_top_k` BM25 documents;
- obtains `dense_top_k` pgvector documents;
- `reciprocal_rank_fuse_documents()` assigns each document
  `1 / (60 + rank)` per branch;
- duplicate identity is `chunk_id`, then `id`, then page content;
- scores are summed and the final list is truncated to `result_limit`.

**Output**

- equal-weight RRF fused documents.

### 4.7 Weighted RRF, candidate union, and reranker

These are not part of production query execution:

- `evaluation.retrieval.weighted_rrf()` implements weighted dense/lexical RRF
  for saved study configurations.
- `evaluation.retrieval.deduplicated_union()` creates dense+BM25 candidate
  unions for reranker experiments.
- `evaluation.reranker.CrossEncoderReranker` lazily loads a Sentence
  Transformers `CrossEncoder` and scores query/document pairs.

`execute_query()` cannot enable any of these stages.

## 5. Current retrieval configuration

| Setting | Current location | Current behavior |
|---|---|---|
| Production mode | `Retriever.search(mode="hybrid")` default | Not in `.env`; production callers omit it. |
| Final `top_k` | `Retriever(top_k=5)` in `RAGRuntime.retriever` | Hard-coded runtime default; evaluation passes explicit values. |
| Dense branch depth | `dense_top_k=20` in runtime | Mutable instance attribute; studies sometimes retrieve top 30 explicitly. |
| Lexical/BM25 branch depth | `lexical_top_k=20` in runtime | Shared by PostgreSQL FTS and BM25 branches. |
| Production RRF constant | `rrf_k=60` inside `Retriever.search()` / BM25 fusion helper default | Hard-coded. |
| Production fusion weights | none | Equal contribution only. |
| BM25 `k1`, `b` | defaults in `rank_bm25()` | `1.5`, `0.75`; not environment-configurable. |
| Embedding provider | `EMBEDDING_PROVIDER` | `openai`, `huggingface`, or `ollama`. |
| Embedding model/dimension | provider-specific `EMBEDDING_*` fields | Selects active chunks and validates query/chunk vectors. |
| Reranker model/batch/max length | `RERANKER_*` | Evaluation only. |
| Weighted RRF sweep | `FUSION_CONFIGS` in `evaluation/studies.py` | Code-defined experiment matrix. |

Changing the embedding or chat provider is an `.env` change. Changing the
production retrieval mode requires changing a caller or the default argument.
Changing candidate depths requires changing runtime construction or mutating
the retriever instance. Weighted fusion experiments require editing
`FUSION_CONFIGS`; there is no shared retrieval settings object or YAML
experiment configuration.

## 6. Evaluation architecture

### 6.1 Evaluation models

`evaluation/models.py` defines:

- `ReferenceEvidence`: gold text/document/accession/chunk/section/item/page plus
  metadata.
- `EvaluationCase`: normalized dataset case with question, company/filing
  filters, reference answer/evidence, tags, and metadata.
- `RetrievedEvidence`: normalized retrieved passage and rank/score/metadata.
- `EvaluationRunMetrics`: latency/token/cost slots.
- `EvaluationRun`: one case result, generated answer, retrieved evidence,
  citations, metrics, and error.
- `EvaluationScore`: evaluator/metric/score/label/explanation and evaluator
  type.

These are generic models; FinanceBench-specific fields are stored in
`EvaluationCase.metadata` and tags.

### 6.2 Module responsibilities and overlaps

| Module | Current purpose | Called by / overlap |
|---|---|---|
| `evaluation/artifacts.py` | Atomic JSON artifact read/write. | `studies.py`; independent of production. |
| `evaluation/suite.py` | `FinanceBenchSuite`: paths, input validation, accession resolution, case validation, database corpus preflight, embedding metadata. | CLI/studies. Despite the generic filename, it is FinanceBench-specific. |
| `evaluation/datasets/financebench.py` | Reads raw JSONL, normalizes form/accession/company metadata, creates `EvaluationCase`. | `FinanceBenchSuite`. |
| `evaluation/corpus/financebench.py` | Accession cache, EdgarTools-based accession resolution, evidence matching fallback, unsupported-document handling, and corpus ingestion loop. | Suite and corpus-preparation script. It calls production `IngestionPipeline`. |
| `evaluation/evaluators/matching.py` | Text normalization, global/local token recall, and evidence match predicate. | Retrieval metrics and accession-resolution fallback. |
| `evaluation/evaluators/retrieval.py` | Relevance, Hit@K, Recall@K, reciprocal rank, and score construction. | `evaluation/retrieval.py`. |
| `evaluation/retrieval.py` | `RetrievalEvaluator`, per-case query embedding/retrieval, aggregation, evidence conversion, union, weighted RRF, contribution diagnostics. | `FinanceBenchStudies`. It partially recreates the retrieval portion of `application/query.py`. |
| `evaluation/reranker.py` | Evaluation-only cross-encoder lifecycle and ranking. | Reranker study. |
| `evaluation/runner.py` | Full RAG/generation dataset runner using production `execute_query()`. | Present but not called by current retrieval study CLI. |
| `evaluation/studies.py` | Orchestrates baseline, ablations, fusion sweep, and reranker study; embeds fixed experiment constants and artifact names. | Evaluation CLI. |
| `evaluation/cli.py` | Selects study by positional name, configures observability, initializes/closes study runtime. | `python -m ...` and wrapper scripts. |

There is no `rewriter` module, database experiment repository, or experiment
configuration file. Persistence is JSON-only.

The evaluation layer does recreate part of production behavior:

- `RetrievalEvaluator._run_case()` independently times query embedding and
  invokes `runtime.retriever.search()` rather than calling `execute_query()`;
  this is needed for retrieval-only results but duplicates embedding/retrieval
  orchestration and error handling.
- Weighted RRF and candidate union are evaluation implementations, not
  production `Retriever` modes.
- The evaluation reranker is not wired to the production query pipeline.
- `runner.py` provides a second path for full RAG evaluation, while current
  studies use `RetrievalEvaluator`.

## 7. FinanceBench dataset and corpus

### 7.1 Local raw data

The suite expects these ignored local files under
`data/evaluation/financebench/`:

- `financebench_open_source.jsonl`
- `financebench_document_information.jsonl`
- `accession_map.json`

The raw JSONL files are the local FinanceBench open-source question/evidence and
document metadata exports. No runtime dataset download occurs in the current
evaluation CLI.

### 7.2 Loading and normalization

`load_financebench()` joins question rows to document information, normalizes
form type, extracts an accession from available document links when possible,
maps known company names to ticker/CIK tables, and creates `EvaluationCase`
instances. Reference evidence carries text, document ID, accession, page, and
full-page text metadata.

FinanceBench metadata retained includes:

- `financebench_id`
- `doc_name`, period, type, and link
- company CIK
- justification
- dataset subset label
- question type/reasoning as tags

### 7.3 Accession resolution and filtering

`resolve_financebench_accessions()` applies the local cache, document metadata,
EdgarTools candidates, filing period/year/form filters, and evidence-text
matching to resolve exact SEC accessions. It records unresolved and unsupported
documents separately. The suite then keeps cases with an accession and validates
unique IDs, non-empty questions, tickers, supported forms (`10-K`, `10-Q`,
`8-K`), accession syntax, and reference/case accession consistency.

The local artifacts contain 150 source cases, 136 SEC-compatible evaluated
cases, and 14 excluded cases.

### 7.4 Corpus preparation

`scripts/prepare_financebench_corpus.py` loads the suite, initializes the
database and embedding model, then calls `prepare_financebench_corpus()`.
Unique accessions are ingested one at a time through the normal production
`IngestionPipeline.ingest_accession()`. The report counts processed, skipped,
failed, and missing-accession cases.

Before retrieval studies, `FinanceBenchSuite.validate_corpus()` verifies that
every required accession has an ACTIVE `ProcessingVersion` for the currently
configured embedding provider/model/dimension.

FinanceBench-specific logic is isolated in dataset/corpus modules and
`FinanceBenchSuite`; however, `studies.py` itself is wholly FinanceBench-specific
despite living at the evaluation package root.

## 8. Metric system

### 8.1 Evidence matching

`EvidenceMatchConfig` controls normalized text matching. The implementation:

- case-folds and whitespace-normalizes text;
- can match exact chunk IDs/accessions/locations when present;
- computes token recall of gold evidence against retrieved text;
- uses `local_token_recall()` to search windows within long retrieved chunks;
- uses gold full-page text metadata where available.

This matching is deterministic and reused by retrieval evaluation and accession
resolution.

### 8.2 Retrieval metrics

For an `EvaluationCase` and `EvaluationRun`:

- `hit@k`: 1 if at least one reference evidence is matched in the first `k`.
- `recall@k`: fraction of distinct reference evidence items matched in the first
  `k`; this handles multi-evidence questions.
- `reciprocal_rank@k`: reciprocal of the first relevant rank, or zero.
- aggregate `mrr@max_k`: arithmetic mean of per-case reciprocal rank at the
  largest requested k.

`evaluate_retrieval()` returns reusable `EvaluationScore` objects.
`aggregate_runs()` converts them to artifact records and means each metric
across cases. It renames only the largest requested reciprocal-rank metric to
`mrr@K`; records retain `reciprocal_rank@K`.

### 8.3 Generation metrics

There are no answer-correctness, faithfulness, citation-precision, LLM-judge,
or human metrics implemented. `runner.py` can collect generated answers,
citations, latencies, and token usage, but current study commands are retrieval
experiments only.

## 9. Current experiment execution flow

All five compatibility scripts call the single evaluation CLI with a fixed
study name:

```text
scripts/evaluate_financebench*.py
  -> evaluation.cli.main([experiment])
  -> FinanceBenchStudies.initialize()
      -> FinanceBenchSuite.load()
      -> resolve accessions / validate cases
      -> validate ACTIVE database corpus
      -> RAGRuntime.warmup_retrieval()
  -> selected method in FinanceBenchStudies
      -> RetrievalEvaluator.evaluate(mode, top_k, ks)
          -> per case: optional embed_query()
          -> production Retriever.search()
          -> RetrievedEvidence + EvaluationRun
      -> deterministic retrieval metrics
      -> aggregation and experiment-specific diagnostics
  -> write_artifact(...json)
  -> runtime shutdown
```

The reranker command is different: it initializes case data without retrieval
warmup, reads the BM25 ablation artifact, reconstructs candidates, loads the
cross-encoder, runs two reranking variants, aggregates, and writes a new JSON
artifact.

Current commands are:

```bash
uv run scripts/prepare_financebench_corpus.py
uv run scripts/evaluate_financebench.py
uv run scripts/evaluate_financebench_retriever_ablation.py
uv run scripts/evaluate_financebench_bm25_ablation.py
uv run scripts/evaluate_financebench_bm25_fusion.py
uv run scripts/evaluate_financebench_reranker.py

# Equivalent unified interface
uv run python -m rag_sec.evaluation.cli baseline
uv run python -m rag_sec.evaluation.cli fts-ablation
uv run python -m rag_sec.evaluation.cli bm25-ablation
uv run python -m rag_sec.evaluation.cli fusion
uv run python -m rag_sec.evaluation.cli reranker
```

Experiment metadata and per-case results are JSON dictionaries, not instances
of a persisted experiment model. Timestamps, embedding metadata, fixed depths,
model names, metrics, evidence, and errors are stored in the artifact.

`notebooks/evaluation/01_financebench_retrieval_baseline_analysis.ipynb`
directly loads the six named JSON artifacts and the raw questions JSONL. It
contains baseline, question-family, reasoning, evidence-count, failure-bucket,
FTS, BM25, fusion, reranker, and conclusion sections. The notebook depends on
artifact filenames and nested JSON layouts rather than an artifact API.

## 10. Current FinanceBench experiment results

These values come from the local JSON artifacts. They are not recomputed in
this document.

### 10.1 Original top-5 baseline

Artifact: `baseline_retrieval_v1.json`; 136 cases; OpenAI
`text-embedding-3-small`, 1536 dimensions.

| Metric | Value |
|---|---:|
| Hit@1 | 32.35% |
| Hit@3 | 51.47% |
| Hit@5 | 55.88% |
| Recall@5 | 49.51% |
| MRR@5 | 41.03% |

The artifact labels its retrieval configuration with `dense_top_k=20`,
`lexical_top_k=20`, and final `top_k=5`; its name predates the consolidated
study API.

### 10.2 Dense candidate recall at 20

Artifact: `candidate_retrieval_top20_v1.json`.

| Metric | Value |
|---|---:|
| Hit@5 | 55.88% |
| Hit@10 | 70.59% |
| Hit@20 | 79.41% |
| Recall@20 | 77.21% |
| MRR@20 | 43.60% |

### 10.3 PostgreSQL FTS diagnostic

Artifact: `retriever_ablation_top20_v1.json`.

| Branch | Hit@5 | Hit@20 | Recall@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Dense | 56.62% | 79.41% | 77.21% | 43.68% |
| PostgreSQL lexical | 0.00% | 0.00% | 0.00% | 0.00% |
| LangChain hybrid | 55.88% | 79.41% | 77.21% | 43.60% |

The zero lexical gold-match metrics do not, by themselves, prove that the FTS
branch returned zero documents; they mean no gold evidence was recognized in
its evaluated top 20.

### 10.4 Dense, BM25, and equal-RRF comparison

Artifact: `retriever_bm25_ablation_top20_v1.json`.

| Branch | Hit@5 | Hit@20 | Recall@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Dense | 56.62% | 79.41% | 77.21% | 43.68% |
| BM25 | 41.91% | 58.09% | 55.39% | 29.87% |
| Dense + BM25 equal RRF | 54.41% | 75.74% | 73.65% | 38.09% |

BM25 has 79 Hit@20 successes, including 5 cases missed by dense. Equal fusion
has 103 successes, adds 4 versus dense, and loses 9 dense successes.

### 10.5 Weighted RRF sweep

Artifact: `retriever_bm25_fusion_v1.json`; dense/BM25 retrieval depth 30,
final top 20, `rrf_k=60`.

| Configuration | Dense:BM25 | Depths | Hit@5 | Hit@20 | Recall@20 |
|---|---:|---:|---:|---:|---:|
| `rrf_1_1_d20_b20` | 1:1 | 20/20 | 54.41% | 75.74% | 73.65% |
| `rrf_2_1_d20_b20` | 2:1 | 20/20 | 53.68% | 79.41% | 77.21% |
| `rrf_3_1_d20_b20` | 3:1 | 20/20 | 53.68% | 79.41% | 77.21% |
| `rrf_2_1_d30_b20` | 2:1 | 30/20 | 51.47% | 80.15% | 77.70% |
| `rrf_2_1_d30_b30` | 2:1 | 30/30 | 52.21% | 80.15% | 77.94% |

The artifact selects `rrf_2_1_d30_b30` by its code-defined ordering of Hit@20,
Recall@20, then Hit@5. Relative to dense at 20 it gains 3 cases and loses 2;
its early-rank metrics remain below the dense branch.

### 10.6 Cross-encoder reranking

Artifact: `reranker_dense_vs_bm25_union_v1.json`; model
`cross-encoder/ms-marco-MiniLM-L-6-v2`; dense/BM25 candidate depth 20; final
top 5; average deduplicated union size 32.60.

| Variant | Hit@1 | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Dense top-5 baseline | 32.35% | 56.62% | 50.25% | 41.24% |
| Dense top-20 reranked | 27.21% | 52.94% | 50.25% | 36.19% |
| Dense+BM25 union reranked | 25.74% | 49.26% | 46.20% | 33.76% |

The tested generic cross-encoder did not improve the recorded FinanceBench
retrieval metrics.

## 11. Current complexity and pain points

The following are concrete properties of the current code, not proposed
solutions:

1. **Production retrieval configuration is code-scattered.** Mode is a method
   argument/default; top-k values are runtime constructor constants; RRF is
   hard-coded; BM25 parameters are function defaults; weighted fusion matrices
   live in evaluation code.
2. **One `Retriever.search()` branches across five algorithms.** It owns
   pgvector initialization, PostgreSQL FTS configuration, BM25 orchestration,
   two fusion implementations, filters, logging, and tracing.
3. **“Lexical” means two different mechanisms.** `lexical` is PostgreSQL FTS;
   `bm25` is Python Okapi BM25 over database-loaded chunks. Both share the
   `lexical_top_k` instance field.
4. **BM25 is operationally expensive.** It loads and tokenizes all matching
   chunks for each query; there is no persistent BM25 index or corpus cache.
5. **Evaluation duplicates query stages.** `RetrievalEvaluator` duplicates
   embedding, retrieval timing, evidence conversion, and error boundaries from
   the production query workflow, while `runner.py` separately wraps full
   `execute_query()`.
6. **Evaluation contains retrieval variants absent from production.** Weighted
   RRF, candidate union, and reranking can be evaluated but cannot be enabled in
   `execute_query()`.
7. **Study configuration and artifact schema are implicit Python/JSON.** Adding
   a sweep means editing `FUSION_CONFIGS`/study code and often the notebook.
8. **FinanceBench orchestration is spread across several modules.** Dataset
   loading, accession resolution, corpus preparation, suite preflight, study
   execution, wrapper scripts, artifacts, and notebook analysis are distinct
   but tightly filename/schema-coupled.
9. **Some naming suggests broader abstractions than implemented.** `suite.py`
   and `studies.py` are FinanceBench-only; `runner.py` is generic-looking but is
   not used by current study commands.
10. **The Streamlit presentation remains monolithic.** `streamlit_app.py`
    combines styling, rendering, session state, ingestion UI, filing tables,
    and assistant interaction.
11. **No generation-quality evaluation exists.** Generation result models and a
    dataset runner exist, but the active framework measures retrieval only.
12. **Artifacts are local and ignored by Git.** Reproducibility depends on
    separately preserving raw dataset files, accession cache, corpus state, and
    JSON outputs.

## 12. Retriever modularity relative to desired configuration

The current code is partly prepared for selectable modes:

- `RetrievalMode` already enumerates dense, FTS lexical, LangChain hybrid,
  BM25, and BM25 hybrid.
- `Retriever.search()` already has one mode-selection argument and a common
  `Document` output.
- embedding provider/model/dimension already use environment settings.
- dense and BM25 search logic have internal boundaries (`_search_dense`,
  `BM25Store`).

It is not currently configurable in the desired declarative form because:

- `execute_query()` does not expose `mode` or retrieval parameters;
- no `RetrievalSettings` exists in `config.py`;
- branch depths and final top-k are runtime attributes/arguments;
- production RRF has no selectable fusion type or weights;
- weighted RRF exists only in `evaluation/retrieval.py`;
- reranking exists only in `evaluation/reranker.py` and `studies.py`;
- the FTS and BM25 mechanisms share “lexical” terminology and depth state;
- experiment settings are constants distributed across `studies.py`,
  `Retriever`, and artifact metadata.

Adding another mode currently requires changing the `RetrievalMode` literal,
the validation/branch logic in `Retriever.search()`, potentially runtime/query
callers, evaluation studies/CLI, artifact interpretation, and notebook analysis.

## 13. Current component modularity

| Component | Exists? | Current wiring / configurability |
|---|---|---|
| Query Rewriter | No | No production or evaluation implementation. |
| Query Analyzer | No | No classification/routing stage. FinanceBench question metadata is analyzed only after evaluation in the notebook. |
| Retriever | Yes | One production class with five code-selected modes; production always defaults to hybrid. |
| Reranker | Evaluation only | `CrossEncoderReranker`; enabled only by the reranker study command and `.env` reranker settings. |
| Generator | Yes | Always used by `execute_query()`/`answer_query()`; omitted only by retrieval-specific evaluation code. |

There is no common optional-stage pipeline model. Enabling/disabling generation
means choosing between production `execute_query()` and evaluation
`RetrievalEvaluator`; enabling reranking requires using evaluation study code.

## 14. Final source-of-truth summary

### A. Current architecture

Async Python application with lazy runtime services, EdgarTools ingestion,
SQLAlchemy/PostgreSQL persistence, pgvector/FTS/BM25 retrieval, structured LLM
generation, Streamlit/CLI interfaces, and local FinanceBench evaluation.

### B. Current query execution flow

Question -> validated query embedding -> default LangChain pgvector+FTS hybrid
retrieval filtered by ticker/form and embedding profile -> context/cited
structured generation -> `RAGAnswer` with sources, usage, and timings.

### C. Current retrieval modes

`dense`, PostgreSQL FTS `lexical`, LangChain equal-RRF `hybrid`, local Python
`bm25`, and equal-RRF `bm25_hybrid`. Weighted RRF and reranking are evaluation
only.

### D. Current evaluation execution flow

FinanceBench JSONL -> accession resolution/cache -> SEC-compatible case
validation -> active-corpus preflight -> production retriever calls ->
deterministic evidence matching/metrics -> local JSON artifact -> notebook.

### E. Current configuration mechanism

Pydantic settings from `.env` cover providers, model names/dimensions, database,
EDGAR, observability, and evaluation reranker. Retrieval mode, depths, RRF, BM25
parameters, and study sweeps remain code constants/arguments.

### F. Current experimental results

Dense candidate Hit@20 is 79.41%. BM25 Hit@20 is 58.09% and contributes five
unique successes. Equal dense+BM25 RRF is worse than dense. The selected 2:1
weighted depth-30 fusion reaches 80.15% Hit@20 but degrades early ranking. The
tested generic cross-encoder degrades top-5/MRR. PostgreSQL lexical gold-match
metrics are zero in the stored ablation.

### G. Current technical-debt hotspots

Scattered retrieval/study configuration, branching `Retriever`, evaluation-only
retrieval stages, duplicated retrieval orchestration, per-query in-memory BM25,
FinanceBench/artifact/notebook coupling, and monolithic Streamlit presentation.

### H. Files to inspect before retrieval/evaluation changes

At minimum:

- `src/rag_sec/retrieval/retriever.py`
- `src/rag_sec/retrieval/bm25.py`
- `src/rag_sec/retrieval/bm25_store.py`
- `src/rag_sec/application/query.py`
- `src/rag_sec/application/runtime.py`
- `src/rag_sec/config.py`
- `src/rag_sec/database/manager.py`
- `src/rag_sec/models/processing_version.py`
- `src/rag_sec/evaluation/retrieval.py`
- `src/rag_sec/evaluation/studies.py`
- `src/rag_sec/evaluation/suite.py`
- `src/rag_sec/evaluation/runner.py`
- `src/rag_sec/evaluation/evaluators/matching.py`
- `src/rag_sec/evaluation/evaluators/retrieval.py`
- `src/rag_sec/evaluation/datasets/financebench.py`
- `src/rag_sec/evaluation/corpus/financebench.py`
- `scripts/evaluate_financebench*.py`
- `notebooks/evaluation/01_financebench_retrieval_baseline_analysis.ipynb`
- the JSON artifacts under `data/evaluation/financebench/`
