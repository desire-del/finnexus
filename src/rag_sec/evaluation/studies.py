from datetime import UTC, datetime
from typing import Any, Literal

from rag_sec.application.runtime import RAGRuntime, get_runtime
from rag_sec.config import get_settings
from rag_sec.evaluation.artifacts import write_artifact
from rag_sec.evaluation.models import EvaluationCase
from rag_sec.evaluation.retrieval import (
    RetrievalEvaluator,
    RetrievalResult,
    contribution_diagnostics,
)
from rag_sec.evaluation.suite import FinanceBenchSuite, LoadedSuite
from rag_sec.retrieval.retriever import RetrievalMode

DEFAULT_KS = (1, 3, 5, 10, 20)


class FinanceBenchStudies:
    """Single orchestration surface for all FinanceBench retrieval studies."""

    def __init__(
        self,
        suite: FinanceBenchSuite | None = None,
        runtime: RAGRuntime | None = None,
    ) -> None:
        self.suite = suite or FinanceBenchSuite()
        self.runtime = runtime or get_runtime()
        self.evaluator = RetrievalEvaluator(self.runtime)
        self.loaded: LoadedSuite | None = None

    async def initialize(
        self, *, require_retrieval: bool = True
    ) -> list[EvaluationCase]:
        self.loaded = await self.suite.load()
        if require_retrieval:
            await self.suite.validate_corpus(self.loaded.cases)
            await self.runtime.warmup_retrieval()
        return self.loaded.cases

    async def close(self) -> None:
        await self.runtime.shutdown()

    async def baseline(self, *, top_k: int = 20) -> dict[str, Any]:
        cases = self._cases
        result = await self._evaluate("dense", cases, top_k=top_k)
        retrieval = get_settings().retrieval
        payload = {
            "experiment": self._metadata(
                "financebench-dense-pipeline-v2",
                cases=len(cases),
                top_k=top_k,
                dense_candidate_k=retrieval.dense_candidate_k,
                excluded_cases=self._loaded.excluded_cases,
            ),
            "metrics": result.metrics,
            "results": result.records,
        }
        write_artifact(
            self.suite.artifact_path("retrieval_dense_pipeline_v2.json"), payload
        )
        return payload

    async def fts_baseline(self, *, top_k: int = 20) -> dict[str, Any]:
        cases = self._cases
        result = await self._evaluate("fts", cases, top_k=top_k)
        payload = {
            "experiment": self._metadata(
                "financebench-postgres-fts-pipeline-v2",
                cases=len(cases),
                top_k=top_k,
                fts_candidate_k=get_settings().retrieval.fts_candidate_k,
                query_parser="websearch_to_tsquery_disjunction",
                text_search_config="pg_catalog.english",
                excluded_cases=self._loaded.excluded_cases,
            ),
            "metrics": result.metrics,
            "results": result.records,
        }
        write_artifact(
            self.suite.artifact_path("retrieval_postgres_fts_pipeline_v2.json"),
            payload,
        )
        return payload

    async def bm25_baseline(self, *, top_k: int = 20) -> dict[str, Any]:
        cases = self._cases
        result = await self._evaluate("bm25", cases, top_k=top_k)
        payload = {
            "experiment": self._metadata(
                "financebench-pgsearch-bm25-pipeline-v2",
                cases=len(cases),
                top_k=top_k,
                bm25_candidate_k=get_settings().retrieval.bm25_candidate_k,
                backend="pg_search",
                excluded_cases=self._loaded.excluded_cases,
            ),
            "metrics": result.metrics,
            "results": result.records,
        }
        write_artifact(
            self.suite.artifact_path("retrieval_pgsearch_bm25_pipeline_v2.json"),
            payload,
        )
        return payload

    async def hybrid_baseline(
        self,
        *,
        lexical_backend: Literal["fts", "bm25"],
        top_k: int = 20,
    ) -> dict[str, Any]:
        cases = self._cases
        mode: RetrievalMode = "hybrid" if lexical_backend == "fts" else "bm25_hybrid"
        result = await self._evaluate(mode, cases, top_k=top_k)
        retrieval = get_settings().retrieval
        payload = {
            "experiment": self._metadata(
                f"financebench-hybrid-{lexical_backend}-pipeline-v2",
                cases=len(cases),
                top_k=top_k,
                lexical_backend=lexical_backend,
                dense_candidate_k=retrieval.dense_candidate_k,
                lexical_candidate_k=(
                    retrieval.fts_candidate_k
                    if lexical_backend == "fts"
                    else retrieval.bm25_candidate_k
                ),
                rrf_k=retrieval.rrf_k,
                dense_weight=retrieval.dense_weight,
                lexical_weight=retrieval.lexical_weight,
                excluded_cases=self._loaded.excluded_cases,
            ),
            "metrics": result.metrics,
            "results": result.records,
        }
        write_artifact(
            self.suite.artifact_path(
                f"retrieval_hybrid_{lexical_backend}_pipeline_v2.json"
            ),
            payload,
        )
        return payload

    async def ablation(self, *, lexical: RetrievalMode) -> dict[str, Any]:
        if lexical not in {"lexical", "bm25"}:
            raise ValueError("Ablation lexical mode must be 'lexical' or 'bm25'.")
        cases = self._cases
        hybrid: RetrievalMode = "hybrid" if lexical == "lexical" else "bm25_hybrid"
        modes: tuple[RetrievalMode, ...] = ("dense", lexical, hybrid)
        results = {mode: await self._evaluate(mode, cases, top_k=20) for mode in modes}
        contribution = contribution_diagnostics(
            results["dense"],
            results[lexical],
            results[hybrid],
            lexical_label=lexical,
        )
        payload = {
            "experiment": self._metadata(
                f"financebench-{lexical}-ablation-top20-v1",
                cases=len(cases),
                top_k=20,
                modes=list(modes),
            ),
            "modes": {
                mode: self._result_payload(result) for mode, result in results.items()
            },
            "contribution": contribution,
        }
        filename = (
            "retriever_ablation_top20_v1.json"
            if lexical == "lexical"
            else "retriever_bm25_ablation_top20_v1.json"
        )
        write_artifact(self.suite.artifact_path(filename), payload)
        return payload

    @property
    def _cases(self) -> list[EvaluationCase]:
        return self._loaded.cases

    @property
    def _loaded(self) -> LoadedSuite:
        if self.loaded is None:
            raise RuntimeError("Initialize FinanceBenchStudies before running a study.")
        return self.loaded

    async def _evaluate(
        self,
        mode: RetrievalMode,
        cases: list[EvaluationCase],
        *,
        top_k: int,
    ) -> RetrievalResult:
        print(f"Evaluating {mode} retrieval...")
        result = await self.evaluator.evaluate(
            cases, mode=mode, top_k=top_k, ks=DEFAULT_KS
        )
        print(
            f"{mode}: Hit@{top_k}={result.metrics[f'hit@{top_k}']:.4f}, "
            f"Recall@{top_k}={result.metrics[f'recall@{top_k}']:.4f}"
        )
        return result

    def _metadata(self, name: str, **values: Any) -> dict[str, Any]:
        return {
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": "financebench",
            "subset": "sec-compatible",
            **values,
            **self.suite.embedding_metadata(),
        }

    @staticmethod
    def _result_payload(result: RetrievalResult) -> dict[str, Any]:
        return {"metrics": result.metrics, "results": result.records}
