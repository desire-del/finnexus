import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from time import perf_counter
from typing import Any

from rag_sec.application.runtime import RAGRuntime, get_runtime
from rag_sec.config import get_settings
from rag_sec.evaluation.artifacts import read_artifact, write_artifact
from rag_sec.evaluation.models import (
    EvaluationCase,
    EvaluationRun,
    EvaluationRunMetrics,
    RetrievedEvidence,
)
from rag_sec.evaluation.reranker import CrossEncoderReranker
from rag_sec.evaluation.retrieval import (
    RetrievalEvaluator,
    RetrievalResult,
    aggregate_runs,
    contribution_diagnostics,
    deduplicated_union,
    weighted_rrf,
)
from rag_sec.evaluation.suite import FinanceBenchSuite, LoadedSuite
from rag_sec.retrieval.retriever import RetrievalMode

DEFAULT_KS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class FusionConfig:
    name: str
    dense_depth: int
    lexical_depth: int
    dense_weight: float
    lexical_weight: float


FUSION_CONFIGS = (
    FusionConfig("rrf_1_1_d20_b20", 20, 20, 1, 1),
    FusionConfig("rrf_2_1_d20_b20", 20, 20, 2, 1),
    FusionConfig("rrf_3_1_d20_b20", 20, 20, 3, 1),
    FusionConfig("rrf_2_1_d30_b20", 30, 20, 2, 1),
    FusionConfig("rrf_2_1_d30_b30", 30, 30, 2, 1),
)


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
                "financebench-dense-refactor-v1",
                cases=len(cases),
                top_k=top_k,
                dense_candidate_k=retrieval.dense_candidate_k,
                excluded_cases=self._loaded.excluded_cases,
            ),
            "metrics": result.metrics,
            "results": result.records,
        }
        write_artifact(
            self.suite.artifact_path("retrieval_dense_refactor_v1.json"), payload
        )
        return payload

    async def fts_baseline(self, *, top_k: int = 20) -> dict[str, Any]:
        cases = self._cases
        result = await self._evaluate("fts", cases, top_k=top_k)
        payload = {
            "experiment": self._metadata(
                "financebench-postgres-fts-v1",
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
            self.suite.artifact_path("retrieval_postgres_fts_v1.json"), payload
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

    async def fusion(self) -> dict[str, Any]:
        cases = self._cases
        depth, final_k, rrf_k = 30, 20, 60
        dense = await self._evaluate("dense", cases, top_k=depth)
        lexical = await self._evaluate("bm25", cases, top_k=depth)
        configurations: dict[str, dict[str, Any]] = {}
        configuration_results: dict[str, RetrievalResult] = {}

        for config in FUSION_CONFIGS:
            runs = [
                weighted_rrf(
                    dense_run,
                    lexical_run,
                    dense_depth=config.dense_depth,
                    lexical_depth=config.lexical_depth,
                    dense_weight=config.dense_weight,
                    lexical_weight=config.lexical_weight,
                    rrf_k=rrf_k,
                    top_k=final_k,
                )
                for dense_run, lexical_run in zip(dense.runs, lexical.runs, strict=True)
            ]
            metrics, records = aggregate_runs(cases, runs, ks=DEFAULT_KS)
            result = RetrievalResult(metrics=metrics, records=records, runs=runs)
            configuration_results[config.name] = result
            configurations[config.name] = {
                "parameters": config.__dict__,
                **self._result_payload(result),
            }

        best_name = max(
            configurations,
            key=lambda name: (
                configurations[name]["metrics"]["hit@20"],
                configurations[name]["metrics"]["recall@20"],
                configurations[name]["metrics"]["hit@5"],
            ),
        )
        contribution = contribution_diagnostics(
            dense,
            lexical,
            configuration_results[best_name],
            lexical_label="bm25",
        )
        payload = {
            "experiment": self._metadata(
                "financebench-bm25-fusion-v1",
                cases=len(cases),
                retrieval_depth=depth,
                final_top_k=final_k,
                rrf_k=rrf_k,
            ),
            "branches": {
                "dense_top30": self._result_payload(dense),
                "bm25_top30": self._result_payload(lexical),
            },
            "configurations": configurations,
            "best_configuration": best_name,
            "best_contribution": contribution,
        }
        write_artifact(
            self.suite.artifact_path("retriever_bm25_fusion_v1.json"), payload
        )
        return payload

    async def reranker(self) -> dict[str, Any]:
        cases = self._cases
        candidate_path = self.suite.artifact_path(
            "retriever_bm25_ablation_top20_v1.json"
        )
        artifact = read_artifact(candidate_path)
        dense_records = self._records_by_id(artifact, "dense")
        bm25_records = self._records_by_id(artifact, "bm25")
        cases = [case for case in cases if case.id in dense_records]
        dense = {case.id: self._evidence(dense_records[case.id]) for case in cases}
        union = {
            case.id: deduplicated_union(
                dense[case.id], self._evidence(bm25_records[case.id])
            )
            for case in cases
        }
        settings = get_settings().reranker
        reranker = CrossEncoderReranker(settings)
        await asyncio.to_thread(reranker.warmup)
        experiments = {
            "dense_baseline_top5": [
                EvaluationRun(case_id=case.id, retrieved_evidence=dense[case.id][:5])
                for case in cases
            ],
            "dense_reranked_top5": await self._rerank(reranker, cases, dense),
            "dense_bm25_union_reranked_top5": await self._rerank(
                reranker, cases, union
            ),
        }
        results = {}
        for name, runs in experiments.items():
            metrics, records = aggregate_runs(cases, runs, ks=(1, 3, 5))
            results[name] = {"metrics": metrics, "results": records}
        union_sizes = [len(union[case.id]) for case in cases]
        payload = {
            "experiment": self._metadata(
                "financebench-reranker-dense-vs-bm25-union-v1",
                cases=len(cases),
                candidate_artifact=candidate_path.name,
                final_top_k=5,
                reranker_model=settings.model_name,
                average_union_size=mean(union_sizes),
                minimum_union_size=min(union_sizes),
                maximum_union_size=max(union_sizes),
            ),
            "experiments": results,
        }
        write_artifact(
            self.suite.artifact_path("reranker_dense_vs_bm25_union_v1.json"), payload
        )
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

    @staticmethod
    async def _rerank(
        reranker: CrossEncoderReranker,
        cases: list[EvaluationCase],
        candidates: dict[str, list[RetrievedEvidence]],
    ) -> list[EvaluationRun]:
        runs = []
        for case in cases:
            started = perf_counter()
            evidence = await reranker.rerank(
                case.question, candidates[case.id], top_k=5
            )
            runs.append(
                EvaluationRun(
                    case_id=case.id,
                    retrieved_evidence=evidence,
                    metrics=EvaluationRunMetrics(
                        total_latency_ms=(perf_counter() - started) * 1000
                    ),
                )
            )
        return runs

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

    @staticmethod
    def _records_by_id(artifact: dict, mode: str) -> dict[str, dict]:
        return {
            record["case_id"]: record for record in artifact["modes"][mode]["results"]
        }

    @staticmethod
    def _evidence(record: dict) -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence.model_validate(value)
            for value in record["retrieved_evidence"]
        ]
