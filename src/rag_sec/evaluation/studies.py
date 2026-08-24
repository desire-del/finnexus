from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from rag_sec.application.runtime import RAGRuntime, get_runtime
from rag_sec.config import RetrievalSettings, get_settings
from rag_sec.evaluation.artifacts import write_artifact
from rag_sec.evaluation.models import EvaluationCase
from rag_sec.evaluation.retrieval import RetrievalEvaluator
from rag_sec.evaluation.suite import FinanceBenchSuite, LoadedSuite
from rag_sec.retrieval.retriever import RetrievalMode

DEFAULT_KS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class RetrievalExperimentConfig:
    """Values that distinguish one retrieval experiment."""

    name: str
    artifact_name: str
    mode: RetrievalMode
    lexical_backend: Literal["fts", "bm25"] = "fts"
    top_k: int = 20

    def retrieval_settings(self) -> RetrievalSettings:
        return get_settings().retrieval.model_copy(
            update={
                "mode": self.mode,
                "hybrid_lexical_backend": self.lexical_backend,
            }
        )


class FinanceBenchStudies:
    """Evaluate configured production retrieval against FinanceBench."""

    def __init__(
        self,
        suite: FinanceBenchSuite | None = None,
        runtime: RAGRuntime | None = None,
    ) -> None:
        self.suite = suite or FinanceBenchSuite()
        self.runtime = runtime or get_runtime()
        self.evaluator = RetrievalEvaluator(self.runtime)
        self.loaded: LoadedSuite | None = None

    async def initialize(self) -> list[EvaluationCase]:
        self.loaded = await self.suite.load()
        await self.suite.validate_corpus(self.loaded.cases)
        await self.runtime.warmup_retrieval()
        return self.loaded.cases

    async def close(self) -> None:
        await self.runtime.shutdown()

    async def run(self, experiment: RetrievalExperimentConfig) -> dict[str, Any]:
        retrieval = experiment.retrieval_settings()
        self.runtime.retriever.settings = retrieval

        backend = (
            f" with {experiment.lexical_backend}" if experiment.mode == "hybrid" else ""
        )
        print(f"Evaluating {experiment.mode} retrieval{backend}...")
        result = await self.evaluator.evaluate(
            self._loaded.cases,
            top_k=experiment.top_k,
            ks=DEFAULT_KS,
        )
        print(
            f"{experiment.name}: "
            f"Hit@{experiment.top_k}="
            f"{result.metrics[f'hit@{experiment.top_k}']:.4f}, "
            f"Recall@{experiment.top_k}="
            f"{result.metrics[f'recall@{experiment.top_k}']:.4f}"
        )

        payload = {
            "experiment": self._metadata(experiment, retrieval),
            "metrics": result.metrics,
            "results": result.records,
        }
        write_artifact(self.suite.artifact_path(experiment.artifact_name), payload)
        return payload

    @property
    def _loaded(self) -> LoadedSuite:
        if self.loaded is None:
            raise RuntimeError("Initialize FinanceBenchStudies before evaluation.")
        return self.loaded

    def _metadata(
        self,
        experiment: RetrievalExperimentConfig,
        retrieval: RetrievalSettings,
    ) -> dict[str, Any]:
        return {
            "name": experiment.name,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": "financebench",
            "subset": "sec-compatible",
            "cases": len(self._loaded.cases),
            "excluded_cases": self._loaded.excluded_cases,
            "top_k": experiment.top_k,
            "retrieval": retrieval.model_dump(mode="json"),
            **self.suite.embedding_metadata(),
        }


def experiment(
    *,
    name: str,
    artifact_name: str,
    mode: RetrievalMode,
    lexical_backend: Literal["fts", "bm25"] = "fts",
) -> RetrievalExperimentConfig:
    return RetrievalExperimentConfig(
        name=name,
        artifact_name=artifact_name,
        mode=mode,
        lexical_backend=lexical_backend,
    )
