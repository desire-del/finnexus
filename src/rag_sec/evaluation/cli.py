import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rag_sec.config import RetrievalSettings, get_settings
from rag_sec.evaluation.artifacts import save_result
from rag_sec.evaluation.datasets import FinanceBench
from rag_sec.evaluation.evaluation import EvaluationResult, evaluate
from rag_sec.evaluation.metrics import retrieval_metrics
from rag_sec.observability import configure_observability, shutdown_observability
from rag_sec.retrieval.retriever import RetrievalMode

DEFAULT_KS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class CLIExperiment:
    """Backward-compatible name for one configuration-driven CLI run."""

    name: str
    artifact_name: str
    mode: RetrievalMode
    lexical_backend: Literal["fts", "bm25"] = "fts"

    def settings(self) -> RetrievalSettings:
        return get_settings().retrieval.model_copy(
            update={
                "mode": self.mode,
                "top_k": 20,
                "hybrid_lexical_backend": self.lexical_backend,
            }
        )

EXPERIMENTS = {
    "baseline": CLIExperiment(
        name="financebench-dense-configured-v1",
        artifact_name="retrieval_dense_configured_v1.json",
        mode="dense",
    ),
    "fts": CLIExperiment(
        name="financebench-postgres-fts-configured-v1",
        artifact_name="retrieval_postgres_fts_configured_v1.json",
        mode="fts",
    ),
    "bm25": CLIExperiment(
        name="financebench-pgsearch-bm25-configured-v1",
        artifact_name="retrieval_pgsearch_bm25_configured_v1.json",
        mode="bm25",
    ),
    "hybrid-fts": CLIExperiment(
        name="financebench-hybrid-fts-configured-v1",
        artifact_name="retrieval_hybrid_fts_configured_v1.json",
        mode="hybrid",
        lexical_backend="fts",
    ),
    "hybrid-bm25": CLIExperiment(
        name="financebench-hybrid-bm25-configured-v1",
        artifact_name="retrieval_hybrid_bm25_configured_v1.json",
        mode="hybrid",
        lexical_backend="bm25",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible FinanceBench retrieval studies."
    )
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    return parser


async def run(experiment: str) -> tuple[EvaluationResult, Path]:
    """Run and explicitly persist one legacy named CLI configuration."""
    configuration = EXPERIMENTS[experiment]
    dataset = await FinanceBench.load()
    result = await evaluate(
        dataset=dataset,
        settings=configuration.settings(),
        metrics=retrieval_metrics(DEFAULT_KS),
    )
    path = save_result(result, dataset.root / configuration.artifact_name)
    return result, path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure_observability()
    try:
        result, artifact_path = asyncio.run(run(arguments.experiment))
        print("\nMetrics:")
        for name, value in result.aggregate_metrics.items():
            print(f"{name}: {value:.4f}")
        print("Experiment completed:", EXPERIMENTS[arguments.experiment].name)
        print("Artifact:", artifact_path)
        return 0
    finally:
        shutdown_observability()


if __name__ == "__main__":
    raise SystemExit(main())
