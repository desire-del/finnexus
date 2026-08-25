"""Compatibility CLI for named FinanceBench retrieval configurations."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rag_sec.config import get_settings
from rag_sec.evaluation import (
    EvaluationResult,
    evaluate,
    retrieval_metrics,
    save_result,
)
from rag_sec.evaluation.datasets import FinanceBench
from rag_sec.observability import configure_observability, shutdown_observability
from rag_sec.retrieval.retriever import RetrievalMode

DEFAULT_KS = (1, 3, 5, 10, 20)
CLIConfig = tuple[RetrievalMode, Literal["fts", "bm25"], str, str]

CONFIGURATIONS: dict[str, CLIConfig] = {
    "baseline": (
        "dense",
        "fts",
        "financebench-dense-configured-v1",
        "retrieval_dense_configured_v1.json",
    ),
    "fts": (
        "fts",
        "fts",
        "financebench-postgres-fts-configured-v1",
        "retrieval_postgres_fts_configured_v1.json",
    ),
    "bm25": (
        "bm25",
        "fts",
        "financebench-pgsearch-bm25-configured-v1",
        "retrieval_pgsearch_bm25_configured_v1.json",
    ),
    "hybrid-fts": (
        "hybrid",
        "fts",
        "financebench-hybrid-fts-configured-v1",
        "retrieval_hybrid_fts_configured_v1.json",
    ),
    "hybrid-bm25": (
        "hybrid",
        "bm25",
        "financebench-hybrid-bm25-configured-v1",
        "retrieval_hybrid_bm25_configured_v1.json",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a named FinanceBench retrieval configuration."
    )
    parser.add_argument("configuration", choices=tuple(CONFIGURATIONS))
    return parser


async def run(configuration_name: str) -> tuple[EvaluationResult, Path]:
    mode, lexical_backend, _, filename = CONFIGURATIONS[configuration_name]
    dataset = await FinanceBench.load()
    settings = get_settings().retrieval.model_copy(
        update={
            "mode": mode,
            "top_k": 20,
            "hybrid_lexical_backend": lexical_backend,
        }
    )
    result = await evaluate(
        dataset=dataset,
        settings=settings,
        metrics=retrieval_metrics(DEFAULT_KS),
    )
    return result, save_result(result, dataset.root / filename)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure_observability()
    try:
        result, result_path = asyncio.run(run(arguments.configuration))
        print("\nMetrics:")
        for name, value in result.aggregate_metrics.items():
            print(f"{name}: {value:.4f}")

        configuration = CONFIGURATIONS[arguments.configuration]
        print("Evaluation completed:", configuration[2])
        print("Result:", result_path)
        return 0
    finally:
        shutdown_observability()

if __name__ == "__main__":
    arguments = sys.argv[1:] or ["baseline"]
    raise SystemExit(main(arguments))
