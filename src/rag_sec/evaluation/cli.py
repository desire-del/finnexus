import argparse
import asyncio
from collections.abc import Sequence
from typing import Any

from rag_sec.evaluation.studies import FinanceBenchStudies, experiment
from rag_sec.observability import configure_observability, shutdown_observability

EXPERIMENTS = {
    "baseline": experiment(
        name="financebench-dense-configured-v1",
        artifact_name="retrieval_dense_configured_v1.json",
        mode="dense",
    ),
    "fts": experiment(
        name="financebench-postgres-fts-configured-v1",
        artifact_name="retrieval_postgres_fts_configured_v1.json",
        mode="fts",
    ),
    "bm25": experiment(
        name="financebench-pgsearch-bm25-configured-v1",
        artifact_name="retrieval_pgsearch_bm25_configured_v1.json",
        mode="bm25",
    ),
    "hybrid-fts": experiment(
        name="financebench-hybrid-fts-configured-v1",
        artifact_name="retrieval_hybrid_fts_configured_v1.json",
        mode="hybrid",
        lexical_backend="fts",
    ),
    "hybrid-bm25": experiment(
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


async def run(experiment: str) -> dict[str, Any]:
    studies = FinanceBenchStudies()
    try:
        await studies.initialize()
        return await studies.run(EXPERIMENTS[experiment])
    finally:
        await studies.close()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configure_observability()
    try:
        payload = asyncio.run(run(arguments.experiment))
        print("\nMetrics:")
        metrics = payload.get("metrics")
        if metrics:
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
        print("Experiment completed:", payload["experiment"]["name"])
        return 0
    finally:
        shutdown_observability()


if __name__ == "__main__":
    raise SystemExit(main())
