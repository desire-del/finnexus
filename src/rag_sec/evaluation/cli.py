import argparse
import asyncio
from collections.abc import Sequence
from typing import Any

from rag_sec.evaluation.studies import FinanceBenchStudies
from rag_sec.observability import configure_observability, shutdown_observability

EXPERIMENTS = ("baseline", "fts-ablation", "bm25-ablation", "fusion", "reranker")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible FinanceBench retrieval studies."
    )
    parser.add_argument("experiment", choices=EXPERIMENTS)
    return parser


async def run(experiment: str) -> dict[str, Any]:
    studies = FinanceBenchStudies()
    try:
        await studies.initialize(require_retrieval=experiment != "reranker")
        if experiment == "baseline":
            return await studies.baseline()
        if experiment == "fts-ablation":
            return await studies.ablation(lexical="lexical")
        if experiment == "bm25-ablation":
            return await studies.ablation(lexical="bm25")
        if experiment == "fusion":
            return await studies.fusion()
        return await studies.reranker()
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
