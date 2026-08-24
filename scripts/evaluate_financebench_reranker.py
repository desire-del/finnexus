"""Compatibility entrypoint for the FinanceBench reranker study."""

from rag_sec.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reranker"]))
