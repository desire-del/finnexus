"""Compatibility entrypoint for the dense FinanceBench baseline."""

from rag_sec.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["baseline"]))
