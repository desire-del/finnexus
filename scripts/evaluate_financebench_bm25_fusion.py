"""Compatibility entrypoint for weighted BM25 fusion tuning."""

from rag_sec.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fusion"]))
