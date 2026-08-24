"""Compatibility entrypoint for the PostgreSQL FTS ablation."""

from rag_sec.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fts-ablation"]))
