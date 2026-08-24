"""Compatibility entrypoint for the BM25 ablation."""

from rag_sec.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["bm25-ablation"]))
