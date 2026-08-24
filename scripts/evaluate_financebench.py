"""Compatibility entrypoint for the dense FinanceBench baseline."""

import sys

from rag_sec.evaluation.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:] or ["baseline"]
    raise SystemExit(main(arguments))
