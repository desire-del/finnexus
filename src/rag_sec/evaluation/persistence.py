from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_sec.evaluation.evaluation import EvaluationResult


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_result(result: EvaluationResult, path: str | Path) -> Path:
    """Explicitly persist an evaluation result as JSON without overwriting."""
    result_path = Path(path)
    payload = {
        "schema_version": 1,
        "dataset": result.dataset_metadata,
        "settings": result.settings.model_dump(mode="json"),
        "metrics": result.aggregate_metrics,
        "results": [
            {
                "case": case_result.case.model_dump(mode="json"),
                "run": case_result.run.model_dump(mode="json"),
                "scores": [
                    score.model_dump(mode="json") for score in case_result.scores
                ],
            }
            for case_result in result.cases
        ],
    }
    _write_json(result_path, payload)
    return result_path
