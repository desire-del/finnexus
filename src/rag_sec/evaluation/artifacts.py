import json
from pathlib import Path
from typing import Any


def read_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist an evaluation artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
