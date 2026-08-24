import json
from pathlib import Path
from typing import Any


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist an evaluation artifact."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
