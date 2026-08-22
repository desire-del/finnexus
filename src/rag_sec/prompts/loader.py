from functools import cache
from importlib.resources import files


@cache
def load_prompt(*path: str) -> str:
    prompt = files("rag_sec.prompts").joinpath(*path)

    if not prompt.is_file():
        raise FileNotFoundError(
            f"Prompt file not found: {'/'.join(path)}"
        )

    return prompt.read_text(encoding="utf-8").strip()


def render_prompt(
    *path: str,
    **variables: str,
) -> str:
    template = load_prompt(*path)

    try:
        return template.format(**variables)
    except KeyError as exc:
        missing_variable = exc.args[0]
        raise ValueError(
            "Missing variable "
            f"'{missing_variable}' for prompt "
            f"{'/'.join(path)}."
        ) from exc
