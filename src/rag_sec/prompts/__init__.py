from .loader import load_prompt, render_prompt

GENERATION_SYSTEM_PROMPT = load_prompt(
    "generation",
    "system.md",
)


def build_generation_user_prompt(
    *,
    question: str,
    context: str,
) -> str:
    return render_prompt(
        "generation",
        "user.md",
        question=question,
        context=context,
    )


__all__ = [
    "GENERATION_SYSTEM_PROMPT",
    "build_generation_user_prompt",
    "load_prompt",
    "render_prompt",
]
