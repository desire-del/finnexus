from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from rag_sec.generation.models import TokenUsage


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def extract_token_usage(
    raw_message,
    messages: Sequence[BaseMessage],
    answer: str,
) -> TokenUsage:
    usage = getattr(raw_message, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is not None and output_tokens is not None:
        total_tokens = usage.get("total_tokens")
        return TokenUsage(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(total_tokens or input_tokens + output_tokens),
        )

    estimated_input = estimate_token_count(
        "\n".join(str(message.content) for message in messages)
    )
    estimated_output = estimate_token_count(answer)
    return TokenUsage(
        input_tokens=estimated_input,
        output_tokens=estimated_output,
        total_tokens=estimated_input + estimated_output,
        estimated=True,
    )
