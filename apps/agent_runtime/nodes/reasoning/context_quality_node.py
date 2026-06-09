import logging
from typing import Any

from openai import AsyncOpenAI

from apps.agent_runtime.llms.prompts.task_note_orchestration_prompt import (
    QUALITY_GATE_CHAT_PROMPT,
)
from apps.agent_runtime.nodes.chunk_prompting import render_chunk_context
from apps.agent_runtime.state.task_note_state import TaskNoteState, append_error
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import ContextQualityOutput

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _average_relevance(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    return round(
        sum(float(chunk.get("relevance_score") or 0.0) for chunk in chunks)
        / len(chunks),
        4,
    )


def _high_relevance_chunk_count(chunks: list[dict[str, Any]]) -> int:
    return sum(
        1
        for chunk in chunks
        if float(chunk.get("relevance_score") or 0.0) >= 0.65
    )


def _should_generate_from_quality(
    parsed: ContextQualityOutput,
    *,
    chunks: list[dict[str, Any]],
    average_relevance: float,
) -> bool:
    """Allow generation for high-signal context even when LLM booleans are conservative."""
    if parsed.is_contradictory:
        return False

    return (
        parsed.context_quality_score >= 0.45
        or average_relevance >= 0.55
        or _high_relevance_chunk_count(chunks) >= 2
    )


async def check_context_quality(state: TaskNoteState) -> dict[str, Any]:
    """Use an LLM quality gate to decide whether generation should run."""
    chunks = state.get("reranked_chunks", [])
    best_chunks = chunks[:12]
    if not best_chunks:
        return {
            "context_quality_score": 0.0,
            "should_generate": False,
            "errors": [
                *state.get("errors", []),
                "Skipped generation: no useful context",
            ],
        }

    average_relevance = _average_relevance(best_chunks)
    messages = QUALITY_GATE_CHAT_PROMPT.format_messages(
        user_id=state["user_id"],
        space_id=state["space_id"],
        average_relevance=str(average_relevance),
        chunks=render_chunk_context(best_chunks, limit=12),
    )

    try:
        response = await client.chat.completions.parse(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_format=ContextQualityOutput,
            messages=messages,
        )
        print("response in contet quality node --->> ", response)
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise ValueError("OpenAI returned no parsed context quality output.")

        should_generate = _should_generate_from_quality(
            parsed,
            chunks=best_chunks,
            average_relevance=average_relevance,
        )
        reasons = parsed.reasons
        logger.info(
            "LLM checked context quality.",
            extra={
                "user_id": state["user_id"],
                "space_id": state["space_id"],
                "score": parsed.context_quality_score,
                "should_generate": should_generate,
                "reasons": reasons,
            },
        )

        patch: dict[str, Any] = {
            "context_quality_score": parsed.context_quality_score,
            "should_generate": should_generate,
        }
        if not should_generate:
            reason_text = ", ".join(reasons) if reasons else "LLM rejected weak context"
            patch["errors"] = [
                *state.get("errors", []),
                f"Skipped generation: {reason_text}",
            ]

        return patch
    except Exception as error:
        logger.exception(
            "LLM context quality check failed.",
            extra={"user_id": state["user_id"], "space_id": state["space_id"]},
        )
        return {
            **append_error(state, f"check_context_quality failed: {error}"),
            "context_quality_score": 0.0,
            "should_generate": False,
        }
