from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from services.schedule_extraction.context import ExtractionContext
from services.schedule_extraction.llm import ExtractionCaller
from services.schedule_extraction.log import extraction_log
from services.schedule_extraction.schemas import PipelineOutcome


async def extract_windows(
    name: str,
    ctx: ExtractionContext,
    caller: ExtractionCaller,
    prompt: str,
    schema: type[BaseModel],
    field: str,
    concurrency: int,
) -> tuple[list[Any], PipelineOutcome]:
    """Run one prompt over every transcript window; a failed window never sinks the others."""
    outcome = PipelineOutcome(name=name, windows=len(ctx.windows))
    semaphore = asyncio.Semaphore(max(1, concurrency))
    anchor = ctx.anchor()

    async def run(index: int, text: str) -> list[Any] | None:
        payload = {**anchor, "windowIndex": index, "windowCount": len(ctx.windows), "transcript": text}
        async with semaphore:
            try:
                result = await caller.generate(prompt, schema, payload)
            except Exception as error:
                extraction_log(
                    "schedule_extraction_window_failed",
                    pipeline=name,
                    conversationId=ctx.conversation_id,
                    windowIndex=index,
                    error=f"{type(error).__name__}: {str(error)[:200]}",
                )
                return None
        return list(getattr(result, field, []) or [])

    results = await asyncio.gather(*(run(index, text) for index, text in enumerate(ctx.windows)))
    candidates: list[Any] = []
    for items in results:
        if items is None:
            outcome.failedWindows += 1
        else:
            candidates.extend(items)
    outcome.rawCandidates = len(candidates)
    if outcome.failed:
        outcome.error = "all_windows_failed"
    return candidates, outcome
