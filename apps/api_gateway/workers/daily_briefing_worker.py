"""Daily briefing is not part of the worker process.

Meeting processing, speech, and the other queues are unchanged. Importing
this module must not scan users or call a model.
"""

from __future__ import annotations

from services.queue.streams import EventEnvelope


async def handle_daily_briefing_event(event: EventEnvelope) -> None:
    del event


async def run_daily_briefing_scheduler() -> None:
    return
