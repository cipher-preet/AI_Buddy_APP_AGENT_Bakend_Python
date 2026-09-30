"""Calendar pipeline: future meetings, calls, appointments and deadlines."""

from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.schedule_extraction.base import extract_windows
from services.schedule_extraction.context import ExtractionContext
from services.schedule_extraction.llm import ExtractionCaller
from services.schedule_extraction.schemas import EventExtraction, PipelineOutcome
from services.schedule_extraction.validation import CandidateValidator

EVENTS_PROMPT = "meeting-calendar-extractor-v1"


class CalendarEventPipeline:
    name = "calendar_events"

    def __init__(self, caller: ExtractionCaller):
        self.caller = caller

    async def run(self, ctx: ExtractionContext, validator: CandidateValidator) -> PipelineOutcome:
        candidates, outcome = await extract_windows(
            self.name,
            ctx,
            self.caller,
            EVENTS_PROMPT,
            EventExtraction,
            "events",
            settings.SCHEDULE_EXTRACTION_WINDOW_CONCURRENCY,
        )
        outcome.items = validator.events(candidates, outcome)
        return outcome
