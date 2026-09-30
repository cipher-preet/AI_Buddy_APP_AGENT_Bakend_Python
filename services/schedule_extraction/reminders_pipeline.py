"""Reminder pipeline: explicit "remind me / don't forget" requests with a moment attached."""

from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.schedule_extraction.base import extract_windows
from services.schedule_extraction.context import ExtractionContext
from services.schedule_extraction.llm import ExtractionCaller
from services.schedule_extraction.schemas import PipelineOutcome, ReminderExtraction
from services.schedule_extraction.validation import CandidateValidator

REMINDERS_PROMPT = "meeting-reminder-extractor-v1"


class ReminderPipeline:
    name = "reminders"

    def __init__(self, caller: ExtractionCaller):
        self.caller = caller

    async def run(self, ctx: ExtractionContext, validator: CandidateValidator) -> PipelineOutcome:
        candidates, outcome = await extract_windows(
            self.name,
            ctx,
            self.caller,
            REMINDERS_PROMPT,
            ReminderExtraction,
            "reminders",
            settings.SCHEDULE_EXTRACTION_WINDOW_CONCURRENCY,
        )
        outcome.items = validator.reminders(candidates, outcome)
        return outcome
