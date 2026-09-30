"""Orchestrates the calendar + reminder pipelines for one finished recording."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError

from apps.api_gateway.config.setting import settings
from services.schedule_extraction.context import load_context
from services.schedule_extraction.events_pipeline import CalendarEventPipeline
from services.schedule_extraction.llm import ExtractionCaller, RoutedExtractionCaller
from services.schedule_extraction.log import extraction_log
from services.schedule_extraction.reminder_schedule import ReminderScheduler
from services.schedule_extraction.reminders_pipeline import ReminderPipeline
from services.schedule_extraction.schemas import PipelineOutcome
from services.schedule_extraction.validation import CandidateValidator, drop_reminders_covered_by_events
from services.schedule_extraction.writer import ScheduleWriter

CLAIM_STALE = timedelta(minutes=15)
TERMINAL = {"READY", "SKIPPED"}


class RetryableExtractionError(RuntimeError):
    pass


class ScheduleExtractionJob:
    def __init__(
        self,
        repository,
        database,
        caller: ExtractionCaller | None = None,
        scheduler: ReminderScheduler | None = None,
        now: datetime | None = None,
    ):
        self.repository = repository
        self.db = database
        self.jobs = database.schedule_extractions
        caller = caller or RoutedExtractionCaller()
        self.events = CalendarEventPipeline(caller)
        self.reminders = ReminderPipeline(caller)
        self.scheduler = scheduler or ReminderScheduler()
        self.now = now

    def _now(self) -> datetime:
        return self.now or datetime.now(timezone.utc)

    async def _claim(self, conversation_id: str, job_id: str) -> tuple[str, dict[str, Any]]:
        now = self._now()
        existing = await self.jobs.find_one({"conversationId": conversation_id})
        if existing and existing.get("status") in TERMINAL:
            return "exists", existing
        claimed_at = (existing or {}).get("claimedAt")
        if isinstance(claimed_at, datetime) and claimed_at.tzinfo is None:
            claimed_at = claimed_at.replace(tzinfo=timezone.utc)
        if (
            existing
            and existing.get("status") == "PROCESSING"
            and existing.get("claimedBy") != job_id
            and isinstance(claimed_at, datetime)
            and claimed_at > now - CLAIM_STALE
        ):
            return "busy", existing
        update = {"status": "PROCESSING", "claimedBy": job_id, "claimedAt": now, "updatedAt": now}
        if existing is None:
            doc = {"conversationId": conversation_id, "completedPipelines": [], "attempts": 1, "createdAt": now, **update}
            try:
                await self.jobs.insert_one(doc)
            except DuplicateKeyError:
                return "busy", {}
            return "claimed", doc
        await self.jobs.update_one(
            {"conversationId": conversation_id},
            {"$set": {**update, "attempts": int(existing.get("attempts") or 0) + 1}},
        )
        return "claimed", {**existing, **update}

    async def _finish(self, conversation_id: str, status: str, **fields: Any) -> None:
        await self.jobs.update_one(
            {"conversationId": conversation_id},
            {"$set": {"status": status, "updatedAt": self._now(), **fields}},
        )

    async def run(self, conversation_id: str, job_id: str, final_attempt: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        state, job = await self._claim(conversation_id, job_id)
        if state != "claimed":
            return {"status": state}

        ctx = await load_context(
            self.repository,
            self.db,
            conversation_id,
            settings.SCHEDULE_EXTRACTION_WINDOW_TOKENS,
            settings.SCHEDULE_EXTRACTION_MAX_WINDOWS,
        )
        if ctx is None or not ctx.windows:
            reason = "conversation_missing" if ctx is None else "empty_transcript"
            await self._finish(conversation_id, "SKIPPED", skipReason=reason)
            return {"status": "SKIPPED", "reason": reason}

        done = set(job.get("completedPipelines") or [])
        validator = CandidateValidator(
            conversation_id,
            ctx.transcript,
            ctx.recorded_at_local,
            ctx.timezone_name,
            settings.SCHEDULE_EXTRACTION_MIN_CONFIDENCE,
            settings.SCHEDULE_EXTRACTION_MAX_ITEMS,
            now=self._now(),
        )
        pipelines = [pipeline for pipeline in (self.events, self.reminders) if pipeline.name not in done]
        results = await asyncio.gather(*(pipeline.run(ctx, validator) for pipeline in pipelines), return_exceptions=True)
        outcomes: dict[str, PipelineOutcome] = {}
        for pipeline, result in zip(pipelines, results):
            if isinstance(result, BaseException):
                result = PipelineOutcome(name=pipeline.name, error=f"{type(result).__name__}: {str(result)[:200]}")
            outcomes[pipeline.name] = result

        writer = ScheduleWriter(
            self.db,
            self.scheduler,
            ai_calling=settings.SCHEDULE_EXTRACTION_AI_CALLING,
            meeting_remind_before=settings.SCHEDULE_EXTRACTION_MEETING_REMIND_BEFORE_MINUTES,
            deadline_remind_before=settings.SCHEDULE_EXTRACTION_DEADLINE_REMIND_BEFORE_MINUTES,
            now=self._now(),
        )
        written: dict[str, list[str]] = {}
        events_outcome = outcomes.get(self.events.name)
        if events_outcome and not events_outcome.error:
            written["eventIds"] = await writer.write_events(
                ctx.user_id, conversation_id, ctx.timezone_name, events_outcome.items, events_outcome
            )
            done.add(self.events.name)
        reminders_outcome = outcomes.get(self.reminders.name)
        if reminders_outcome and not reminders_outcome.error:
            items = drop_reminders_covered_by_events(
                reminders_outcome.items,
                events_outcome.items if events_outcome and not events_outcome.error else [],
                reminders_outcome,
            )
            written["reminderIds"] = await writer.write_reminders(
                ctx.user_id, conversation_id, ctx.timezone_name, items, reminders_outcome
            )
            done.add(self.reminders.name)

        stats = {
            name: {
                "windows": outcome.windows,
                "failedWindows": outcome.failedWindows,
                "candidates": outcome.rawCandidates,
                "kept": len(outcome.items),
                "dropped": outcome.dropped,
                "error": outcome.error,
            }
            for name, outcome in outcomes.items()
        }
        failed = [name for name, outcome in outcomes.items() if outcome.error]
        duration = int((time.perf_counter() - started) * 1000)
        extraction_log(
            "schedule_extraction_completed",
            conversationId=conversation_id,
            userId=ctx.user_id,
            jobId=job_id,
            windows=len(ctx.windows),
            transcriptTruncated=ctx.truncated,
            failedPipelines=failed,
            stats=stats,
            written={key: len(value) for key, value in written.items()},
            duration=duration,
        )
        status = "READY" if not failed else ("FAILED" if final_attempt else "RETRY_PENDING")
        await self._finish(
            conversation_id,
            status,
            completedPipelines=sorted(done),
            userId=ctx.user_id,
            stats=stats,
            durationMs=duration,
            **{key: value for key, value in written.items()},
        )
        if failed and not final_attempt:
            raise RetryableExtractionError(f"schedule_extraction_failed pipelines={failed}")
        return {"status": status, **written, "stats": stats}
