from __future__ import annotations

import time
from apps.api_gateway.config.setting import settings
from services.daily_briefing.log import briefing_log
from services.daily_briefing.pipeline import DailyBriefingPipeline, PendingTranscriptError
from services.daily_briefing.prepare import filter_and_order_transcripts
from services.daily_briefing.schemas import SourceStats
from services.daily_briefing.sources import ActivitySource
from services.daily_briefing.store import DailyBriefingStore
from services.daily_briefing.timezones import local_day_bounds_utc
from services.observability.diagnostics import briefing_duplicates, diag_log
from services.queue.streams import EventEnvelope, NonRetryableQueueError


class DailyBriefingJobHandler:
    def __init__(self, database, pipeline: DailyBriefingPipeline | None = None):
        self.database = database
        self.store = DailyBriefingStore(database)
        self.sources = ActivitySource(database)
        self.pipeline = pipeline or DailyBriefingPipeline()

    async def handle(self, event: EventEnvelope) -> None:
        payload = event.payload or {}
        user_id = event.userId
        date_key = str(payload.get("dateKey") or "")
        timezone_name = str(payload.get("timezone") or "")
        if not user_id or not date_key or not timezone_name:
            raise NonRetryableQueueError("daily briefing event is missing userId/dateKey/timezone")
        first_event_id = briefing_duplicates.begin(user_id, date_key, event.eventId)
        if first_event_id:
            diag_log(
                "duplicate_briefing_execution_detected",
                user_id=user_id,
                date_key=date_key,
                first_event_id=first_event_id,
                second_event_id=event.eventId,
            )
        period_start, period_end = local_day_bounds_utc(date_key, timezone_name)
        started = time.perf_counter()
        finish_status = "started"
        transcript_count = None
        window_count = None
        briefing_log(
            "daily_briefing_started",
            userId=user_id,
            dateKey=date_key,
            timezone=timezone_name,
            jobId=event.eventId,
            retryCount=event.attempt,
        )
        diag_log(
            "daily_briefing_job_start",
            event_id=event.eventId,
            user_id=user_id,
            date_key=date_key,
            attempt=event.attempt,
        )
        try:
            claim_state, _doc = await self.store.claim(
                user_id,
                date_key,
                timezone_name,
                period_start,
                period_end,
                event.eventId,
            )
            if claim_state in {"exists", "busy"}:
                finish_status = claim_state
                return
            try:
                bundle = await self.sources.load(user_id, date_key, period_start, period_end)
                allow_pending = event.attempt >= settings.DAILY_BRIEFING_MAX_RETRIES
                useful_transcripts = filter_and_order_transcripts(bundle.transcripts)
                has_activity = bool(
                    useful_transcripts
                    or bundle.tasks
                    or bundle.notes
                    or bundle.events
                    or bundle.reminders
                )
                if bundle.pendingTranscriptCount > 0 and not allow_pending:
                    raise PendingTranscriptError(
                        f"pending_transcripts={bundle.pendingTranscriptCount}"
                    )
                if not has_activity:
                    stats = SourceStats(pendingTranscriptCount=bundle.pendingTranscriptCount)
                    await self.store.mark_skipped(user_id, date_key, "no_activity", stats)
                    briefing_log(
                        "daily_briefing_skipped_no_activity",
                        userId=user_id,
                        dateKey=date_key,
                        timezone=timezone_name,
                        jobId=event.eventId,
                    )
                    finish_status = "SKIPPED"
                    transcript_count = 0
                    return
                synthesis, stats, window_count = await self.pipeline.generate(
                    user_id=user_id,
                    date_key=date_key,
                    timezone_name=timezone_name,
                    bundle=bundle,
                    attempt=event.attempt,
                    job_id=event.eventId,
                    allow_pending=allow_pending,
                )
                await self.store.save_ready(
                    user_id,
                    date_key,
                    timezone_name,
                    period_start,
                    period_end,
                    synthesis,
                    stats,
                )
                transcript_count = stats.transcriptCount
                finish_status = "READY"
                briefing_log(
                    "daily_briefing_completed",
                    userId=user_id,
                    dateKey=date_key,
                    timezone=timezone_name,
                    jobId=event.eventId,
                    windowCount=window_count,
                    transcriptCount=stats.transcriptCount,
                    taskCount=stats.taskCount,
                    noteCount=stats.noteCount,
                    duration=int((time.perf_counter() - started) * 1000),
                    retryCount=event.attempt,
                )
            except PendingTranscriptError as error:
                finish_status = "pending_transcripts"
                briefing_log(
                    "daily_briefing_failed",
                    userId=user_id,
                    dateKey=date_key,
                    timezone=timezone_name,
                    jobId=event.eventId,
                    retryCount=event.attempt,
                    error="pending_transcripts",
                )
                diag_log(
                    "daily_briefing_job_failed",
                    event_id=event.eventId,
                    user_id=user_id,
                    date_key=date_key,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    exception_type=type(error).__name__,
                    retryable=True,
                )
                raise error
            except Exception as error:
                finish_status = "FAILED"
                await self.store.mark_failed(user_id, date_key, str(error))
                briefing_log(
                    "daily_briefing_failed",
                    userId=user_id,
                    dateKey=date_key,
                    timezone=timezone_name,
                    jobId=event.eventId,
                    retryCount=event.attempt,
                    error=type(error).__name__,
                )
                diag_log(
                    "daily_briefing_job_failed",
                    event_id=event.eventId,
                    user_id=user_id,
                    date_key=date_key,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    exception_type=type(error).__name__,
                    retryable=not isinstance(error, NonRetryableQueueError),
                )
                raise
        finally:
            briefing_duplicates.end(user_id, date_key, event.eventId)
            if finish_status not in {"pending_transcripts", "FAILED"}:
                diag_log(
                    "daily_briefing_job_finish",
                    event_id=event.eventId,
                    user_id=user_id,
                    date_key=date_key,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    status=finish_status,
                    transcript_count=transcript_count,
                    window_count=window_count,
                )
