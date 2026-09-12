from __future__ import annotations

import asyncio
from typing import Any, Protocol

from pydantic import ValidationError

from apps.api_gateway.config.setting import settings
from services.daily_briefing.log import briefing_log
from services.daily_briefing.prepare import build_windows, filter_and_order_transcripts
from services.daily_briefing.schemas import (
    DailyBriefingSynthesis,
    SourceStats,
    TaskCard,
    MeetingCard,
    TranscriptWindow,
    ValidatorResult,
    WindowIntelligence,
)
from services.daily_briefing.sources import ActivityBundle, source_stats
from services.llm.router import LLMCapability, get_llm_router


class PendingTranscriptError(RuntimeError):
    pass


class StructuredCaller(Protocol):
    async def generate(
        self,
        capability: LLMCapability,
        prompt_name: str,
        schema: type,
        payload: dict[str, Any],
    ) -> Any:
        ...


class DefaultStructuredCaller:
    async def generate(self, capability, prompt_name, schema, payload):
        from services.conversation.meeting_pipeline.llm import generate_structured

        result, _provider, _model = await generate_structured(
            get_llm_router(),
            capability,
            prompt_name,
            schema,
            payload,
            stage=prompt_name,
        )
        return result


def _structured_context(bundle: ActivityBundle) -> dict[str, Any]:
    return {
        "tasks": bundle.tasks,
        "notes": bundle.notes,
        "reminders": bundle.reminders,
        "events": bundle.events,
    }


def _fallback_synthesis(date_key: str, bundle: ActivityBundle, windows: list[WindowIntelligence]) -> DailyBriefingSynthesis:
    tasks = [
        TaskCard(
            id=str(item.get("id")),
            title=str(item.get("title") or "Task"),
            meta=str(item.get("dueDate") or item.get("status") or "Task"),
        )
        for item in bundle.tasks
        if item.get("title")
    ]
    meetings = [
        MeetingCard(
            id=str(item.get("id")),
            time=str(item.get("timeLabel") or ""),
            title=str(item.get("title") or "Event"),
            meta=str(item.get("meta") or item.get("detail") or "Calendar"),
        )
        for item in [*bundle.events, *bundle.reminders]
        if item.get("title")
    ]
    overview_parts = [window.summary for window in windows if window.summary]
    if not overview_parts and bundle.notes:
        overview_parts = [str(item.get("title") or item.get("detail") or "") for item in bundle.notes[:3]]
    return DailyBriefingSynthesis(
        headline=f"Daily briefing for {date_key}",
        overview=" ".join(overview_parts)[:800],
        highlights=[item for window in windows for item in window.highlights][:8],
        importantMoments=[item for window in windows for item in window.importantMoments][:8],
        completed=[item for window in windows for item in window.completedItems][:8],
        pendingTasks=[item for window in windows for item in window.pendingItems][:8],
        decisions=[item for window in windows for item in window.decisions][:8],
        followUps=[item for window in windows for item in window.followUps][:8],
        people=sorted({person for window in windows for person in window.people}),
        topics=sorted({topic for window in windows for topic in window.topics}),
        tasks=tasks,
        meetings=meetings,
    )


class DailyBriefingPipeline:
    def __init__(self, caller: StructuredCaller | None = None):
        self.caller = caller or DefaultStructuredCaller()
        self.max_retries = settings.DAILY_BRIEFING_MAX_RETRIES
        self.window_concurrency = settings.DAILY_BRIEFING_MAX_CONCURRENCY

    async def generate(
        self,
        *,
        user_id: str,
        date_key: str,
        timezone_name: str,
        bundle: ActivityBundle,
        attempt: int = 0,
        job_id: str | None = None,
        allow_pending: bool = False,
    ) -> tuple[DailyBriefingSynthesis, SourceStats, int]:
        ordered = filter_and_order_transcripts(bundle.transcripts)
        stats = source_stats(bundle, len(ordered))
        if bundle.pendingTranscriptCount > 0 and not allow_pending and attempt < self.max_retries:
            raise PendingTranscriptError(
                f"pending_transcripts={bundle.pendingTranscriptCount}"
            )
        if not ordered and not bundle.tasks and not bundle.notes and not bundle.events and not bundle.reminders:
            return DailyBriefingSynthesis(), stats, 0

        windows = build_windows(
            ordered,
            settings.DAILY_BRIEFING_WINDOW_TARGET_TOKENS,
            settings.DAILY_BRIEFING_WINDOW_MAX_TOKENS,
        )
        analyzed = await self._analyze_windows(windows, date_key, timezone_name, bundle)
        briefing_log(
            "daily_briefing_window_complete",
            userId=user_id,
            dateKey=date_key,
            timezone=timezone_name,
            jobId=job_id,
            windowCount=len(windows),
            transcriptCount=stats.transcriptCount,
            taskCount=stats.taskCount,
            noteCount=stats.noteCount,
            retryCount=attempt,
        )
        synthesis = await self._synthesize(date_key, timezone_name, bundle, analyzed)
        briefing_log(
            "daily_briefing_synthesis_complete",
            userId=user_id,
            dateKey=date_key,
            timezone=timezone_name,
            jobId=job_id,
            windowCount=len(windows),
        )
        validated = await self._validate(date_key, timezone_name, bundle, analyzed, synthesis)
        return validated, stats, len(windows)

    async def _analyze_windows(
        self,
        windows: list[TranscriptWindow],
        date_key: str,
        timezone_name: str,
        bundle: ActivityBundle,
    ) -> list[WindowIntelligence]:
        if not windows:
            return []
        semaphore = asyncio.Semaphore(self.window_concurrency)

        async def run(window: TranscriptWindow) -> WindowIntelligence:
            async with semaphore:
                payload = {
                    "dateKey": date_key,
                    "timezone": timezone_name,
                    "windowIndex": window.index,
                    "transcripts": window.items,
                    "structuredContext": _structured_context(bundle) if window.index == 0 else {},
                }
                try:
                    result = await self.caller.generate(
                        LLMCapability.SEMANTIC_EXTRACTION,
                        "daily-briefing-window-v1",
                        WindowIntelligence,
                        payload,
                    )
                    if isinstance(result, WindowIntelligence):
                        return result
                    return WindowIntelligence.model_validate(result)
                except (ValidationError, TypeError, ValueError):
                    return WindowIntelligence(summary="")

        return list(await asyncio.gather(*(run(window) for window in windows)))

    async def _synthesize(
        self,
        date_key: str,
        timezone_name: str,
        bundle: ActivityBundle,
        windows: list[WindowIntelligence],
    ) -> DailyBriefingSynthesis:
        payload = {
            "dateKey": date_key,
            "timezone": timezone_name,
            "windows": [item.model_dump() for item in windows],
            "structuredContext": _structured_context(bundle),
        }
        try:
            result = await self.caller.generate(
                LLMCapability.FINAL_SYNTHESIS,
                "daily-briefing-synthesis-v1",
                DailyBriefingSynthesis,
                payload,
            )
            if isinstance(result, DailyBriefingSynthesis):
                return result
            return DailyBriefingSynthesis.model_validate(result)
        except (ValidationError, TypeError, ValueError):
            return _fallback_synthesis(date_key, bundle, windows)

    async def _validate(
        self,
        date_key: str,
        timezone_name: str,
        bundle: ActivityBundle,
        windows: list[WindowIntelligence],
        synthesis: DailyBriefingSynthesis,
    ) -> DailyBriefingSynthesis:
        last = synthesis
        for round_index in range(self.max_retries + 1):
            payload = {
                "dateKey": date_key,
                "timezone": timezone_name,
                "briefing": last.model_dump(),
                "windows": [item.model_dump() for item in windows],
                "structuredContext": _structured_context(bundle),
            }
            try:
                result = await self.caller.generate(
                    LLMCapability.VALIDATION,
                    "daily-briefing-validator-v1",
                    ValidatorResult,
                    payload,
                )
                report = result if isinstance(result, ValidatorResult) else ValidatorResult.model_validate(result)
            except (ValidationError, TypeError, ValueError):
                briefing_log(
                    "daily_briefing_validation_failed",
                    dateKey=date_key,
                    reasons=["malformed_validator_output"],
                    retryCount=round_index,
                )
                return last
            if report.accepted:
                return report.repaired or last
            briefing_log(
                "daily_briefing_validation_failed",
                dateKey=date_key,
                reasons=report.reasons,
                retryCount=round_index,
            )
            if report.repaired is not None:
                last = report.repaired
                continue
            if round_index < self.max_retries:
                last = await self._synthesize(date_key, timezone_name, bundle, windows)
                continue
            raise ValueError("validator_rejected")
        return last
