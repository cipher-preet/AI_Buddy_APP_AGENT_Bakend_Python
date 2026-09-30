"""v2 daily briefing: rich context -> (optional window digests) -> Krutrim synthesis -> grounding."""

from __future__ import annotations

import asyncio
import time

from apps.api_gateway.config.setting import settings
from services.conversation.transcript import estimate_tokens
from services.daily_briefing.briefing_llm import (
    STAGE_SYNTHESIS,
    STAGE_WINDOW,
    BriefingCaller,
    KrutrimBriefingCaller,
)
from services.daily_briefing.context import BriefingContext, build_context, timeline_payload
from services.daily_briefing.grounding import fallback_draft, ground_briefing
from services.daily_briefing.log import briefing_log
from services.daily_briefing.pipeline import PendingTranscriptError
from services.daily_briefing.prepare import build_windows
from services.daily_briefing.schemas import (
    ActivityBundle,
    BriefingDraft,
    DailyBriefingSynthesis,
    SourceStats,
    WindowDigest,
)
from services.daily_briefing.sources import source_stats

PIPELINE_VERSION_V2 = "daily-briefing-v2"
WINDOW_PROMPT = "daily-briefing-window-v2"
SYNTHESIS_PROMPT = "daily-briefing-synthesis-v2"


def _within_budget(items: list[dict], budget: int) -> list[dict]:
    kept: list[dict] = []
    used = 0
    for item in items:
        tokens = estimate_tokens(item["text"])
        if kept and used + tokens > budget:
            break
        kept.append(item)
        used += tokens
    return kept


class DailyBriefingPipelineV2:
    version = PIPELINE_VERSION_V2

    def __init__(self, caller: BriefingCaller | None = None):
        self.caller = caller or KrutrimBriefingCaller()
        self.max_retries = settings.DAILY_BRIEFING_MAX_RETRIES
        self.window_concurrency = max(1, settings.DAILY_BRIEFING_MAX_CONCURRENCY)

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
        if bundle.pendingTranscriptCount > 0 and not allow_pending and attempt < self.max_retries:
            raise PendingTranscriptError(f"pending_transcripts={bundle.pendingTranscriptCount}")

        started = time.perf_counter()
        ctx = build_context(bundle, date_key, timezone_name, settings.DAILY_BRIEFING_OPEN_TASK_LIMIT)
        stats = source_stats(bundle, sum(1 for item in ctx.timeline if item["source"] == "voice"))
        if ctx.empty:
            return DailyBriefingSynthesis(stats=ctx.stats, planDateKey=ctx.plan_date_key), stats, 0

        timeline_tokens = sum(estimate_tokens(item["text"]) for item in ctx.timeline)
        digests: list[WindowDigest] = []
        window_count = 0
        if timeline_tokens > settings.DAILY_BRIEFING_DIRECT_SYNTHESIS_TOKENS:
            windows = build_windows(
                ctx.timeline,
                settings.DAILY_BRIEFING_WINDOW_TARGET_TOKENS,
                settings.DAILY_BRIEFING_WINDOW_MAX_TOKENS,
            )
            window_count = len(windows)
            digests = await self._digest_windows(ctx, [window.items for window in windows])

        draft = await self._synthesize(ctx, digests, windowed=window_count > 0)
        used_fallback = draft is None
        synthesis = ground_briefing(draft or fallback_draft(ctx, digests), ctx, digests)
        briefing_log(
            "daily_briefing_v2_complete",
            userId=user_id,
            dateKey=date_key,
            timezone=timezone_name,
            jobId=job_id,
            windowCount=window_count,
            timelineItems=len(ctx.timeline),
            timelineTokens=timeline_tokens,
            openTasks=len(ctx.open_tasks),
            agendaItems=len(ctx.agenda),
            fallback=used_fallback,
            duration=int((time.perf_counter() - started) * 1000),
            retryCount=attempt,
        )
        return synthesis, stats, window_count

    async def _digest_windows(self, ctx: BriefingContext, windows: list[list[dict]]) -> list[WindowDigest]:
        semaphore = asyncio.Semaphore(self.window_concurrency)
        known_tasks = [{"ref": task["ref"], "title": task["title"]} for task in ctx.open_tasks]

        async def run(index: int, items: list[dict]) -> WindowDigest | None:
            payload = {
                "dateKey": ctx.date_key,
                "timezone": ctx.timezone_name,
                "windowIndex": index,
                "windowCount": len(windows),
                "knownTasks": known_tasks,
                "timeline": timeline_payload(items),
            }
            async with semaphore:
                try:
                    result = await self.caller.generate(STAGE_WINDOW, WINDOW_PROMPT, WindowDigest, payload)
                    return result if isinstance(result, WindowDigest) else WindowDigest.model_validate(result)
                except Exception as error:
                    briefing_log(
                        "daily_briefing_window_failed",
                        dateKey=ctx.date_key,
                        windowIndex=index,
                        error=f"{type(error).__name__}: {str(error)[:200]}",
                    )
                    return None

        results = await asyncio.gather(*(run(index, items) for index, items in enumerate(windows)))
        return [item for item in results if item is not None]

    async def _synthesize(self, ctx: BriefingContext, digests: list[WindowDigest], windowed: bool) -> BriefingDraft | None:
        payload = ctx.facts()
        if digests:
            payload["windowDigests"] = [digest.model_dump(exclude_defaults=True) for digest in digests]
        elif windowed:
            payload["timeline"] = timeline_payload(_within_budget(ctx.timeline, settings.DAILY_BRIEFING_DIRECT_SYNTHESIS_TOKENS))
        else:
            payload["timeline"] = timeline_payload(ctx.timeline)
        try:
            result = await self.caller.generate(STAGE_SYNTHESIS, SYNTHESIS_PROMPT, BriefingDraft, payload)
            return result if isinstance(result, BriefingDraft) else BriefingDraft.model_validate(result)
        except Exception as error:
            briefing_log(
                "daily_briefing_synthesis_failed",
                dateKey=ctx.date_key,
                error=f"{type(error).__name__}: {str(error)[:200]}",
            )
            return None
