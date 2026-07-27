import asyncio
import logging

from apps.api_gateway.config.setting import settings
from services.queue.redis_queue import (
    iter_idle_transcript_sessions,
    mark_transcript_session_ended,
    maybe_queue_transcript_analysis_for_session,
)

logger = logging.getLogger(__name__)


async def start_transcript_session_consumer():
    """End idle listening sessions and trigger transcript analysis once."""
    logger.info("Transcript session worker started.")

    while True:
        try:
            sessions = await iter_idle_transcript_sessions(
                idle_seconds=settings.TRANSCRIPT_SESSION_IDLE_SECONDS
            )
            for session in sessions:
                user_id = str(session.get("user_id") or "").strip()
                space_id = str(session.get("space_id") or "").strip()
                if not user_id or not space_id:
                    continue

                ended_session = await mark_transcript_session_ended(user_id, space_id)
                queued = await maybe_queue_transcript_analysis_for_session(
                    user_id=user_id,
                    space_id=space_id,
                    reason="idle_timeout",
                    session=ended_session,
                )
                if queued:
                    logger.info(
                        "Queued transcript analysis after idle timeout.",
                        extra={"user_id": user_id, "space_id": space_id},
                    )
        except Exception:
            logger.exception("Transcript session worker error.")

        await asyncio.sleep(settings.TRANSCRIPT_SESSION_IDLE_CHECK_SECONDS)
