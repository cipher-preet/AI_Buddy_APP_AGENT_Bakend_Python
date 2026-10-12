from __future__ import annotations

import json
from typing import Any

from apps.api_gateway.config.setting import settings
from services.goals.redis_client import get_goal_redis_client


async def publish_goal_event(
    *,
    job_id: str,
    job_type: str,
    goal_id: str,
    user_id: str,
    payload: dict[str, Any] | None = None,
) -> None:
    stream = settings.REDIS_GOAL_INTERACTIVE_STREAM if job_type == "clarify" else settings.REDIS_GOAL_JOBS_STREAM
    envelope = {
        "eventId": job_id,
        "eventType": f"goal.{job_type}",
        "eventVersion": 1,
        "correlationId": goal_id,
        "causationId": None,
        "userId": user_id,
        "spaceId": "",
        "conversationId": goal_id,
        "payload": {"goalId": goal_id, "jobId": job_id, "jobType": job_type, **(payload or {})},
        "attempt": 0,
    }
    redis = get_goal_redis_client()
    await redis.xadd(stream, {"event": json.dumps(envelope)}, maxlen=settings.REDIS_GOAL_STREAM_MAXLEN, approximate=True)
