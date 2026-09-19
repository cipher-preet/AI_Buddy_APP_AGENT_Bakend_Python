"""Requeue uploaded meeting chunks that failed STT so the worker can retry.

Run inside buddy-worker after deploying the WebM header repair:

  docker compose -f docker-compose.aws.yml --env-file .env.aws exec buddy-worker \\
    python -m scripts.requeue_failed_meeting_chunks --session 6aae640ee6b1324e29e324c4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from bson import ObjectId

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.api_gateway.config.setting import settings
from services.db.mongo import close_mongo_client, get_database
from services.queue.redis_queue import test_redis_connection
from services.queue.streams import EventEnvelope, RedisStreamProducer


def _oid(value: str):
    try:
        return ObjectId(str(value))
    except Exception:
        return value


async def requeue(session_id: str | None) -> int:
    await test_redis_connection()
    db = get_database()
    query: dict = {
        "uploadStatus": "UPLOADED",
        "processingStatus": {"$in": ["FAILED", "CORRUPT_MEDIA"]},
        "mediaKind": {"$ne": "video"},
    }
    if session_id:
        query["meetingSessionId"] = _oid(session_id)

    chunks = [row async for row in db.meeting_recording_chunks.find(query)]
    if not chunks:
        print("no failed uploaded meeting chunks found", flush=True)
        return 0

    producer = RedisStreamProducer()
    now = datetime.now(timezone.utc)
    published = 0
    for chunk in chunks:
        meeting_session_id = str(chunk.get("meetingSessionId") or "")
        sequence = int(chunk.get("sequence") or 0)
        user_id = str(chunk.get("userId") or "")
        media_kind = str(chunk.get("mediaKind") or "muxed")
        s3_key = str(chunk.get("s3Key") or "")
        if not meeting_session_id or sequence < 1 or not s3_key:
            continue
        session = await db.meeting_sessions.find_one({"_id": _oid(meeting_session_id)})
        space_id = str((session or {}).get("spaceId") or user_id)
        job_id = str(chunk.get("jobId") or f"meeting:{meeting_session_id}:{media_kind}:{sequence}")
        await db.meeting_recording_chunks.update_one(
            {"_id": chunk["_id"]},
            {
                "$set": {
                    "processingEnqueued": True,
                    "processingStatus": "ENQUEUED",
                    "transcriptionStatus": "PENDING",
                    "failureCode": None,
                    "failureMessage": None,
                    "updatedAt": now,
                }
            },
        )
        await db.transcript_chunks.update_one(
            {
                "conversationId": {"$in": [_oid(meeting_session_id), meeting_session_id]},
                "sequenceNumber": sequence,
            },
            {
                "$set": {
                    "sttStatus": "PENDING",
                    "terminal": False,
                    "failureType": None,
                    "lastError": None,
                    "updatedAt": now,
                }
            },
        )
        started_at = (session or {}).get("startedAt")
        await producer.publish(
            settings.REDIS_STT_STREAM,
            EventEnvelope(
                eventId=f"{job_id}:retry:{uuid4().hex[:8]}",
                eventType="meeting.video.chunk.ready",
                correlationId=meeting_session_id,
                userId=user_id,
                spaceId=space_id,
                conversationId=meeting_session_id,
                payload={
                    "jobType": "meeting_video_chunk_ready",
                    "meetingSessionId": meeting_session_id,
                    "chunkId": job_id,
                    "sequence": sequence,
                    "sequenceNumber": sequence,
                    "mediaKind": media_kind,
                    "userId": user_id,
                    "spaceId": space_id,
                    "s3Key": s3_key,
                    "s3Bucket": settings.S3_AUDIO_BUCKET,
                    "startOffsetMs": int(chunk.get("startOffsetMs") or 0),
                    "endOffsetMs": int(chunk.get("endOffsetMs") or 0),
                    "durationMs": int(chunk.get("durationMs") or 0),
                    "sourceType": "meeting_extension",
                    "startedAt": started_at.isoformat() if hasattr(started_at, "isoformat") else started_at,
                    "mimeType": chunk.get("mimeType") or "video/webm",
                },
            ),
        )
        published += 1
        print(f"requeued sequence {sequence} session {meeting_session_id}", flush=True)
    return published


async def main() -> None:
    parser = argparse.ArgumentParser(description="Requeue failed meeting chunks for STT")
    parser.add_argument("--session", help="meetingSessionId to requeue", default="")
    args = parser.parse_args()
    try:
        count = await requeue(args.session.strip() or None)
        print(f"requeued {count} chunk(s)", flush=True)
    finally:
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
