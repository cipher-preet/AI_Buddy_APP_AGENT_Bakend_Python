from __future__ import annotations

import shutil
from datetime import datetime, timezone
from uuid import uuid4

from bson import ObjectId

from apps.api_gateway.config.setting import settings
from services.meeting_extension.ffmpeg_audio import MeetingAudioExtractionError, concat_webm_chunks
from services.meeting_extension.s3_keys import meeting_chunk_object_key, meeting_final_object_key, validate_meeting_object_key
from services.observability.diagnostics import diag_log
from services.queue.streams import EventEnvelope, NonRetryableQueueError
from services.storage.s3_audio_storage import get_s3_audio_storage, temp_audio_root


async def process_meeting_video_merge(event: EventEnvelope) -> None:
    if not settings.MEETING_VIDEO_FINALIZATION_ENABLED:
        return
    payload = event.payload or {}
    meeting_session_id = str(payload.get("meetingSessionId") or event.conversationId)
    user_id = str(payload.get("userId") or event.userId)
    expected = int(payload.get("expectedFinalSequence") or 0)
    final_key = str(payload.get("finalRecordingS3Key") or meeting_final_object_key(user_id, meeting_session_id))
    validate_meeting_object_key(object_key=final_key, user_id=user_id, meeting_session_id=meeting_session_id)
    db = get_database_safe()
    job_dir = temp_audio_root() / f"meeting-merge-{meeting_session_id}-{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    diag_log("meeting_video_merge_started", meetingSessionId=meeting_session_id, userId=user_id)
    try:
        await db.meeting_sessions.update_one(
            {"_id": _oid(meeting_session_id)},
            {"$set": {"videoMergeStatus": "RUNNING", "updatedAt": datetime.now(timezone.utc)}},
        )
        storage = get_s3_audio_storage()
        chunk_paths = []
        missing_sequences: list[int] = []
        for sequence in range(1, expected + 1):
            destination = job_dir / f"{sequence:06d}.webm"
            try:
                await _download_merge_chunk(storage, user_id, meeting_session_id, sequence, destination)
            except Exception:
                # Late/missing uploads (often seq 1) must not fail the whole merge.
                missing_sequences.append(sequence)
                diag_log(
                    "meeting_video_merge_chunk_skipped",
                    meetingSessionId=meeting_session_id,
                    userId=user_id,
                    chunkSequence=sequence,
                )
                continue
            chunk_paths.append(destination)
        if not chunk_paths:
            raise MeetingAudioExtractionError("No uploaded video chunks available to merge", corrupt=True)
        if missing_sequences:
            diag_log(
                "meeting_video_merge_partial",
                meetingSessionId=meeting_session_id,
                userId=user_id,
                missingSequences=missing_sequences,
                presentCount=len(chunk_paths),
            )
        output = job_dir / "meeting.webm"
        await concat_webm_chunks(chunk_paths, output)
        await storage.upload_file(output, final_key, content_type="video/webm")
        now = datetime.now(timezone.utc)
        await db.meeting_sessions.update_one(
            {"_id": _oid(meeting_session_id)},
            {
                "$set": {
                    "videoMergeStatus": "COMPLETED",
                    "finalRecordingS3Key": final_key,
                    "updatedAt": now,
                }
            },
        )
        diag_log("meeting_video_merge_completed", meetingSessionId=meeting_session_id, userId=user_id)
    except Exception as error:
        await db.meeting_sessions.update_one(
            {"_id": _oid(meeting_session_id)},
            {
                "$set": {
                    "videoMergeStatus": "FAILED",
                    "lastErrorCode": "MEETING_VIDEO_MERGE_FAILED",
                    "lastErrorMessage": str(error)[:200],
                    "updatedAt": datetime.now(timezone.utc),
                }
            },
        )
        diag_log(
            "meeting_video_merge_failed",
            meetingSessionId=meeting_session_id,
            userId=user_id,
            errorCode=type(error).__name__,
        )
        if isinstance(error, MeetingAudioExtractionError) and error.corrupt:
            raise NonRetryableQueueError(str(error)) from error
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def get_database_safe():
    from services.db.mongo import get_database

    return get_database()


async def _download_merge_chunk(storage, user_id: str, meeting_session_id: str, sequence: int, destination) -> None:
    video_key = meeting_chunk_object_key(user_id, meeting_session_id, sequence, media_kind="video")
    muxed_key = meeting_chunk_object_key(user_id, meeting_session_id, sequence, media_kind="muxed")
    try:
        await storage.download_file(bucket=storage.bucket, object_key=video_key, destination=destination)
        return
    except Exception:
        await storage.download_file(bucket=storage.bucket, object_key=muxed_key, destination=destination)


def _oid(value: str):
    try:
        return ObjectId(value)
    except Exception:
        return value
