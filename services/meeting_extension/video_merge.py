from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from bson import ObjectId

from apps.api_gateway.config.setting import settings
from services.meeting_extension.ffmpeg_audio import MeetingAudioExtractionError, concat_webm_chunks
from services.meeting_extension.s3_keys import meeting_chunk_object_key, meeting_final_object_key, validate_meeting_object_key
from services.observability.diagnostics import diag_log
from services.queue.streams import EventEnvelope, NonRetryableQueueError, RedisStreamProducer
from services.storage.s3_audio_storage import get_s3_audio_storage, temp_audio_root

# Late chunks can land while a merge runs; re-run a bounded number of times so they are included.
MAX_MERGE_PASSES = 3
# Stop only after FFmpeg itself has timed out this many times. A killed worker is not that case.
_MAX_TIMEOUT_ATTEMPTS = 2
# Meetings left RUNNING by a dead process are picked up again on the next worker start.
_STUCK_RESUME_LIMIT = 5
# This process is actually concatenating these meetings. A RUNNING row in Mongo is not
# proof of that: the last worker may have been killed and left the row behind.
_ACTIVE_MERGES: set[str] = set()


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
    if await _merge_should_stand_down(db, meeting_session_id):
        return
    _ACTIVE_MERGES.add(meeting_session_id)
    try:
        await _merge_passes(db, meeting_session_id, user_id, expected, final_key)
    finally:
        _ACTIVE_MERGES.discard(meeting_session_id)


async def _merge_passes(db, meeting_session_id: str, user_id: str, expected: int, final_key: str) -> None:
    for merge_pass in range(1, MAX_MERGE_PASSES + 1):
        # The server can raise expectedFinalSequence after enqueue (late/auto-finalized STOP).
        expected = await _current_expected_sequence(db, meeting_session_id, expected)
        await _merge_once(db, meeting_session_id, user_id, expected, final_key)
        # Clear the flag atomically; only loop if a late chunk asked for it during this pass.
        rerun = await db.meeting_sessions.find_one_and_update(
            {"_id": _oid(meeting_session_id), "videoRemergeRequested": True},
            {"$set": {"videoRemergeRequested": False}},
        )
        if not rerun:
            return
        diag_log(
            "meeting_video_remerge_pass",
            meetingSessionId=meeting_session_id,
            userId=user_id,
            nextPass=merge_pass + 1,
        )


def merge_work_root() -> Path:
    """Video chunks must not land on the worker's RAM disk.

    The AWS worker mounts /tmp as a 1GB tmpfs. A meeting's chunks, their cleaned
    copies, and the concat output all counted as memory and the kernel killed the
    worker (exit 137) before the recording could be saved.
    """
    configured = str(getattr(settings, "MEETING_MERGE_WORK_ROOT", "") or "").strip()
    if configured:
        root = Path(configured)
    else:
        root = Path("/var/tmp/buddy-merge")
    try:
        root.mkdir(parents=True, exist_ok=True)
        return root
    except OSError:
        fallback = temp_audio_root() / "meeting-merge"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


async def _merge_once(db, meeting_session_id: str, user_id: str, expected: int, final_key: str) -> None:
    job_dir = merge_work_root() / f"meeting-merge-{meeting_session_id}-{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    diag_log("meeting_video_merge_started", meetingSessionId=meeting_session_id, userId=user_id)
    try:
        previous = await db.meeting_sessions.find_one_and_update(
            {"_id": _oid(meeting_session_id)},
            {
                "$set": {"videoMergeStatus": "RUNNING", "updatedAt": datetime.now(timezone.utc)},
                "$inc": {"videoMergeAttempts": 1},
            },
        )
        previous = previous or {}
        storage = get_s3_audio_storage()
        chunk_paths = []
        missing_sequences: list[int] = []
        present_sequences: list[int] = []
        for sequence in range(1, expected + 1):
            destination = job_dir / f"{sequence:06d}.webm"
            try:
                await _download_merge_chunk(storage, user_id, meeting_session_id, sequence, destination)
                if not destination.exists() or destination.stat().st_size <= 0:
                    raise MeetingAudioExtractionError("Downloaded chunk is empty", corrupt=True)
            except Exception:
                # Lost/empty chunks are skipped — final video continues from the next good chunk.
                missing_sequences.append(sequence)
                diag_log(
                    "meeting_video_merge_chunk_skipped",
                    meetingSessionId=meeting_session_id,
                    userId=user_id,
                    chunkSequence=sequence,
                )
                continue
            chunk_paths.append(destination)
            present_sequences.append(sequence)
        if not chunk_paths:
            raise MeetingAudioExtractionError("No uploaded video chunks available to merge", corrupt=True)

        previous_count = previous.get("mergePresentChunkCount")
        if (
            previous.get("finalRecordingS3Key")
            and isinstance(previous_count, int)
            and len(chunk_paths) < previous_count
        ):
            # Source chunks expired from S3 since the last merge — never replace a fuller recording.
            await db.meeting_sessions.update_one(
                {"_id": _oid(meeting_session_id)},
                {"$set": {"videoMergeStatus": "COMPLETED", "updatedAt": datetime.now(timezone.utc)}},
            )
            diag_log(
                "meeting_video_remerge_skipped_fewer_chunks",
                meetingSessionId=meeting_session_id,
                userId=user_id,
                presentCount=len(chunk_paths),
                previousCount=previous_count,
            )
            return

        # Gap-tolerant: concat only present chunks in sequence order (already ascending).
        if missing_sequences:
            diag_log(
                "meeting_video_merge_partial",
                meetingSessionId=meeting_session_id,
                userId=user_id,
                missingSequences=missing_sequences,
                presentSequences=present_sequences,
                presentCount=len(chunk_paths),
                expected=expected,
            )
        output = job_dir / "meeting.webm"
        output = await concat_webm_chunks(chunk_paths, output)
        if not output.exists() or output.stat().st_size < 8_192:
            raise MeetingAudioExtractionError(
                "Merged recording is empty or too small to play",
                corrupt=True,
            )
        with output.open("rb") as recorded:
            probe = recorded.read(64)
        is_webm = probe.startswith(bytes([0x1A, 0x45, 0xDF, 0xA3]))
        is_mp4 = b"ftyp" in probe
        if is_webm:
            final_key = final_key.rsplit(".", 1)[0] + ".webm"
            await storage.upload_file(output, final_key, content_type="video/webm")
        elif is_mp4:
            final_key = final_key.rsplit(".", 1)[0] + ".mp4"
            await storage.upload_file(output, final_key, content_type="video/mp4")
        else:
            raise MeetingAudioExtractionError(
                "Merged recording is not a recognized MP4/WebM container",
                corrupt=True,
            )
        now = datetime.now(timezone.utc)
        await db.meeting_sessions.update_one(
            {"_id": _oid(meeting_session_id)},
            {
                "$set": {
                    "videoMergeStatus": "COMPLETED",
                    "finalRecordingS3Key": final_key,
                    "updatedAt": now,
                    "mergeMissingSequences": missing_sequences,
                    "mergePresentChunkCount": len(chunk_paths),
                    "mergeExpectedSequence": expected,
                    "videoMergeAttempts": 0,
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


async def _merge_should_stand_down(db, meeting_session_id: str) -> bool:
    """Skip only a merge this process is already doing.

    A RUNNING status left by a killed worker must be resumed. Treating that row as
    busy acknowledges the queue message and the recording stays on "processing" forever.
    Repeated FFmpeg timeouts are the one case that stops, so a bad file cannot fill the disk.
    """
    if meeting_session_id in _ACTIVE_MERGES:
        diag_log("meeting_video_merge_skipped_busy", meetingSessionId=meeting_session_id)
        return True
    try:
        existing = await db.meeting_sessions.find_one(
            {"_id": _oid(meeting_session_id)},
            {"videoMergeAttempts": 1, "lastErrorMessage": 1},
        )
    except Exception:
        return False
    if not existing:
        return False
    attempts = int(existing.get("videoMergeAttempts") or 0)
    message = str(existing.get("lastErrorMessage") or "")
    if attempts >= _MAX_TIMEOUT_ATTEMPTS and "timed out" in message.casefold():
        diag_log(
            "meeting_video_merge_stopped",
            meetingSessionId=meeting_session_id,
            attempts=attempts,
        )
        raise NonRetryableQueueError("Meeting merge timed out repeatedly; leaving the server alone")
    return False


async def resume_stuck_video_merges() -> int:
    """Put meetings stuck on RUNNING back on the merge queue.

    The worker that was concatenating them was killed, and the queue message was
    already acknowledged, so nothing else will finish the recording.
    """
    if not settings.MEETING_VIDEO_FINALIZATION_ENABLED:
        return 0
    db = get_database_safe()
    try:
        cursor = db.meeting_sessions.find(
            {"videoMergeStatus": "RUNNING"},
            {"userId": 1, "spaceId": 1, "expectedFinalSequence": 1},
        )
        docs = await cursor.to_list(_STUCK_RESUME_LIMIT)
    except Exception as error:
        diag_log("meeting_video_merge_resume_failed", error=type(error).__name__)
        return 0
    producer = RedisStreamProducer()
    resumed = 0
    for doc in docs:
        session_id = str(doc.get("_id") or "")
        user_id = str(doc.get("userId") or "")
        if not session_id or not user_id or session_id in _ACTIVE_MERGES:
            continue
        event = EventEnvelope(
            eventType="meeting.video.merge.requested",
            correlationId=session_id,
            userId=user_id,
            spaceId=str(doc.get("spaceId") or ""),
            conversationId=session_id,
            payload={
                "meetingSessionId": session_id,
                "userId": user_id,
                "expectedFinalSequence": int(doc.get("expectedFinalSequence") or 0),
            },
        )
        await producer.publish(settings.REDIS_MEETING_MERGE_STREAM, event)
        diag_log("meeting_video_merge_resumed", meetingSessionId=session_id, userId=user_id)
        resumed += 1
    return resumed


async def _current_expected_sequence(db, meeting_session_id: str, fallback: int) -> int:
    try:
        doc = await db.meeting_sessions.find_one({"_id": _oid(meeting_session_id)}, {"expectedFinalSequence": 1})
        latest = int((doc or {}).get("expectedFinalSequence") or 0)
    except Exception:
        return fallback
    return max(latest, fallback)


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
