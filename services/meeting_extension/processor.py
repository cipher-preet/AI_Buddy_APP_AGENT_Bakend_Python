from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from bson import ObjectId

from apps.api_gateway.config.setting import settings
from services.conversation.models import ConversationStatus, STTStatus
from services.conversation.repository import ConversationRepository
from services.db.mongo import get_database
from services.meeting_extension.ffmpeg_audio import (
    MeetingAudioExtractionError,
    extract_audio_for_stt,
    extract_webm_init,
    has_webm_header,
    write_standalone_webm,
)
from services.meeting_extension.s3_keys import meeting_chunk_object_key, validate_meeting_object_key
from services.meeting_extension.segments import extract_stt_segments
from services.observability.diagnostics import diag_log
from services.queue.streams import EventEnvelope, NonRetryableQueueError, RedisStreamProducer
from services.speech.errors import STTPermanentAudioError
from services.speech.transcription_router import transcribe_from_path_with_fallback
from services.storage.s3_audio_storage import PermanentS3StorageError, get_s3_audio_storage, temp_audio_root


def _object_id(value: str) -> Any:
    try:
        return ObjectId(str(value))
    except Exception:
        return value


async def process_meeting_video_chunk(event: EventEnvelope) -> None:
    payload = event.payload or {}
    meeting_session_id = str(payload.get("meetingSessionId") or event.conversationId)
    sequence = int(payload.get("sequence") or payload.get("sequenceNumber") or 0)
    user_id = str(payload.get("userId") or event.userId)
    space_id = str(payload.get("spaceId") or event.spaceId)
    chunk_id = str(payload.get("chunkId") or f"meeting:{meeting_session_id}:chunk:{sequence}")
    object_key = str(payload.get("s3Key") or payload.get("objectKey") or "")
    bucket = str(payload.get("s3Bucket") or payload.get("bucket") or "")
    start_offset_ms = int(payload.get("startOffsetMs") or 0)
    end_offset_ms = int(payload.get("endOffsetMs") or start_offset_ms)
    started_at = payload.get("startedAt")
    mime_type = str(payload.get("mimeType") or "video/webm")
    media_kind = str(payload.get("mediaKind") or "").strip().lower()

    repository = ConversationRepository(get_database())
    existing = await repository.get_transcript_chunk(meeting_session_id, sequence)
    if existing and existing.sttStatus == STTStatus.COMPLETED:
        diag_log(
            "meeting_chunk_stt_duplicate",
            meetingSessionId=meeting_session_id,
            chunkSequence=sequence,
            userId=user_id,
        )
        return
    await repository.mark_transcript_chunk_processing(meeting_session_id, sequence)

    job_dir = temp_audio_root() / f"meeting-{meeting_session_id}-{sequence}-{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    diag_log(
        "meeting_chunk_processing_started",
        meetingSessionId=meeting_session_id,
        chunkSequence=sequence,
        userId=user_id,
        jobId=chunk_id,
        sourceType="meeting_extension",
    )
    try:
        object_key = validate_meeting_object_key(
            object_key=object_key,
            user_id=user_id,
            meeting_session_id=meeting_session_id,
        )
        if not bucket:
            bucket = get_s3_audio_storage().bucket
        video_path = job_dir / Path(object_key).name
        await get_s3_audio_storage().download_file(bucket=bucket, object_key=object_key, destination=video_path)
        audio_already_separated = media_kind == "audio" or mime_type.lower().startswith("audio/")
        if audio_already_separated:
            audio_path = video_path
            content_type = mime_type.split(";", 1)[0].strip() or "audio/webm"
            diag_log(
                "meeting_audio_chunk_reused",
                meetingSessionId=meeting_session_id,
                chunkSequence=sequence,
                userId=user_id,
            )
        else:
            video_path = await _ensure_standalone_webm(
                video_path,
                sequence=sequence,
                user_id=user_id,
                meeting_session_id=meeting_session_id,
                media_kind=media_kind or "muxed",
                bucket=bucket,
                job_dir=job_dir,
            )
            audio_path = await extract_audio_for_stt(video_path, job_dir / "audio")
            diag_log(
                "meeting_audio_extracted",
                meetingSessionId=meeting_session_id,
                chunkSequence=sequence,
                userId=user_id,
            )
            content_type = "audio/wav" if audio_path.suffix == ".wav" else "audio/webm"
        result = await transcribe_from_path_with_fallback(
            file_path=str(audio_path),
            filename=audio_path.name,
            content_type=content_type,
            job_id=chunk_id,
            stream_attempt=event.attempt,
        )
        segments = extract_stt_segments(
            result,
            chunk_start_offset_ms=start_offset_ms,
            chunk_end_offset_ms=end_offset_ms,
            started_at=str(started_at) if started_at else None,
            sequence=sequence,
            chunk_id=chunk_id,
        )
        first_start = segments[0]["startOffsetMs"] if segments else start_offset_ms
        last_end = segments[-1]["endOffsetMs"] if segments else end_offset_ms
        await repository.complete_transcript_chunk(
            conversation_id=meeting_session_id,
            sequence_number=sequence,
            raw_text=str(result.get("transcript") or ""),
            language_code=result.get("language_code"),
            request_id=result.get("request_id"),
            provider=result.get("provider") or "unknown",
            extra_fields={
                "startTimeMs": first_start,
                "endTimeMs": last_end,
                "sourceType": "meeting_extension",
                "source": "meeting_extension",
                "meetingSessionId": meeting_session_id,
                "segments": segments,
                "sttRequestId": result.get("request_id"),
            },
        )
        await _mark_chunk_processed(meeting_session_id, sequence, "COMPLETED", "COMPLETED", media_kind=media_kind)
        diag_log(
            "meeting_stt_completed",
            meetingSessionId=meeting_session_id,
            chunkSequence=sequence,
            userId=user_id,
            processingDurationMs=int((datetime.now(timezone.utc) - started).total_seconds() * 1000),
        )
        diag_log(
            "meeting_chunk_transcript_saved",
            meetingSessionId=meeting_session_id,
            chunkSequence=sequence,
            userId=user_id,
        )
        producer = RedisStreamProducer()
        await producer.publish(
            settings.REDIS_TRANSCRIPT_READY_STREAM,
            EventEnvelope(
                eventType="conversation.transcript.ready",
                correlationId=meeting_session_id,
                userId=user_id,
                spaceId=space_id,
                conversationId=meeting_session_id,
                causationId=event.eventId,
                payload={"conversationId": meeting_session_id, "sequenceNumber": sequence},
            ),
        )
        conversation = await repository.get_conversation(meeting_session_id)
        if conversation and conversation.expectedLastSequence is not None:
            await producer.publish(
                settings.REDIS_FINALIZATION_STREAM,
                EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=meeting_session_id,
                    userId=user_id,
                    spaceId=space_id,
                    conversationId=meeting_session_id,
                    causationId=event.eventId,
                    payload={"expectedLastSequence": conversation.expectedLastSequence},
                ),
            )
    except (MeetingAudioExtractionError, STTPermanentAudioError, PermanentS3StorageError, ValueError) as error:
        corrupt = isinstance(error, MeetingAudioExtractionError) and error.corrupt
        await _mark_chunk_processed(
            meeting_session_id,
            sequence,
            "CORRUPT_MEDIA" if corrupt else "FAILED",
            "FAILED",
            str(error)[:200],
            media_kind=media_kind,
        )
        await repository.fail_transcript_chunk(
            meeting_session_id,
            sequence,
            str(error)[:200],
            job_id=chunk_id,
            failure_stage="meeting_video",
            failure_type="CORRUPT_MEDIA" if corrupt else type(error).__name__,
            terminal=corrupt or isinstance(error, STTPermanentAudioError),
        )
        diag_log(
            "meeting_processing_failed",
            meetingSessionId=meeting_session_id,
            chunkSequence=sequence,
            userId=user_id,
            errorCode="CORRUPT_MEDIA" if corrupt else type(error).__name__,
        )
        if corrupt or isinstance(error, STTPermanentAudioError):
            raise NonRetryableQueueError(str(error)) from error
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


async def _mark_chunk_processed(
    meeting_session_id: str,
    sequence: int,
    processing_status: str,
    transcription_status: str,
    failure_message: str | None = None,
    media_kind: str | None = None,
) -> None:
    db = get_database()
    now = datetime.now(timezone.utc)
    update = {
        "processingStatus": processing_status,
        "transcriptionStatus": transcription_status,
        "processedAt": now,
        "updatedAt": now,
    }
    if failure_message:
        update["failureMessage"] = failure_message
        update["failureCode"] = processing_status
    elif processing_status == "COMPLETED":
        update["failureMessage"] = None
        update["failureCode"] = None
    query: dict[str, Any] = {
        "meetingSessionId": _object_id(meeting_session_id),
        "sequence": sequence,
    }
    if media_kind:
        query["mediaKind"] = media_kind
    await db.meeting_recording_chunks.update_one(
        query,
        {"$set": update},
    )
    if processing_status == "COMPLETED":
        await db.meeting_sessions.update_one(
            {"_id": _object_id(meeting_session_id)},
            {"$inc": {"processedChunks": 1}, "$set": {"updatedAt": now, "transcriptStatus": "processing"}},
        )


async def _ensure_standalone_webm(
    video_path: Path,
    *,
    sequence: int,
    user_id: str,
    meeting_session_id: str,
    media_kind: str,
    bucket: str,
    job_dir: Path,
) -> Path:
    payload = video_path.read_bytes()
    if has_webm_header(payload) or sequence <= 1:
        return video_path
    init_key = meeting_chunk_object_key(
        user_id,
        meeting_session_id,
        1,
        media_kind=media_kind if media_kind in {"audio", "video", "muxed"} else "muxed",
    )
    init_path = job_dir / "webm-init-source.webm"
    try:
        await get_s3_audio_storage().download_file(bucket=bucket, object_key=init_key, destination=init_path)
    except Exception as error:
        diag_log(
            "meeting_webm_init_missing",
            meetingSessionId=meeting_session_id,
            chunkSequence=sequence,
            error=str(error)[:200],
        )
        return video_path
    init_bytes = extract_webm_init(init_path.read_bytes())
    if not init_bytes:
        return video_path
    repaired = write_standalone_webm(video_path, init_bytes, job_dir / f"{video_path.stem}.standalone.webm")
    diag_log(
        "meeting_webm_init_prepended",
        meetingSessionId=meeting_session_id,
        chunkSequence=sequence,
        initBytes=len(init_bytes),
    )
    return repaired
