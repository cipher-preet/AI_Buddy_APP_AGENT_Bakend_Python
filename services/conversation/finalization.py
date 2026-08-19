from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlparse

from apps.api_gateway.config.setting import settings
from services.conversation.incremental import IncrementalMeetingProcessor
from services.conversation.models import ConversationStatus, STTStatus, WindowProcessingStatus, as_utc, utc_now
from services.conversation.repository import ConversationRepository, to_mongo_id
from services.conversation.transcript import detect_missing_sequences
from services.conversation.windowing import is_useful_chunk
from services.queue.streams import EventEnvelope, RedisStreamProducer
from services.storage.s3_audio_storage import build_audio_object_key, use_s3_storage


_TERMINAL_CONVERSATION_STATUSES = {
    ConversationStatus.READY_FOR_PROCESSING,
    ConversationStatus.VALIDATING,
    ConversationStatus.COMPLETED,
    ConversationStatus.PROCESSING,
}


class ConversationFinalizationCoordinator:
    def __init__(
        self,
        repository: ConversationRepository,
        producer: RedisStreamProducer | None = None,
    ):
        self.repository = repository
        self.producer = producer or RedisStreamProducer()

    async def finalize(self, conversation_id: str) -> None:
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation or conversation.expectedLastSequence is None:
            raise ValueError("Conversation is not ready for finalization")
        if conversation.status in _TERMINAL_CONVERSATION_STATUSES:
            return
        if conversation.status == ConversationStatus.PARTIAL:
            run = (
                await self.repository.get_extraction_run(conversation.activeExtractionRunId)
                if conversation.activeExtractionRunId is not None
                else None
            )
            if run and run.status.value == "PUBLISHED":
                return

        chunks = await self.repository.list_transcript_chunks(conversation_id)
        windows = await self.repository.list_conversation_windows(conversation_id)
        skippable, unresolved, accounting = _session_readiness(conversation, chunks, windows)

        await self._persist_accounting(conversation_id, accounting)

        if unresolved:
            retryable_count = await self._requeue_lost_stt(conversation_id, chunks)
            if conversation.status != ConversationStatus.WAITING_FOR_TRANSCRIPTS:
                await self._safe_transition(
                    conversation_id,
                    ConversationStatus.WAITING_FOR_TRANSCRIPTS,
                    {
                        "missingSequences": accounting["missingSequences"],
                        "lastAccounting": accounting,
                    },
                )
            print(
                "Finalization waiting for transcripts:",
                {
                    "conversationId": conversation_id,
                    "unresolvedExpected": accounting["unresolvedExpected"],
                    "pending": accounting["pendingTranscripts"],
                    "retryableCount": retryable_count,
                    **_public_accounting(accounting),
                },
            )
            return

        if settings.ENABLE_INCREMENTAL_MEETING_PROCESSING:
            processor = IncrementalMeetingProcessor(self.repository, self.producer)
            await processor.close_ready_windows(
                conversation_id,
                force_final=True,
                through_sequence=conversation.expectedLastSequence,
                skippable_sequences=skippable,
            )
            unwindowed = await self.repository.count_unwindowed_non_empty_transcripts(
                conversation_id,
                through_sequence=conversation.expectedLastSequence,
            )
            windows = await self.repository.list_conversation_windows(conversation_id)
            stale_before = utc_now() - timedelta(seconds=settings.WINDOW_PROCESSING_STALE_TIMEOUT_SECONDS)
            reclaimed = await self.repository.reclaim_stale_processing_windows(conversation_id, stale_before)
            if reclaimed:
                accounting["windowStaleRecoveredCount"] = len(reclaimed)
                print(
                    "Stale processing windows reclaimed:",
                    {"conversationId": conversation_id, "count": len(reclaimed)},
                )
            windows = await self.repository.list_conversation_windows(conversation_id)
            incomplete = [window for window in windows if window.status != WindowProcessingStatus.COMPLETED]
            queued_stale_before = utc_now() - timedelta(seconds=settings.WINDOW_PROCESSING_STALE_TIMEOUT_SECONDS)
            for window in incomplete:
                if not _should_publish_window_job(window, queued_stale_before):
                    continue
                await self.repository.mark_window_queued(window.id)
                await self.producer.publish(
                    settings.REDIS_WINDOW_EXTRACTION_STREAM,
                    EventEnvelope(
                        eventType="conversation.window.extraction.requested",
                        correlationId=conversation_id,
                        userId=str(window.userId),
                        spaceId=str(window.spaceId),
                        conversationId=conversation_id,
                        payload={"windowId": str(window.id), "windowIndex": window.windowIndex},
                    ),
                )
            chunks = await self.repository.list_transcript_chunks(conversation_id)
            _, _, accounting = _session_readiness(conversation, chunks, windows)
            accounting["validUnwindowed"] = unwindowed
            await self._persist_accounting(conversation_id, accounting)
            if unwindowed > 0 or incomplete:
                if conversation.status != ConversationStatus.FINALIZING:
                    await self._safe_transition(
                        conversation_id,
                        ConversationStatus.FINALIZING,
                        {
                            "missingSequences": accounting["missingSequences"],
                            "lastAccounting": accounting,
                        },
                    )
                print(
                    "Finalization waiting for windows:",
                    {
                        "conversationId": conversation_id,
                        "unwindowedNonEmpty": unwindowed,
                        "incompleteWindows": len(incomplete),
                        **_public_accounting(accounting),
                    },
                )
                return
            persistence_gaps = [
                window
                for window in windows
                if window.status == WindowProcessingStatus.COMPLETED
                and window.artifactPersistenceOk is False
            ]
            if persistence_gaps:
                for window in persistence_gaps:
                    await self.repository.fail_window(window.id, "artifact persistence incomplete")
                    await self.repository.mark_window_queued(window.id)
                    await self.producer.publish(
                        settings.REDIS_WINDOW_EXTRACTION_STREAM,
                        EventEnvelope(
                            eventType="conversation.window.extraction.requested",
                            correlationId=conversation_id,
                            userId=str(window.userId),
                            spaceId=str(window.spaceId),
                            conversationId=conversation_id,
                            payload={"windowId": str(window.id), "windowIndex": window.windowIndex, "recovery": True},
                        ),
                    )
                await self._safe_transition(
                    conversation_id,
                    ConversationStatus.FINALIZING,
                    {"lastAccounting": accounting},
                )
                return

        if accounting["validUnwindowed"] > 0:
            await self._safe_transition(
                conversation_id,
                ConversationStatus.FINALIZING,
                {"lastAccounting": accounting},
            )
            return

        print("Finalization accounting ready:", {"conversationId": conversation_id, **_public_accounting(accounting)})
        await self.repository.append_meeting_debug_trace(
            conversation_id,
            conversation.userId,
            conversation.spaceId,
            "pre_finalization_accounting",
            accounting,
        )
        transitioned = await self._safe_transition(
            conversation_id,
            ConversationStatus.READY_FOR_PROCESSING,
            {
                "missingSequences": accounting["missingSequences"] if accounting["permanentFailures"] else [],
                "lastAccounting": accounting,
            },
        )
        if not transitioned:
            return
        await self.producer.publish(
            settings.REDIS_PROCESSING_STREAM,
            EventEnvelope(
                eventType="conversation.processing.requested",
                correlationId=conversation_id,
                userId=str(conversation.userId),
                spaceId=str(conversation.spaceId),
                conversationId=conversation_id,
                payload={
                    "processingVersion": conversation.processingVersion,
                    "partial": bool(accounting["permanentFailures"]),
                    "missingSequences": accounting["missingSequences"],
                },
            ),
        )

    async def _requeue_lost_stt(self, conversation_id: str, chunks) -> int:
        stale_before = utc_now() - timedelta(seconds=settings.STT_PROCESSING_STALE_TIMEOUT_SECONDS)
        reclaimed = await self.repository.reclaim_stale_stt_chunks(conversation_id, stale_before)
        retry_ids = {chunk.sequenceNumber for chunk in reclaimed}
        for chunk in chunks:
            if chunk.sequenceNumber in retry_ids:
                continue
            if not _is_lost_stt_job(chunk, stale_before):
                continue
            retry_ids.add(chunk.sequenceNumber)
            reclaimed.append(chunk)
        retryable_count = 0
        for chunk in reclaimed:
            if chunk.sttStatus == STTStatus.FAILED and _is_permanent_audio_failure(chunk):
                continue
            audio_chunk = await self.repository.get_audio_chunk(str(chunk.conversationId), chunk.sequenceNumber)
            if chunk.sttStatus == STTStatus.FAILED and chunk.sttAttempts >= settings.WORKER_MAX_RETRIES:
                audio_fields = _stt_payload_audio_fields(chunk, audio_chunk)
                if not _is_recoverable_s3_retry(chunk, audio_fields):
                    continue
            if not chunk.audioFilePath and not audio_chunk:
                continue
            audio_fields = _stt_payload_audio_fields(chunk, audio_chunk)
            retryable_count += 1
            await self.producer.publish(
                settings.REDIS_STT_STREAM,
                EventEnvelope(
                    eventType="stt.requested",
                    correlationId=conversation_id,
                    userId=str(chunk.userId),
                    spaceId=str(chunk.spaceId),
                    conversationId=conversation_id,
                    payload={
                        "conversationId": str(chunk.conversationId),
                        "userId": str(chunk.userId),
                        "spaceId": str(chunk.spaceId),
                        "chunkId": chunk.chunkId,
                        "sequenceNumber": chunk.sequenceNumber,
                        **audio_fields,
                    },
                ),
            )
        return retryable_count

    async def _persist_accounting(self, conversation_id: str, accounting: dict) -> None:
        await self.repository.db.conversations.update_one(
            {"_id": to_mongo_id(conversation_id)},
            {"$set": {"lastAccounting": accounting, "updatedAt": utc_now()}},
        )

    async def _safe_transition(self, conversation_id: str, target: ConversationStatus, updates: dict | None = None) -> bool:
        current = await self.repository.get_conversation(conversation_id)
        if not current or current.status in _TERMINAL_CONVERSATION_STATUSES:
            return False
        if current.status == target:
            if updates:
                await self.repository.db.conversations.update_one(
                    {"_id": current.id, "status": current.status.value},
                    {"$set": {**updates, "updatedAt": utc_now()}},
                )
            return False
        try:
            await self.repository.transition(conversation_id, target, updates)
            return True
        except ValueError as error:
            print("Finalization transition skipped:", {"conversationId": conversation_id, "target": target.value, "error": str(error)})
            return False


def _session_readiness(conversation, chunks, windows) -> tuple[set[int], bool, dict]:
    expected_last = int(conversation.expectedLastSequence)
    present = {chunk.sequenceNumber for chunk in chunks}
    missing = detect_missing_sequences(list(present), expected_last)
    pending = [chunk for chunk in chunks if chunk.sttStatus in {STTStatus.PENDING, STTStatus.PROCESSING}]
    failed = [chunk for chunk in chunks if chunk.sttStatus == STTStatus.FAILED]
    missing_timeout = _missing_sequences_are_terminal(conversation)
    terminal_failed = [chunk.sequenceNumber for chunk in failed if _is_terminal_failed_chunk(chunk)]
    retryable_failed = [chunk.sequenceNumber for chunk in failed if chunk.sequenceNumber not in terminal_failed]
    skippable = {
        chunk.sequenceNumber
        for chunk in chunks
        if (chunk.sttStatus == STTStatus.COMPLETED and not is_useful_chunk(chunk))
        or chunk.exclusionReason
        or chunk.sequenceNumber in terminal_failed
    }
    if missing_timeout:
        skippable.update(missing)
    unresolved = bool(pending or retryable_failed or (missing and not missing_timeout))
    empty = [chunk for chunk in chunks if chunk.sttStatus == STTStatus.COMPLETED and not is_useful_chunk(chunk)]
    useful = [chunk for chunk in chunks if chunk.sttStatus == STTStatus.COMPLETED and is_useful_chunk(chunk)]
    useful_windowed = [chunk for chunk in useful if chunk.processingWindowId is not None]
    useful_excluded = [chunk for chunk in useful if chunk.exclusionReason]
    useful_unwindowed = [
        chunk for chunk in useful if chunk.processingWindowId is None and not chunk.exclusionReason
    ]
    incomplete_windows = [window for window in windows if window.status != WindowProcessingStatus.COMPLETED]
    accounting = {
        "expectedSequences": expected_last + 1,
        "accountedSequences": len(present) + (len(missing) if missing_timeout else 0),
        "missingSequences": missing,
        "emptyTranscripts": len(empty),
        "failedTranscripts": len(failed),
        "pendingTranscripts": len(pending),
        "validTranscripts": len(useful),
        "validWindowed": len(useful_windowed),
        "validExcluded": len(useful_excluded),
        "validUnwindowed": len(useful_unwindowed),
        "windowsCreated": len(windows),
        "windowsCompleted": sum(1 for window in windows if window.status == WindowProcessingStatus.COMPLETED),
        "windowsPending": sum(1 for window in windows if window.status == WindowProcessingStatus.PENDING),
        "windowsProcessing": sum(1 for window in windows if window.status == WindowProcessingStatus.PROCESSING),
        "windowsFailed": sum(1 for window in windows if window.status == WindowProcessingStatus.FAILED),
        "windowStaleRecoveredCount": 0,
        "unresolvedExpected": len(pending) + len(retryable_failed) + (0 if missing_timeout else len(missing)),
        "permanentFailures": len(terminal_failed) + (len(missing) if missing_timeout else 0),
        "artifactPersistenceGaps": sum(
            1 for window in windows if window.status == WindowProcessingStatus.COMPLETED and window.artifactPersistenceOk is False
        ),
        "incompleteWindows": len(incomplete_windows),
    }
    return skippable, unresolved, accounting


def _public_accounting(accounting: dict) -> dict:
    return {
        "expectedSequences": accounting["expectedSequences"],
        "accountedSequences": accounting["accountedSequences"],
        "emptyTranscripts": accounting["emptyTranscripts"],
        "failedTranscripts": accounting["failedTranscripts"],
        "validTranscripts": accounting["validTranscripts"],
        "validWindowed": accounting["validWindowed"],
        "validUnwindowed": accounting["validUnwindowed"],
        "windowsCreated": accounting["windowsCreated"],
        "windowsCompleted": accounting["windowsCompleted"],
        "windowsPending": accounting["windowsPending"],
        "windowsProcessing": accounting["windowsProcessing"],
        "windowsFailed": accounting["windowsFailed"],
        "unresolvedExpected": accounting["unresolvedExpected"],
    }


def _missing_sequences_are_terminal(conversation) -> bool:
    stopped_at = as_utc(conversation.stoppedAt)
    if stopped_at is None:
        return False
    elapsed = utc_now() - stopped_at
    return elapsed.total_seconds() >= settings.FINALIZATION_MISSING_SEQUENCE_TIMEOUT_SECONDS


def _should_publish_window_job(window, stale_before) -> bool:
    if window.status == WindowProcessingStatus.FAILED:
        return True
    if window.status != WindowProcessingStatus.PENDING:
        return False
    if window.queuedAt is None:
        return True
    updated_at = as_utc(window.updatedAt)
    if updated_at is None:
        return True
    return updated_at <= as_utc(stale_before)


def _is_lost_stt_job(chunk, stale_before) -> bool:
    if chunk.sttStatus == STTStatus.PROCESSING:
        return False
    if chunk.sttStatus == STTStatus.COMPLETED:
        return False
    if chunk.sttStatus == STTStatus.FAILED and _is_permanent_audio_failure(chunk):
        return False
    if chunk.sttStatus == STTStatus.FAILED and chunk.sttAttempts >= settings.WORKER_MAX_RETRIES:
        return False
    updated_at = as_utc(getattr(chunk, "updatedAt", None))
    if updated_at is None:
        return True
    return updated_at <= as_utc(stale_before)


def _is_terminal_failed_chunk(chunk) -> bool:
    if _is_permanent_audio_failure(chunk):
        return True
    return int(chunk.sttAttempts or 0) >= settings.WORKER_MAX_RETRIES


def _stt_payload_audio_fields(chunk, audio_chunk: dict | None) -> dict:
    if audio_chunk:
        payload = {
            "filename": audio_chunk.get("filename") or f"{chunk.chunkId}.audio",
            "contentType": audio_chunk.get("contentType") or "audio/wav",
        }
        storage_provider = str(audio_chunk.get("storageProvider") or "").lower()
        s3_bucket = audio_chunk.get("s3Bucket")
        s3_object_key = audio_chunk.get("s3ObjectKey")
        if storage_provider == "s3" or s3_bucket or s3_object_key:
            payload.update(
                {
                    "storageProvider": "s3",
                    "bucket": s3_bucket,
                    "objectKey": s3_object_key,
                }
            )
            return payload
        if use_s3_storage():
            payload.update(_legacy_s3_reference_payload(chunk, audio_chunk))
            return payload
        payload.update(_audio_reference_payload(audio_chunk.get("filePath") or chunk.audioFilePath))
        return payload

    if use_s3_storage():
        return {
            "filename": f"{chunk.chunkId}.audio",
            "contentType": "audio/wav",
            **_legacy_s3_reference_payload(chunk, None),
        }
    return {
        "filename": f"{chunk.chunkId}.audio",
        "contentType": "audio/wav",
        **_audio_reference_payload(chunk.audioFilePath),
    }


def _audio_reference_payload(audio_file_path: str | None) -> dict:
    value = str(audio_file_path or "").strip()
    if value.startswith("s3://"):
        parsed = urlparse(value)
        bucket = parsed.netloc
        object_key = parsed.path.lstrip("/")
        return {
            "storageProvider": "s3",
            "bucket": bucket,
            "objectKey": object_key,
        }
    return {"filePath": value}


def _legacy_s3_reference_payload(chunk, audio_chunk: dict | None) -> dict:
    filename = str((audio_chunk or {}).get("filename") or f"{chunk.chunkId}.audio")
    object_key = build_audio_object_key(
        user_id=str(chunk.userId),
        space_id=str(chunk.spaceId),
        session_id=str(chunk.conversationId),
        job_id=str(chunk.chunkId),
        filename=filename,
    )
    return {
        "storageProvider": "s3",
        "bucket": settings.S3_AUDIO_BUCKET or settings.S3_BUCKET,
        "objectKey": object_key,
    }


def _is_recoverable_s3_retry(chunk, audio_fields: dict) -> bool:
    if str(audio_fields.get("storageProvider") or "").lower() != "s3":
        return False
    error = str(chunk.lastError or "").lower()
    return "missing or empty" in error or "no such file" in error or "resources/audio_jobs" in error


def _is_permanent_audio_failure(chunk) -> bool:
    error = str(chunk.lastError or "").lower()
    permanent_markers = (
        "audio duration exceeds",
        "exceeds the maximum limit",
        "failed to read the file",
        "audio format",
        "invalid audio",
        "file too large",
        "batch api",
        "unsupported audio content type",
    )
    return any(marker in error for marker in permanent_markers)
