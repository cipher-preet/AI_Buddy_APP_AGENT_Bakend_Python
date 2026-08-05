# apps/api_gateway/workers/vector_worker.py

import asyncio
from pathlib import Path

from apps.api_gateway.config.setting import settings
from services.conversation.repository import ConversationRepository
from services.db.mongo import get_database
from services.queue.redis_queue import (
    delete_speech_job,
    get_job_result,
    pop_completed_speech_job,
)
from services.queue.streams import EventEnvelope, RedisStreamProducer

from services.vector.vector_service import (
    store_transcript_in_vector_db,
)

UPLOAD_DIR = Path("resources/audio_jobs").resolve()


def _remove_processed_audio_file(file_path: str | None) -> bool:
    if not file_path:
        return False

    path = Path(file_path).resolve()
    if UPLOAD_DIR not in path.parents or not path.is_file():
        return False

    try:
        path.unlink()
        return True
    except OSError as error:
        print(
            "Processed audio cleanup failed:",
            {
                "file_path": str(path),
                "error": str(error),
            },
        )
        return False


async def process_completed_speech_job(job_id: str) -> None:
    job = await get_job_result(job_id)

    if not job:
        print("Completed speech job already cleaned up or missing:", job_id)
        return

    if job.get("status") != "completed":
        print("Completed speech job is not ready for vector processing:", job_id)
        return

    result = job.get("result") or {}

    transcript = str(result.get("transcript") or "").strip()
    language_code = result.get("language_code")
    request_id = result.get("request_id")

    user_id = str(job.get("user_id") or "").strip()
    space_id = str(job.get("space_id") or "").strip()

    if not user_id or not space_id:
        print("Invalid completed job data:", {"job_id": job_id})
        return

    conversation_id = str(job.get("conversation_id") or "").strip()
    sequence_number = job.get("sequence_number")
    if conversation_id and sequence_number is not None:
        repository = ConversationRepository(get_database())
        await repository.complete_transcript_chunk(
            conversation_id=conversation_id,
            sequence_number=int(sequence_number),
            raw_text=str(transcript or ""),
            language_code=language_code,
            request_id=request_id,
            provider="sarvam",
        )
        conversation = await repository.get_conversation(conversation_id)
        if conversation and conversation.expectedLastSequence is not None:
            await RedisStreamProducer().publish(
                settings.REDIS_FINALIZATION_STREAM,
                EventEnvelope(
                    eventType="conversation.finalization.requested",
                    correlationId=conversation_id,
                    userId=user_id,
                    spaceId=space_id,
                    conversationId=conversation_id,
                    payload={"expectedLastSequence": conversation.expectedLastSequence},
                ),
            )

    if not transcript:
        audio_removed = _remove_processed_audio_file(job.get("file_path"))
        print(
            "Transcript vector job skipped empty transcript:",
            {
                "job_id": job_id,
                "user_id": user_id,
                "space_id": space_id,
                "conversation_id": conversation_id or None,
                "sequence_number": sequence_number,
                "audio_removed": audio_removed,
            },
        )
        await delete_speech_job(job_id)
        return

    await store_transcript_in_vector_db(
        user_id=user_id,
        space_id=space_id,
        job_id=job_id,
        transcript=transcript,
        language_code=language_code,
        request_id=request_id,
    )
    audio_removed = _remove_processed_audio_file(job.get("file_path"))

    print(
        "Transcript vector job completed:",
        {
            "job_id": job_id,
            "user_id": user_id,
            "space_id": space_id,
            "conversation_id": conversation_id or None,
            "sequence_number": sequence_number,
            "audio_removed": audio_removed,
        },
    )

    await delete_speech_job(job_id)


async def start_vector_consumer():
    print("Vector worker started...")

    while True:
        try:
            job_id = await pop_completed_speech_job()

            if not job_id:
                await asyncio.sleep(1)
                continue

            await process_completed_speech_job(job_id)

        except Exception as error:
            print("Vector worker error:", str(error))
            await asyncio.sleep(2)
