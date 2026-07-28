import os
import uuid
import aiofiles
from fastapi import UploadFile

from services.conversation.models import AudioChunkMetadata, utc_now
from services.conversation.repository import ConversationRepository
from services.conversation.service import ConversationService
from services.db.mongo import get_database
from services.queue.redis_queue import (
    get_job_result,
    push_speech_job,
    redis_client,
)

UPLOAD_DIR = "resources/audio_jobs"


async def transcribe_audio_service(file: UploadFile, user_id: str, space_id: str):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    user_id = user_id.strip()
    space_id = space_id.strip()
    job_id = str(uuid.uuid4())
    filename = file.filename or f"{job_id}.audio"
    file_extension = filename.rsplit(".", 1)[-1] if "." in filename else "audio"
    saved_filename = f"{job_id}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    active_session = await _get_or_create_listening_session(user_id, space_id)
    conversation_id = active_session.get("conversation_id")
    sequence_number = None
    if conversation_id:
        sequence_number = await _next_listening_sequence(user_id, space_id)
        metadata = AudioChunkMetadata(
            conversationId=conversation_id,
            userId=user_id,
            spaceId=space_id,
            chunkId=job_id,
            sequenceNumber=sequence_number,
            filePath=file_path,
            filename=filename,
            contentType=file.content_type,
        )
        await ConversationRepository(get_database()).record_audio_chunk(metadata)

    job = {
        "job_id": job_id,
        "user_id": user_id,
        "space_id": space_id,
        "file_path": file_path,
        "filename": filename,
        "content_type": file.content_type,
        "status": "queued",
    }
    if conversation_id:
        job["conversation_id"] = conversation_id
        job["sequence_number"] = sequence_number

    await redis_client.hset(f"speech_job:{job_id}", mapping=job)

    await push_speech_job(job)
    print(
        "Transcript speech job queued:",
        {
            "job_id": job_id,
            "user_id": user_id,
            "space_id": space_id,
        },
    )

    return {
        "job_id": job_id,
        "user_id": user_id,
        "space_id": space_id,
        "conversation_id": conversation_id,
        "sequence_number": sequence_number,
        "filename": filename,
        "status": "queued",
    }


async def transcribe_audio_batch_service(
    files: list[UploadFile],
    user_id: str,
    space_id: str,
):
    """Persist multiple audio files and enqueue one transcription job per file."""
    jobs = []

    for file in files:
        jobs.append(
            await transcribe_audio_service(
                file=file,
                user_id=user_id,
                space_id=space_id,
            )
        )

    return {
        "user_id": user_id,
        "space_id": space_id,
        "total_files": len(jobs),
        "jobs": jobs,
    }


async def get_transcribe_job_result_service(job_id: str):
    return await get_job_result(job_id)


def _listening_session_key(user_id: str, space_id: str) -> str:
    return f"speech_listening_session:{user_id}:{space_id}"


async def start_listening_session_service(user_id: str, space_id: str):
    existing = await redis_client.hgetall(_listening_session_key(user_id, space_id))
    if existing.get("status") == "listening" and existing.get("conversation_id"):
        return {
            "success": True,
            "message": "Listening session already started.",
            "data": existing,
        }

    conversation = await ConversationService().start(user_id, space_id)
    session = {
        "user_id": user_id,
        "space_id": space_id,
        "conversation_id": conversation["conversationId"],
        "status": "listening",
        "started_at": utc_now().isoformat(),
        "last_sequence_number": "-1",
    }
    await redis_client.hset(_listening_session_key(user_id, space_id), mapping=session)

    return {
        "success": True,
        "message": "Listening session started.",
        "data": session,
    }


async def end_listening_session_service(
    user_id: str,
    space_id: str,
    conversation_id: str | None = None,
    last_sequence_number: int | None = None,
    stopped_at_client=None,
):
    if conversation_id:
        data = await ConversationService().stop(
            conversation_id=conversation_id,
            user_id=user_id,
            space_id=space_id,
            last_sequence_number=last_sequence_number or 0,
            stopped_at_client=stopped_at_client,
        )
        return {
            "success": True,
            "message": "Conversation stop accepted; processing will continue asynchronously.",
            "data": data,
        }

    key = _listening_session_key(user_id, space_id)
    existing = await redis_client.hgetall(key)
    active_conversation_id = existing.get("conversation_id")
    if active_conversation_id and existing.get("status") == "listening":
        data = await ConversationService().stop(
            conversation_id=active_conversation_id,
            user_id=user_id,
            space_id=space_id,
            last_sequence_number=int(existing.get("last_sequence_number") or -1),
            stopped_at_client=stopped_at_client,
        )
    else:
        data = {
            "user_id": user_id,
            "space_id": space_id,
            "accepted": True,
        }

    stopped_at = utc_now().isoformat()
    await redis_client.hset(
        key,
        mapping={
            "user_id": user_id,
            "space_id": space_id,
            "conversation_id": active_conversation_id or "",
            "status": "stopped",
            "stopped_at": stopped_at,
        },
    )
    await redis_client.expire(key, 86400)

    return {
        "success": True,
        "message": "Listening session ended.",
        "data": {
            "user_id": user_id,
            "space_id": space_id,
            "status": "stopped",
            "stopped_at": stopped_at,
            "had_active_session": bool(existing),
            "conversation_id": active_conversation_id,
            "conversation": data,
        },
    }


async def _get_or_create_listening_session(user_id: str, space_id: str) -> dict:
    key = _listening_session_key(user_id, space_id)
    existing = await redis_client.hgetall(key)
    if existing.get("status") == "listening" and existing.get("conversation_id"):
        return existing

    result = await start_listening_session_service(user_id, space_id)
    return result["data"]


async def _next_listening_sequence(user_id: str, space_id: str) -> int:
    key = _listening_session_key(user_id, space_id)
    return int(await redis_client.hincrby(key, "last_sequence_number", 1))
