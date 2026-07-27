import os
import uuid
import aiofiles
from fastapi import UploadFile

from services.queue.redis_queue import (
    get_job_result,
    mark_transcript_session_ended,
    mark_transcript_session_started,
    maybe_queue_transcript_analysis_for_session,
    push_speech_job,
    redis_client,
    register_transcript_job_queued,
)

UPLOAD_DIR = "resources/audio_jobs"


async def transcribe_audio_service(file: UploadFile, user_id: str, space_id: str):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    job_id = str(uuid.uuid4())
    filename = file.filename or f"{job_id}.audio"
    file_extension = filename.rsplit(".", 1)[-1] if "." in filename else "audio"
    saved_filename = f"{job_id}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    session = await register_transcript_job_queued(user_id, space_id)
    job = {
        "job_id": job_id,
        "user_id": user_id,
        "space_id": space_id,
        "session_id": session.get("session_id", ""),
        "file_path": file_path,
        "filename": filename,
        "content_type": file.content_type,
        "status": "queued",
    }

    await redis_client.hset(f"speech_job:{job_id}", mapping=job)

    await push_speech_job(job)
    print(
        "Transcript speech job queued:",
        {
            "job_id": job_id,
            "user_id": user_id,
            "space_id": space_id,
            "session_id": session.get("session_id"),
            "pending_jobs": int(session.get("pending_jobs") or 0),
        },
    )

    return {
        "job_id": job_id,
        "user_id": user_id,
        "space_id": space_id,
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


async def start_listening_session_service(user_id: str, space_id: str):
    user_id = user_id.strip()
    space_id = space_id.strip()
    session = await mark_transcript_session_started(user_id, space_id)
    print(
        "Transcript session started:",
        {
            "user_id": user_id,
            "space_id": space_id,
            "session_id": session.get("session_id"),
        },
    )
    return {
        "user_id": user_id,
        "space_id": space_id,
        "session_id": session.get("session_id"),
        "status": session.get("status", "active"),
    }


async def end_listening_session_service(user_id: str, space_id: str):
    user_id = user_id.strip()
    space_id = space_id.strip()
    session = await mark_transcript_session_ended(user_id, space_id)
    queued = await maybe_queue_transcript_analysis_for_session(
        user_id=user_id,
        space_id=space_id,
        reason="session_end",
        session=session,
        force=True,
    )
    print(
        "Transcript session ended:",
        {
            "user_id": user_id,
            "space_id": space_id,
            "session_id": session.get("session_id"),
            "pending_jobs": int(session.get("pending_jobs") or 0),
            "analysis_queued": queued,
        },
    )
    return {
        "user_id": user_id,
        "space_id": space_id,
        "session_id": session.get("session_id"),
        "status": session.get("status", "ended"),
        "pending_jobs": int(session.get("pending_jobs") or 0),
        "analysis_queued": queued,
    }
