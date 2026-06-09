import os
import uuid
import aiofiles
from fastapi import UploadFile

from services.queue.redis_queue import push_speech_job, redis_client, get_job_result

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

    job = {
        "job_id": job_id,
        "user_id": user_id,
        "space_id": space_id,
        "file_path": file_path,
        "filename": filename,
        "content_type": file.content_type,
        "status": "queued",
    }

    await redis_client.hset(f"speech_job:{job_id}", mapping=job)

    await push_speech_job(job)

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
