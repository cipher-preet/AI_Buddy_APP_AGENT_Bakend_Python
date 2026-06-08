import os
import uuid
import aiofiles

from services.queue.redis_queue import push_speech_job, redis_client, get_job_result

UPLOAD_DIR = "resources/audio_jobs"


async def transcribe_audio_service(file, user_id: str, space_id: str):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    job_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
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
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "queued",
    }

    await redis_client.hset(f"speech_job:{job_id}", mapping=job)

    await push_speech_job(job)

    return {
        "job_id": job_id,
        "user_id": user_id,
        "space_id": space_id,
        "status": "queued",
    }


async def get_transcribe_job_result_service(job_id: str):
    return await get_job_result(job_id)
