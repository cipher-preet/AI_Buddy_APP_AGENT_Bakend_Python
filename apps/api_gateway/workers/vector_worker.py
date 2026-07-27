# apps/api_gateway/workers/vector_worker.py

import asyncio
from pathlib import Path

from services.queue.redis_queue import (
    delete_speech_job,
    get_job_result,
    pop_completed_speech_job,
)

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


async def start_vector_consumer():
    print("Vector worker started...")

    while True:
        job = None
        try:
            job_id = await pop_completed_speech_job()

            if not job_id:
                await asyncio.sleep(1)
                continue


            job = await get_job_result(job_id)

            if not job:
                continue

            if job.get("status") != "completed":
                continue

            result = job.get("result") or {}

            transcript = result.get("transcript")
            language_code = result.get("language_code")
            request_id = result.get("request_id")

            user_id = str(job.get("user_id") or "").strip()
            space_id = str(job.get("space_id") or "").strip()

            if not transcript or not user_id or not space_id:
                print("Invalid completed job data:", job)
                continue

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
                    "audio_removed": audio_removed,
                },
            )

            await delete_speech_job(job_id)

        except Exception as error:
            print("Vector worker error:", str(error))
            await asyncio.sleep(2)
