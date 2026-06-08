# apps/api_gateway/workers/vector_worker.py

import asyncio

from services.queue.redis_queue import (
    pop_completed_speech_job,
    get_job_result,
    delete_speech_job,
)

from services.vector.vector_service import (
    store_transcript_in_vector_db,
)


async def start_vector_consumer():
    print("Vector worker started...")

    while True:
        try:
            job_id = await pop_completed_speech_job()

            if not job_id:
                await asyncio.sleep(1)
                continue

            print("Vector worker picked job:", job_id)

            job = await get_job_result(job_id)

            if not job:
                print("Job not found in Redis:", job_id)
                continue

            if job.get("status") != "completed":
                print("Job not completed yet:", job_id)
                continue

            result = job.get("result") or {}

            transcript = result.get("transcript")
            language_code = result.get("language_code")
            request_id = result.get("request_id")

            user_id = job.get("user_id")
            space_id = job.get("space_id")

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

            await delete_speech_job(job_id)

            print("Job stored in vector DB and deleted from Redis:", job_id)

        except Exception as error:
            print("Vector worker error:", str(error))
            await asyncio.sleep(2)
