import asyncio

from services.queue.redis_queue import (
    mark_job_processing,
    mark_job_failed,
    pop_speech_job,
    push_completed_speech_job,
    save_job_result,
)

from services.speech.providers.sarvam_provider import (
    sarvam_transcribe_from_path,
)


async def start_speech_consumer():
    print("Speech worker started...")

    while True:
        job_id = None
        job = None
        try:
            job = await pop_speech_job()

            if not job:
                await asyncio.sleep(1)
                continue

            job_id = job["job_id"]

            print("Processing speech job:", job_id)

            await mark_job_processing(job_id)

            result = await sarvam_transcribe_from_path(
                file_path=job["file_path"],
                filename=job["filename"],
                content_type=job["content_type"],
            )

            await save_job_result(job_id, result)

            await push_completed_speech_job(job_id)

            print("Speech job completed:", job_id)

        except Exception as error:
            print("Speech worker error:", str(error))

            if job_id:
                await mark_job_failed(job_id, str(error))

            await asyncio.sleep(2)
