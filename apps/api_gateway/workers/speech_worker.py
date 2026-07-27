import asyncio

from services.queue.redis_queue import (
    mark_job_processing,
    mark_job_failed,
    mark_transcript_job_vectorized,
    maybe_queue_transcript_analysis_for_session,
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
            if job:
                user_id = str(job.get("user_id") or "").strip()
                space_id = str(job.get("space_id") or "").strip()
                session_id = str(job.get("session_id") or "").strip() or None
                if user_id and space_id:
                    session = await mark_transcript_job_vectorized(
                        user_id,
                        space_id,
                        session_id=session_id,
                    )
                    await maybe_queue_transcript_analysis_for_session(
                        user_id=user_id,
                        space_id=space_id,
                        reason="speech_job_failed",
                        session=session,
                    )

            await asyncio.sleep(2)
