# apps/api_gateway/workers/vector_worker.py

import asyncio

from services.queue.redis_queue import (
    delete_speech_job,
    get_job_result,
    mark_transcript_job_vectorized,
    maybe_queue_transcript_analysis_for_session,
    pop_completed_speech_job,
)

from services.vector.vector_service import (
    store_transcript_in_vector_db,
)


async def start_vector_consumer():
    print("Vector worker started...")

    while True:
        job = None
        pending_released = False
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
            session_id = str(job.get("session_id") or "").strip() or None

            if not transcript or not user_id or not space_id:
                print("Invalid completed job data:", job)
                if user_id and space_id:
                    session = await mark_transcript_job_vectorized(
                        user_id,
                        space_id,
                        session_id=session_id,
                    )
                    pending_released = True
                    await maybe_queue_transcript_analysis_for_session(
                        user_id=user_id,
                        space_id=space_id,
                        reason="invalid_completed_job",
                        session=session,
                    )
                continue

            await store_transcript_in_vector_db(
                user_id=user_id,
                space_id=space_id,
                job_id=job_id,
                transcript=transcript,
                language_code=language_code,
                request_id=request_id,
                session_id=session_id,
            )

            session = await mark_transcript_job_vectorized(
                user_id,
                space_id,
                session_id=session_id,
            )
            pending_released = True
            queued = await maybe_queue_transcript_analysis_for_session(
                user_id=user_id,
                space_id=space_id,
                request_id=request_id,
                reason="session_end_after_vectorized",
                session=session,
            )
            print(
                "Transcript vector job completed:",
                {
                    "job_id": job_id,
                    "user_id": user_id,
                    "space_id": space_id,
                    "session_id": session_id,
                    "session_status": session.get("status"),
                    "pending_jobs": int(session.get("pending_jobs") or 0),
                    "analysis_queued": queued,
                },
            )

            await delete_speech_job(job_id)

        except Exception as error:
            print("Vector worker error:", str(error))
            if job and not pending_released:
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
                        reason="vector_job_failed",
                        session=session,
                    )
            await asyncio.sleep(2)
