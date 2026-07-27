from services.vector.transcript_vector_service import (
    store_transcript_in_vector_db as save_to_qdrant,
)


async def store_transcript_in_vector_db(
    user_id: str,
    space_id: str,
    job_id: str,
    transcript: str,
    language_code: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
):

    result = await save_to_qdrant(
        user_id=user_id,
        space_id=space_id,
        job_id=job_id,
        transcript=transcript,
        language_code=language_code,
        request_id=request_id,
        session_id=session_id,
    )

    return result
