from services.speech.speech_service import (
    get_transcribe_job_result_service,
    start_listening_session_service,
    end_listening_session_service,
    transcribe_audio_batch_service,
)


async def transcribe_audio_controller(files, user_id: str, space_id: str):
    user_id = user_id.strip()
    space_id = space_id.strip()

    result = await transcribe_audio_batch_service(
        files=files, user_id=user_id, space_id=space_id
    )

    return {
        "success": True,
        "message": "Audio files added to transcription queue.",
        "data": result,
    }


async def get_transcribe_result_controller(job_id: str):

    result = await get_transcribe_job_result_service(job_id)

    if not result:
        return {"success": False, "message": "Job not found."}

    return {"success": True, "data": result}


async def end_listening_controller(
    user_id: str,
    space_id: str,
    conversation_id: str | None = None,
    last_sequence_number: int | None = None,
    stopped_at_client=None,
):
    return await end_listening_session_service(
        user_id=user_id.strip(),
        space_id=space_id.strip(),
        conversation_id=conversation_id.strip() if conversation_id else None,
        last_sequence_number=last_sequence_number,
        stopped_at_client=stopped_at_client,
    )


async def start_listening_controller(user_id: str, space_id: str):
    return await start_listening_session_service(
        user_id=user_id.strip(),
        space_id=space_id.strip(),
    )
