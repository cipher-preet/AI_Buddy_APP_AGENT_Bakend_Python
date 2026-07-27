from services.speech.speech_service import (
    end_listening_session_service,
    get_transcribe_job_result_service,
    start_listening_session_service,
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


async def start_listening_session_controller(user_id: str, space_id: str):
    result = await start_listening_session_service(user_id=user_id, space_id=space_id)
    return {
        "success": True,
        "message": "Listening session started.",
        "data": result,
    }


async def end_listening_session_controller(user_id: str, space_id: str):
    result = await end_listening_session_service(user_id=user_id, space_id=space_id)
    return {
        "success": True,
        "message": "Listening session ended.",
        "data": result,
    }
