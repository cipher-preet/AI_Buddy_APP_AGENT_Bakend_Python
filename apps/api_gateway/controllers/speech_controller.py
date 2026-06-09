from services.speech.speech_service import (
    transcribe_audio_batch_service,
    get_transcribe_job_result_service,
)


async def transcribe_audio_controller(files, user_id: str, space_id: str):
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
