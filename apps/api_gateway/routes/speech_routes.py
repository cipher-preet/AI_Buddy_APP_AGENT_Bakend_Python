from fastapi import APIRouter, UploadFile, File, Form
from apps.api_gateway.controllers.speech_controller import (
    transcribe_audio_controller,
    get_transcribe_result_controller,
)

router = APIRouter()


@router.post("/transcripting")
async def transcribe(
    user_id: str = Form(...),
    space_id: str = Form(...),
    files: list[UploadFile] = File(..., alias="file"),
):
    return await transcribe_audio_controller(
        files=files, user_id=user_id, space_id=space_id
    )


@router.get("/transcribe/{job_id}")
async def get_transcribe_result(job_id: str):
    return await get_transcribe_result_controller(job_id)
