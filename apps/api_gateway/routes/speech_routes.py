from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel, Field

from apps.api_gateway.controllers.speech_controller import (
    end_listening_session_controller,
    get_transcribe_result_controller,
    start_listening_session_controller,
    transcribe_audio_controller,
)

router = APIRouter()


class ListeningSessionPayload(BaseModel):
    user_id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)


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


@router.post("/listening/start")
async def start_listening_session(payload: ListeningSessionPayload):
    return await start_listening_session_controller(
        user_id=payload.user_id,
        space_id=payload.space_id,
    )


@router.post("/listening/end")
async def end_listening_session(payload: ListeningSessionPayload):
    return await end_listening_session_controller(
        user_id=payload.user_id,
        space_id=payload.space_id,
    )
