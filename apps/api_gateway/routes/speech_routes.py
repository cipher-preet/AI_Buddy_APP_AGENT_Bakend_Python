from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, HTTPException, UploadFile, File, Form

from apps.api_gateway.controllers.speech_controller import (
    end_listening_controller,
    get_transcribe_result_controller,
    start_listening_controller,
    transcribe_audio_controller,
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


@router.post("/listening/start")
async def start_listening(request: dict[str, Any] | None = Body(default=None)):
    request = request or {}
    user_id = _first_present(request, "userId", "user_id")
    space_id = _first_present(request, "spaceId", "space_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required to start listening.")
    if not space_id:
        raise HTTPException(status_code=400, detail="spaceId is required to start listening.")

    return await start_listening_controller(user_id=user_id, space_id=space_id)


@router.post("/listening/end")
async def end_listening(request: dict[str, Any] | None = Body(default=None)):
    request = request or {}
    conversation_id = _first_present(request, "conversationId", "conversation_id")
    user_id = _first_present(request, "userId", "user_id")
    space_id = _first_present(request, "spaceId", "space_id")
    last_sequence_number = _first_present(
        request, "lastSequenceNumber", "last_sequence_number", "sequenceNumber", "sequence_number"
    )
    stopped_at_client = request.get("stoppedAtClient", request.get("stopped_at_client"))

    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required to stop listening.")
    if not space_id:
        raise HTTPException(status_code=400, detail="spaceId is required to stop listening.")

    try:
        if last_sequence_number is not None:
            last_sequence_number = int(last_sequence_number)
            if last_sequence_number < 0:
                raise ValueError("lastSequenceNumber must be greater than or equal to 0")
        stopped_at_client = (
            datetime.fromisoformat(stopped_at_client)
            if isinstance(stopped_at_client, str)
            else stopped_at_client
        )
        return await end_listening_controller(
            user_id=user_id,
            space_id=space_id,
            conversation_id=conversation_id,
            last_sequence_number=last_sequence_number,
            stopped_at_client=stopped_at_client,
        )
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _first_present(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None
