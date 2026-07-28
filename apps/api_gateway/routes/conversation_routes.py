from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from apps.api_gateway.controllers.conversation_controller import (
    conversation_status_controller,
    ingest_audio_controller,
    start_conversation_controller,
    stop_conversation_controller,
)


router = APIRouter()


class StartConversationRequest(BaseModel):
    userId: str
    spaceId: str


class StopConversationRequest(BaseModel):
    userId: str
    spaceId: str
    lastSequenceNumber: int = Field(ge=0)
    stoppedAtClient: datetime | None = None


@router.post("/start")
async def start_conversation(request: StartConversationRequest):
    return await _call(start_conversation_controller(request.userId, request.spaceId))


@router.post("/{conversation_id}/audio")
async def ingest_audio(
    conversation_id: str,
    user_id: str = Form(...),
    space_id: str = Form(...),
    chunk_id: str = Form(...),
    sequence_number: int = Form(...),
    captured_at: datetime | None = Form(default=None),
    duration_ms: int | None = Form(default=None),
    file: UploadFile = File(...),
):
    return await _call(
        ingest_audio_controller(
            conversation_id=conversation_id,
            user_id=user_id,
            space_id=space_id,
            chunk_id=chunk_id,
            sequence_number=sequence_number,
            captured_at=captured_at,
            duration_ms=duration_ms,
            file=file,
        )
    )


@router.post("/{conversation_id}/stop")
async def stop_conversation(conversation_id: str, request: StopConversationRequest):
    return await _call(
        stop_conversation_controller(
            conversation_id=conversation_id,
            user_id=request.userId,
            space_id=request.spaceId,
            last_sequence_number=request.lastSequenceNumber,
            stopped_at_client=request.stoppedAtClient,
        )
    )


@router.get("/{conversation_id}/status")
async def conversation_status(conversation_id: str, user_id: str, space_id: str):
    return await _call(conversation_status_controller(conversation_id, user_id, space_id))


async def _call(awaitable):
    try:
        return await awaitable
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
