from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File, Form

from apps.api_gateway.auth import raise_for_access, require_matching_user, require_user
from apps.api_gateway.controllers.speech_controller import (
    end_listening_controller,
    get_transcribe_result_controller,
    start_listening_controller,
    transcribe_audio_controller,
)
from services.auth.identity import AuthenticatedUser
from services.auth.space_access import AccessDenied, require_space_operation
from services.db.mongo import get_database

router = APIRouter(dependencies=[Depends(require_user)])


async def _require_listening_owner(actor: AuthenticatedUser, user_id: Any, space_id: Any) -> str:
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required.")
    if not space_id:
        raise HTTPException(status_code=400, detail="spaceId is required.")
    verified_user_id = require_matching_user(actor, user_id)
    try:
        await require_space_operation(get_database(), verified_user_id, str(space_id).strip(), "write")
    except AccessDenied as error:
        raise_for_access(error)
    return verified_user_id


@router.post("/transcripting")
async def transcribe(
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Form(...),
    space_id: str = Form(...),
    files: list[UploadFile] = File(..., alias="file"),
):
    verified_user_id = await _require_listening_owner(actor, user_id, space_id)
    return await transcribe_audio_controller(
        files=files, user_id=verified_user_id, space_id=space_id.strip()
    )


@router.get("/transcribe/{job_id}")
async def get_transcribe_result(
    job_id: str,
    actor: AuthenticatedUser = Depends(require_user),
):
    return await get_transcribe_result_controller(job_id, actor.user_id)


@router.post("/listening/start")
async def start_listening(
    actor: AuthenticatedUser = Depends(require_user),
    request: dict[str, Any] | None = Body(default=None),
):
    request = request or {}
    user_id = _first_present(request, "userId", "user_id")
    space_id = _first_present(request, "spaceId", "space_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required to start listening.")
    if not space_id:
        raise HTTPException(status_code=400, detail="spaceId is required to start listening.")
    verified_user_id = await _require_listening_owner(actor, user_id, space_id)
    return await start_listening_controller(user_id=verified_user_id, space_id=str(space_id).strip())


@router.post("/listening/end")
async def end_listening(
    actor: AuthenticatedUser = Depends(require_user),
    request: dict[str, Any] | None = Body(default=None),
):
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
    verified_user_id = await _require_listening_owner(actor, user_id, space_id)

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
            user_id=verified_user_id,
            space_id=str(space_id).strip(),
            conversation_id=str(conversation_id).strip() if conversation_id else None,
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
