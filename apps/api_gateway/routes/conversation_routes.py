from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from apps.api_gateway.auth import raise_for_access, require_matching_user, require_user
from apps.api_gateway.controllers.conversation_controller import (
    conversation_status_controller,
    complete_audio_upload_controller,
    create_audio_upload_url_controller,
    ingest_audio_controller,
    start_conversation_controller,
    stop_conversation_controller,
)
from services.auth.identity import AuthenticatedUser
from services.auth.space_access import AccessDenied, require_space_operation
from services.db.mongo import get_database
from services.storage.s3_audio_storage import PermanentS3StorageError


router = APIRouter(dependencies=[Depends(require_user)])


class StartConversationRequest(BaseModel):
    userId: str
    spaceId: str


class StopConversationRequest(BaseModel):
    userId: str
    spaceId: str
    lastSequenceNumber: int = Field(ge=0)
    stoppedAtClient: datetime | None = None


class AudioUploadUrlRequest(BaseModel):
    userId: str
    spaceId: str
    sequenceNumber: int = Field(ge=0)
    contentType: str
    extension: str
    expectedSizeBytes: int = Field(gt=0)
    chunkId: str | None = None


class AudioUploadCompleteRequest(BaseModel):
    userId: str
    spaceId: str
    chunkId: str
    sequenceNumber: int = Field(ge=0)
    objectKey: str
    contentType: str
    sizeBytes: int = Field(gt=0)
    capturedAt: datetime | None = None
    durationMs: int | None = Field(default=None, ge=0)


async def _owner(actor: AuthenticatedUser, supplied_user_id: str, space_id: str) -> str:
    user_id = require_matching_user(actor, supplied_user_id)
    try:
        await require_space_operation(get_database(), user_id, space_id.strip(), "write")
    except AccessDenied as error:
        raise_for_access(error)
    return user_id


@router.post("/start")
async def start_conversation(
    request: StartConversationRequest,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = await _owner(actor, request.userId, request.spaceId)
    return await _call(start_conversation_controller(user_id, request.spaceId.strip()))


@router.post("/{conversation_id}/audio")
async def ingest_audio(
    conversation_id: str,
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Form(...),
    space_id: str = Form(...),
    chunk_id: str = Form(...),
    sequence_number: int = Form(...),
    captured_at: datetime | None = Form(default=None),
    duration_ms: int | None = Form(default=None),
    file: UploadFile = File(...),
):
    verified_user_id = await _owner(actor, user_id, space_id)
    return await _call(
        ingest_audio_controller(
            conversation_id=conversation_id,
            user_id=verified_user_id,
            space_id=space_id.strip(),
            chunk_id=chunk_id,
            sequence_number=sequence_number,
            captured_at=captured_at,
            duration_ms=duration_ms,
            file=file,
        )
    )


@router.post("/{conversation_id}/audio/upload-url")
async def create_audio_upload_url(
    conversation_id: str,
    request: AudioUploadUrlRequest,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = await _owner(actor, request.userId, request.spaceId)
    return await _call(
        create_audio_upload_url_controller(
            conversation_id=conversation_id,
            user_id=user_id,
            space_id=request.spaceId.strip(),
            sequence_number=request.sequenceNumber,
            content_type=request.contentType,
            extension=request.extension,
            expected_size_bytes=request.expectedSizeBytes,
            chunk_id=request.chunkId,
        )
    )


@router.post("/{conversation_id}/audio/complete")
async def complete_audio_upload(
    conversation_id: str,
    request: AudioUploadCompleteRequest,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = await _owner(actor, request.userId, request.spaceId)
    return await _call(
        complete_audio_upload_controller(
            conversation_id=conversation_id,
            user_id=user_id,
            space_id=request.spaceId.strip(),
            chunk_id=request.chunkId,
            sequence_number=request.sequenceNumber,
            object_key=request.objectKey,
            content_type=request.contentType,
            size_bytes=request.sizeBytes,
            captured_at=request.capturedAt,
            duration_ms=request.durationMs,
        )
    )


@router.post("/{conversation_id}/stop")
async def stop_conversation(
    conversation_id: str,
    request: StopConversationRequest,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = await _owner(actor, request.userId, request.spaceId)
    return await _call(
        stop_conversation_controller(
            conversation_id=conversation_id,
            user_id=user_id,
            space_id=request.spaceId.strip(),
            last_sequence_number=request.lastSequenceNumber,
            stopped_at_client=request.stoppedAtClient,
        )
    )


@router.get("/{conversation_id}/status")
async def conversation_status(
    conversation_id: str,
    user_id: str,
    space_id: str,
    actor: AuthenticatedUser = Depends(require_user),
):
    verified_user_id = require_matching_user(actor, user_id)
    return await _call(conversation_status_controller(conversation_id, verified_user_id, space_id))


async def _call(awaitable):
    try:
        return await awaitable
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except PermanentS3StorageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
