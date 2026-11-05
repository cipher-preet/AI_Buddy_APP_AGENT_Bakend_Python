from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError

from apps.api_gateway.auth import forwarded_user_token, require_matching_user, require_user
from apps.api_gateway.controllers.chat_controller import (
    ask_chat_controller,
    create_chat_session_controller,
    list_chats_controller,
    load_chat_controller,
)
from services.auth.identity import AuthenticatedUser
from services.auth.space_access import AccessDenied


router = APIRouter(dependencies=[Depends(require_user)])


class AskChatRequest(BaseModel):
    userId: str
    question: str = Field(min_length=1)
    spaceId: str | None = None
    spaceIds: list[str] = Field(default_factory=list)
    chatId: str | None = None
    authToken: str | None = None
    meetingId: str | None = None


class CreateChatSessionRequest(BaseModel):
    userId: str
    spaceId: str | None = None
    meetingId: str | None = None


@router.post("/sessions")
async def create_chat_session(
    request: CreateChatSessionRequest,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = require_matching_user(actor, request.userId)
    return await _call(create_chat_session_controller(user_id, request.spaceId, request.meetingId))


@router.post("/ask")
async def ask_chat(
    request: AskChatRequest,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = require_matching_user(actor, request.userId)
    auth_token = forwarded_user_token(actor, request.authToken)
    return await _call(
        ask_chat_controller(
            user_id=user_id,
            space_id=request.spaceId,
            space_ids=request.spaceIds,
            chat_id=request.chatId,
            question=request.question,
            auth_token=auth_token,
            meeting_id=request.meetingId,
        )
    )


def _card_view(view: str | None) -> bool:
    return view == "card"


@router.get("/sessions")
async def list_chat_sessions(
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Query(..., alias="userId"),
    space_id: str | None = Query(default=None, alias="spaceId"),
    meeting_id: str | None = Query(default=None, alias="meetingId"),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    view: str | None = Query(default=None),
):
    verified_user_id = require_matching_user(actor, user_id)
    return await _call(
        list_chats_controller(
            verified_user_id,
            space_id,
            limit,
            cursor,
            meeting_id,
            card=_card_view(view),
        )
    )


@router.get("/sessions/{session_id}")
async def load_chat_by_session_id(
    session_id: str,
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Query(..., alias="userId"),
    view: str | None = Query(default=None),
):
    verified_user_id = require_matching_user(actor, user_id)
    return await _call(load_chat_controller(verified_user_id, session_id, card=_card_view(view)))


@router.get("")
async def list_chats(
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Query(..., alias="userId"),
    space_id: str | None = Query(default=None, alias="spaceId"),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    view: str | None = Query(default=None),
):
    verified_user_id = require_matching_user(actor, user_id)
    return await _call(
        list_chats_controller(verified_user_id, space_id, limit, cursor, card=_card_view(view))
    )


@router.get("/{chat_id}")
async def load_chat(
    chat_id: str,
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Query(..., alias="userId"),
    view: str | None = Query(default=None),
):
    verified_user_id = require_matching_user(actor, user_id)
    return await _call(load_chat_controller(verified_user_id, chat_id, card=_card_view(view)))


async def _call(awaitable):
    try:
        return await awaitable
    except AccessDenied as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (AutoReconnect, ServerSelectionTimeoutError) as error:
        raise HTTPException(
            status_code=503,
            detail="Database connection was interrupted. Please try again.",
        ) from error
