from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api_gateway.controllers.chat_controller import (
    ask_chat_controller,
    create_chat_session_controller,
    list_chats_controller,
    load_chat_controller,
)


router = APIRouter()


class AskChatRequest(BaseModel):
    userId: str
    question: str = Field(min_length=1)
    spaceId: str | None = None
    chatId: str | None = None


class CreateChatSessionRequest(BaseModel):
    userId: str
    spaceId: str | None = None


@router.post("/sessions")
async def create_chat_session(request: CreateChatSessionRequest):
    return await _call(create_chat_session_controller(request.userId, request.spaceId))


@router.post("/ask")
async def ask_chat(request: AskChatRequest):
    return await _call(
        ask_chat_controller(
            user_id=request.userId,
            space_id=request.spaceId,
            chat_id=request.chatId,
            question=request.question,
        )
    )


@router.get("/sessions")
async def list_chat_sessions(
    user_id: str = Query(..., alias="userId"),
    space_id: str | None = Query(default=None, alias="spaceId"),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
):
    return await _call(list_chats_controller(user_id, space_id, limit, cursor))


@router.get("/sessions/{session_id}")
async def load_chat_by_session_id(
    session_id: str,
    user_id: str = Query(..., alias="userId"),
):
    return await _call(load_chat_controller(user_id, session_id))


@router.get("")
async def list_chats(
    user_id: str = Query(..., alias="userId"),
    space_id: str | None = Query(default=None, alias="spaceId"),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
):
    return await _call(list_chats_controller(user_id, space_id, limit, cursor))


@router.get("/{chat_id}")
async def load_chat(chat_id: str, user_id: str = Query(..., alias="userId")):
    return await _call(load_chat_controller(user_id, chat_id))


async def _call(awaitable):
    try:
        return await awaitable
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
