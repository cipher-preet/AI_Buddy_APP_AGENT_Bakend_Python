from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, IndexModel

from apps.api_gateway.config.setting import settings
from services.chat.langchain_history import MotorMongoChatMessageHistory
from services.chat.models import (
    MAX_CHAT_MESSAGES,
    ChatSessionDocument,
    utc_now,
)
from services.db.mongo import get_database

try:
    from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
except ImportError:
    MongoDBChatMessageHistory = None


class ChatRepository:
    def __init__(self, db: AsyncIOMotorDatabase | None = None):
        self.db = db or get_database()

    async def get_session(self, chat_id: str) -> ChatSessionDocument | None:
        document = await self.db.chat_sessions.find_one(_id_query(chat_id))
        return ChatSessionDocument.model_validate(document) if document else None

    async def get_active_or_create_session(
        self,
        user_id: str,
        space_id: str | None,
        chat_id: str | None = None,
    ) -> ChatSessionDocument:
        if chat_id:
            session = await self.get_session(chat_id)
            if not session:
                raise ValueError("Chat session not found")
            if not same_mongo_id(session.userId, user_id):
                raise PermissionError("Chat session does not belong to this user or space")
            if space_id is not None and not same_mongo_id(session.spaceId, space_id):
                raise PermissionError("Chat session does not belong to this user or space")
            session_space_id = space_id if space_id is not None else str(session.spaceId) if session.spaceId is not None else None
            if not is_native_object_id(session.id) or not is_native_object_id(session.userId):
                await self.archive_session(session.id)
                return await self.create_session(user_id, session_space_id)
            if session.messageCount + 2 <= MAX_CHAT_MESSAGES:
                return session
            await self.archive_session(session.id)
            return await self.create_session(user_id, session_space_id)

        query = {
            "userId": to_mongo_id(user_id),
            "status": "active",
            "_id": {"$type": "objectId"},
            "messageCount": {"$lte": MAX_CHAT_MESSAGES - 2},
        }
        if space_id is not None:
            query["spaceId"] = to_mongo_id(space_id)
        document = await self.db.chat_sessions.find_one(
            query,
            sort=[("updatedAt", -1)],
        )
        if document:
            return ChatSessionDocument.model_validate(document)
        return await self.create_session(user_id, space_id)

    async def create_session(self, user_id: str, space_id: str | None) -> ChatSessionDocument:
        session = ChatSessionDocument(userId=to_mongo_id(user_id), spaceId=to_mongo_id(space_id))
        await self.db.chat_sessions.insert_one(session.model_dump(by_alias=True))
        return session

    async def archive_session(self, chat_id: Any) -> None:
        await self.db.chat_sessions.update_one(
            _id_query(chat_id),
            {"$set": {"status": "archived", "updatedAt": utc_now()}},
        )

    async def ensure_chat_history_indexes(self) -> None:
        indexes = await self.db.chat_message_store.index_information()
        session_index = indexes.get("SessionId_1")
        if session_index and session_index.get("unique"):
            await self.db.chat_message_store.drop_index("SessionId_1")
        await self.db.chat_message_store.create_indexes(
            [
                IndexModel([("SessionId", ASCENDING)]),
                IndexModel([("SessionId", ASCENDING), ("createdAt", ASCENDING)]),
            ]
        )

    def get_message_history(self, chat_id: Any, history_size: int = MAX_CHAT_MESSAGES) -> MotorMongoChatMessageHistory:
        session_id = str(chat_id)
        if MongoDBChatMessageHistory is not None:
            return MongoDBChatMessageHistory(
                connection_string=settings.MONGODB_URL,
                session_id=session_id,
                database_name=settings.MONGODB_DATABASE,
                collection_name="chat_message_store",
                session_id_key="SessionId",
                history_key="History",
                create_index=False,
                history_size=history_size,
                index_kwargs={"unique": False},
            )
        return MotorMongoChatMessageHistory(session_id, db=self.db, history_size=history_size)

    async def list_recent_messages(self, chat_id: Any, limit: int = MAX_CHAT_MESSAGES):
        history = self.get_message_history(chat_id, limit)
        return await history.aget_messages()

    async def sync_message_count(self, chat_id: Any) -> int:
        count = len(await self.list_recent_messages(chat_id, MAX_CHAT_MESSAGES))
        await self.db.chat_sessions.update_one(
            _id_query(chat_id),
            {"$set": {"messageCount": count, "updatedAt": utc_now()}},
        )
        return count

    async def list_sessions(
        self,
        user_id: str,
        space_id: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> list[ChatSessionDocument]:
        query: dict[str, Any] = {
            "userId": to_mongo_id(user_id),
            "_id": {"$type": "objectId"},
        }
        if space_id is not None:
            query["spaceId"] = to_mongo_id(space_id)
        if cursor:
            cursor_updated_at, cursor_id = decode_sessions_cursor(cursor)
            query["$or"] = [
                {"updatedAt": {"$lt": cursor_updated_at}},
                {"updatedAt": cursor_updated_at, "_id": {"$lt": cursor_id}},
            ]
        mongo_cursor = (
            self.db.chat_sessions.find(query)
            .sort([("updatedAt", -1), ("_id", -1)])
            .limit(limit)
        )
        return [ChatSessionDocument.model_validate(document) async for document in mongo_cursor]

    async def touch_title(self, chat_id: Any, title: str) -> None:
        await self.db.chat_sessions.update_one(
            {**_id_query(chat_id), "title": None},
            {"$set": {"title": title[:80], "updatedAt": utc_now()}},
        )

    async def set_pending_action(self, chat_id: Any, pending_action: dict[str, Any]) -> None:
        await self.db.chat_sessions.update_one(
            _id_query(chat_id),
            {"$set": {"pendingAction": pending_action, "updatedAt": utc_now()}},
        )

    async def clear_pending_action(self, chat_id: Any) -> None:
        await self.db.chat_sessions.update_one(
            _id_query(chat_id),
            {"$unset": {"pendingAction": ""}, "$set": {"updatedAt": utc_now()}},
        )

    async def set_session_space(self, chat_id: Any, space_id: str) -> None:
        await self.db.chat_sessions.update_one(
            _id_query(chat_id),
            {"$set": {"spaceId": to_mongo_id(space_id), "updatedAt": utc_now()}},
        )


def to_mongo_id(value: Any) -> Any:
    if value is None or isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    return value


def same_mongo_id(left: Any, right: Any) -> bool:
    return to_mongo_id(left) == to_mongo_id(right)


def is_native_object_id(value: Any) -> bool:
    return isinstance(value, ObjectId)


def _id_query(value: Any) -> dict[str, Any]:
    return {"_id": to_mongo_id(value)}


def encode_sessions_cursor(session: ChatSessionDocument) -> str:
    payload = {
        "updatedAt": session.updatedAt.isoformat(),
        "id": str(session.id),
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_sessions_cursor(cursor: str) -> tuple[datetime, ObjectId]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8"))
        updated_at = datetime.fromisoformat(str(payload["updatedAt"]))
        cursor_id = to_mongo_id(payload["id"])
    except Exception as error:
        raise ValueError("Invalid chat sessions cursor") from error
    if not isinstance(cursor_id, ObjectId):
        raise ValueError("Invalid chat sessions cursor")
    return updated_at, cursor_id
