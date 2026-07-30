from __future__ import annotations

import json
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from motor.motor_asyncio import AsyncIOMotorDatabase

from services.chat.models import MAX_CHAT_MESSAGES, utc_now
from services.db.mongo import get_database


class MotorMongoChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        db: AsyncIOMotorDatabase | None = None,
        collection_name: str = "chat_message_store",
        history_size: int = MAX_CHAT_MESSAGES,
    ):
        self.session_id = session_id
        self.db = db or get_database()
        self.collection = self.db[collection_name]
        self.history_size = history_size

    @property
    def messages(self) -> list[BaseMessage]:
        raise RuntimeError("Use aget_messages() for async Mongo chat history")

    async def aget_messages(self) -> list[BaseMessage]:
        skip_count = 0
        if self.history_size:
            count = await self.collection.count_documents({"SessionId": self.session_id})
            skip_count = max(0, count - self.history_size)
        cursor = self.collection.find({"SessionId": self.session_id}).sort("createdAt", 1).skip(skip_count)
        items = []
        async for document in cursor:
            history = document.get("History")
            if isinstance(history, str):
                items.append(json.loads(history))
            elif isinstance(history, list):
                items.extend(history)
            elif isinstance(history, dict):
                items.append(history)
        return messages_from_dict(items[-self.history_size :])

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        now = utc_now()
        await self.collection.insert_many(
            [
                {
                    "SessionId": self.session_id,
                    "History": json.dumps(message_to_dict(message)),
                    "createdAt": now,
                }
                for message in messages
            ]
        )

    async def aclear(self) -> None:
        await self.collection.delete_many({"SessionId": self.session_id})

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        raise RuntimeError("Use aadd_messages() for async Mongo chat history")

    def clear(self) -> None:
        raise RuntimeError("Use aclear() for async Mongo chat history")
