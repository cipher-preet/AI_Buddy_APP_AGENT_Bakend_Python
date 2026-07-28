from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from apps.api_gateway.config.setting import settings


_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URL, uuidRepresentation="standard")
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_mongo_client()[settings.MONGODB_DATABASE]


async def close_mongo_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def ensure_mongo_indexes(db: AsyncIOMotorDatabase | None = None) -> None:
    database = db or get_database()

    await database.conversations.create_indexes(
        [
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("createdAt", DESCENDING)]),
            IndexModel([("status", ASCENDING), ("updatedAt", ASCENDING)]),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("status", ASCENDING)]),
        ]
    )
    await database.transcript_chunks.create_indexes(
        [
            IndexModel([("conversationId", ASCENDING), ("sequenceNumber", ASCENDING)], unique=True),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("createdAt", DESCENDING)]),
            IndexModel([("conversationId", ASCENDING), ("sttStatus", ASCENDING)]),
            IndexModel([("conversationId", ASCENDING), ("processingStatus", ASCENDING)]),
            IndexModel([("expiresAt", ASCENDING)], expireAfterSeconds=0),
        ]
    )
    await database.audio_chunks.create_indexes(
        [
            IndexModel([("conversationId", ASCENDING), ("sequenceNumber", ASCENDING)], unique=True),
            IndexModel([("conversationId", ASCENDING), ("chunkId", ASCENDING)], unique=True),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("createdAt", DESCENDING)]),
        ]
    )
    await database.conversation_summaries.create_indexes(
        [
            IndexModel([("conversationId", ASCENDING)], unique=True),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("createdAt", DESCENDING)]),
        ]
    )
    await database.space_memory.create_indexes(
        [IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING)], unique=True)]
    )
    await database.extraction_runs.create_indexes(
        [
            IndexModel([("conversationId", ASCENDING), ("processingVersion", ASCENDING)], unique=True),
            IndexModel([("status", ASCENDING), ("updatedAt", ASCENDING)]),
        ]
    )
    await database.tasks.create_indexes(
        [
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("sourceConversationId", ASCENDING)]),
            IndexModel([("fingerprint", ASCENDING)], unique=True, sparse=True),
        ]
    )
