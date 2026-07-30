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
    await database.chat_sessions.create_indexes(
        [
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("status", ASCENDING), ("updatedAt", DESCENDING)]),
            IndexModel([("userId", ASCENDING), ("updatedAt", DESCENDING)]),
        ]
    )
    chat_history_indexes = await database.chat_message_store.index_information()
    session_index = chat_history_indexes.get("SessionId_1")
    if session_index and session_index.get("unique"):
        await database.chat_message_store.drop_index("SessionId_1")
    await database.chat_message_store.create_indexes(
        [
            IndexModel([("SessionId", ASCENDING)]),
            IndexModel([("SessionId", ASCENDING), ("createdAt", ASCENDING)]),
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
    await database["stagedTasks"].create_indexes(
        [
            IndexModel([("extractionRunId", ASCENDING)]),
            IndexModel([("conversationId", ASCENDING), ("processingVersion", ASCENDING)]),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("updatedAt", DESCENDING)]),
            IndexModel([("fingerprint", ASCENDING)], sparse=True),
        ]
    )
    await database["stagedNotes"].create_indexes(
        [
            IndexModel([("extractionRunId", ASCENDING)]),
            IndexModel([("conversationId", ASCENDING), ("processingVersion", ASCENDING)]),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("updatedAt", DESCENDING)]),
            IndexModel([("fingerprint", ASCENDING)], sparse=True),
        ]
    )
    await database["stagedDecisions"].create_indexes(
        [
            IndexModel([("extractionRunId", ASCENDING)]),
            IndexModel([("conversationId", ASCENDING), ("processingVersion", ASCENDING)]),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("updatedAt", DESCENDING)]),
        ]
    )
    await database["stagedIssues"].create_indexes(
        [
            IndexModel([("extractionRunId", ASCENDING)]),
            IndexModel([("conversationId", ASCENDING), ("processingVersion", ASCENDING)]),
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("updatedAt", DESCENDING)]),
        ]
    )
    await database.tasks.create_indexes(
        [
            IndexModel([("userId", ASCENDING), ("spaceId", ASCENDING), ("status", ASCENDING)]),
            IndexModel([("sourceConversationId", ASCENDING)]),
            IndexModel([("fingerprint", ASCENDING)], unique=True, sparse=True),
        ]
    )
