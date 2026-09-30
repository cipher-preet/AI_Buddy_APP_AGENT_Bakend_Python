from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from apps.api_gateway.config.setting import settings
from services.chat.meeting.context import MeetingContext, TranscriptSegment, format_timestamp
from services.chat.models import utc_now
from services.db.mongo import get_database
from services.observability.diagnostics import diag_log
from services.vector.embedding_service import generate_embeddings
from services.vector.qdrant_client import qdrant_client

MEETING_COLLECTION = settings.MEETING_CHAT_QDRANT_COLLECTION
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 128

_collection_ready = False
_collection_lock = asyncio.Lock()
_meeting_locks: dict[str, asyncio.Lock] = {}


@dataclass
class MeetingChunk:
    index: int
    text: str
    start_ms: int | None
    end_ms: int | None
    speakers: list[str] = field(default_factory=list)

    @property
    def time_range(self) -> str:
        return f"{format_timestamp(self.start_ms)}-{format_timestamp(self.end_ms)}"


@dataclass
class MeetingCorpus:
    chunks: list[MeetingChunk]
    vector_ready: bool


def build_chunks(segments: list[TranscriptSegment], max_chars: int) -> list[MeetingChunk]:
    """Group transcript segments into timestamped windows with one-segment overlap."""
    chunks: list[MeetingChunk] = []
    window: list[TranscriptSegment] = []
    size = 0

    def emit() -> None:
        if not window:
            return
        speakers: list[str] = []
        for segment in window:
            if segment.speaker and segment.speaker not in speakers:
                speakers.append(segment.speaker)
        chunks.append(
            MeetingChunk(
                index=len(chunks),
                text="\n".join(_segment_line(segment) for segment in window),
                start_ms=window[0].start_ms,
                end_ms=window[-1].end_ms if window[-1].end_ms is not None else window[-1].start_ms,
                speakers=speakers,
            )
        )

    for segment in segments:
        line_size = len(segment.text) + 24
        if window and size + line_size > max_chars:
            emit()
            carry = window[-1]
            window = [carry] if len(carry.text) < max_chars // 2 else []
            size = sum(len(item.text) + 24 for item in window)
        window.append(segment)
        size += line_size
    emit()
    return chunks


def _segment_line(segment: TranscriptSegment) -> str:
    speaker = f"{segment.speaker}: " if segment.speaker else ""
    return f"[{format_timestamp(segment.start_ms)}] {speaker}{segment.text}"


def _fingerprint(chunks: list[MeetingChunk]) -> str:
    digest = sha1(settings.EMBEDDING_MODEL.encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()


def _point_id(meeting_id: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"buddy-meeting:{meeting_id}:{index}"))


def _meeting_filter(meeting_id: str, user_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="meetingId", match=MatchValue(value=meeting_id)),
            FieldCondition(key="userId", match=MatchValue(value=user_id)),
        ]
    )


class MeetingTranscriptIndex:
    """Meeting-scoped vector index stored in its own Qdrant collection."""

    def __init__(self, db: AsyncIOMotorDatabase | None = None):
        self.db = db or get_database()

    async def ensure_corpus(self, context: MeetingContext) -> MeetingCorpus:
        chunks = build_chunks(context.segments, settings.MEETING_CHAT_CHUNK_CHARS)
        if not chunks:
            # Raw transcript chunks expire via TTL; the vector index is the durable copy.
            stored = await self._load_stored_chunks(context.meeting_id, context.user_id)
            return MeetingCorpus(chunks=stored, vector_ready=bool(stored))

        fingerprint = _fingerprint(chunks)
        lock = _meeting_locks.setdefault(context.meeting_id, asyncio.Lock())
        async with lock:
            state = await self.db.meeting_chat_index.find_one({"_id": context.meeting_id})
            if state and state.get("fingerprint") == fingerprint:
                return MeetingCorpus(chunks=chunks, vector_ready=True)
            try:
                await self._index(context, chunks, fingerprint)
                return MeetingCorpus(chunks=chunks, vector_ready=True)
            except Exception as error:
                diag_log(
                    "meeting_chat_index_failed",
                    meetingId=context.meeting_id,
                    chunkCount=len(chunks),
                    error=str(error)[:300],
                )
                return MeetingCorpus(chunks=chunks, vector_ready=False)

    async def search(
        self,
        meeting_id: str,
        user_id: str,
        queries: list[str],
        limit: int,
    ) -> list[list[int]]:
        """Return one ranked list of chunk indexes per query."""
        if not queries:
            return []
        vectors = await generate_embeddings(queries)
        search_filter = _meeting_filter(meeting_id, user_id)
        results = await asyncio.gather(
            *(self._search_vector(vector, search_filter, limit) for vector in vectors),
            return_exceptions=True,
        )
        ranked: list[list[int]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            indexes = []
            for hit in result:
                chunk_index = (hit.payload or {}).get("chunkIndex")
                if isinstance(chunk_index, int):
                    indexes.append(chunk_index)
            ranked.append(indexes)
        return ranked

    async def _index(self, context: MeetingContext, chunks: list[MeetingChunk], fingerprint: str) -> None:
        await _ensure_collection()
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors.extend(await generate_embeddings([_embedding_text(context, chunk) for chunk in batch]))

        await qdrant_client.delete(
            collection_name=MEETING_COLLECTION,
            points_selector=FilterSelector(filter=_meeting_filter(context.meeting_id, context.user_id)),
        )
        created_at = utc_now().isoformat()
        points = [
            PointStruct(
                id=_point_id(context.meeting_id, chunk.index),
                vector=vector,
                payload={
                    "meetingId": context.meeting_id,
                    "conversationId": context.meeting_id,
                    "userId": context.user_id,
                    "spaceId": context.space_id,
                    "chunkIndex": chunk.index,
                    "text": chunk.text,
                    "startMs": chunk.start_ms,
                    "endMs": chunk.end_ms,
                    "speakers": chunk.speakers,
                    "meetingTitle": context.title,
                    "sourceType": "meeting_extension",
                    "fingerprint": fingerprint,
                    "createdAt": created_at,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            await qdrant_client.upsert(
                collection_name=MEETING_COLLECTION,
                points=points[start : start + UPSERT_BATCH_SIZE],
                wait=True,
            )
        await self.db.meeting_chat_index.update_one(
            {"_id": context.meeting_id},
            {
                "$set": {
                    "userId": context.user_id,
                    "fingerprint": fingerprint,
                    "chunkCount": len(chunks),
                    "embeddingModel": settings.EMBEDDING_MODEL,
                    "collection": MEETING_COLLECTION,
                    "indexedAt": utc_now(),
                }
            },
            upsert=True,
        )
        diag_log("meeting_chat_indexed", meetingId=context.meeting_id, chunkCount=len(chunks))

    async def _load_stored_chunks(self, meeting_id: str, user_id: str) -> list[MeetingChunk]:
        try:
            await _ensure_collection()
            chunks: list[MeetingChunk] = []
            offset = None
            while True:
                records, offset = await qdrant_client.scroll(
                    collection_name=MEETING_COLLECTION,
                    scroll_filter=_meeting_filter(meeting_id, user_id),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = dict(record.payload or {})
                    chunks.append(
                        MeetingChunk(
                            index=int(payload.get("chunkIndex") or 0),
                            text=str(payload.get("text") or ""),
                            start_ms=payload.get("startMs"),
                            end_ms=payload.get("endMs"),
                            speakers=list(payload.get("speakers") or []),
                        )
                    )
                if offset is None:
                    break
            return sorted(chunks, key=lambda chunk: chunk.index)
        except Exception as error:
            diag_log("meeting_chat_stored_chunks_failed", meetingId=meeting_id, error=str(error)[:300])
            return []

    async def _search_vector(self, vector: list[float], search_filter: Filter, limit: int) -> list[Any]:
        if hasattr(qdrant_client, "query_points"):
            response = await qdrant_client.query_points(
                collection_name=MEETING_COLLECTION,
                query=vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(response, "points", response))
        return await qdrant_client.search(
            collection_name=MEETING_COLLECTION,
            query_vector=vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )


def _embedding_text(context: MeetingContext, chunk: MeetingChunk) -> str:
    speakers = ", ".join(chunk.speakers) if chunk.speakers else "unknown speakers"
    return f"Meeting: {context.title}\nTime: {chunk.time_range}\nSpeakers: {speakers}\n{chunk.text}"


async def _ensure_collection() -> None:
    global _collection_ready
    if _collection_ready:
        return
    async with _collection_lock:
        if _collection_ready:
            return
        collections = await qdrant_client.get_collections()
        existing = {collection.name for collection in collections.collections}
        if MEETING_COLLECTION not in existing:
            await qdrant_client.create_collection(
                collection_name=MEETING_COLLECTION,
                vectors_config=VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE),
            )
        for field_name, schema in (
            ("meetingId", PayloadSchemaType.KEYWORD),
            ("userId", PayloadSchemaType.KEYWORD),
            ("chunkIndex", PayloadSchemaType.INTEGER),
        ):
            try:
                await qdrant_client.create_payload_index(
                    collection_name=MEETING_COLLECTION,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as error:
                message = str(error).lower()
                if "already exists" not in message and "already has" not in message:
                    raise
        _collection_ready = True
