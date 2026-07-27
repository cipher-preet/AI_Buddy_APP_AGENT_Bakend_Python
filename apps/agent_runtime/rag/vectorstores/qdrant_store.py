"""Qdrant access helpers for memory analysis."""

from dataclasses import dataclass
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType

from services.vector.qdrant_client import (
    QDRANT_COLLECTION,
    ensure_collection_exists,
    qdrant_client,
)

_payload_indexes_ready = False


@dataclass(frozen=True, slots=True)
class MemoryVector:
    """A transcript vector payload used by the analysis layer."""

    point_id: str
    text: str
    request_id: str | None
    payload: dict[str, Any]

    @property
    def chunk_id(self) -> str:
        """Stable chunk id exposed to generation and validation."""
        return str(self.payload.get("chunkId") or self.point_id)

    @property
    def created_at(self) -> str:
        return str(self.payload.get("createdAt") or "")


def canonical_user_id(payload: dict[str, Any]) -> str | None:
    """Read either legacy or canonical user id payload fields."""
    value = payload.get("user_id") or payload.get("userId")
    return str(value) if value else None


def canonical_space_id(payload: dict[str, Any]) -> str | None:
    """Read either legacy or canonical space id payload fields."""
    value = payload.get("space_id") or payload.get("spaceId")
    return str(value) if value else None


def is_eligible_speech_payload(payload: dict[str, Any], *, require_unpublished: bool) -> bool:
    """Return whether a Qdrant payload is eligible for transcript analysis."""
    source_type = payload.get("sourceType") or payload.get("source")
    if source_type != "speech":
        return False
    if require_unpublished and payload.get("isPublish", False) is not False:
        return False
    return (
        payload.get("chunkStatus", "active") == "active"
        and payload.get("isDamaged", False) is False
        and payload.get("isUseful", True) is True
    )


def _is_missing_collection_error(error: Exception) -> bool:
    message = str(error)
    return (
        "doesn't exist" in message
        or "does not exist" in message
        or "Not found: Collection" in message
    )


def _is_missing_payload_index_error(error: Exception) -> bool:
    return "Index required but not found" in str(error)


def _is_existing_payload_index_error(error: Exception) -> bool:
    return "already exists" in str(error).lower()


def _is_recoverable_qdrant_storage_error(error: Exception) -> bool:
    return _is_missing_collection_error(error) or _is_missing_payload_index_error(error)


async def ensure_memory_payload_indexes(*, force: bool = False) -> None:
    """Create Qdrant payload indexes required by analysis filters."""
    global _payload_indexes_ready

    if _payload_indexes_ready and not force:
        return

    await ensure_collection_exists()
    collection = await qdrant_client.get_collection(QDRANT_COLLECTION)
    existing_indexes = set((collection.payload_schema or {}).keys())

    required_indexes = {
        "user_id": PayloadSchemaType.KEYWORD,
        "userId": PayloadSchemaType.KEYWORD,
        "space_id": PayloadSchemaType.KEYWORD,
        "spaceId": PayloadSchemaType.KEYWORD,
        "isPublish": PayloadSchemaType.BOOL,
        "isDamaged": PayloadSchemaType.BOOL,
        "isUseful": PayloadSchemaType.BOOL,
        "chunkStatus": PayloadSchemaType.KEYWORD,
        "sourceType": PayloadSchemaType.KEYWORD,
        "source": PayloadSchemaType.KEYWORD,
    }

    for field_name, field_schema in required_indexes.items():
        if field_name in existing_indexes:
            continue

        try:
            await qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception as error:
            if not _is_existing_payload_index_error(error):
                raise

    _payload_indexes_ready = True


async def _recover_qdrant_storage_indexes() -> None:
    global _payload_indexes_ready

    _payload_indexes_ready = False
    await ensure_collection_exists()
    await ensure_memory_payload_indexes(force=True)


def _match_any_payload_name(names: list[str], value: str) -> Filter:
    return Filter(
        should=[
            FieldCondition(key=name, match=MatchValue(value=value)) for name in names
        ]
    )


def _user_space_filter(
    user_id: str,
    space_id: str,
    *,
    only_unpublished: bool,
) -> Filter:
    must: list[Filter | FieldCondition] = [
        _match_any_payload_name(["user_id", "userId"], user_id),
        _match_any_payload_name(["space_id", "spaceId"], space_id),
    ]

    if only_unpublished:
        must.append(FieldCondition(key="isPublish", match=MatchValue(value=False)))

    return Filter(must=must)


def _strict_unpublished_chunk_filter(user_id: str, space_id: str) -> Filter:
    return Filter(
        must=[
            _match_any_payload_name(["user_id", "userId"], user_id),
            _match_any_payload_name(["space_id", "spaceId"], space_id),
            _match_any_payload_name(["sourceType", "source"], "speech"),
            FieldCondition(key="isPublish", match=MatchValue(value=False)),
            FieldCondition(key="isDamaged", match=MatchValue(value=False)),
            FieldCondition(key="isUseful", match=MatchValue(value=True)),
            FieldCondition(key="chunkStatus", match=MatchValue(value="active")),
        ]
    )


def _memory_vector_from_point(point: Any) -> MemoryVector | None:
    payload = point.payload or {}
    text = payload.get("text")

    if not text or not canonical_user_id(payload) or not canonical_space_id(payload):
        return None

    return MemoryVector(
        point_id=str(point.id),
        text=str(text),
        request_id=payload.get("request_id") or payload.get("requestId"),
        payload=payload,
    )


async def _scroll_vectors(filter_: Filter, limit: int) -> list[MemoryVector]:
    await ensure_collection_exists()
    await ensure_memory_payload_indexes()

    vectors: list[MemoryVector] = []
    offset = None

    while True:
        try:
            points, offset = await qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=filter_,
                limit=min(limit - len(vectors), 256),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as error:
            if not _is_recoverable_qdrant_storage_error(error):
                raise
            await _recover_qdrant_storage_indexes()
            points, offset = await qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=filter_,
                limit=min(limit - len(vectors), 256),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

        for point in points:
            vector = _memory_vector_from_point(point)
            if vector:
                vectors.append(vector)

        if not offset or len(vectors) >= limit:
            return vectors


async def fetch_unpublished_vectors(
    user_id: str,
    space_id: str,
    *,
    limit: int = 1000,
) -> list[MemoryVector]:
    """Fetch new vectors that have not yet been processed."""
    return await _scroll_vectors(
        _user_space_filter(user_id, space_id, only_unpublished=True),
        limit=limit,
    )


async def fetch_strict_unpublished_chunks(
    user_id: str,
    space_id: str,
    *,
    limit: int = 1000,
) -> list[MemoryVector]:
    """Fetch unpublished, useful, active, non-damaged chunks for orchestration."""
    return await _scroll_vectors(
        _strict_unpublished_chunk_filter(user_id, space_id),
        limit=limit,
    )


async def fetch_recent_speech_chunks(
    user_id: str,
    space_id: str,
    *,
    limit: int,
) -> list[MemoryVector]:
    """Fetch recent eligible same-space speech chunks for chronological context."""
    vectors = await _scroll_vectors(
        Filter(
            must=[
                _match_any_payload_name(["user_id", "userId"], user_id),
                _match_any_payload_name(["space_id", "spaceId"], space_id),
                _match_any_payload_name(["sourceType", "source"], "speech"),
                FieldCondition(key="isDamaged", match=MatchValue(value=False)),
                FieldCondition(key="isUseful", match=MatchValue(value=True)),
                FieldCondition(key="chunkStatus", match=MatchValue(value="active")),
            ]
        ),
        limit=max(limit * 4, limit),
    )
    vectors = [vector for vector in vectors if is_eligible_speech_payload(vector.payload, require_unpublished=False)]
    return sort_vectors_chronologically(vectors)[-limit:]


async def search_relevant_speech_chunks(
    user_id: str,
    space_id: str,
    *,
    query_vector: list[float],
    exclude_chunk_ids: set[str],
    limit: int,
) -> list[tuple[MemoryVector, float]]:
    """Search semantically relevant older same-space speech chunks."""
    await ensure_collection_exists()
    await ensure_memory_payload_indexes()
    filter_ = Filter(
        must=[
            _match_any_payload_name(["user_id", "userId"], user_id),
            _match_any_payload_name(["space_id", "spaceId"], space_id),
            _match_any_payload_name(["sourceType", "source"], "speech"),
            FieldCondition(key="isDamaged", match=MatchValue(value=False)),
            FieldCondition(key="isUseful", match=MatchValue(value=True)),
            FieldCondition(key="chunkStatus", match=MatchValue(value="active")),
        ]
    )

    try:
        try:
            results = await qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                query_filter=filter_,
                limit=max(limit * 3, limit),
                with_payload=True,
                with_vectors=False,
            )
        except AttributeError:
            query_result = await qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                query_filter=filter_,
                limit=max(limit * 3, limit),
                with_payload=True,
                with_vectors=False,
            )
            results = query_result.points
    except Exception as error:
        if not _is_recoverable_qdrant_storage_error(error):
            raise
        await _recover_qdrant_storage_indexes()
        try:
            results = await qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                query_filter=filter_,
                limit=max(limit * 3, limit),
                with_payload=True,
                with_vectors=False,
            )
        except AttributeError:
            query_result = await qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                query_filter=filter_,
                limit=max(limit * 3, limit),
                with_payload=True,
                with_vectors=False,
            )
            results = query_result.points

    matches: list[tuple[MemoryVector, float]] = []
    seen_texts: set[str] = set()
    for point in results:
        vector = _memory_vector_from_point(point)
        if not vector or vector.chunk_id in exclude_chunk_ids:
            continue
        if not is_eligible_speech_payload(vector.payload, require_unpublished=False):
            continue
        text_key = vector.text.strip().casefold()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        matches.append((vector, float(getattr(point, "score", 0.0) or 0.0)))
        if len(matches) >= limit:
            break
    return matches


async def fetch_full_context_vectors(
    user_id: str,
    space_id: str,
    *,
    limit: int = 2000,
) -> list[MemoryVector]:
    """Fetch all vectors for the same user and space for LLM context."""
    return await _scroll_vectors(
        _user_space_filter(user_id, space_id, only_unpublished=False),
        limit=limit,
    )


async def mark_vectors_as_published(point_ids: list[str]) -> None:
    """Mark processed vectors as published after downstream saves succeed."""
    if not point_ids:
        return

    await ensure_collection_exists()
    await qdrant_client.set_payload(
        collection_name=QDRANT_COLLECTION,
        payload={"isPublish": True},
        points=point_ids,
    )


async def mark_vectors_analysis_completed(
    point_ids: list[str],
    *,
    window_id: str,
    published_at: str,
) -> None:
    """Mark processed vectors as published with analysis metadata."""
    if not point_ids:
        return

    await ensure_collection_exists()
    await qdrant_client.set_payload(
        collection_name=QDRANT_COLLECTION,
        payload={
            "isPublish": True,
            "publishedAt": published_at,
            "analysisWindowId": window_id,
            "analysisStatus": "completed",
        },
        points=point_ids,
    )


def sort_vectors_chronologically(vectors: list[MemoryVector]) -> list[MemoryVector]:
    """Sort vectors by createdAt, chunkIndex, then chunk id."""
    return sorted(
        vectors,
        key=lambda vector: (
            str(vector.payload.get("createdAt") or ""),
            int(vector.payload.get("chunkIndex") or 0),
            vector.chunk_id,
        ),
    )
