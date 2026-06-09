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


async def ensure_memory_payload_indexes() -> None:
    """Create Qdrant payload indexes required by analysis filters."""
    global _payload_indexes_ready

    if _payload_indexes_ready:
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
    }

    for field_name, field_schema in required_indexes.items():
        if field_name in existing_indexes:
            continue

        await qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=field_schema,
        )

    _payload_indexes_ready = True


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
            FieldCondition(key="userId", match=MatchValue(value=user_id)),
            FieldCondition(key="spaceId", match=MatchValue(value=space_id)),
            FieldCondition(key="isPublish", match=MatchValue(value=False)),
            FieldCondition(key="isDamaged", match=MatchValue(value=False)),
            FieldCondition(key="isUseful", match=MatchValue(value=True)),
            FieldCondition(key="chunkStatus", match=MatchValue(value="active")),
        ]
    )


def _memory_vector_from_point(point: Any) -> MemoryVector | None:
    payload = point.payload or {}
    text = payload.get("text")

    if not text:
        return None

    return MemoryVector(
        point_id=str(point.id),
        text=str(text),
        request_id=payload.get("request_id") or payload.get("requestId"),
        payload=payload,
    )


async def _scroll_vectors(filter_: Filter, limit: int) -> list[MemoryVector]:
    await ensure_memory_payload_indexes()

    vectors: list[MemoryVector] = []
    offset = None

    while True:
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

    await qdrant_client.set_payload(
        collection_name=QDRANT_COLLECTION,
        payload={"isPublish": True},
        points=point_ids,
    )
