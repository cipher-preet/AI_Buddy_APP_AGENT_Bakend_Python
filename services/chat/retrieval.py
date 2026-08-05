from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from services.chat.models import RetrievedContext
from services.vector.embedding_service import generate_embedding
from services.vector.qdrant_client import QDRANT_COLLECTION, ensure_collection_exists, qdrant_client


class ChatRetriever:
    def __init__(self, child_limit: int = 8, parent_window: int = 1):
        self.child_limit = child_limit
        self.parent_window = parent_window

    async def retrieve(self, question: str, user_id: str, space_id: str | None = None) -> list[RetrievedContext]:
        return await self.retrieve_many([question], user_id, space_id)

    async def retrieve_many(
        self,
        queries: list[str],
        user_id: str,
        space_id: str | None = None,
    ) -> list[RetrievedContext]:
        await ensure_collection_exists()
        search_filter = _user_space_filter(user_id, space_id)
        hits = []
        for query in _dedupe_queries(queries):
            query_vector = await generate_embedding(query)
            hits.extend(await _search_points(query_vector, search_filter, self.child_limit))
        children = [_context_from_hit(hit) for hit in hits]
        parents = await self._expand_parent_context(children, user_id, space_id)
        merged = _rank_contexts(_dedupe_contexts([*children, *parents]), queries)
        return merged[: self.child_limit * (self.parent_window * 2 + 1)]

    async def _expand_parent_context(
        self,
        children: list[RetrievedContext],
        user_id: str,
        space_id: str | None,
    ) -> list[RetrievedContext]:
        by_job: dict[str, set[int]] = defaultdict(set)
        for child in children:
            if child.jobId is None or child.chunkIndex is None:
                continue
            for index in range(child.chunkIndex - self.parent_window, child.chunkIndex + self.parent_window + 1):
                if index >= 0:
                    by_job[child.jobId].add(index)

        contexts: list[RetrievedContext] = []
        child_scores = {
            (child.jobId, child.chunkIndex): child.score
            for child in children
            if child.jobId is not None and child.chunkIndex is not None and child.score is not None
        }
        for job_id, indexes in by_job.items():
            scroll_filter = _parent_filter(user_id, space_id, job_id, sorted(indexes))
            records, _ = await qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=scroll_filter,
                limit=max(len(indexes), 1),
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                context = _context_from_record(record)
                nearby_scores = [
                    score
                    for (score_job_id, score_index), score in child_scores.items()
                    if score_job_id == context.jobId
                    and context.chunkIndex is not None
                    and abs(score_index - context.chunkIndex) <= self.parent_window
                ]
                if nearby_scores and context.score is None:
                    context.score = max(nearby_scores) * 0.92
                contexts.append(context)
        return sorted(contexts, key=lambda item: (item.jobId or "", item.chunkIndex or 0))


def format_context(contexts: list[RetrievedContext]) -> str:
    if not contexts:
        return "No retrieved context was found for this user."
    lines = []
    for index, context in enumerate(contexts, start=1):
        lines.append(f"Context note {index}: {context.text}")
    return "\n\n".join(lines)


def _user_space_filter(user_id: str, space_id: str | None) -> Filter:
    conditions = [
        FieldCondition(key="userId", match=MatchValue(value=user_id)),
    ]
    if space_id is not None:
        conditions.append(FieldCondition(key="spaceId", match=MatchValue(value=space_id)))
    return Filter(must=conditions)


def _parent_filter(user_id: str, space_id: str | None, job_id: str, indexes: list[int]) -> Filter:
    conditions = [
        FieldCondition(key="userId", match=MatchValue(value=user_id)),
        FieldCondition(key="job_id", match=MatchValue(value=job_id)),
        FieldCondition(key="chunkIndex", match=MatchAny(any=indexes)),
    ]
    if space_id is not None:
        conditions.append(FieldCondition(key="spaceId", match=MatchValue(value=space_id)))
    return Filter(must=conditions)


def _context_from_hit(hit: Any) -> RetrievedContext:
    payload = dict(hit.payload or {})
    return RetrievedContext(
        text=str(payload.get("text") or ""),
        score=float(hit.score) if getattr(hit, "score", None) is not None else None,
        sourceId=str(hit.id) if getattr(hit, "id", None) is not None else None,
        jobId=payload.get("job_id"),
        chunkIndex=payload.get("chunkIndex"),
        payload=payload,
    )


async def _search_points(query_vector: list[float], search_filter: Filter, limit: int) -> list[Any]:
    if hasattr(qdrant_client, "search"):
        return await qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

    response = await qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True,
    )
    return list(getattr(response, "points", response))


def _context_from_record(record: Any) -> RetrievedContext:
    payload = dict(record.payload or {})
    return RetrievedContext(
        text=str(payload.get("text") or ""),
        sourceId=str(record.id) if getattr(record, "id", None) is not None else None,
        jobId=payload.get("job_id"),
        chunkIndex=payload.get("chunkIndex"),
        payload=payload,
    )


def _dedupe_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    seen = set()
    unique = []
    for context in contexts:
        key = context.sourceId or (context.jobId, context.chunkIndex, context.text)
        if key in seen or not context.text.strip():
            continue
        seen.add(key)
        unique.append(context)
    return unique


def _rank_contexts(contexts: list[RetrievedContext], queries: list[str]) -> list[RetrievedContext]:
    query_terms = _query_terms(queries)

    def rank_key(context: RetrievedContext) -> tuple[float, int, str, int]:
        vector_score = float(context.score or 0)
        lexical_score = _lexical_overlap(context.text, query_terms)
        # Prefer strong semantic matches, then exact term overlap, then local transcript order.
        combined = vector_score + (0.04 * lexical_score)
        return (combined, lexical_score, context.jobId or "", -(context.chunkIndex or 0))

    return sorted(contexts, key=rank_key, reverse=True)


def _query_terms(queries: list[str]) -> set[str]:
    terms: set[str] = set()
    for query in queries:
        terms.update(_tokenize(query))
    return {term for term in terms if len(term) > 2}


def _lexical_overlap(text: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    terms = set(_tokenize(text))
    return len(query_terms & terms)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", (text or "").lower(), flags=re.UNICODE)


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen = set()
    unique = []
    for query in queries:
        normalized = " ".join((query or "").strip().split())
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique
