from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from apps.api_gateway.config.setting import settings
from services.chat.meeting.index import MeetingChunk, MeetingCorpus, MeetingTranscriptIndex
from services.observability.diagnostics import diag_log

RRF_K = 60
NEIGHBOR_WINDOW = 1
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be",
    "it", "this", "that", "with", "as", "at", "by", "we", "you", "i", "he", "she", "they", "what",
    "who", "when", "where", "why", "how", "did", "do", "does", "about", "meeting", "said", "tell",
    "me", "please", "can", "could", "would", "should", "there", "their", "from", "any", "all",
}


@dataclass
class MeetingRetrieval:
    chunks: list[MeetingChunk]
    mode: str  # "full" | "hybrid" | "lexical" | "empty"


class MeetingRetriever:
    def __init__(self, index: MeetingTranscriptIndex | None = None):
        self.index = index or MeetingTranscriptIndex()

    async def retrieve(
        self,
        meeting_id: str,
        user_id: str,
        corpus: MeetingCorpus,
        queries: list[str],
        top_k: int | None = None,
    ) -> MeetingRetrieval:
        chunks = corpus.chunks
        if not chunks:
            return MeetingRetrieval(chunks=[], mode="empty")

        total_chars = sum(len(chunk.text) for chunk in chunks)
        if total_chars <= settings.MEETING_CHAT_FULL_CONTEXT_CHARS:
            return MeetingRetrieval(chunks=_dedupe_overlap(chunks), mode="full")

        top_k = top_k or settings.MEETING_CHAT_TOP_K
        ranked_lists: list[list[int]] = [_bm25_rank(chunks, query)[: top_k * 2] for query in queries]
        mode = "lexical"
        if corpus.vector_ready:
            try:
                dense = await self.index.search(meeting_id, user_id, queries, limit=top_k * 2)
                if any(dense):
                    ranked_lists.extend(dense)
                    mode = "hybrid"
            except Exception as error:
                diag_log("meeting_chat_dense_search_failed", meetingId=meeting_id, error=str(error)[:300])

        fused = _reciprocal_rank_fusion(ranked_lists)
        selected = set(fused[:top_k])
        if not selected:
            selected = set(range(min(top_k, len(chunks))))
        expanded: set[int] = set()
        for index in selected:
            for neighbor in range(index - NEIGHBOR_WINDOW, index + NEIGHBOR_WINDOW + 1):
                if 0 <= neighbor < len(chunks):
                    expanded.add(neighbor)
        by_index = {chunk.index: chunk for chunk in chunks}
        ordered = [by_index[index] for index in sorted(expanded) if index in by_index]
        return MeetingRetrieval(chunks=_dedupe_overlap(ordered), mode=mode)


def _reciprocal_rank_fusion(ranked_lists: list[list[int]]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, index in enumerate(ranking):
            scores[index] = scores.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [index for index, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall((text or "").lower()) if token not in _STOPWORDS and len(token) > 1]


def _bm25_rank(chunks: list[MeetingChunk], query: str, k1: float = 1.4, b: float = 0.75) -> list[int]:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return []
    docs = [_tokenize(chunk.text) for chunk in chunks]
    doc_count = len(docs)
    avg_len = (sum(len(doc) for doc in docs) / doc_count) if doc_count else 0
    document_frequency: Counter[str] = Counter()
    for doc in docs:
        document_frequency.update(set(doc) & query_terms)

    scored: list[tuple[float, int]] = []
    for chunk, doc in zip(chunks, docs):
        if not doc:
            continue
        frequencies = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(doc) / (avg_len or 1)))
        if score > 0:
            scored.append((score, chunk.index))
    scored.sort(reverse=True)
    return [index for _, index in scored]


def _dedupe_overlap(chunks: list[MeetingChunk]) -> list[MeetingChunk]:
    """Drop lines repeated by the one-segment overlap between consecutive chunks."""
    result: list[MeetingChunk] = []
    previous_lines: set[str] = set()
    previous_index: int | None = None
    for chunk in chunks:
        lines = chunk.text.split("\n")
        if previous_index is not None and chunk.index == previous_index + 1:
            lines = [line for line in lines if line not in previous_lines]
        previous_lines = set(chunk.text.split("\n"))
        previous_index = chunk.index
        if not lines:
            continue
        result.append(
            MeetingChunk(
                index=chunk.index,
                text="\n".join(lines),
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                speakers=chunk.speakers,
            )
        )
    return result
