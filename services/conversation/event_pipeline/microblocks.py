"""Build small local conversational micro-blocks from adjacent useful chunks.

Embeddings are a grouping signal, not semantic truth. Blocks stay small:
2–5 useful turns or ~250–500 tokens, with a hard maximum.
"""

from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.embeddings import CachedEmbedder, default_embedder
from services.conversation.event_pipeline.schemas import CleanedTranscriptRecord, MicroBlock
from services.conversation.event_pipeline.textutil import (
    cosine_similarity,
    extract_entities,
    entity_overlap,
    stable_id,
    token_count,
)


async def build_micro_blocks(
    records: list[CleanedTranscriptRecord],
    embedder: CachedEmbedder | None = None,
) -> list[MicroBlock]:
    if not records:
        return []
    embedder = embedder or default_embedder()
    min_turns = int(getattr(settings, "EVENT_PIPELINE_MICROBLOCK_MIN_TURNS", 2))
    max_turns = int(getattr(settings, "EVENT_PIPELINE_MICROBLOCK_MAX_TURNS", 5))
    min_tokens = int(getattr(settings, "EVENT_PIPELINE_MICROBLOCK_MIN_TOKENS", 250))
    max_tokens = int(getattr(settings, "EVENT_PIPELINE_MICROBLOCK_MAX_TOKENS", 500))
    topic_break_threshold = float(getattr(settings, "MICROBLOCK_SIMILARITY_THRESHOLD", 0.34))
    hard_break_threshold = min(0.28, topic_break_threshold)
    entity_break_threshold = max(topic_break_threshold, 0.62)
    vectors = await embedder.embed_many([record.rawText for record in records])
    blocks: list[MicroBlock] = []
    current: list[int] = [0]
    current_tokens = token_count(records[0].rawText)
    pending_overlap: list[int] = []

    def flush(overlap_from: int | None = None) -> None:
        nonlocal current, current_tokens, pending_overlap
        if not current:
            return
        blocks.append(_block_from_indices(records, current, vectors, pending_overlap))
        if overlap_from is not None:
            pending_overlap = [records[overlap_from].sequenceId]
            current = [overlap_from]
            current_tokens = token_count(records[overlap_from].rawText)
        else:
            pending_overlap = []
            current = []
            current_tokens = 0

    for index in range(1, len(records)):
        next_tokens = token_count(records[index].rawText)
        would_turns = len(current) + 1
        would_tokens = current_tokens + next_tokens
        score = _pair_score(records, vectors, current[-1], index)
        force_max = would_turns > max_turns or would_tokens > max_tokens
        can_close = len(current) >= min_turns or current_tokens >= min_tokens
        named_conflict = _named_entity_conflict_block(records, current, index)
        hard_break = score < hard_break_threshold and named_conflict
        entity_break = named_conflict and score < entity_break_threshold and (can_close or named_conflict)
        topic_break = (score < topic_break_threshold and can_close) or hard_break or entity_break
        if force_max or topic_break:
            overlap = (
                current[-1]
                if hard_break_threshold <= score < max(0.55, topic_break_threshold + 0.2)
                and not force_max
                and not hard_break
                and not entity_break
                else None
            )
            flush(overlap)
            if overlap is None:
                current = [index]
                current_tokens = next_tokens
            else:
                current.append(index)
                current_tokens += next_tokens
            continue
        current.append(index)
        current_tokens = would_tokens

    flush()
    return blocks


def _block_from_indices(
    records: list[CleanedTranscriptRecord],
    indices: list[int],
    vectors: list[list[float]],
    overlap_ids: list[int],
) -> MicroBlock:
    selected = [records[index] for index in indices]
    sequence_ids = [item.sequenceId for item in selected]
    source_ids = [item.sourceId for item in selected]
    lines = [f"[{item.sequenceId}] {item.rawText}" for item in selected]
    text = "\n".join(lines)
    overlap = list(overlap_ids)
    embedding = vectors[indices[0]]
    if len(indices) > 1:
        from services.conversation.event_pipeline.textutil import mean_vector

        embedding = mean_vector([vectors[index] for index in indices])
    return MicroBlock(
        microBlockId=stable_id("MB", selected[0].sessionId, sequence_ids[0], sequence_ids[-1], source_ids[0]),
        sequenceStart=sequence_ids[0],
        sequenceEnd=sequence_ids[-1],
        sequenceIds=sequence_ids,
        sourceIds=source_ids,
        text=text,
        tokenCount=token_count(text),
        embedding=embedding,
        overlapSequenceIds=[seq for seq in overlap if seq in sequence_ids],
        speakerIds=[item.speaker for item in selected if item.speaker],
    )


def _pair_score(
    records: list[CleanedTranscriptRecord],
    vectors: list[list[float]],
    left_index: int,
    right_index: int,
) -> float:
    left = records[left_index]
    right = records[right_index]
    semantic = cosine_similarity(vectors[left_index], vectors[right_index])
    adjacency = 1.0 if right.sequenceId - left.sequenceId <= 2 else max(0.0, 1.0 - (right.sequenceId - left.sequenceId) / 8)
    time_score = _time_score(left.timestampMs, right.timestampMs)
    speaker_score = 1.0 if left.speaker and left.speaker == right.speaker else 0.45
    entities = entity_overlap(extract_entities(left.rawText), extract_entities(right.rawText))
    short_follow = 1.0 if token_count(right.rawText) <= 18 else 0.0
    return (
        0.38 * semantic
        + 0.22 * adjacency
        + 0.12 * time_score
        + 0.08 * speaker_score
        + 0.12 * entities
        + 0.08 * short_follow
    )


def _named_entity_conflict_block(records: list[CleanedTranscriptRecord], current: list[int], next_index: int) -> bool:
    block_entities = {item.casefold() for index in current for item in extract_entities(records[index].rawText)}
    next_entities = {item.casefold() for item in extract_entities(records[next_index].rawText)}
    return bool(block_entities and next_entities and block_entities.isdisjoint(next_entities))


def _named_entity_conflict(left: CleanedTranscriptRecord, right: CleanedTranscriptRecord) -> bool:
    left_entities = {item.casefold() for item in extract_entities(left.rawText)}
    right_entities = {item.casefold() for item in extract_entities(right.rawText)}
    return bool(left_entities and right_entities and left_entities.isdisjoint(right_entities))


def _distinct_entities(left: CleanedTranscriptRecord, right: CleanedTranscriptRecord) -> bool:
    left_entities = {item.casefold() for item in extract_entities(left.rawText)}
    right_entities = {item.casefold() for item in extract_entities(right.rawText)}
    if not left_entities or not right_entities:
        return token_count(left.rawText) >= 6 and token_count(right.rawText) >= 6
    return left_entities.isdisjoint(right_entities) and token_count(right.rawText) >= 6


def _time_score(left_ms: int | None, right_ms: int | None) -> float:
    if left_ms is None or right_ms is None:
        return 0.5
    gap = abs(right_ms - left_ms)
    if gap <= 15_000:
        return 1.0
    if gap >= 180_000:
        return 0.0
    return max(0.0, 1.0 - gap / 180_000)
