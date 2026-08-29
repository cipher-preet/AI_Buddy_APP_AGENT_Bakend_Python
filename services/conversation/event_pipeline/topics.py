"""Adjacent micro-blocks → local topics. Global threads are a later stage."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.conversation.event_pipeline.embeddings import CachedEmbedder, default_embedder
from services.conversation.event_pipeline.flags import (
    topic_coherence_drop_threshold,
    topic_continue_similarity_threshold,
    topic_filler_density_threshold,
    topic_object_discontinuity_max_overlap,
    topic_object_discontinuity_max_similarity,
    topic_safety_continue_similarity,
    topic_safety_max_micro_blocks,
    topic_safety_max_tokens,
)
from services.conversation.event_pipeline.observability import current_observability
from services.conversation.event_pipeline.schemas import ACTION_PRONOUNS, DEICTIC_OR_TIME, LocalTopic, MicroBlock
from services.conversation.event_pipeline.textutil import (
    content_tokens,
    cosine_similarity,
    entity_overlap,
    extract_entities,
    information_density,
    is_low_information_text,
    mean_vector,
    token_count,
    token_jaccard,
)
from services.llm.router import LLMRouter


async def segment_local_topics(
    blocks: list[MicroBlock],
    embedder: CachedEmbedder | None = None,
    router: LLMRouter | None = None,
) -> list[LocalTopic]:
    if not blocks:
        return []
    embedder = embedder or default_embedder()
    missing = [block.text for block in blocks if not block.embedding]
    if missing:
        vectors = await embedder.embed_many(missing)
        iterator = iter(vectors)
        for block in blocks:
            if not block.embedding:
                block.embedding = next(iterator)
    groups: list[list[int]] = [[0]]
    reasons: list[str] = ["topic_start"]
    for index in range(1, len(blocks)):
        should_continue, reason = _should_continue_topic(
            [blocks[member] for member in groups[-1]],
            blocks[index],
        )
        if should_continue:
            groups[-1].append(index)
        else:
            groups.append([index])
            reasons.append(reason)
    groups, reasons = _split_groups_on_filler_bridges(blocks, groups, reasons)
    topics: list[LocalTopic] = []
    for offset, indices in enumerate(groups, start=1):
        selected = [blocks[index] for index in indices]
        sequence_ids: list[int] = []
        seen: set[int] = set()
        for block in selected:
            for sequence in block.sequenceIds:
                if sequence not in seen:
                    seen.add(sequence)
                    sequence_ids.append(sequence)
        entities: list[str] = []
        entity_seen: set[str] = set()
        for block in selected:
            if not block.informationDensity:
                block.informationDensity = information_density(block.text)
            for entity in extract_entities(block.text):
                key = entity.casefold()
                if key not in entity_seen:
                    entity_seen.add(key)
                    entities.append(entity)
        label = _label(entities, selected[0].text)
        coherence = _topic_coherence(selected)
        boundary = reasons[offset - 1] if offset <= len(reasons) else "topic_start"
        topic = LocalTopic(
            topicId=f"T{offset}",
            label=label,
            microBlockIds=[block.microBlockId for block in selected],
            sequenceStart=sequence_ids[0],
            sequenceEnd=sequence_ids[-1],
            sequenceIds=sequence_ids,
            entities=entities,
            embedding=mean_vector([block.embedding for block in selected if block.embedding]),
            text="\n".join(block.text for block in selected),
            coherence=coherence,
            boundaryReason=boundary,
            tokenCount=sum(block.tokenCount or token_count(block.text) for block in selected),
        )
        topics.append(topic)
        _log_topic_segment(topic)
    if router is not None:
        await label_ambiguous_topics(topics, router)
    return topics


def _should_continue_topic(topic_blocks: list[MicroBlock], candidate: MicroBlock) -> tuple[bool, str]:
    last = topic_blocks[-1]
    content_blocks = [block for block in topic_blocks if not _is_filler_block(block)]
    anchor = content_blocks[-1] if content_blocks else last
    candidate_is_filler = _is_filler_block(candidate)
    filler_bridge = (not candidate_is_filler) and bool(topic_blocks) and _is_filler_block(last)
    sim_last = cosine_similarity(anchor.embedding, candidate.embedding)
    centroid_source = content_blocks or topic_blocks
    centroid = mean_vector([block.embedding for block in centroid_source if block.embedding])
    sim_centroid = cosine_similarity(centroid, candidate.embedding)
    topic_text = " ".join(block.text for block in (content_blocks or topic_blocks))
    object_continuity = max(
        token_jaccard(topic_text, candidate.text),
        token_jaccard(anchor.text, candidate.text),
        entity_overlap(extract_entities(topic_text), extract_entities(candidate.text)),
    )
    gap = candidate.sequenceStart - last.sequenceEnd if last.sequenceEnd and candidate.sequenceStart else 99
    adjacent = gap <= 2
    # Filler is a weak bridge: adjacency must not dominate A vs B similarity.
    adjacency = 0.0 if filler_bridge else (1.0 if adjacent else 0.2)
    time_score = 0.0 if filler_bridge else (1.0 if gap <= 1 else max(0.0, 1.0 - gap / 6) if gap < 99 else 0.5)
    continue_threshold = topic_continue_similarity_threshold()
    drop_threshold = topic_coherence_drop_threshold()
    projected_tokens = sum(block.tokenCount or token_count(block.text) for block in topic_blocks) + (
        candidate.tokenCount or token_count(candidate.text)
    )
    projected_blocks = len(topic_blocks) + 1

    if projected_blocks > topic_safety_max_micro_blocks() or projected_tokens > topic_safety_max_tokens():
        if candidate_is_filler and adjacent:
            return True, "safety_continue_filler"
        if sim_centroid >= topic_safety_continue_similarity() and (adjacent or object_continuity >= 0.25) and not filler_bridge:
            return True, "safety_continue_high_coherence"
        return False, "safety_bound"

    if candidate_is_filler and adjacent:
        return True, "weak_bridge_filler"

    if filler_bridge:
        sim_anchors = cosine_similarity(anchor.embedding, candidate.embedding)
        if (
            _has_object_tokens(anchor.text)
            and _has_object_tokens(candidate.text)
            and object_continuity <= max(topic_object_discontinuity_max_overlap(), 0.22)
            and sim_anchors < 0.74
        ):
            return False, "FILLER_BRIDGE_SEMANTIC_BREAK"
        if sim_anchors < continue_threshold and object_continuity < 0.25:
            return False, "FILLER_BRIDGE_SEMANTIC_BREAK"

    if (
        _has_object_tokens(topic_text)
        and _has_object_tokens(candidate.text)
        and object_continuity <= topic_object_discontinuity_max_overlap()
        and sim_centroid < topic_object_discontinuity_max_similarity()
    ):
        return False, "object_discontinuity"

    if len(content_blocks) >= 2:
        old_coherence = _topic_coherence(content_blocks)
        new_coherence = _topic_coherence([*content_blocks, candidate])
        if old_coherence - new_coherence >= drop_threshold and sim_centroid < 0.68:
            return False, "coherence_drop"

    continue_score = (
        0.45 * sim_centroid
        + 0.20 * sim_last
        + 0.15 * adjacency
        + 0.10 * time_score
        + 0.10 * object_continuity
    )
    if continue_score >= continue_threshold:
        return True, "semantic_continue"
    return False, "below_continue_threshold"


def _split_groups_on_filler_bridges(
    blocks: list[MicroBlock],
    groups: list[list[int]],
    reasons: list[str],
) -> tuple[list[list[int]], list[str]]:
    """Split A → filler → B when content anchors are semantically different.

    Filler stays accounted with its neighboring topic; it is not deleted.
    Adjacent content without a filler span is left together.
    """
    new_groups: list[list[int]] = []
    new_reasons: list[str] = []
    for group, reason in zip(groups, reasons):
        pieces = _content_anchor_pieces(blocks, group)
        for offset, piece in enumerate(pieces):
            new_groups.append(piece)
            new_reasons.append(reason if offset == 0 else "FILLER_BRIDGE_SEMANTIC_BREAK")
    return new_groups, new_reasons


def _content_anchor_pieces(blocks: list[MicroBlock], indices: list[int]) -> list[list[int]]:
    content_pos = [index for index in indices if not _is_filler_block(blocks[index])]
    if len(content_pos) < 2:
        return [indices]
    cuts: list[int] = []
    for left, right in zip(content_pos, content_pos[1:]):
        if indices.index(right) - indices.index(left) <= 1:
            continue
        left_text = _content_only_text(blocks[left])
        right_text = _content_only_text(blocks[right])
        overlap = token_jaccard(left_text, right_text)
        entities = entity_overlap(extract_entities(left_text), extract_entities(right_text))
        # Cosine of mixed blocks must not veto a content-anchor break.
        if overlap < 0.22 and entities < 0.25:
            cuts.append(right)
    if not cuts:
        return [indices]
    pieces: list[list[int]] = []
    current: list[int] = []
    cut_set = set(cuts)
    for index in indices:
        if index in cut_set and current:
            pieces.append(current)
            current = [index]
            continue
        current.append(index)
    if current:
        pieces.append(current)
    return pieces or [indices]


def _content_only_text(block: MicroBlock) -> str:
    lines = [line.split("]", 1)[-1].strip() for line in (block.text or "").splitlines() if line.strip()]
    content = [line for line in lines if not is_low_information_text(line, topic_filler_density_threshold())]
    return "\n".join(content) if content else (block.text or "")


def _topic_coherence(blocks: list[MicroBlock]) -> float:
    content = [block for block in blocks if not _is_filler_block(block)]
    use = content or blocks
    if not use:
        return 0.0
    if len(use) == 1:
        return 1.0
    centroid = mean_vector([block.embedding for block in use if block.embedding])
    scores = [cosine_similarity(block.embedding, centroid) for block in use if block.embedding]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _is_filler_block(block: MicroBlock) -> bool:
    lines = [line.split("]", 1)[-1].strip() for line in (block.text or "").splitlines() if line.strip()]
    if lines and all(is_low_information_text(line, topic_filler_density_threshold()) for line in lines):
        return True
    density = block.informationDensity or information_density(block.text)
    if not lines and density < topic_filler_density_threshold():
        return True
    return _mostly_referential(block.text) or (not _has_object_tokens(block.text) and density < topic_filler_density_threshold())


def _has_object_tokens(text: str) -> bool:
    tokens = [token.casefold() for token in content_tokens(text)]
    distinctive = [token for token in tokens if token not in DEICTIC_OR_TIME and token not in ACTION_PRONOUNS]
    return len(distinctive) >= 2


def _mostly_referential(text: str) -> bool:
    tokens = [token.casefold() for token in content_tokens(text)]
    if not tokens:
        return False
    referential = DEICTIC_OR_TIME | ACTION_PRONOUNS
    distinctive = [token for token in tokens if token not in referential]
    return len(distinctive) <= 1


def _label(entities: list[str], fallback_text: str) -> str:
    if entities:
        return " / ".join(entities[:4])
    line = fallback_text.splitlines()[0] if fallback_text else "topic"
    return line[:80]


def _log_topic_segment(topic: LocalTopic) -> None:
    sequences = ",".join(str(sequence) for sequence in topic.sequenceIds[:12])
    if len(topic.sequenceIds) > 12:
        sequences += ",..."
    line = (
        f"[TOPIC_SEGMENT] topicId={topic.topicId} "
        f"microBlocks={len(topic.microBlockIds)} "
        f"sequences={sequences} "
        f"coherence={topic.coherence:.3f} "
        f"boundaryReason={topic.boundaryReason}"
    )
    print(line)
    obs = current_observability()
    if obs is not None:
        obs.logs.append(line)


class TopicLabelResponse(BaseModel):
    label: str = ""
    entities: list[str] = Field(default_factory=list)


async def label_ambiguous_topics(topics: list[LocalTopic], router: LLMRouter) -> None:
    from services.conversation.event_pipeline.llm import generate_structured_for_stage
    from services.conversation.event_pipeline.routing import PipelineStage

    for topic in topics:
        if topic.entities:
            continue
        try:
            response, _, _ = await generate_structured_for_stage(
                router,
                PipelineStage.TOPIC_LABEL,
                "topic-label-v1",
                TopicLabelResponse,
                {"topicId": topic.topicId, "text": topic.text},
            )
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime
            from services.conversation.event_pipeline.observability import record_failure

            record_failure(error)
            reraise_if_hard_runtime(error)
            continue
        if response.label:
            topic.label = response.label[:80]
        if response.entities:
            topic.entities = response.entities[:8]
