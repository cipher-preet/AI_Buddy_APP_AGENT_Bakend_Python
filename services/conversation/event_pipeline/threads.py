"""Global thread linking over atomic events, not raw chunks.

Candidate generation is top-k embedding retrieval plus entity/object filters.
Membership is validated semantically; vector similarity alone never links.
Generative LLM is used only for ambiguous membership, with rare high-accuracy
escalation. Pairwise gpt-oss-120b comparison of every event is forbidden.
"""

from __future__ import annotations

from pydantic import BaseModel

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.embeddings import CachedEmbedder, default_embedder, top_k_similar
from services.conversation.event_pipeline.flags import (
    thread_candidate_similarity_threshold,
    thread_entityless_min_similarity,
)
from services.conversation.event_pipeline.schemas import (
    AtomicEvent,
    EventKind,
    GlobalThread,
    ThreadLink,
    ThreadRelation,
)
from services.conversation.event_pipeline.textutil import (
    cosine_similarity,
    extract_entities,
    mean_vector,
    stable_id,
    token_jaccard,
)
from services.llm.router import LLMCapability, LLMRouter


RELATION_BY_KIND = {
    EventKind.COMPLETION: ThreadRelation.COMPLETES,
    EventKind.CANCELLATION: ThreadRelation.CANCELS,
    EventKind.CONTRADICTION: ThreadRelation.CONTRADICTS,
}


async def link_global_threads(
    events: list[AtomicEvent],
    embedder: CachedEmbedder | None = None,
    router: LLMRouter | None = None,
    verifier: "ThreadMembershipVerifier | None" = None,
) -> tuple[list[GlobalThread], list[ThreadLink], int]:
    if not events:
        return [], [], 0
    embedder = embedder or default_embedder()
    missing = [event.meaning for event in events if not event.embedding]
    if missing:
        vectors = await embedder.embed_many(missing)
        iterator = iter(vectors)
        for event in events:
            if not event.embedding:
                event.embedding = next(iterator)
            if not event.entities:
                event.entities = extract_entities(" ".join([event.meaning, event.object or "", " ".join(span.text for span in event.evidence)]))
    ordered = sorted(events, key=lambda event: (min(event.sequenceIds) if event.sequenceIds else 0, event.eventId))
    threads: list[GlobalThread] = []
    links: list[ThreadLink] = []
    comparisons = 0
    top_k = int(getattr(settings, "EVENT_PIPELINE_THREAD_TOP_K", 8))
    verifier = verifier or (ThreadMembershipVerifier(router) if router is not None else None)

    for event in ordered:
        candidates = _candidate_threads(event, threads, top_k)
        comparisons += max(len(threads), 1) if len(threads) <= top_k else top_k
        chosen: GlobalThread | None = None
        relation = ThreadRelation.SAME_THREAD
        best_score = 0.0
        related_edges: list[tuple[GlobalThread, ThreadRelation, float]] = []
        for thread, score in candidates:
            accepted, candidate_relation = _validate_membership(event, thread, score)
            if verifier is not None and _ambiguous_membership(event, thread, score, accepted, candidate_relation):
                accepted, candidate_relation = await verifier.verify(event, thread, score, accepted, candidate_relation)
            semantic = _semantic_relation(candidate_relation, accepted)
            if semantic == "RELATED_BUT_DISTINCT":
                related_edges.append((thread, candidate_relation, score))
                continue
            if semantic == "UNRELATED":
                continue
            if accepted and semantic == "SAME_THREAD" and score >= best_score:
                chosen = thread
                relation = candidate_relation if candidate_relation != ThreadRelation.RELATED_TO else ThreadRelation.SAME_THREAD
                best_score = score
        if chosen is None:
            chosen = _new_thread(event)
            threads.append(chosen)
        else:
            _attach(chosen, event)
            if chosen.eventIds:
                previous = chosen.eventIds[-2] if len(chosen.eventIds) > 1 else chosen.eventIds[0]
                links.append(
                    ThreadLink(
                        fromEventId=previous,
                        toEventId=event.eventId,
                        relation=relation,
                        score=best_score,
                        crossWindow=abs((min(event.sequenceIds) if event.sequenceIds else 0) - chosen.sequenceStart) >= 8,
                    )
                )
        for thread, edge_relation, edge_score in related_edges:
            links.append(
                ThreadLink(
                    fromEventId=thread.eventIds[-1] if thread.eventIds else event.eventId,
                    toEventId=event.eventId,
                    relation=edge_relation if edge_relation != ThreadRelation.RELATED_TO else ThreadRelation.RELATED_BUT_DISTINCT,
                    score=edge_score,
                    crossWindow=True,
                )
            )
        event.threadId = chosen.threadId
    return threads, links, comparisons


def _candidate_threads(event: AtomicEvent, threads: list[GlobalThread], k: int) -> list[tuple[GlobalThread, float]]:
    if not threads:
        return []
    retrieved = top_k_similar(
        event.embedding or [],
        ((thread.threadId, thread.embedding) for thread in threads),
        k=k,
        min_score=thread_candidate_similarity_threshold(),
    )
    by_id = {thread.threadId: thread for thread in threads}
    ranked: list[tuple[GlobalThread, float]] = []
    entityless_min = thread_entityless_min_similarity()
    for thread_id, score in retrieved:
        thread = by_id[thread_id]
        entity_score = _distinctive_entity_overlap(event.entities, thread.entities)
        object_score = token_jaccard(event.object or event.meaning, thread.label)
        if entity_score <= 0 and object_score <= 0 and score < entityless_min:
            continue
        ranked.append((thread, 0.45 * score + 0.35 * entity_score + 0.2 * object_score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


WEAK_THREAD_ENTITIES = frozenset(
    {
        "server",
        "connection",
        "issue",
        "problem",
        "id",
        "network",
        "setup",
        "app",
        "service",
        "system",
    }
)


def _validate_membership(event: AtomicEvent, thread: GlobalThread, score: float) -> tuple[bool, ThreadRelation]:
    distinctive_entities = _distinctive_entity_overlap(event.entities, thread.entities)
    objects = token_jaccard(event.object or "", thread.label)
    meaning = cosine_similarity(event.embedding, thread.embedding)
    same_object = objects >= 0.45 or _same_action_object(event, thread)
    same_issue = distinctive_entities >= 0.34 and (objects >= 0.2 or meaning >= 0.55)
    broad_domain_only = (
        (distinctive_entities <= 0 or objects < 0.2)
        and meaning >= 0.45
        and not same_object
    )
    if same_object or same_issue:
        relation = RELATION_BY_KIND.get(event.kind, ThreadRelation.SAME_THREAD)
        if event.kind in {EventKind.STATE, EventKind.RESULT, EventKind.FACT, EventKind.ISSUE} and score >= 0.4:
            relation = ThreadRelation.UPDATES
        return True, relation
    if broad_domain_only or (score >= 0.62 and (distinctive_entities > 0 or objects > 0)):
        return False, ThreadRelation.RELATED_BUT_DISTINCT
    return False, ThreadRelation.UNRELATED


def _same_action_object(event: AtomicEvent, thread: GlobalThread) -> bool:
    left = (event.object or (event.actionSignal.object if event.actionSignal else "") or "").strip()
    right = (thread.label or "").strip()
    if not left or not right:
        return False
    return token_jaccard(left, right) >= 0.5


def _semantic_relation(relation: ThreadRelation, accepted: bool) -> str:
    if relation == ThreadRelation.RELATED_BUT_DISTINCT:
        return "RELATED_BUT_DISTINCT"
    if relation == ThreadRelation.UNRELATED:
        return "UNRELATED"
    if relation == ThreadRelation.RELATED_TO:
        return "RELATED_BUT_DISTINCT"
    if accepted and relation in {
        ThreadRelation.SAME_THREAD,
        ThreadRelation.UPDATES,
        ThreadRelation.SUPPORTS,
        ThreadRelation.CONTRADICTS,
        ThreadRelation.COMPLETES,
        ThreadRelation.CANCELS,
        ThreadRelation.SUPERSEDES,
    }:
        return "SAME_THREAD"
    if accepted:
        return "SAME_THREAD"
    return "UNRELATED"


def _distinctive_entity_overlap(left, right) -> float:
    left_set = {item.casefold() for item in left if item and item.casefold() not in WEAK_THREAD_ENTITIES}
    right_set = {item.casefold() for item in right if item and item.casefold() not in WEAK_THREAD_ENTITIES}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _new_thread(event: AtomicEvent) -> GlobalThread:
    sequences = event.sequenceIds or [0]
    label = event.object or (event.entities[0] if event.entities else event.meaning[:80])
    thread = GlobalThread(
        threadId=stable_id("THREAD", event.conversationId or event.eventId, label),
        label=label,
        eventIds=[event.eventId],
        entities=list(event.entities),
        latestState=event.meaning,
        sequenceStart=min(sequences),
        sequenceEnd=max(sequences),
        embedding=event.embedding,
    )
    return thread


def _attach(thread: GlobalThread, event: AtomicEvent) -> None:
    if event.eventId not in thread.eventIds:
        thread.eventIds.append(event.eventId)
    for entity in event.entities:
        if entity.casefold() not in {item.casefold() for item in thread.entities}:
            thread.entities.append(entity)
    sequences = event.sequenceIds or [thread.sequenceEnd]
    thread.sequenceStart = min(thread.sequenceStart, min(sequences))
    thread.sequenceEnd = max(thread.sequenceEnd, max(sequences))
    thread.latestState = event.meaning
    thread.embedding = mean_vector([item for item in [thread.embedding, event.embedding] if item]) or thread.embedding


def _ambiguous_membership(
    event: AtomicEvent,
    thread: GlobalThread,
    score: float,
    accepted: bool,
    relation: ThreadRelation | None = None,
) -> bool:
    if relation in {ThreadRelation.RELATED_BUT_DISTINCT, ThreadRelation.UNRELATED}:
        return False
    entities = _distinctive_entity_overlap(event.entities, thread.entities)
    objects = token_jaccard(event.object or event.meaning, thread.label)
    if accepted:
        return False
    if 0.42 <= score < 0.78 and (entities > 0 or objects > 0):
        return True
    if score >= 0.62 and entities <= 0 and objects < 0.25:
        return True
    return False


class ThreadMembershipVerdict(BaseModel):
    sameThread: bool = False
    relation: ThreadRelation = ThreadRelation.UNRELATED
    semanticRelation: str = "UNRELATED"
    confidence: float = 0.5
    ambiguous: bool = False
    reason: str = ""


class ThreadMembershipVerifier:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.verify_calls = 0
        self.hard_escalations = 0
        self.requested_capabilities: list[LLMCapability] = []

    async def verify(
        self,
        event: AtomicEvent,
        thread: GlobalThread,
        score: float,
        accepted: bool,
        relation: ThreadRelation,
    ) -> tuple[bool, ThreadRelation]:
        from services.conversation.event_pipeline.llm import compact_event, compact_thread, generate_structured_for_stage
        from services.conversation.event_pipeline.routing import PipelineStage, capability_for_stage

        self.verify_calls += 1
        self.requested_capabilities.append(capability_for_stage(PipelineStage.THREAD_VERIFY))
        payload = {
            "event": compact_event(event),
            "thread": compact_thread(thread),
            "retrievalScore": round(score, 4),
        }
        try:
            verdict, _, _ = await generate_structured_for_stage(
                self.router,
                PipelineStage.THREAD_VERIFY,
                "thread-membership-v1",
                ThreadMembershipVerdict,
                payload,
            )
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime
            from services.conversation.event_pipeline.observability import record_failure

            record_failure(error)
            reraise_if_hard_runtime(error)
            return accepted, relation
        if verdict.ambiguous or verdict.confidence < 0.45:
            max_hard = int(getattr(settings, "EVENT_PIPELINE_THREAD_HARD_MAX_ESCALATIONS", 3))
            if self.hard_escalations < max_hard:
                self.hard_escalations += 1
                self.requested_capabilities.append(capability_for_stage(PipelineStage.THREAD_HARD))
                try:
                    verdict, _, _ = await generate_structured_for_stage(
                        self.router,
                        PipelineStage.THREAD_HARD,
                        "thread-membership-v1",
                        ThreadMembershipVerdict,
                        payload,
                    )
                except Exception:
                    return accepted, relation
        semantic = str(verdict.semanticRelation or "").strip().upper()
        if semantic == "RELATED_BUT_DISTINCT":
            return False, ThreadRelation.RELATED_BUT_DISTINCT
        if semantic == "UNRELATED":
            return False, ThreadRelation.UNRELATED
        if verdict.sameThread and semantic != "RELATED_BUT_DISTINCT":
            joined = verdict.relation or relation
            if joined in {ThreadRelation.RELATED_TO, ThreadRelation.RELATED_BUT_DISTINCT, ThreadRelation.UNRELATED}:
                joined = ThreadRelation.SAME_THREAD
            return True, joined
        if verdict.sameThread:
            return False, ThreadRelation.RELATED_BUT_DISTINCT
        return False, verdict.relation or ThreadRelation.UNRELATED

