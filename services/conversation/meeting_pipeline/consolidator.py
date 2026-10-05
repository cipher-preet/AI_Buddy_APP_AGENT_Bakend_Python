"""Global consolidation of the candidate ledger into tasks and notes."""

from __future__ import annotations

import asyncio
import json

from services.conversation.event_pipeline.textutil import token_jaccard
from services.conversation.meeting_pipeline.flags import (
    consolidation_max_candidates,
    consolidation_partition_tokens,
    coverage_batch_candidates,
    max_extraction_concurrency,
    output_language,
)
from services.conversation.meeting_pipeline.harness import (
    CandidateAliases,
    clean_inline,
    clean_multiline,
    normalize_priority,
    normalize_topic,
    same_topic,
    strip_internal_references,
    topic_key,
)
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.llm import generate_structured
from services.conversation.meeting_pipeline.observability import log_pipeline
from services.conversation.meeting_pipeline.schemas import (
    ArtifactClaim,
    CandidateKind,
    MeetingCandidate,
    MeetingConsolidatorResponse,
    MeetingCoverageResponse,
)
from services.conversation.transcript import estimate_tokens
from services.llm.async_runtime import reraise_if_hard_runtime
from services.llm.router import LLMCapability, LLMRouter

_NOTE_KINDS = frozenset(
    {
        CandidateKind.REQUIREMENT,
        CandidateKind.DECISION,
        CandidateKind.FACT,
        CandidateKind.RATIONALE,
        CandidateKind.ISSUE,
        CandidateKind.IDEA,
        CandidateKind.QUESTION,
    }
)
_TITLE_PARAPHRASE = 0.72
_NOTE_PARAPHRASE = 0.55

_WRITING_CONTRACT = {
    "taskTitle": "Imperative, specific deliverable (verb + object), e.g. 'Add configurable X to Y'. Not a topic label, not 'Speaker will...'.",
    "taskDescription": "2-4 grounded sentences: what to build/do/decide, the concrete rules and values it must honor, current status or open points. Never copy the title.",
    "acceptanceCriteria": "One sentence: the observable result that proves the task is done.",
    "priority": "High | Medium | Low, judged from this meeting: High = core to the meeting's main objective, blocking, correctness of core logic, or stated urgent; Medium = needed but secondary; Low = explicitly deferred/later/nice-to-have.",
    "noteTitle": "Specific topic name (the subject), not a sentence about a speaker.",
    "noteBody": "Lead sentence, then '- ' bullet lines with every grounded rule, field, option, value, example, and flow (use 'â†’' for sequences). Mark unresolved items 'Open decision:' and tentative ones 'Tentative:'.",
    "rules": [
        "Do not invent owners, dates, numbers, or facts.",
        "Speaker N labels are diarization, never owners or note subjects.",
        "One note per topic; merge every candidate about the same topic into it.",
        "Do not fold every non-action meaning into a task and return notes=[].",
        "Do not return tasks=[] when the meeting implies real work.",
        "Do not publish two tasks for the same work. Merge paraphrases.",
        "List noise candidate IDs in discardedCandidateIds.",
        "Missing a real independent action, decision, rule, or requirement is worse than a near-duplicate.",
    ],
}


class GlobalArtifactConsolidator:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.last_provider = "none"
        self.last_model = "none"
        self.last_discarded_ids: set[str] = set()
        self.last_partition_count = 0
        self.unresolved_ids = 0
        self.evidence_fallbacks = 0
        self.dropped_claims = 0
        self.coverage_calls = 0
        self.coverage_uncovered = 0
        self.coverage_attached = 0

    async def consolidate(
        self,
        ledger: CandidateLedger,
        sequence_text: dict[int, str],
    ) -> tuple[list[ArtifactClaim], str, list[str]]:
        self.last_discarded_ids = set()
        self.last_partition_count = 0
        self.unresolved_ids = 0
        self.evidence_fallbacks = 0
        self.dropped_claims = 0
        self.coverage_calls = 0
        self.coverage_uncovered = 0
        self.coverage_attached = 0
        if not ledger.candidates:
            self.calls += 1
            return [], "", []
        budget = consolidation_partition_tokens()
        size = consolidation_max_candidates()
        partitions = partition_candidates(ledger.candidates, sequence_text, budget, size)
        if len(partitions) == 1:
            try:
                result = await self._consolidate_partitions(partitions, ledger, sequence_text)
            except Exception as error:
                reraise_if_hard_runtime(error)
                smaller = partition_candidates(ledger.candidates, sequence_text, max(2000, budget // 2), max(8, size // 2))
                if len(smaller) <= 1:
                    raise
                log_pipeline({"event": "consolidator_single_call_failed_repartitioning", "partitions": len(smaller), "error": str(error)[:300]})
                result = await self._consolidate_partitions(smaller, ledger, sequence_text)
        else:
            result = await self._consolidate_partitions(partitions, ledger, sequence_text)
        artifacts, summary, topics = result
        artifacts = await self._ensure_coverage(artifacts, ledger, sequence_text)
        return artifacts, summary, topics

    async def _ensure_coverage(
        self,
        artifacts: list[ArtifactClaim],
        ledger: CandidateLedger,
        sequence_text: dict[int, str],
    ) -> list[ArtifactClaim]:
        """Every candidate must end up cited, attached, or explicitly discarded.

        Candidates the consolidator neither used nor discarded get one targeted
        pass that may attach them to existing artifacts, publish new ones, or
        discard them. Failure is non-fatal: later recovery stages still run.
        """
        cited = {candidate_id for item in artifacts for candidate_id in item.sourceCandidateIds}
        uncovered = [
            candidate
            for candidate in ledger.candidates
            if candidate.candidateId not in cited and candidate.candidateId not in self.last_discarded_ids
        ]
        self.coverage_uncovered = len(uncovered)
        if not uncovered or not artifacts:
            return artifacts
        updated = list(artifacts)
        batch = coverage_batch_candidates()
        for offset in range(0, len(uncovered), batch):
            try:
                updated = await self._coverage_call(updated, uncovered[offset : offset + batch], ledger, sequence_text, offset)
            except Exception as error:
                reraise_if_hard_runtime(error)
                log_pipeline({"event": "consolidator_coverage_failed", "uncovered": len(uncovered), "error": str(error)[:300]})
        cited = {candidate_id for item in updated for candidate_id in item.sourceCandidateIds}
        self.last_discarded_ids -= cited
        return updated

    async def _coverage_call(
        self,
        artifacts: list[ArtifactClaim],
        uncovered: list[MeetingCandidate],
        ledger: CandidateLedger,
        sequence_text: dict[int, str],
        offset: int,
    ) -> list[ArtifactClaim]:
        self.calls += 1
        self.coverage_calls += 1
        aliases = CandidateAliases(uncovered)
        artifact_ids = {f"A{position + 1}": position for position in range(len(artifacts))}
        payload = {
            "outputLanguage": output_language(),
            "existingArtifacts": [
                {
                    "artifactId": artifact_id,
                    "kind": artifacts[position].kind,
                    "title": artifacts[position].title,
                    "topic": artifacts[position].topic,
                    "body": artifacts[position].body[:600],
                }
                for artifact_id, position in artifact_ids.items()
            ],
            "uncoveredCandidates": [_candidate_payload(item, aliases.alias(item.candidateId)) for item in uncovered],
            "citedTranscript": _cited_lines(uncovered, sequence_text),
            "writingContract": _WRITING_CONTRACT,
        }
        response, provider, model = await generate_structured(
            self.router,
            LLMCapability.FINAL_SYNTHESIS,
            "meeting-artifact-coverage-v1",
            MeetingCoverageResponse,
            payload,
            stage="consolidator_coverage",
        )
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        known_ids = set(ledger.by_id())
        meeting_sequences = set(sequence_text)
        updated = list(artifacts)
        attached: set[str] = set()
        for attachment in response.attachments or []:
            position = artifact_ids.get(str(attachment.artifactId or "").strip().upper())
            sources = [cid for cid in aliases.resolve_many(attachment.candidateIds) if cid not in attached]
            if position is None or not sources:
                continue
            updated[position] = _attach(updated[position], sources, attachment.addition, ledger, meeting_sequences)
            attached.update(sources)
        prefix = f"cov{offset}:"
        for index, item in enumerate(response.tasks or []):
            item.sourceCandidateIds = aliases.resolve_many(item.sourceCandidateIds, item.evidenceSequences)
            claim = _task_claim(item, index, known_ids, ledger, meeting_sequences, prefix=prefix)
            if claim is not None:
                updated.append(claim)
        for index, item in enumerate(response.notes or []):
            item.sourceCandidateIds = aliases.resolve_many(item.sourceCandidateIds, item.evidenceSequences)
            claim = _note_claim(item, index, known_ids, ledger, meeting_sequences, prefix=prefix)
            if claim is not None:
                updated.append(claim)
        self.last_discarded_ids.update(aliases.resolve_many(response.discardedCandidateIds))
        self.coverage_attached += len(attached)
        return updated

    async def _consolidate_partitions(
        self,
        partitions: list[list[MeetingCandidate]],
        ledger: CandidateLedger,
        sequence_text: dict[int, str],
    ) -> tuple[list[ArtifactClaim], str, list[str]]:
        self.last_partition_count = len(partitions)
        semaphore = asyncio.Semaphore(max_extraction_concurrency())

        async def run(index: int, candidates: list[MeetingCandidate]) -> MeetingConsolidatorResponse:
            async with semaphore:
                return await self._call(candidates, sequence_text, index, len(partitions))

        responses = await asyncio.gather(*(run(index, part) for index, part in enumerate(partitions)))
        known_ids = set(ledger.by_id())
        meeting_sequences = set(sequence_text)
        artifacts: list[ArtifactClaim] = []
        summaries: list[str] = []
        topics: list[str] = []
        for part_index, response in enumerate(responses):
            prefix = f"p{part_index}:" if len(partitions) > 1 else ""
            for index, item in enumerate(response.tasks or []):
                claim = _task_claim(item, index, known_ids, ledger, meeting_sequences, prefix=prefix)
                if claim is not None:
                    artifacts.append(claim)
                else:
                    self.dropped_claims += 1
            for index, item in enumerate(response.notes or []):
                claim = _note_claim(item, index, known_ids, ledger, meeting_sequences, prefix=prefix)
                if claim is not None:
                    artifacts.append(claim)
                else:
                    self.dropped_claims += 1
            self.last_discarded_ids.update(
                str(value) for value in (response.discardedCandidateIds or []) if str(value) in known_ids
            )
            if str(response.summary or "").strip():
                summaries.append(str(response.summary).strip())
            for topic in response.topics or []:
                text = str(topic).strip()
                if text and text.casefold() not in {item.casefold() for item in topics}:
                    topics.append(text)
        cited = {candidate_id for item in artifacts for candidate_id in item.sourceCandidateIds}
        self.last_discarded_ids -= cited
        return artifacts, " ".join(summaries), topics

    async def _call(
        self,
        candidates: list[MeetingCandidate],
        sequence_text: dict[int, str],
        index: int,
        total: int,
    ) -> MeetingConsolidatorResponse:
        self.calls += 1
        aliases = CandidateAliases(candidates)
        payload = {
            "outputLanguage": output_language(),
            "partition": {"index": index, "total": total} if total > 1 else None,
            "candidates": [_candidate_payload(item, aliases.alias(item.candidateId)) for item in candidates],
            "citedTranscript": _cited_lines(candidates, sequence_text),
            "writingContract": _WRITING_CONTRACT,
        }
        response, provider, model = await generate_structured(
            self.router,
            LLMCapability.FINAL_SYNTHESIS,
            "meeting-artifact-consolidator-v1",
            MeetingConsolidatorResponse,
            payload,
            stage="consolidator",
            meta={"partition": index, "partitions": total},
        )
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        for item in [*(response.tasks or []), *(response.notes or [])]:
            item.sourceCandidateIds = aliases.resolve_many(item.sourceCandidateIds, item.evidenceSequences)
        response.discardedCandidateIds = aliases.resolve_many(response.discardedCandidateIds)
        self.unresolved_ids += aliases.unresolved
        self.evidence_fallbacks += aliases.evidence_fallbacks
        return response


def partition_candidates(
    candidates: list[MeetingCandidate],
    sequence_text: dict[int, str],
    budget_tokens: int,
    max_candidates: int | None = None,
) -> list[list[MeetingCandidate]]:
    """Pack whole topic groups into partitions under a token and size budget.

    Topic groups never split unless a single group alone exceeds the budget;
    groups keep transcript order so cross-window references stay together.
    """
    ordered = list(candidates)
    limit = max_candidates or len(ordered) or 1

    def over(items: list[MeetingCandidate]) -> bool:
        return len(items) > limit or _payload_tokens(items, sequence_text) > budget_tokens

    if not over(ordered):
        return [ordered]
    groups: list[list[MeetingCandidate]] = []
    labels: list[str | None] = []
    for candidate in ordered:
        label = candidate.topic
        slot = None
        if topic_key(label):
            slot = next((i for i, existing in enumerate(labels) if same_topic(existing, label)), None)
        if slot is None:
            if not topic_key(label) and groups and not topic_key(labels[-1]) and groups[-1][-1].sourceWindowIndex == candidate.sourceWindowIndex:
                slot = len(groups) - 1
            else:
                groups.append([])
                labels.append(label)
                slot = len(groups) - 1
        groups[slot].append(candidate)
    partitions: list[list[MeetingCandidate]] = []
    current: list[MeetingCandidate] = []
    for group in groups:
        pieces = _split_group(group, over) if over(group) else [group]
        for piece in pieces:
            if current and over([*current, *piece]):
                partitions.append(current)
                current = []
            current = [*current, *piece]
    if current:
        partitions.append(current)
    return partitions or [ordered]


def _split_group(group: list[MeetingCandidate], over) -> list[list[MeetingCandidate]]:
    pieces: list[list[MeetingCandidate]] = []
    current: list[MeetingCandidate] = []
    for candidate in group:
        if current and over([*current, candidate]):
            pieces.append(current)
            current = []
        current.append(candidate)
    if current:
        pieces.append(current)
    return pieces


def _payload_tokens(candidates: list[MeetingCandidate], sequence_text: dict[int, str]) -> int:
    payload = {
        "candidates": [_candidate_payload(item) for item in candidates],
        "citedTranscript": _cited_lines(candidates, sequence_text),
    }
    return estimate_tokens(json.dumps(payload, ensure_ascii=False))


def _candidate_payload(item: MeetingCandidate, alias: str | None = None) -> dict:
    return {
        "candidateId": alias or item.candidateId,
        "kind": item.kind.value if hasattr(item.kind, "value") else str(item.kind),
        "topic": item.topic,
        "meaning": item.meaning,
        "evidenceSequences": list(item.evidenceSequences),
        "owner": item.owner,
        "dueDate": item.dueDate,
        "sourceWindowIndex": item.sourceWindowIndex,
    }


def _cited_lines(candidates: list[MeetingCandidate], sequence_text: dict[int, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for candidate in candidates:
        for sequence in candidate.evidenceSequences:
            if sequence in sequence_text:
                lookup[str(sequence)] = sequence_text[sequence]
    return dict(sorted(lookup.items(), key=lambda pair: int(pair[0])))


def recover_unpublished_notes(
    artifacts: list[ArtifactClaim],
    ledger: CandidateLedger,
    meeting_sequences: set[int],
    *,
    discarded: set[str] | None = None,
) -> tuple[list[ArtifactClaim], int]:
    """Publish leftover memory as notes when consolidation folded it into tasks.

    Candidates the consolidator explicitly discarded as noise stay discarded.
    Leftovers that share a topic with a published note are appended to it;
    other leftovers sharing a topic are grouped into one note per topic.
    """
    skip = set(discarded or ())
    recovered = list(artifacts)
    cited_by_notes = {
        str(candidate_id)
        for item in recovered
        if item.kind == "note"
        for candidate_id in item.sourceCandidateIds
    }
    if cited_by_notes:
        # The consolidator published notes, so its task/note split is deliberate:
        # candidates it used for tasks are not re-published as standalone notes.
        skip |= {
            str(candidate_id)
            for item in recovered
            if item.kind == "task"
            for candidate_id in item.sourceCandidateIds
        }
    index = ledger.by_id()
    pending = [
        candidate
        for candidate in ledger.candidates
        if _is_note_kind(candidate) and candidate.candidateId not in cited_by_notes and candidate.candidateId not in skip
    ]
    if not any(item.kind == "note" for item in recovered) and not pending:
        pending = _supporting_candidates_from_tasks(recovered, index, cited_by_notes | skip)

    added = 0
    for offset, candidate in enumerate(pending):
        if candidate.candidateId in cited_by_notes:
            continue
        if _memory_already_published(candidate.meaning, recovered):
            continue
        target = _topic_note_index(candidate, recovered)
        if target is not None:
            recovered[target] = _append_to_note(recovered[target], candidate, ledger, meeting_sequences)
            cited_by_notes.add(candidate.candidateId)
            added += 1
            continue
        claim = _note_from_candidate(candidate, offset, ledger, meeting_sequences)
        if claim is None:
            continue
        recovered.append(claim)
        cited_by_notes.add(candidate.candidateId)
        added += 1
    return recovered, added


def _topic_note_index(candidate: MeetingCandidate, artifacts: list[ArtifactClaim]) -> int | None:
    if not topic_key(candidate.topic):
        return None
    for position, item in enumerate(artifacts):
        if item.kind == "note" and same_topic(item.topic, candidate.topic):
            return position
    return None


def _append_to_note(
    note: ArtifactClaim,
    candidate: MeetingCandidate,
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> ArtifactClaim:
    meaning = clean_inline(candidate.meaning)
    body = clean_multiline(note.body)
    if meaning and meaning.casefold() not in body.casefold():
        body = f"{body}\n- {meaning}" if body else meaning
    source_ids = [*note.sourceCandidateIds]
    if candidate.candidateId not in source_ids:
        source_ids.append(candidate.candidateId)
    evidence = list(note.evidenceSequences)
    for sequence in _clip_evidence(candidate.evidenceSequences, [candidate.candidateId], ledger, meeting_sequences):
        if sequence not in evidence:
            evidence.append(sequence)
    return note.model_copy(update={"body": body, "sourceCandidateIds": source_ids, "evidenceSequences": evidence})


def _attach(
    artifact: ArtifactClaim,
    sources: list[str],
    addition: str,
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> ArtifactClaim:
    text = strip_internal_references(clean_multiline(addition))
    body = clean_multiline(artifact.body)
    if text and text.casefold() not in body.casefold():
        if artifact.kind == "note":
            lines = [line if line.startswith(("- ", "Open decision:", "Tentative:", "Deferred:")) else f"- {line}" for line in text.splitlines() if line]
            body = "\n".join([body, *lines]) if body else "\n".join(lines)
        else:
            body = f"{body} {clean_inline(text)}".strip()
    source_ids = list(artifact.sourceCandidateIds)
    evidence = list(artifact.evidenceSequences)
    for candidate_id in sources:
        if candidate_id not in source_ids:
            source_ids.append(candidate_id)
        for sequence in _clip_evidence([], [candidate_id], ledger, meeting_sequences):
            if sequence not in evidence:
                evidence.append(sequence)
    return artifact.model_copy(update={"body": body, "sourceCandidateIds": source_ids, "evidenceSequences": evidence})


def _task_claim(
    item,
    index: int,
    known_ids: set[str],
    ledger: CandidateLedger,
    meeting_sequences: set[int],
    *,
    prefix: str = "",
) -> ArtifactClaim | None:
    title = strip_internal_references(clean_inline(item.title))
    body = strip_internal_references(clean_multiline(item.description))
    if not title:
        return None
    source_ids = [str(value) for value in (item.sourceCandidateIds or []) if str(value) in known_ids]
    if not source_ids:
        return None
    evidence = _clip_evidence(item.evidenceSequences, source_ids, ledger, meeting_sequences)
    if not evidence:
        return None
    return ArtifactClaim(
        artifactKey=f"task:{prefix}{index}:{title}",
        kind="task",
        title=title,
        body=body,
        owner=_optional_text(item.owner),
        dueDate=_optional_text(item.dueDate),
        priority=normalize_priority(getattr(item, "priority", None)),
        acceptanceCriteria=strip_internal_references(clean_inline(getattr(item, "acceptanceCriteria", ""))),
        topic=normalize_topic(getattr(item, "topic", None)) or _dominant_topic(source_ids, ledger),
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _note_claim(
    item,
    index: int,
    known_ids: set[str],
    ledger: CandidateLedger,
    meeting_sequences: set[int],
    *,
    prefix: str = "",
) -> ArtifactClaim | None:
    title = strip_internal_references(clean_inline(item.title))
    body = strip_internal_references(clean_multiline(item.body)) or title
    if not title or not body:
        return None
    source_ids = [str(value) for value in (item.sourceCandidateIds or []) if str(value) in known_ids]
    if not source_ids:
        return None
    evidence = _clip_evidence(item.evidenceSequences, source_ids, ledger, meeting_sequences)
    if not evidence:
        return None
    return ArtifactClaim(
        artifactKey=f"note:{prefix}{index}:{title}",
        kind="note",
        title=title,
        body=body,
        topic=normalize_topic(getattr(item, "topic", None)) or _dominant_topic(source_ids, ledger) or title,
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _dominant_topic(source_ids: list[str], ledger: CandidateLedger) -> str | None:
    index = ledger.by_id()
    counts: dict[str, tuple[int, str]] = {}
    for candidate_id in source_ids:
        candidate = index.get(candidate_id)
        key = topic_key(candidate.topic) if candidate else ""
        if not key:
            continue
        count, label = counts.get(key, (0, candidate.topic or ""))
        counts[key] = (count + 1, label)
    if not counts:
        return None
    return normalize_topic(max(counts.values(), key=lambda pair: pair[0])[1])


def _note_from_candidate(
    candidate: MeetingCandidate,
    offset: int,
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> ArtifactClaim | None:
    meaning = clean_inline(candidate.meaning)
    topic = normalize_topic(candidate.topic)
    title = topic or _title_from_meaning(meaning)
    if not title or not meaning:
        return None
    source_ids = [candidate.candidateId]
    evidence = _clip_evidence(candidate.evidenceSequences, source_ids, ledger, meeting_sequences)
    if not evidence:
        return None
    return ArtifactClaim(
        artifactKey=f"note:recovered:{offset}:{title}",
        kind="note",
        title=title,
        body=f"- {meaning}" if topic else meaning,
        topic=topic,
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _clip_evidence(values, source_ids: list[str], ledger: CandidateLedger, meeting_sequences: set[int]) -> list[int]:
    if not source_ids:
        return []
    allowed = set(ledger.evidence_union(source_ids))
    provided = list(values or [])
    if not provided:
        return [sequence for sequence in ledger.evidence_union(source_ids) if sequence in meeting_sequences]
    sequences: list[int] = []
    seen: set[int] = set()
    for value in provided:
        try:
            sequence = int(value)
        except (TypeError, ValueError):
            continue
        if sequence not in meeting_sequences or sequence not in allowed or sequence in seen:
            continue
        seen.add(sequence)
        sequences.append(sequence)
    if not sequences:
        return [sequence for sequence in ledger.evidence_union(source_ids) if sequence in meeting_sequences]
    return sequences


def _supporting_candidates_from_tasks(
    artifacts: list[ArtifactClaim],
    index: dict[str, MeetingCandidate],
    cited_by_notes: set[str],
) -> list[MeetingCandidate]:
    pending: list[MeetingCandidate] = []
    seen: set[str] = set()
    for item in artifacts:
        if item.kind != "task":
            continue
        cited = [index[candidate_id] for candidate_id in item.sourceCandidateIds if candidate_id in index]
        if len(cited) < 2:
            continue
        primary = max(cited, key=lambda candidate: token_jaccard(candidate.meaning, item.title))
        for candidate in cited:
            if candidate.candidateId == primary.candidateId:
                continue
            if candidate.candidateId in cited_by_notes or candidate.candidateId in seen:
                continue
            seen.add(candidate.candidateId)
            pending.append(candidate)
    return pending


def _memory_already_published(meaning: str, artifacts: list[ArtifactClaim]) -> bool:
    for item in artifacts:
        if item.kind == "task":
            if token_jaccard(meaning, item.title) >= _TITLE_PARAPHRASE:
                return True
            continue
        blob = f"{item.title} {item.body}".strip()
        if token_jaccard(meaning, item.title) >= _TITLE_PARAPHRASE or token_jaccard(meaning, blob) >= _NOTE_PARAPHRASE:
            return True
        if clean_inline(meaning).casefold() in clean_inline(item.body).casefold():
            return True
    return False


def _is_note_kind(candidate: MeetingCandidate) -> bool:
    kind = candidate.kind
    if isinstance(kind, CandidateKind):
        return kind in _NOTE_KINDS
    try:
        return CandidateKind(str(kind)) in _NOTE_KINDS
    except ValueError:
        return False


def _title_from_meaning(meaning: str) -> str:
    text = " ".join(str(meaning or "").split())
    if not text:
        return ""
    if len(text) <= 80:
        return text
    for separator in (". ", "à¥¤ ", "? ", "! "):
        index = text.find(separator)
        if 12 <= index <= 80:
            return text[:index]
    clipped = text[:80]
    if " " in clipped:
        return clipped.rsplit(" ", 1)[0]
    return clipped


def _optional_text(value: str | None) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None
