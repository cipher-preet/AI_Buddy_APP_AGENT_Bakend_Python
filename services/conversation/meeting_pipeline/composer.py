"""General-purpose composition of user-facing Tasks and Notes.

Atomic candidates are useful for recall. Users need composed artifacts:
- a Task title names one action; the body explains grounded context
- a Note groups related memory into a later-readable explanation

This layer does not invent facts, owners, dates, or sequence IDs. It only
rearranges candidate meanings that already exist on the ledger.
It has no domain keywords, product rules, or meeting-specific templates.
"""

from __future__ import annotations

from services.conversation.event_pipeline.textutil import content_tokens, normalize_text, token_jaccard
from services.conversation.meeting_pipeline.consolidator import _is_note_kind, _title_from_meaning
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.schemas import ArtifactClaim, CandidateKind, MeetingCandidate

_RELATED = 0.24
_CLUSTER = 0.42
_TITLE_ECHO = 0.82
_MIN_SHARED = 2
_MAX_EXTRA = 4
_MAX_CLUSTER = 8
_NEIGHBOR_GAP = 2
_CONTEXT_KINDS = frozenset(
    {
        CandidateKind.REQUIREMENT,
        CandidateKind.DECISION,
        CandidateKind.FACT,
        CandidateKind.RATIONALE,
        CandidateKind.ISSUE,
    }
)


def compose_artifacts(
    artifacts: list[ArtifactClaim],
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> tuple[list[ArtifactClaim], dict[str, int]]:
    tasks = [item for item in artifacts if item.kind == "task"]
    notes = [item for item in artifacts if item.kind == "note"]
    enriched = 0
    composed_tasks: list[ArtifactClaim] = []
    for task in tasks:
        updated, changed = _enrich_task(task, notes, ledger, meeting_sequences)
        composed_tasks.append(updated)
        enriched += int(changed)
    clustered, merged = _cluster_notes(notes)
    return [*composed_tasks, *clustered], {
        "tasksEnriched": enriched,
        "notesClustered": merged,
        "composedTaskCount": len(composed_tasks),
        "composedNoteCount": len(clustered),
    }


def _enrich_task(
    task: ArtifactClaim,
    notes: list[ArtifactClaim],
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> tuple[ArtifactClaim, bool]:
    related = _related_context(task, notes, ledger)
    if not related:
        if _is_shallow(task.title, task.body) and task.body:
            body = _as_sentence(task.body)
            if body != task.body:
                return task.model_copy(update={"body": body}), True
        return task, False
    extras: list[str] = []
    source_ids = list(task.sourceCandidateIds)
    evidence = list(task.evidenceSequences)
    seen_ids = set(source_ids)
    seen_seq = set(evidence)
    for item in related:
        meaning = _meaning_of(item)
        if not meaning or _already_covered(meaning, f"{task.title} {task.body} {' '.join(extras)}"):
            continue
        extras.append(meaning)
        for candidate_id in _source_ids_of(item):
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            source_ids.append(candidate_id)
        for sequence in _sequences_of(item):
            if sequence not in meeting_sequences or sequence in seen_seq:
                continue
            seen_seq.add(sequence)
            evidence.append(sequence)
        if len(extras) >= _MAX_EXTRA:
            break
    body = _compose_body(task.title, task.body, extras)
    changed = body != normalize_text(task.body) or source_ids != list(task.sourceCandidateIds)
    if not changed:
        return task, False
    return task.model_copy(update={"body": body, "sourceCandidateIds": source_ids, "evidenceSequences": evidence}), True


def _related_context(
    task: ArtifactClaim,
    notes: list[ArtifactClaim],
    ledger: CandidateLedger,
) -> list[MeetingCandidate | ArtifactClaim]:
    index = ledger.by_id()
    seed = f"{task.title} {task.body}"
    found: list[MeetingCandidate | ArtifactClaim] = []
    seen: set[str] = set()

    def take(item: MeetingCandidate | ArtifactClaim, key: str) -> None:
        if key in seen:
            return
        meaning = _meaning_of(item)
        if not meaning:
            return
        if _already_covered(meaning, seed) and not _is_shallow(task.title, task.body):
            return
        seen.add(key)
        found.append(item)

    for candidate_id in task.sourceCandidateIds:
        candidate = index.get(str(candidate_id))
        if candidate is None or not _is_context_kind(candidate):
            continue
        take(candidate, candidate.candidateId)

    for candidate in ledger.candidates:
        if candidate.candidateId in seen or not _is_context_kind(candidate):
            continue
        if _related_to(seed, candidate.meaning, task.evidenceSequences, candidate.evidenceSequences):
            take(candidate, candidate.candidateId)

    for note in notes:
        key = note.artifactKey or note.title
        if key in seen or _is_action_workstream(note, index):
            continue
        blob = f"{note.title} {note.body}"
        if _related_to(seed, blob, task.evidenceSequences, note.evidenceSequences):
            take(note, key)
    return found


def _cluster_notes(notes: list[ArtifactClaim]) -> tuple[list[ArtifactClaim], int]:
    if len(notes) <= 1:
        return list(notes), 0
    groups = _union_groups(notes, _notes_related)
    clustered: list[ArtifactClaim] = []
    merged = 0
    for group in groups:
        if len(group) == 1:
            clustered.append(group[0])
            continue
        group = group[:_MAX_CLUSTER]
        clustered.append(_merge_note_group(group))
        merged += len(group) - 1
    return clustered, merged


def _notes_related(left: ArtifactClaim, right: ArtifactClaim) -> bool:
    return _related_to(
        f"{left.title} {left.body}",
        f"{right.title} {right.body}",
        left.evidenceSequences,
        right.evidenceSequences,
        cluster=True,
    )


def _merge_note_group(group: list[ArtifactClaim]) -> ArtifactClaim:
    ordered = sorted(group, key=lambda item: min(item.evidenceSequences or [10**9]))
    extras: list[str] = []
    source_ids: list[str] = []
    evidence: list[int] = []
    seen_ids: set[str] = set()
    seen_seq: set[int] = set()
    lead = ""
    for item in ordered:
        text = normalize_text(item.body) or normalize_text(item.title)
        if not lead:
            lead = text
        elif not _already_covered(text, f"{lead} {' '.join(extras)}"):
            extras.append(text)
        for candidate_id in item.sourceCandidateIds:
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            source_ids.append(candidate_id)
        for sequence in item.evidenceSequences:
            if sequence in seen_seq:
                continue
            seen_seq.add(sequence)
            evidence.append(sequence)
    title = _cluster_title(ordered)
    body = _compose_body(title, lead, extras)
    return ArtifactClaim(
        artifactKey=ordered[0].artifactKey,
        kind="note",
        title=title,
        body=body,
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _cluster_title(group: list[ArtifactClaim]) -> str:
    ranked = sorted(
        group,
        key=lambda item: (
            -len(content_tokens(item.title)),
            len(normalize_text(item.title)),
        ),
    )
    title = normalize_text(ranked[0].title)
    if title and len(title) <= 80:
        return title
    return _title_from_meaning(group[0].body or group[0].title)


def _related_to(
    left: str,
    right: str,
    left_sequences: list[int] | None,
    right_sequences: list[int] | None,
    *,
    cluster: bool = False,
) -> bool:
    shared = _shared_tokens(left, right)
    if not shared:
        return False
    score = token_jaccard(left, right)
    nearby = _nearby(left_sequences, right_sequences)
    if cluster:
        if len(shared) < _MIN_SHARED:
            return False
        return score >= _CLUSTER or nearby
    if score >= _RELATED and len(shared) >= _MIN_SHARED:
        return True
    distinctive = {token for token in shared if len(token) >= 5}
    return nearby and bool(distinctive)


def _shared_tokens(left: str, right: str) -> set[str]:
    return {token.casefold() for token in content_tokens(left)} & {token.casefold() for token in content_tokens(right)}


def _nearby(left: list[int] | None, right: list[int] | None) -> bool:
    if not left or not right:
        return False
    return min(abs(a - b) for a in left for b in right) <= _NEIGHBOR_GAP


def _is_shallow(title: str, body: str) -> bool:
    heading = normalize_text(title)
    text = normalize_text(body)
    if not heading:
        return False
    if not text or text.casefold() == heading.casefold():
        return True
    if token_jaccard(heading, text) >= _TITLE_ECHO and len(text) <= len(heading) + 48:
        return True
    return len(text) <= max(len(heading) + 8, int(len(heading) * 1.2))


def _already_covered(meaning: str, blob: str) -> bool:
    text = normalize_text(meaning)
    if not text:
        return True
    haystack = normalize_text(blob).casefold()
    if text.casefold() in haystack:
        return True
    return token_jaccard(text, blob) >= 0.78


def _compose_body(title: str, lead: str, extras: list[str]) -> str:
    parts: list[str] = []
    for raw in [lead, *extras]:
        text = normalize_text(raw)
        if not text:
            continue
        if any(_already_covered(text, prev) for prev in parts):
            continue
        parts.append(_as_sentence(text))
    if extras:
        detailed = [part for part in parts if not _is_shallow(title, part)]
        if detailed:
            parts = detailed
    if not parts:
        fallback = normalize_text(lead) or normalize_text(title)
        return _as_sentence(fallback)
    return " ".join(parts)


def _as_sentence(text: str) -> str:
    value = normalize_text(text)
    if not value:
        return ""
    if value[0].isalpha() and value[0].islower():
        value = value[0].upper() + value[1:]
    if value[-1] not in ".!?।":
        value += "."
    return value


def _is_context_kind(candidate: MeetingCandidate) -> bool:
    kind = candidate.kind
    if isinstance(kind, CandidateKind):
        return kind in _CONTEXT_KINDS
    try:
        return CandidateKind(str(kind)) in _CONTEXT_KINDS
    except ValueError:
        return _is_note_kind(candidate)


def _is_action_workstream(note: ArtifactClaim, index: dict[str, MeetingCandidate]) -> bool:
    candidates = [index[candidate_id] for candidate_id in note.sourceCandidateIds if candidate_id in index]
    if not candidates:
        return False
    return all(not _is_context_kind(candidate) for candidate in candidates)


def _meaning_of(item: MeetingCandidate | ArtifactClaim) -> str:
    if isinstance(item, MeetingCandidate):
        return normalize_text(item.meaning)
    return normalize_text(item.body) or normalize_text(item.title)


def _source_ids_of(item: MeetingCandidate | ArtifactClaim) -> list[str]:
    if isinstance(item, MeetingCandidate):
        return [item.candidateId]
    return list(item.sourceCandidateIds or [])


def _sequences_of(item: MeetingCandidate | ArtifactClaim) -> list[int]:
    return list(item.evidenceSequences or [])


def _union_groups(items: list[ArtifactClaim], related) -> list[list[ArtifactClaim]]:
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(items):
        for j, right in enumerate(items[i + 1 :], start=i + 1):
            if related(left, right):
                union(i, j)
    groups: dict[int, list[ArtifactClaim]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)
    return list(groups.values())
