"""General-purpose composition of user-facing Tasks and Notes.

Atomic candidates are useful for recall. Users need composed artifacts:
- a Task title names one action; the body explains grounded context
- a Note groups related memory into a later-readable explanation

This layer does not invent facts, owners, dates, or sequence IDs. It only
rearranges candidate meanings that already exist on the ledger.
It has no domain keywords, product rules, or meeting-specific templates.

Relatedness is judged with meeting-local token statistics: tokens shared by a
large share of the meeting (speaker labels, the product's own name) carry no
signal, so they never make two unrelated meanings look related. Topic labels
assigned by the extractor/consolidator LLMs take precedence over lexical
heuristics. LLM-written task descriptions are never padded with ledger text.
"""

from __future__ import annotations

import re

from services.conversation.event_pipeline.textutil import content_tokens, normalize_text, token_jaccard
from services.conversation.meeting_pipeline.consolidator import _is_note_kind, _title_from_meaning
from services.conversation.meeting_pipeline.harness import TokenProfile, clean_multiline, same_topic, topic_key
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.schemas import ArtifactClaim, CandidateKind, MeetingCandidate

_RELATED = 0.3
_CLUSTER = 0.5
_TITLE_ECHO = 0.82
_MIN_SHARED = 2
_MAX_EXTRA = 4
_MAX_CLUSTER = 12
_RESTATE = 0.4
_CONTAINED = 0.6
# Pipeline-internal kind labels the model sometimes prefixes; decisions and questions stay readable.
_KIND_LABEL_RE = re.compile(
    r"^(?:%s)\s*:\s*"
    % "|".join(kind.value for kind in CandidateKind if kind not in {CandidateKind.DECISION, CandidateKind.QUESTION}),
    re.IGNORECASE,
)
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
    profile = TokenProfile(
        [item.meaning for item in ledger.candidates]
        + [f"{item.title} {item.body}" for item in artifacts]
    )
    enriched = 0
    composed_tasks: list[ArtifactClaim] = []
    for task in tasks:
        updated, changed = _enrich_task(task, ledger, meeting_sequences, profile)
        composed_tasks.append(updated)
        enriched += int(changed)
    distinct_notes = [note for note in notes if not _restates_task(note, composed_tasks, profile)]
    clustered, merged = _cluster_notes(distinct_notes, profile)
    return [*composed_tasks, *clustered], {
        "tasksEnriched": enriched,
        "notesClustered": merged,
        "redundantNotesDropped": len(notes) - len(distinct_notes),
        "composedTaskCount": len(composed_tasks),
        "composedNoteCount": len(clustered),
    }


def _restates_task(note: ArtifactClaim, tasks: list[ArtifactClaim], profile: TokenProfile) -> bool:
    """A note built only from one task's candidates that mostly repeats that task adds no memory."""
    sources = set(note.sourceCandidateIds)
    if not sources:
        return False
    text = f"{note.title} {note.body}"
    for task in tasks:
        if sources <= set(task.sourceCandidateIds) and profile.jaccard(text, f"{task.title} {task.body}") >= _RESTATE:
            return True
    return False


def _enrich_task(
    task: ArtifactClaim,
    ledger: CandidateLedger,
    meeting_sequences: set[int],
    profile: TokenProfile,
) -> tuple[ArtifactClaim, bool]:
    if not _is_shallow(task.title, task.body):
        return task, False
    related = _related_context(task, ledger, profile)
    if not related:
        if task.body:
            body = _as_sentence(task.body)
            if body != task.body:
                return task.model_copy(update={"body": body}), True
        return task, False
    extras: list[str] = []
    source_ids = list(task.sourceCandidateIds)
    evidence = list(task.evidenceSequences)
    seen_ids = set(source_ids)
    seen_seq = set(evidence)
    for candidate in related:
        meaning = normalize_text(candidate.meaning)
        if not meaning or _already_covered(meaning, f"{task.title} {task.body} {' '.join(extras)}"):
            continue
        extras.append(meaning)
        if candidate.candidateId not in seen_ids:
            seen_ids.add(candidate.candidateId)
            source_ids.append(candidate.candidateId)
        for sequence in candidate.evidenceSequences:
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
    ledger: CandidateLedger,
    profile: TokenProfile,
) -> list[MeetingCandidate]:
    """Context for a shallow task: its own cited context first, then same-topic memory."""
    index = ledger.by_id()
    seed = f"{task.title} {task.body}"
    found: list[MeetingCandidate] = []
    seen: set[str] = set()

    def take(candidate: MeetingCandidate) -> None:
        if candidate.candidateId in seen or not normalize_text(candidate.meaning):
            return
        seen.add(candidate.candidateId)
        found.append(candidate)

    for candidate_id in task.sourceCandidateIds:
        candidate = index.get(str(candidate_id))
        if candidate is not None and _is_context_kind(candidate):
            take(candidate)

    for candidate in ledger.candidates:
        if candidate.candidateId in seen or not _is_context_kind(candidate):
            continue
        if _related(seed, candidate.meaning, task.topic, candidate.topic, profile, _RELATED):
            take(candidate)
    return found


def _cluster_notes(notes: list[ArtifactClaim], profile: TokenProfile) -> tuple[list[ArtifactClaim], int]:
    if len(notes) <= 1:
        return list(notes), 0

    def related(left: ArtifactClaim, right: ArtifactClaim) -> bool:
        return _related(
            f"{left.title} {left.body}",
            f"{right.title} {right.body}",
            left.topic,
            right.topic,
            profile,
            _CLUSTER,
        )

    groups = _union_groups(notes, related)
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


def _related(
    left: str,
    right: str,
    left_topic: str | None,
    right_topic: str | None,
    profile: TokenProfile,
    threshold: float,
) -> bool:
    if topic_key(left_topic) and topic_key(right_topic):
        return same_topic(left_topic, right_topic)
    shared = profile.distinctive(left) & profile.distinctive(right)
    if len(shared) < _MIN_SHARED:
        return False
    return profile.jaccard(left, right) >= threshold


def _merge_note_group(group: list[ArtifactClaim]) -> ArtifactClaim:
    ordered = sorted(group, key=lambda item: min(item.evidenceSequences or [10**9]))
    first = _note_lines(ordered[0])
    lead = first[0] if first else normalize_text(ordered[0].title)
    lines: list[str] = []
    for line in first[1:]:
        if not _restated(line, [lead, *lines]):
            lines.append(line)
    source_ids: list[str] = []
    evidence: list[int] = []
    seen_ids: set[str] = set()
    seen_seq: set[int] = set()
    for position, item in enumerate(ordered):
        if position:
            for line in _note_lines(item):
                if not _restated(line, [lead, *lines]):
                    lines.append(line)
        for candidate_id in item.sourceCandidateIds:
            if candidate_id not in seen_ids:
                seen_ids.add(candidate_id)
                source_ids.append(candidate_id)
        for sequence in item.evidenceSequences:
            if sequence not in seen_seq:
                seen_seq.add(sequence)
                evidence.append(sequence)
    topic = next((item.topic for item in ordered if topic_key(item.topic)), None)
    title = _cluster_title(ordered, topic)
    body = "\n".join([lead, *(f"- {line}" for line in lines)]) if lines else lead
    return ArtifactClaim(
        artifactKey=ordered[0].artifactKey,
        kind="note",
        title=title,
        body=body,
        topic=topic,
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _note_lines(item: ArtifactClaim) -> list[str]:
    body = clean_multiline(item.body)
    if not body:
        return [normalize_text(item.title)] if normalize_text(item.title) else []
    out: list[str] = []
    for line in body.splitlines():
        text = line[2:].strip() if line.startswith("- ") else line.strip()
        text = _KIND_LABEL_RE.sub("", text).strip()
        if text:
            out.append(text[0].upper() + text[1:])
    return out


def _restated(line: str, previous: list[str]) -> bool:
    """A line whose content tokens are mostly inside one earlier line repeats it."""
    if _already_covered(line, " ".join(previous)):
        return True
    tokens = _folded_tokens(line)
    if len(tokens) < 2:
        return False
    numbers = {token for token in tokens if token.isdigit()}
    for prev in previous:
        prior = _folded_tokens(prev)
        if numbers - prior:
            continue
        if len(tokens & prior) / len(tokens) >= _CONTAINED:
            return True
    return False


def _folded_tokens(text: str) -> set[str]:
    return {_stem(token) for token in content_tokens(text)}


def _stem(token: str) -> str:
    value = token.casefold()
    if value.isdigit() or len(value) <= 4:
        return value
    return value.rstrip("s")[:6]


def _cluster_title(group: list[ArtifactClaim], topic: str | None) -> str:
    if topic and len(topic) <= 80:
        return topic
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
