"""Meeting-wide outline pass over composed artifacts.

Partitioned consolidation sees one slice of the meeting at a time, so it tends to
publish one note per sub-point and to rate priority locally. This pass sees every
draft artifact at once (titles, short bodies, and the transcript lines they cite) and returns structural
decisions: which notes form one topic section, which tasks are the same unit of
work, which artifacts are side conversation, and meeting-relative priorities.

The LLM never rewrites content here. Merges are applied deterministically from
existing bodies, sources, and evidence; the evidence verifier runs afterwards.
Any failure leaves the artifacts unchanged.
"""

from __future__ import annotations

import math
import re

from services.conversation.event_pipeline.textutil import content_tokens, normalize_text
from services.conversation.meeting_pipeline.composer import _already_covered, _as_sentence, _merge_note_group
from services.conversation.meeting_pipeline.flags import output_language
from services.conversation.meeting_pipeline.harness import (
    TokenProfile,
    clean_inline,
    normalize_priority,
    normalize_topic,
    strip_internal_references,
)
from services.conversation.meeting_pipeline.llm import generate_structured
from services.conversation.meeting_pipeline.observability import log_pipeline
from services.conversation.meeting_pipeline.schemas import (
    ArtifactClaim,
    MeetingLogisticsResponse,
    MeetingOutlineResponse,
)
from services.llm.async_runtime import reraise_if_hard_runtime
from services.llm.router import LLMCapability, LLMRouter

_MIN_ARTIFACTS = 3
_BODY_PREVIEW = 320
_EVIDENCE_LINES = 3
_EVIDENCE_CHARS = 240
_MAX_OFF_TOPIC_SHARE = 0.25
# Merging more tasks than this produces an umbrella task that is no longer assignable.
_MAX_TASK_MERGE = 3
# Near-duplicate tasks share distinctive wording or cited candidates. Generic words
# (the meeting's own main nouns) are ignored by TokenProfile, so no topic list is used.
_DUP_OVERLAP = 0.55
_DUP_SHARED = 4
_DUP_REPEAT = 0.62
_DUP_SOURCE = 0.5
# Priority is a rank, not a label the model can apply to everything.
_MIN_BANDS = 4
_HIGH_SHARE = 0.25
_NUMBER_RE = re.compile(r"\d+")
_PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


class MeetingOutlineOrganizer:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.last_provider = "none"
        self.last_model = "none"

    async def organize(
        self,
        artifacts: list[ArtifactClaim],
        sequence_text: dict[int, str] | None = None,
    ) -> tuple[list[ArtifactClaim], dict[str, int]]:
        stats = _empty_stats()
        if len(artifacts) < _MIN_ARTIFACTS:
            return artifacts, stats
        notes = [item for item in artifacts if item.kind == "note"]
        tasks = [item for item in artifacts if item.kind == "task"]
        note_ids = {f"N{index}": item for index, item in enumerate(notes, start=1)}
        task_ids = {f"T{index}": item for index, item in enumerate(tasks, start=1)}
        payload = {
            "outputLanguage": output_language(),
            "notes": [_preview(key, item, sequence_text) for key, item in note_ids.items()],
            "tasks": [_preview(key, item, sequence_text) for key, item in task_ids.items()],
        }
        self.calls += 1
        try:
            response, provider, model = await generate_structured(
                self.router,
                LLMCapability.FINAL_SYNTHESIS,
                "meeting-outline-organizer-v1",
                MeetingOutlineResponse,
                payload,
                stage="outline_organizer",
            )
        except Exception as error:
            reraise_if_hard_runtime(error)
            stats["failed"] = 1
            log_pipeline({"event": "outline_organizer_failed", "error": str(error)[:400]})
            return artifacts, stats
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        organized, stats = apply_outline(response, note_ids, task_ids, stats)
        return await self._drop_logistics(organized, sequence_text, stats)

    async def _drop_logistics(
        self,
        artifacts: list[ArtifactClaim],
        sequence_text: dict[int, str] | None,
        stats: dict[str, int],
    ) -> tuple[list[ArtifactClaim], dict[str, int]]:
        """Second look at non-core tasks. High-ranked work is not eligible for removal."""
        review = [item for item in artifacts if item.kind == "task" and item.priority != "High"]
        if len(review) < 2 or not sequence_text:
            return artifacts, stats
        notes = [item for item in artifacts if item.kind == "note"]
        ids = {f"L{index}": item for index, item in enumerate(review, start=1)}
        payload = {
            "outputLanguage": output_language(),
            "meetingSubject": [item.title for item in notes if item.title][:24],
            "candidates": [_preview(key, item, sequence_text) for key, item in ids.items()],
        }
        self.calls += 1
        try:
            response, provider, model = await generate_structured(
                self.router,
                LLMCapability.FINAL_SYNTHESIS,
                "meeting-logistics-review-v1",
                MeetingLogisticsResponse,
                payload,
                stage="logistics_review",
            )
        except Exception as error:
            reraise_if_hard_runtime(error)
            stats["logisticsFailed"] = 1
            log_pipeline({"event": "logistics_review_failed", "error": str(error)[:400]})
            return artifacts, stats
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        limit = max(1, int(len(review) * _MAX_OFF_TOPIC_SHARE))
        drop = []
        for value in response.offTopicIds:
            key = _ref(value)
            if key in ids and key not in drop:
                drop.append(key)
        if len(drop) > limit:
            log_pipeline({"event": "logistics_review_capped", "dropped": len(drop), "limit": limit})
            drop = drop[:limit]
        stats["logisticsDropped"] = len(drop)
        removed = {id(ids[key]) for key in drop}
        return [item for item in artifacts if id(item) not in removed], stats


def apply_outline(
    response: MeetingOutlineResponse,
    note_ids: dict[str, ArtifactClaim],
    task_ids: dict[str, ArtifactClaim],
    stats: dict[str, int] | None = None,
) -> tuple[list[ArtifactClaim], dict[str, int]]:
    stats = {**_empty_stats(), **(stats or {})}
    total = len(note_ids) + len(task_ids)
    off_topic = {key for key in (_ref(value) for value in response.offTopicIds) if key in note_ids or key in task_ids}
    if len(off_topic) > max(1, int(total * _MAX_OFF_TOPIC_SHARE)):
        off_topic = set()
    stats["offTopicDropped"] = len(off_topic)

    priorities = {_ref(item.taskId): normalize_priority(item.priority) for item in response.priorities}
    tasks: dict[str, ArtifactClaim] = {}
    for key, task in task_ids.items():
        if key in off_topic:
            continue
        priority = priorities.get(key)
        if priority:
            stats["prioritiesSet"] += 1
            task = task.model_copy(update={"priority": priority})
        tasks[key] = task

    groups: list[tuple[ArtifactClaim, list[str]]] = []
    used: set[str] = set()
    for merge in response.taskMerges:
        keys = _unique(merge.taskIds, tasks, used)
        if len(keys) < 2 or len(keys) > _MAX_TASK_MERGE:
            stats["umbrellaMergesRejected"] += int(len(keys) > _MAX_TASK_MERGE)
            continue
        used.update(keys)
        merged = _merge_tasks([tasks[key] for key in keys], merge.title)
        explicit = next((priorities[key] for key in keys if priorities.get(key)), None)
        groups.append((merged.model_copy(update={"priority": explicit}) if explicit else merged, keys))
        stats["tasksMerged"] += len(keys) - 1
    groups.extend((task, [key]) for key, task in tasks.items() if key not in used)
    groups = _collapse_near_duplicates(groups, stats)
    groups = _apply_priority_bands(groups, response.rankedTaskIds, priorities, stats)
    out_tasks = [task for task, _keys in groups]

    notes = {key: note for key, note in note_ids.items() if key not in off_topic}
    out_notes: list[ArtifactClaim] = []
    used = set()
    for section in response.sections:
        keys = _unique(section.noteIds, notes, used)
        if not keys:
            continue
        used.update(keys)
        title = clean_inline(strip_internal_references(section.title))
        group = [notes[key] for key in keys]
        if title:
            group = [note.model_copy(update={"topic": normalize_topic(title) or note.topic}) for note in group]
        if len(group) == 1:
            note = group[0]
            out_notes.append(note.model_copy(update={"title": title}) if title and len(title) <= 80 else note)
            continue
        out_notes.append(_merge_note_group(group))
        stats["sectionsMerged"] += len(group) - 1
    out_notes.extend(note for key, note in notes.items() if key not in used)
    return [*out_tasks, *out_notes], stats


def _empty_stats() -> dict[str, int]:
    return {
        "sectionsMerged": 0,
        "tasksMerged": 0,
        "nearDuplicatesMerged": 0,
        "umbrellaMergesRejected": 0,
        "offTopicDropped": 0,
        "prioritiesSet": 0,
        "prioritiesBanded": 0,
        "logisticsDropped": 0,
        "logisticsFailed": 0,
        "failed": 0,
    }


def _preview(key: str, item: ArtifactClaim, sequence_text: dict[int, str] | None = None) -> dict:
    body = normalize_text(item.body)
    row = {"id": key, "title": item.title, "topic": item.topic, "body": body[:_BODY_PREVIEW]}
    if item.kind == "task":
        row["priority"] = item.priority
    evidence = _evidence_lines(item, sequence_text)
    if evidence:
        row["evidence"] = evidence
    return row


def _evidence_lines(item: ArtifactClaim, sequence_text: dict[int, str] | None) -> list[str]:
    """Cited transcript lines, so off-topic judgment uses what was said, not the title."""
    if not sequence_text:
        return []
    lines: list[str] = []
    if item.evidenceSequences:
        preceding = normalize_text(sequence_text.get(min(item.evidenceSequences) - 1) or "")
        if preceding:
            lines.append(preceding[:_EVIDENCE_CHARS])
    for sequence in item.evidenceSequences:
        text = normalize_text(sequence_text.get(sequence) or "")
        if not text:
            continue
        lines.append(text[:_EVIDENCE_CHARS])
        if len(lines) >= _EVIDENCE_LINES + 1:
            break
    return lines


def _apply_priority_bands(
    groups: list[tuple[ArtifactClaim, list[str]]],
    ranked_ids: list[str],
    categorical: dict[str, str | None],
    stats: dict[str, int],
) -> list[tuple[ArtifactClaim, list[str]]]:
    """Keep High for the top slice of the meeting's own ranking. Explicit Low stays Low."""
    if len(groups) < _MIN_BANDS:
        return groups
    high_cap = max(1, math.ceil(len(groups) * _HIGH_SHARE))
    rank = {_ref(value): index for index, value in enumerate(ranked_ids)}

    def importance(keys: list[str], claim: ArtifactClaim) -> tuple[int, int]:
        positions = [rank[key] for key in keys if key in rank]
        if positions:
            return (0, min(positions))
        label = categorical.get(keys[0]) or claim.priority or ""
        return (1, _PRIORITY_ORDER.get(label, len(_PRIORITY_ORDER)))

    updated = list(groups)
    high_given = 0
    changed = 0
    for index in sorted(range(len(groups)), key=lambda item: importance(groups[item][1], groups[item][0])):
        claim, keys = groups[index]
        labels = [categorical[key] for key in keys if categorical.get(key)]
        priority = "Low" if labels and all(label == "Low" for label in labels) else None
        if priority is None:
            if high_given < high_cap:
                priority = "High"
                high_given += 1
            else:
                priority = "Medium"
        if priority != claim.priority:
            changed += 1
            claim = claim.model_copy(update={"priority": priority})
        updated[index] = (claim, keys)
    stats["prioritiesBanded"] = changed
    return updated


def _collapse_near_duplicates(
    groups: list[tuple[ArtifactClaim, list[str]]],
    stats: dict[str, int],
) -> list[tuple[ArtifactClaim, list[str]]]:
    """Fold tasks that describe the same deliverable. Groups larger than the cap stay split."""
    if len(groups) < 2:
        return groups
    claims = [claim for claim, _keys in groups]
    profile = TokenProfile([f"{item.title} {item.body}" for item in claims])
    pairs: list[tuple[float, int, int]] = []
    for left in range(len(claims)):
        for right in range(left + 1, len(claims)):
            score = _duplicate_score(claims[left], claims[right], profile)
            if score is not None:
                pairs.append((score, left, right))
    pairs.sort(key=lambda item: item[0], reverse=True)
    parent = list(range(len(claims)))
    members: dict[int, list[int]] = {index: [index] for index in range(len(claims))}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for _score, left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        if len(members[root_left]) + len(members[root_right]) > _MAX_TASK_MERGE:
            stats["umbrellaMergesRejected"] += 1
            continue
        parent[root_right] = root_left
        members[root_left].extend(members.pop(root_right))

    collapsed: list[tuple[ArtifactClaim, list[str]]] = []
    for indexes in members.values():
        if len(indexes) == 1:
            collapsed.append(groups[indexes[0]])
            continue
        ordered = sorted(indexes, key=lambda index: min(groups[index][0].evidenceSequences or [10**9]))
        grouped = [groups[index] for index in ordered]
        title = _specific_title([item for item, _keys in grouped], profile)
        keys = [key for _item, group_keys in grouped for key in group_keys]
        collapsed.append((_merge_tasks([item for item, _keys in grouped], title), keys))
        stats["nearDuplicatesMerged"] += len(grouped) - 1
    collapsed.sort(key=lambda item: min(item[0].evidenceSequences or [10**9]))
    return collapsed


def _duplicate_score(left: ArtifactClaim, right: ArtifactClaim, profile: TokenProfile) -> float | None:
    left_sources = {value for value in left.sourceCandidateIds if value}
    right_sources = {value for value in right.sourceCandidateIds if value}
    if left_sources and right_sources:
        overlap = len(left_sources & right_sources) / min(len(left_sources), len(right_sources))
        if overlap >= _DUP_SOURCE:
            return 2 + overlap
    if _conflicting_numbers(left.body, right.body):
        return None
    generic = {_fold_token(token) for token in profile.generic}
    left_text = f"{left.title} {left.body}"
    right_text = f"{right.title} {right.body}"
    left_tokens = _folded_tokens(left_text, generic)
    right_tokens = _folded_tokens(right_text, generic)
    smaller = min(len(left_tokens), len(right_tokens))
    shared = left_tokens & right_tokens
    if smaller and len(shared) >= _DUP_SHARED and len(shared) / smaller >= _DUP_OVERLAP:
        return len(shared) / smaller
    # Wording repeated across the meeting is marked generic, which hides duplicates.
    if smaller < _DUP_SHARED:
        full_left = _folded_tokens(left_text, set())
        full_right = _folded_tokens(right_text, set())
        union = full_left | full_right
        intersection = full_left & full_right
        if union and len(intersection) >= _DUP_SHARED and len(intersection) / len(union) >= _DUP_REPEAT:
            return len(intersection) / len(union)
    return None


def _folded_tokens(text: str, generic: set[str]) -> set[str]:
    return {
        folded
        for token in content_tokens(text)
        if not token.isdigit() and (folded := _fold_token(token)) not in generic and len(folded) > 1
    }


def _fold_token(token: str) -> str:
    value = token.casefold()
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _conflicting_numbers(left: str, right: str) -> bool:
    left_numbers = set(_NUMBER_RE.findall(left or ""))
    right_numbers = set(_NUMBER_RE.findall(right or ""))
    return bool(left_numbers and right_numbers and not left_numbers & right_numbers)


def _specific_title(claims: list[ArtifactClaim], profile: TokenProfile) -> str:
    ranked = sorted(
        claims,
        key=lambda item: (len(profile.distinctive(item.title)), len(normalize_text(item.title))),
        reverse=True,
    )
    return ranked[0].title


def _ref(value: object) -> str:
    return str(value or "").strip().upper()


def _unique(values: list[str], pool: dict[str, ArtifactClaim], used: set[str]) -> list[str]:
    keys: list[str] = []
    for value in values:
        key = _ref(value)
        if key in pool and key not in used and key not in keys:
            keys.append(key)
    return keys


def _merge_tasks(group: list[ArtifactClaim], title: str) -> ArtifactClaim:
    ordered = sorted(group, key=lambda item: min(item.evidenceSequences or [10**9]))
    lead = ordered[0]
    parts: list[str] = []
    criteria: list[str] = []
    source_ids: list[str] = []
    evidence: list[int] = []
    for item in ordered:
        text = normalize_text(item.body) or normalize_text(item.title)
        if text and not any(_already_covered(text, prev) for prev in parts):
            parts.append(_as_sentence(text))
        accept = normalize_text(item.acceptanceCriteria or "")
        if accept and not any(_already_covered(accept, prev) for prev in criteria):
            criteria.append(_as_sentence(accept))
        source_ids.extend(value for value in item.sourceCandidateIds if value not in source_ids)
        evidence.extend(value for value in item.evidenceSequences if value not in evidence)
    owners = {item.owner for item in ordered if item.owner}
    dues = {item.dueDate for item in ordered if item.dueDate}
    ranked = [item.priority for item in ordered if item.priority in _PRIORITY_ORDER]
    clean_title = clean_inline(strip_internal_references(title))
    return lead.model_copy(
        update={
            "title": clean_title if clean_title and len(clean_title) <= 120 else lead.title,
            "body": " ".join(parts),
            "acceptanceCriteria": " ".join(criteria),
            "sourceCandidateIds": source_ids,
            "evidenceSequences": evidence,
            "owner": owners.pop() if len(owners) == 1 else None,
            "dueDate": dues.pop() if len(dues) == 1 else None,
            "priority": min(ranked, key=_PRIORITY_ORDER.__getitem__) if ranked else lead.priority,
        }
    )
