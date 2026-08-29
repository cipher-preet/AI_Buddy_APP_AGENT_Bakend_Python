"""Shadow-mode comparison: new pipeline runs, legacy still publishes."""

from __future__ import annotations

from typing import Any, Iterable

from services.conversation.event_pipeline.channels import is_generic_task_text
from services.conversation.event_pipeline.textutil import evidence_sequence_ids, token_jaccard
from services.conversation.event_pipeline.validation import mixed_thread_rate


def compare_pipeline_outputs(
    *,
    legacy_tasks: Iterable,
    legacy_notes: Iterable,
    new_tasks: Iterable,
    new_notes: Iterable,
    new_events: Iterable | None = None,
) -> dict[str, Any]:
    legacy_task_list = list(legacy_tasks or [])
    legacy_note_list = list(legacy_notes or [])
    new_task_list = list(new_tasks or [])
    new_note_list = list(new_notes or [])
    events = list(new_events or [])
    missing_tasks = _unmatched(legacy_task_list, new_task_list)
    extra_tasks = _unmatched(new_task_list, legacy_task_list)
    missing_notes = _unmatched(legacy_note_list, new_note_list)
    extra_notes = _unmatched(new_note_list, legacy_note_list)
    extra_invalid_tasks = [item for item in extra_tasks if is_generic_task_text(getattr(item, "title", ""), getattr(item, "body", ""))]
    extra_invalid_notes = [item for item in extra_notes if not (getattr(item, "evidence", None) or [])]
    generic = sum(1 for task in new_task_list if is_generic_task_text(task.title, task.body))
    mixed = mixed_thread_rate([*new_task_list, *new_note_list], events) if events else 0.0
    evidence_ok = 0
    evidence_n = 0
    for item in [*new_task_list, *new_note_list]:
        spans = getattr(item, "evidence", None) or []
        evidence_n += 1
        if spans:
            evidence_ok += 1
    return {
        "legacyTaskCount": len(legacy_task_list),
        "newTaskCount": len(new_task_list),
        "legacyNoteCount": len(legacy_note_list),
        "newNoteCount": len(new_note_list),
        "missingValidTasks": len(missing_tasks),
        "extraInvalidTasks": len(extra_invalid_tasks),
        "missingValidNotes": len(missing_notes),
        "extraInvalidNotes": len(extra_invalid_notes),
        "genericTaskRate": generic / max(len(new_task_list), 1) if new_task_list else 0.0,
        "mixedThreadRate": mixed,
        "evidenceQuality": evidence_ok / evidence_n if evidence_n else 1.0,
        "missingTaskTitles": [getattr(item, "title", "") for item in missing_tasks[:12]],
        "extraTaskTitles": [getattr(item, "title", "") for item in extra_tasks[:12]],
        "missingNoteTitles": [getattr(item, "title", "") for item in missing_notes[:12]],
        "extraNoteTitles": [getattr(item, "title", "") for item in extra_notes[:12]],
        "publishedFrom": "legacy",
    }


def _unmatched(left: list, right: list) -> list:
    remaining = list(right)
    missing = []
    for item in left:
        index = _best_index(item, remaining)
        if index is None:
            missing.append(item)
            continue
        remaining.pop(index)
    return missing


def _best_index(item, candidates: list) -> int | None:
    meaning = f"{getattr(item, 'title', '')} {getattr(item, 'body', '')}"
    sequences = set(evidence_sequence_ids(getattr(item, "evidence", [])))
    best_i = None
    best = 0.0
    for index, other in enumerate(candidates):
        other_meaning = f"{getattr(other, 'title', '')} {getattr(other, 'body', '')}"
        score = token_jaccard(meaning, other_meaning)
        other_seqs = set(evidence_sequence_ids(getattr(other, "evidence", [])))
        if sequences and other_seqs and sequences & other_seqs:
            score += 0.3
        if score > best:
            best = score
            best_i = index
    return best_i if best >= 0.28 else None
