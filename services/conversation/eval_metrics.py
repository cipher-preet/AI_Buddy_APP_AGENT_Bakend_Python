from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


Kind = Literal["task", "note"]


@dataclass(frozen=True)
class GoldItem:
    id: str
    kind: Kind
    meaning: str
    evidenceSequences: list[int]
    ownerText: str | None = None
    dueDateText: str | None = None
    state: str | None = None


@dataclass(frozen=True)
class PredictedItem:
    kind: Kind
    meaning: str
    evidenceSequences: list[int]
    ownerText: str | None = None
    dueDateText: str | None = None
    state: str | None = None
    artifactId: str | None = None


@dataclass
class Alignment:
    gold_id: str
    predicted_index: int | None
    overlap: int = 0


@dataclass
class CaseScore:
    caseId: str
    category: str
    taskRecall: float
    taskPrecision: float
    noteRecall: float
    notePrecision: float
    duplicateRate: float
    falseTaskRate: float
    ownerAccuracy: float | None
    deadlineAccuracy: float | None
    evidenceAccuracy: float | None
    crossWindowUpdateAccuracy: float | None
    goldTaskCount: int
    predictedTaskCount: int
    goldNoteCount: int
    predictedNoteCount: int
    matchedTasks: int
    matchedNotes: int
    duplicateCount: int
    falseTaskCount: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusScore:
    cases: list[CaseScore]
    taskRecall: float
    taskPrecision: float
    noteRecall: float
    notePrecision: float
    duplicateRate: float
    falseTaskRate: float
    ownerAccuracy: float | None
    deadlineAccuracy: float | None
    evidenceAccuracy: float | None
    crossWindowUpdateAccuracy: float | None


def predicted_from_extraction(result) -> list[PredictedItem]:
    items: list[PredictedItem] = []
    for task in getattr(result, "tasks", []) or []:
        items.append(
            PredictedItem(
                kind="task",
                meaning=f"{task.title}\n{task.body}".strip(),
                evidenceSequences=_sequences_from_spans(task.evidence),
                ownerText=task.ownerText,
                dueDateText=task.dueDateText or task.dueDateResolved,
                state=_state_from_operation(getattr(task, "operation", None)),
                artifactId=getattr(task, "artifactId", None),
            )
        )
    for note in getattr(result, "notes", []) or []:
        items.append(
            PredictedItem(
                kind="note",
                meaning=f"{note.title}\n{note.body}".strip(),
                evidenceSequences=_sequences_from_spans(note.evidence),
                artifactId=getattr(note, "artifactId", None),
            )
        )
    return items


def predicted_from_artifacts(artifacts: Iterable[Any]) -> list[PredictedItem]:
    items: list[PredictedItem] = []
    for artifact in artifacts:
        kind: Kind = "task"
        artifact_type = getattr(getattr(artifact, "artifactType", None), "value", "") or ""
        if artifact_type in {"note", "fact", "preference", "idea", "reference", "answer"}:
            kind = "note"
        items.append(
            PredictedItem(
                kind=kind,
                meaning=f"{artifact.title}\n{artifact.content}".strip(),
                evidenceSequences=_sequences_from_spans(getattr(artifact, "evidence", [])),
                ownerText=getattr(artifact, "ownerText", None),
                dueDateText=getattr(artifact, "dueDateText", None) or getattr(artifact, "dueDateResolved", None),
                state=getattr(getattr(artifact, "status", None), "value", None),
                artifactId=str(artifact.id),
            )
        )
    return items


def score_case(
    case: dict[str, Any],
    predicted: list[PredictedItem],
    *,
    expected_artifact_ids: dict[str, str] | None = None,
) -> CaseScore:
    gold_tasks = [GoldItem(**item) if not isinstance(item, GoldItem) else item for item in case.get("goldTasks", [])]
    gold_notes = [GoldItem(**item) if not isinstance(item, GoldItem) else item for item in case.get("goldNotes", [])]
    non_task = set(case.get("nonTaskSequences", []))
    predicted_tasks = [item for item in predicted if item.kind == "task"]
    predicted_notes = [item for item in predicted if item.kind == "note"]

    task_alignments, task_duplicates = _align(gold_tasks, predicted_tasks)
    note_alignments, note_duplicates = _align(gold_notes, predicted_notes)
    matched_tasks = [item for item in task_alignments if item.predicted_index is not None]
    matched_notes = [item for item in note_alignments if item.predicted_index is not None]

    false_tasks = 0
    matched_pred_indexes = {item.predicted_index for item in matched_tasks}
    for index, item in enumerate(predicted_tasks):
        sequences = set(item.evidenceSequences)
        if index in matched_pred_indexes:
            continue
        if sequences & non_task or not any(sequences & set(gold.evidenceSequences) for gold in gold_tasks):
            false_tasks += 1

    owner_pairs = []
    deadline_pairs = []
    evidence_pairs = []
    for alignment in matched_tasks:
        gold = gold_tasks[next(i for i, item in enumerate(gold_tasks) if item.id == alignment.gold_id)]
        pred = predicted_tasks[alignment.predicted_index]  # type: ignore[index]
        if gold.ownerText:
            owner_pairs.append(_names_match(gold.ownerText, pred.ownerText))
        if gold.dueDateText:
            deadline_pairs.append(_dates_match(gold.dueDateText, pred.dueDateText))
        gold_seqs = set(gold.evidenceSequences)
        pred_seqs = set(pred.evidenceSequences)
        transcript_seqs = set(_sequences_in_transcript(case.get("transcript", ""))) or gold_seqs
        evidence_pairs.append(bool(pred_seqs) and bool(pred_seqs & gold_seqs) and pred_seqs.issubset(transcript_seqs))

    update_pairs = []
    for expected in case.get("expectedUpdates", []):
        gold_id = expected["goldId"]
        wanted_state = expected["state"]
        alignment = next((item for item in matched_tasks if item.gold_id == gold_id), None)
        if alignment is None or alignment.predicted_index is None:
            update_pairs.append(False)
            continue
        pred = predicted_tasks[alignment.predicted_index]
        update_pairs.append(_states_match(wanted_state, pred.state))
        if expected_artifact_ids and gold_id in expected_artifact_ids:
            update_pairs[-1] = update_pairs[-1] and pred.artifactId == expected_artifact_ids[gold_id]

    duplicate_count = task_duplicates + note_duplicates
    predicted_total = max(len(predicted_tasks) + len(predicted_notes), 1)
    return CaseScore(
        caseId=str(case["id"]),
        category=str(case.get("category", "")),
        taskRecall=_ratio(len(matched_tasks), len(gold_tasks)),
        taskPrecision=_ratio(len(matched_tasks), len(predicted_tasks)),
        noteRecall=_ratio(len(matched_notes), len(gold_notes)),
        notePrecision=_ratio(len(matched_notes), len(predicted_notes)),
        duplicateRate=_ratio(duplicate_count, predicted_total),
        falseTaskRate=_ratio(false_tasks, len(predicted_tasks)) if predicted_tasks else 0.0,
        ownerAccuracy=_ratio(sum(owner_pairs), len(owner_pairs)) if owner_pairs else None,
        deadlineAccuracy=_ratio(sum(deadline_pairs), len(deadline_pairs)) if deadline_pairs else None,
        evidenceAccuracy=_ratio(sum(evidence_pairs), len(evidence_pairs)) if evidence_pairs else None,
        crossWindowUpdateAccuracy=_ratio(sum(update_pairs), len(update_pairs)) if update_pairs else None,
        goldTaskCount=len(gold_tasks),
        predictedTaskCount=len(predicted_tasks),
        goldNoteCount=len(gold_notes),
        predictedNoteCount=len(predicted_notes),
        matchedTasks=len(matched_tasks),
        matchedNotes=len(matched_notes),
        duplicateCount=duplicate_count,
        falseTaskCount=false_tasks,
        details={
            "taskAlignments": [alignment.__dict__ for alignment in task_alignments],
            "noteAlignments": [alignment.__dict__ for alignment in note_alignments],
        },
    )


def score_corpus(scores: list[CaseScore]) -> CorpusScore:
    def _mean(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return sum(present) / len(present)

    return CorpusScore(
        cases=scores,
        taskRecall=_mean([item.taskRecall for item in scores]) or 0.0,
        taskPrecision=_mean([item.taskPrecision for item in scores]) or 0.0,
        noteRecall=_mean([item.noteRecall for item in scores]) or 0.0,
        notePrecision=_mean([item.notePrecision for item in scores]) or 0.0,
        duplicateRate=_mean([item.duplicateRate for item in scores]) or 0.0,
        falseTaskRate=_mean([item.falseTaskRate for item in scores]) or 0.0,
        ownerAccuracy=_mean([item.ownerAccuracy for item in scores]),
        deadlineAccuracy=_mean([item.deadlineAccuracy for item in scores]),
        evidenceAccuracy=_mean([item.evidenceAccuracy for item in scores]),
        crossWindowUpdateAccuracy=_mean([item.crossWindowUpdateAccuracy for item in scores]),
    )


def _align(gold_items: list[GoldItem], predicted: list[PredictedItem]) -> tuple[list[Alignment], int]:
    remaining = set(range(len(predicted)))
    alignments: list[Alignment] = []
    for gold in gold_items:
        gold_seqs = set(gold.evidenceSequences)
        best_index = None
        best_overlap = 0
        for index in list(remaining):
            overlap = len(gold_seqs & set(predicted[index].evidenceSequences))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is None or best_overlap <= 0:
            alignments.append(Alignment(gold_id=gold.id, predicted_index=None, overlap=0))
            continue
        remaining.discard(best_index)
        alignments.append(Alignment(gold_id=gold.id, predicted_index=best_index, overlap=best_overlap))
    duplicate_count = 0
    gold_seq_sets = [set(item.evidenceSequences) for item in gold_items]
    for index in remaining:
        pred_seqs = set(predicted[index].evidenceSequences)
        if any(pred_seqs & gold_seqs for gold_seqs in gold_seq_sets):
            duplicate_count += 1
    return alignments, duplicate_count


def _sequences_from_spans(spans: Iterable[Any]) -> list[int]:
    values: set[int] = set()
    for span in spans or []:
        start = int(getattr(span, "sequenceStart", span.get("sequenceStart") if isinstance(span, dict) else 0))
        end = int(getattr(span, "sequenceEnd", span.get("sequenceEnd") if isinstance(span, dict) else start))
        for sequence in range(min(start, end), max(start, end) + 1):
            values.add(sequence)
    return sorted(values)


def _state_from_operation(operation: str | None) -> str | None:
    mapping = {"COMPLETE": "completed", "CANCEL": "cancelled", "UPDATE": "active", "CREATE": "proposed"}
    return mapping.get(str(operation or "").upper())


def _names_match(gold: str, predicted: str | None) -> bool:
    if not predicted:
        return False
    return _fold(gold) == _fold(predicted) or _fold(gold) in _fold(predicted) or _fold(predicted) in _fold(gold)


def _dates_match(gold: str, predicted: str | None) -> bool:
    if not predicted:
        return False
    return _fold(gold) == _fold(predicted) or _fold(gold) in _fold(predicted) or _fold(predicted) in _fold(gold)


def _states_match(gold: str, predicted: str | None) -> bool:
    return _fold(gold) == _fold(predicted or "")


def _fold(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _sequences_in_transcript(transcript: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[(\d+)\]", transcript or "")]


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0 if numerator <= 0 else 0.0
    return numerator / denominator
