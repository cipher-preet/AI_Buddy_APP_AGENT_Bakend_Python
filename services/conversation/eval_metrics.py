from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from services.conversation.meeting_pipeline.dates import normalize_supported_due_date


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
    reviewStatus: str | None = "REQUIRED"


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
    unsupportedArtifactRate: float = 0.0
    candidateRecall: float | None = None
    backgroundFalsePositiveRate: float | None = None
    meaningRetentionRecall: float | None = None
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
    unsupportedArtifactRate: float = 0.0
    candidateRecall: float | None = None
    backgroundFalsePositiveRate: float | None = None
    meaningRetentionRecall: float | None = None


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
    predicted_candidates: list[PredictedItem] | None = None,
    meaning_embedding_scores: dict[str, float] | None = None,
) -> CaseScore:
    gold_tasks = [_as_gold_item(item) for item in case.get("goldTasks", [])]
    gold_notes = [_as_gold_item(item) for item in case.get("goldNotes", [])]
    non_task = set(case.get("nonTaskSequences", []))
    predicted_tasks = [item for item in predicted if item.kind == "task"]
    predicted_notes = [item for item in predicted if item.kind == "note"]

    task_alignments, task_duplicates = _align(gold_tasks, predicted_tasks)
    note_alignments, note_duplicates = _align(gold_notes, predicted_notes)
    matched_tasks = [item for item in task_alignments if item.predicted_index is not None]
    matched_notes = [item for item in note_alignments if item.predicted_index is not None]
    required_task_alignments = [item for item in task_alignments if _is_required_gold(gold_tasks, item.gold_id)]
    required_note_alignments = [item for item in note_alignments if _is_required_gold(gold_notes, item.gold_id)]
    matched_required_tasks = [item for item in required_task_alignments if item.predicted_index is not None]
    matched_required_notes = [item for item in required_note_alignments if item.predicted_index is not None]

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
    transcript_seqs = set(_sequences_in_transcript(case.get("transcript", "")))
    background_seqs = set(case.get("backgroundSequences") or [])
    unsupported_count = 0
    background_fp = 0
    for item in predicted:
        seqs = set(item.evidenceSequences or [])
        if seqs and transcript_seqs and not seqs.issubset(transcript_seqs):
            unsupported_count += 1
        elif seqs and background_seqs and seqs <= background_seqs:
            background_fp += 1
            unsupported_count += 1
    required_meanings = [
        gold
        for gold in [*gold_tasks, *gold_notes]
        if _is_required_gold([gold], gold.id)
    ]
    retained = [
        gold
        for gold in required_meanings
        if _meaning_retained(
            gold,
            predicted,
            embedding_score=(meaning_embedding_scores or {}).get(gold.id),
        )
    ]
    meaning_retention = _ratio(len(retained), len(required_meanings)) if required_meanings else 1.0
    candidate_recall = None
    gold_candidates = []
    for index, item in enumerate(case.get("goldCandidates") or []):
        payload = dict(item)
        payload.setdefault("id", f"c{index}")
        if payload.get("kind") not in {"task", "note"}:
            payload["kind"] = "note"
        gold_candidates.append(_as_gold_item(payload))
    if gold_candidates:
        alignments, _ = _align(gold_candidates, predicted_candidates or predicted)
        candidate_recall = _ratio(sum(1 for row in alignments if row.predicted_index is not None), len(alignments))
    return CaseScore(
        caseId=str(case["id"]),
        category=str(case.get("category", "")),
        taskRecall=_ratio(len(matched_required_tasks), len(required_task_alignments)),
        taskPrecision=_ratio(len(matched_tasks), len(predicted_tasks)),
        noteRecall=_ratio(len(matched_required_notes), len(required_note_alignments)),
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
        unsupportedArtifactRate=_ratio(unsupported_count, len(predicted)) if predicted else 0.0,
        candidateRecall=candidate_recall,
        backgroundFalsePositiveRate=_ratio(background_fp, len(predicted)) if predicted else (0.0 if background_seqs else None),
        meaningRetentionRecall=meaning_retention,
        details={
            "taskAlignments": [alignment.__dict__ for alignment in task_alignments],
            "noteAlignments": [alignment.__dict__ for alignment in note_alignments],
            "requiredTaskRecall": _ratio(len(matched_required_tasks), len(required_task_alignments)),
            "requiredNoteRecall": _ratio(len(matched_required_notes), len(required_note_alignments)),
            "retainedGoldIds": [gold.id for gold in retained],
            "unretainedGoldIds": [gold.id for gold in required_meanings if gold not in retained],
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
        unsupportedArtifactRate=_mean([item.unsupportedArtifactRate for item in scores]) or 0.0,
        candidateRecall=_mean([item.candidateRecall for item in scores]),
        backgroundFalsePositiveRate=_mean([item.backgroundFalsePositiveRate for item in scores]),
        meaningRetentionRecall=_mean([item.meaningRetentionRecall for item in scores]),
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
    # Semantic fallback: same action/memory meaning can land on nearby/paraphrased evidence.
    unmatched = [item for item in alignments if item.predicted_index is None]
    for alignment in unmatched:
        gold = next(item for item in gold_items if item.id == alignment.gold_id)
        best_index = None
        best_score = 0.0
        for index in list(remaining):
            score = _semantic_align_score(gold.meaning, predicted[index].meaning)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None or best_score < 0.42:
            continue
        remaining.discard(best_index)
        alignment.predicted_index = best_index
        alignment.overlap = 0
    duplicate_count = 0
    gold_seq_sets = [set(item.evidenceSequences) for item in gold_items]
    for index in remaining:
        pred_seqs = set(predicted[index].evidenceSequences)
        if any(pred_seqs & gold_seqs for gold_seqs in gold_seq_sets):
            duplicate_count += 1
    return alignments, duplicate_count


def _as_gold_item(item: Any) -> GoldItem:
    if isinstance(item, GoldItem):
        return item
    allowed = {field.name for field in fields(GoldItem)}
    return GoldItem(**{key: value for key, value in dict(item).items() if key in allowed})


def _is_required_gold(gold_items: list[GoldItem], gold_id: str) -> bool:
    gold = next((item for item in gold_items if item.id == gold_id), None)
    if gold is None:
        return True
    return str(gold.reviewStatus or "REQUIRED").strip().upper() in {"", "REQUIRED"}


def semantic_align_score(gold_meaning: str, predicted_meaning: str) -> float:
    return _semantic_align_score(gold_meaning, predicted_meaning)


def _meaning_retained(
    gold: GoldItem,
    predicted: list[PredictedItem],
    *,
    embedding_score: float | None = None,
) -> bool:
    if not predicted:
        return False
    blob = "\n".join(item.meaning for item in predicted)
    if _semantic_align_score(gold.meaning, blob) >= 0.42:
        return True
    if any(_semantic_align_score(gold.meaning, item.meaning) >= 0.42 for item in predicted):
        return True
    return embedding_score is not None and embedding_score >= 0.72


def _semantic_align_score(gold_meaning: str, predicted_meaning: str) -> float:
    gold_tokens = _content_tokens(gold_meaning)
    pred_tokens = _content_tokens(predicted_meaning)
    if not gold_tokens or not pred_tokens:
        return 0.0
    shared = gold_tokens & pred_tokens
    jaccard = len(shared) / len(gold_tokens | pred_tokens)
    contained = gold_tokens <= pred_tokens or pred_tokens <= gold_tokens
    fold_gold = _fold(gold_meaning)
    fold_pred = _fold(predicted_meaning)
    substring = fold_gold in fold_pred or fold_pred in fold_gold
    coverage = len(shared) / len(gold_tokens)
    if contained and shared:
        return max(jaccard, 0.55)
    if substring and len(shared) >= 2:
        return max(jaccard, 0.5)
    if coverage >= 0.6 and len(shared) >= 2:
        return max(jaccard, 0.5)
    return jaccard


def _content_tokens(text: str) -> set[str]:
    stop = {
        "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "from",
        "is", "are", "was", "were", "be", "this", "that", "it", "we", "you",
        "at", "please", "should", "will", "would", "need", "their", "them", "they",
        "own", "via", "able", "can", "could", "ke", "ki", "ka", "se", "pe", "tak",
        "hai", "hoga", "karega", "kar", "dega", "apni", "us",
    }
    latin = {token for token in re.findall(r"[A-Za-z0-9]+", (text or "").casefold()) if token not in stop and len(token) > 1}
    devanagari = {token for token in re.findall(r"[\u0900-\u097F]+", text or "") if len(token) > 1}
    return {_canon_token(token) for token in latin} | devanagari


def _canon_token(token: str) -> str:
    aliases = {
        "submit": "enter",
        "submits": "enter",
        "submitted": "enter",
        "submitting": "enter",
        "enters": "enter",
        "entered": "enter",
        "entering": "enter",
        "fill": "enter",
        "fills": "enter",
        "filled": "enter",
        "filling": "enter",
        "information": "detail",
        "info": "detail",
        "details": "detail",
        "generate": "create",
        "generates": "create",
        "generated": "create",
        "generating": "create",
        "creates": "create",
        "created": "create",
        "creating": "create",
        "url": "link",
        "links": "link",
        "receive": "use",
        "receives": "use",
        "received": "use",
        "uses": "use",
        "using": "use",
        "used": "use",
        "reducing": "reduce",
        "reduces": "reduce",
        "reduced": "reduce",
        "remove": "reduce",
        "removes": "reduce",
        "removed": "reduce",
        "removing": "reduce",
        "manually": "manual",
        "building": "implement",
        "builds": "implement",
        "build": "implement",
        "implementing": "implement",
        "implements": "implement",
        "implemented": "implement",
        "adding": "add",
        "adds": "add",
        "added": "add",
        "manual": "manual",
        "form": "detail",
        "forms": "detail",
        "page": "page",
        "button": "action",
        "tomorrow": "tomorrow",
        "kal": "tomorrow",
        "integrate": "integrate",
        "integration": "integrate",
    }
    value = aliases.get(token, token)
    if value.endswith("ing") and len(value) > 5:
        value = value[:-3]
    elif value.endswith("ies") and len(value) > 4:
        value = value[:-3] + "y"
    elif value.endswith("es") and len(value) > 4:
        value = value[:-2]
    elif value.endswith("s") and len(value) > 3 and not value.endswith("ss"):
        value = value[:-1]
    return aliases.get(value, value)


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


_OWNER_ALIASES = {
    "अंशु": "anshu",
    "anshu": "anshu",
}


def _names_match(gold: str, predicted: str | None) -> bool:
    if not predicted:
        return False
    gold_fold = _fold(gold)
    pred_fold = _fold(predicted)
    if gold_fold == pred_fold or gold_fold in pred_fold or pred_fold in gold_fold:
        return True
    gold_alias = _OWNER_ALIASES.get(gold_fold) or _OWNER_ALIASES.get(gold)
    pred_alias = _OWNER_ALIASES.get(pred_fold) or _OWNER_ALIASES.get(predicted)
    return bool(gold_alias and pred_alias and gold_alias == pred_alias)


_EVAL_MEETING_AT = datetime(2026, 8, 28, tzinfo=timezone.utc)


def _dates_match(gold: str, predicted: str | None) -> bool:
    if not predicted:
        return False
    if _fold(gold) == _fold(predicted) or _fold(gold) in _fold(predicted) or _fold(predicted) in _fold(gold):
        return True
    gold_iso = normalize_supported_due_date(gold, _EVAL_MEETING_AT)
    pred_iso = normalize_supported_due_date(predicted, _EVAL_MEETING_AT)
    if gold_iso and gold_iso == predicted.strip():
        return True
    if pred_iso and pred_iso == gold.strip():
        return True
    return bool(gold_iso and pred_iso and gold_iso == pred_iso)


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
