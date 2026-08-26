"""Task-coverage accountability after final synthesis.

Internal only: dispositions and TASK_COVERAGE_CONFLICT stay in diagnostics.
"""

from __future__ import annotations

from typing import Any

from services.conversation.extraction_contract import _ACTION_ROLES, _evidence_sequences, semantic_unit_id
from services.conversation.models import ExtractedTask, SemanticUnit, WindowExtractionResult


CREATED_TASK = "CREATED_TASK"
MERGED_INTO_TASK = "MERGED_INTO_TASK"
ALREADY_COMPLETED = "ALREADY_COMPLETED"
DUPLICATE = "DUPLICATE"
NOT_ACTIONABLE_AFTER_REVIEW = "NOT_ACTIONABLE_AFTER_REVIEW"
UNSUPPORTED = "UNSUPPORTED"
TASK_COVERAGE_CONFLICT = "TASK_COVERAGE_CONFLICT"

_ACTIONABLE_KINDS = _ACTION_ROLES | {"action_candidate", "task"}
_COMPLETED_STATES = {"completed", "cancelled", "done", "resolved"}
_EXAMPLE_ROLES = {"example"}


def is_actionable_semantic_unit(unit: SemanticUnit, threads: list[Any] | None = None) -> bool:
    if not (unit.meaning or "").strip():
        return False
    if _unit_completed(unit) or _unit_speculative(unit):
        return False
    kind = str(unit.kind or "").strip().casefold()
    if kind in _ACTIONABLE_KINDS:
        return True
    unit_sequences = _unit_sequences(unit)
    for thread in threads or []:
        roles = {str(role).casefold() for role in getattr(thread, "roles", None) or []}
        if roles & _EXAMPLE_ROLES and not roles & _ACTION_ROLES:
            continue
        if not roles & _ACTION_ROLES:
            continue
        thread_sequences = {int(turn.sequence) for turn in getattr(thread, "turns", []) or []}
        if unit_sequences & thread_sequences:
            return True
    return False


def merge_uncovered_action_units(
    units: list[SemanticUnit],
    threads: list[Any] | None,
    transcript: str,
) -> list[SemanticUnit]:
    """Keep extractor units, and add grounded action threads the extractor omitted."""
    from services.conversation.extraction_contract import hydrate_and_validate_unit_evidence

    covered: set[int] = set()
    for unit in units:
        covered |= _unit_sequences(unit)
    extras: list[SemanticUnit] = []
    for thread in threads or []:
        roles = {str(role).casefold() for role in getattr(thread, "roles", None) or []}
        if not roles & _ACTION_ROLES:
            continue
        sequences = {int(turn.sequence) for turn in getattr(thread, "turns", []) or []}
        if not sequences or sequences <= covered:
            continue
        action_kind = next(
            (role for role in ("request", "assignment", "action", "instruction", "follow_up", "commitment") if role in roles),
            "request",
        )
        meaning = next(
            (
                str(turn.meaning).strip()
                for turn in getattr(thread, "turns", []) or []
                if set(str(role).casefold() for role in (getattr(turn, "roles", None) or [])) & _ACTION_ROLES
                and str(getattr(turn, "meaning", "") or "").strip()
            ),
            "",
        ) or str(getattr(thread, "topic", "") or "").strip()
        if not meaning:
            continue
        extras.append(
            SemanticUnit(
                semanticKey=str(getattr(thread, "thread_key", "") or f"thread-{min(sequences)}"),
                kind=action_kind,
                meaning=meaning,
                evidence=list(getattr(thread, "evidence", None) or []),
                evidenceIds=sorted(sequences),
                quality={"grounded": True, "independentlyUseful": True, "source": "semantic-thread"},
            )
        )
        covered |= sequences
    if not extras:
        return units
    kept, _ = hydrate_and_validate_unit_evidence(extras, transcript)
    return list(units) + kept


def annotate_semantic_units(units: list[SemanticUnit], threads: list[Any] | None = None) -> list[SemanticUnit]:
    for unit in units:
        quality = dict(unit.quality or {})
        quality["actionable"] = is_actionable_semantic_unit(unit, threads)
        unit.quality = quality
    return units


def evaluate_task_coverage(
    units: list[SemanticUnit],
    result: WindowExtractionResult,
    threads: list[Any] | None = None,
) -> dict[str, Any]:
    indexed = [(semantic_unit_id(unit, index), unit) for index, unit in enumerate(units)]
    dispositions: list[dict[str, Any]] = []
    uncovered: list[SemanticUnit] = []
    covered_keys: set[str] = set()
    for unit_id, unit in indexed:
        disposition, task_ref = _disposition_for_unit(unit_id, unit, result, covered_keys, threads)
        quality = dict(unit.quality or {})
        quality["coverageDisposition"] = disposition
        if task_ref is not None:
            quality["coverageTaskRef"] = task_ref
        unit.quality = quality
        dispositions.append(
            {
                "semanticKey": unit.semanticKey or unit_id,
                "kind": unit.kind,
                "disposition": disposition,
                "taskRef": task_ref,
                "evidenceIds": list(unit.evidenceIds or _unit_sequences(unit)),
            }
        )
        if disposition in {CREATED_TASK, MERGED_INTO_TASK}:
            covered_keys.add(unit.semanticKey or unit_id)
        elif disposition not in {ALREADY_COMPLETED, DUPLICATE, NOT_ACTIONABLE_AFTER_REVIEW, UNSUPPORTED} and quality.get(
            "actionable"
        ):
            uncovered.append(unit)
    actionable_count = sum(1 for unit in units if (unit.quality or {}).get("actionable"))
    conflict = actionable_count > 0 and not result.tasks
    if conflict:
        for item in dispositions:
            unit = next((unit for unit_id, unit in indexed if (unit.semanticKey or unit_id) == item["semanticKey"]), None)
            if unit and (unit.quality or {}).get("actionable") and item["disposition"] not in {
                CREATED_TASK,
                MERGED_INTO_TASK,
                ALREADY_COMPLETED,
                DUPLICATE,
                NOT_ACTIONABLE_AFTER_REVIEW,
                UNSUPPORTED,
            }:
                item["disposition"] = item["disposition"] or UNSUPPORTED
    return {
        "validatedActionableUnitCount": actionable_count,
        "finalTaskCount": len(result.tasks),
        "finalNoteCount": len(result.notes),
        "taskCoverageConflict": conflict,
        "undisposedActionableUnitCount": len(uncovered) if conflict else 0,
        "unitDispositions": dispositions,
        "missedActionableUnits": uncovered,
    }


def coverage_repair_payload(
    missed_units: list[SemanticUnit],
    current_tasks: list[ExtractedTask],
) -> dict[str, Any]:
    return {
        "reason": TASK_COVERAGE_CONFLICT,
        "reviewActions": ["CREATE", "MERGE", "SUPPRESS_WITH_REASON"],
        "missedActionableUnits": [_unit_review_payload(unit) for unit in missed_units],
        "currentTasks": [
            {
                "title": task.title,
                "body": task.body,
                "ownerText": task.ownerText,
                "dueDateText": task.dueDateText,
                "sourceSemanticUnitIds": list((task.changes or {}).get("sourceSemanticUnitIds") or []),
                "evidenceIds": sorted(_evidence_sequences(task.evidence)),
            }
            for task in current_tasks
        ],
    }


def _disposition_for_unit(
    unit_id: str,
    unit: SemanticUnit,
    result: WindowExtractionResult,
    covered_keys: set[str],
    threads: list[Any] | None,
) -> tuple[str | None, str | None]:
    quality = unit.quality or {}
    if quality.get("evidenceOutcome") == "CORE_EVIDENCE_INVALID":
        return UNSUPPORTED, None
    if _unit_completed(unit):
        return ALREADY_COMPLETED, None
    if _unit_speculative(unit) or not quality.get("actionable"):
        return None, None
    key = unit.semanticKey or unit_id
    if key in covered_keys:
        return DUPLICATE, None
    unit_sequences = _unit_sequences(unit)
    for index, task in enumerate(result.tasks):
        metadata = task.changes or {}
        source_ids = {str(item) for item in metadata.get("sourceSemanticUnitIds") or []}
        if unit_id in source_ids or (unit.semanticKey and unit.semanticKey in source_ids):
            ref = str(task.artifactId or task.fingerprint or index)
            return (MERGED_INTO_TASK if len(source_ids) > 1 else CREATED_TASK), ref
        if metadata.get("semanticArtifactKey") and metadata.get("semanticArtifactKey") == unit.semanticKey:
            return CREATED_TASK, str(task.artifactId or task.fingerprint or index)
        if unit_sequences and unit_sequences <= _evidence_sequences(task.evidence):
            return MERGED_INTO_TASK, str(task.artifactId or task.fingerprint or index)
        if unit_sequences and unit_sequences & _evidence_sequences(task.evidence):
            return MERGED_INTO_TASK, str(task.artifactId or task.fingerprint or index)
    if unit.state and str(unit.state).casefold() in _COMPLETED_STATES:
        return ALREADY_COMPLETED, None
    return None, None


def _unit_review_payload(unit: SemanticUnit) -> dict[str, Any]:
    return {
        "semanticKey": unit.semanticKey,
        "kind": unit.kind,
        "meaning": unit.meaning,
        "ownerText": unit.ownerText,
        "dueDateText": unit.dueDateText,
        "evidenceIds": list(unit.evidenceIds or sorted(_unit_sequences(unit))),
        "evidence": [
            {"sequenceStart": span.sequenceStart, "sequenceEnd": span.sequenceEnd, "text": span.text}
            for span in (unit.evidence or [])
        ],
        "nearbyContext": unit.relatedSemanticKeys,
    }


def _unit_sequences(unit: SemanticUnit) -> set[int]:
    sequences = set(int(value) for value in (unit.evidenceIds or []) if str(value).lstrip("-").isdigit())
    sequences |= _evidence_sequences(unit.evidence)
    return sequences


def _unit_completed(unit: SemanticUnit) -> bool:
    return str(unit.state or "").strip().casefold() in _COMPLETED_STATES


def _unit_speculative(unit: SemanticUnit) -> bool:
    quality = unit.quality or {}
    return bool(quality.get("semanticSpeculation") or quality.get("uncertain"))
