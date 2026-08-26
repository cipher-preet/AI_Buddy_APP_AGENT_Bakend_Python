from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any

from pydantic import ValidationError

from services.conversation.models import EvidenceSpan, ExtractionOutcome, ExtractedNote, ExtractedTask, SemanticUnit, WindowExtractionResult
from services.llm.structured_output import STRUCTURED_SCHEMA_ECHO, MALFORMED_STRUCTURED_OUTPUT, is_schema_echo


# Roles already produced by understanding. These are structured type checks, not
# English keyword matching against transcript text.
_ACTION_ROLES = {"action", "commitment", "request", "instruction", "follow_up", "assignment"}
_NOTE_ROLES = {
    "fact",
    "claim",
    "explanation",
    "definition",
    "decision",
    "conclusion",
    "requirement",
    "important_point",
    "problem",
}
_MEANINGFUL_ROLES = _ACTION_ROLES | _NOTE_ROLES
_HIGH_CONFIDENCE = 0.55
CORE_EVIDENCE_INVALID = "CORE_EVIDENCE_INVALID"
OPTIONAL_METADATA_INVALID = "OPTIONAL_METADATA_INVALID"
EVIDENCE_VALID = "VALID"

LAST_EXTRACTION_PARSE_TRACE: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_extraction_parse_trace",
    default=None,
)


def empty_parse_trace() -> dict[str, Any]:
    return {
        "rawResponseKeys": [],
        "aliasKeysApplied": [],
        "finishReason": None,
        "rawSemanticUnitCount": 0,
        "rawTaskCount": 0,
        "rawNoteCount": 0,
        "parsedSemanticUnitCount": 0,
        "parsedTaskCount": 0,
        "parsedNoteCount": 0,
        "schemaRejectedUnitCount": 0,
        "schemaRejectedTaskCount": 0,
        "schemaRejectedNoteCount": 0,
        "schemaRejectedDecisionCount": 0,
        "schemaRejectedIssueCount": 0,
        "schemaRejectedReasons": [],
        "evidenceRejectedUnitCount": 0,
        "qualityRejectedTaskCount": 0,
        "qualityRejectedNoteCount": 0,
        "supportedUnitVerdict": None,
        "zeroOutputRecoveryEligible": False,
        "zeroOutputRecoveryAttempted": False,
        "zeroOutputRecoverySource": None,
        "dropStage": None,
        "schemaEchoDetected": False,
        "structuredOutputOutcome": None,
        "parsingOutcome": None,
        "requestedStructuredMode": None,
        "actualResponseFormatMode": None,
        "topLevelResponseKeys": [],
    }


def alias_extraction_payload(value: Any) -> Any:
    """Map known extractor field aliases onto the Pydantic contract without inventing units."""
    if not isinstance(value, dict):
        return value
    if is_schema_echo(value):
        trace = empty_parse_trace()
        trace["rawResponseKeys"] = sorted(str(key) for key in value.keys())
        trace["schemaEchoDetected"] = True
        trace["structuredOutputOutcome"] = STRUCTURED_SCHEMA_ECHO
        trace["parsingOutcome"] = STRUCTURED_SCHEMA_ECHO
        LAST_EXTRACTION_PARSE_TRACE.set(trace)
        raise ValueError(STRUCTURED_SCHEMA_ECHO)
    payload = dict(value)
    aliases: list[str] = []
    if not payload.get("semanticUnits"):
        for key in ("units", "semantic_units", "semanticUnit"):
            if payload.get(key):
                payload["semanticUnits"] = payload[key]
                aliases.append(key)
                break
    if not payload.get("supportedUnitVerdict"):
        for key in ("unitVerdict", "emptyExtractionVerdict", "extractionVerdict"):
            if payload.get(key):
                payload["supportedUnitVerdict"] = payload[key]
                aliases.append(key)
                break
    payload["semanticUnits"] = [_alias_unit(item) for item in _as_list(payload.get("semanticUnits"))]
    payload["tasks"] = [_alias_task(item) for item in _as_list(payload.get("tasks"))]
    payload["notes"] = [_alias_note(item) for item in _as_list(payload.get("notes"))]
    payload["decisions"] = [_alias_decision(item) for item in _as_list(payload.get("decisions"))]
    payload["issues"] = [_alias_issue(item) for item in _as_list(payload.get("issues"))]
    trace = empty_parse_trace()
    trace["rawResponseKeys"] = sorted(str(key) for key in value.keys())
    trace["aliasKeysApplied"] = aliases
    trace["rawSemanticUnitCount"] = len(
        _as_list(value.get("semanticUnits") or value.get("units") or value.get("semantic_units"))
    )
    trace["rawTaskCount"] = len(_as_list(value.get("tasks")))
    trace["rawNoteCount"] = len(_as_list(value.get("notes")))
    LAST_EXTRACTION_PARSE_TRACE.set(trace)
    return payload


def coerce_extraction_lists(
    payload: dict[str, Any],
    unit_cls=SemanticUnit,
    task_cls=None,
    note_cls=None,
    decision_cls=None,
    issue_cls=None,
    update_trace: bool = True,
) -> dict[str, Any]:
    """Keep valid items when a sibling item fails schema, instead of dropping the whole extraction."""
    payload = dict(payload)
    trace = LAST_EXTRACTION_PARSE_TRACE.get() or empty_parse_trace()
    unit_rejected = 0
    task_rejected = 0
    note_rejected = 0
    decision_rejected = 0
    issue_rejected = 0
    reasons: list[str] = []
    if unit_cls is not None:
        payload["semanticUnits"], unit_rejected, unit_reasons = _coerce_items(payload.get("semanticUnits"), unit_cls)
        reasons.extend(unit_reasons)
    if task_cls is not None:
        payload["tasks"], task_rejected, task_reasons = _coerce_items(payload.get("tasks"), task_cls)
        reasons.extend(task_reasons)
    if note_cls is not None:
        payload["notes"], note_rejected, note_reasons = _coerce_items(payload.get("notes"), note_cls)
        reasons.extend(note_reasons)
    if decision_cls is not None:
        payload["decisions"], decision_rejected, decision_reasons = _coerce_items(payload.get("decisions"), decision_cls)
        reasons.extend(decision_reasons)
    if issue_cls is not None:
        payload["issues"], issue_rejected, issue_reasons = _coerce_items(payload.get("issues"), issue_cls)
        reasons.extend(issue_reasons)
    trace["parsedSemanticUnitCount"] = len(payload.get("semanticUnits") or [])
    trace["parsedTaskCount"] = len(payload.get("tasks") or [])
    trace["parsedNoteCount"] = len(payload.get("notes") or [])
    trace["schemaRejectedUnitCount"] = unit_rejected
    trace["schemaRejectedTaskCount"] = task_rejected
    trace["schemaRejectedNoteCount"] = note_rejected
    trace["schemaRejectedDecisionCount"] = decision_rejected
    trace["schemaRejectedIssueCount"] = issue_rejected
    trace["schemaRejectedReasons"] = reasons[:12]
    trace["supportedUnitVerdict"] = payload.get("supportedUnitVerdict")
    if update_trace:
        LAST_EXTRACTION_PARSE_TRACE.set(trace)
    return payload


def hydrate_and_validate_unit_evidence(units: list[SemanticUnit], transcript: str) -> tuple[list[SemanticUnit], int]:
    """Keep a unit when its core meaning is grounded by the union of cited chunks.

    Optional owner/deadline/priority metadata is stripped when unsupported.
    It must not discard the unit.
    """
    lines = _sequence_map(transcript)
    kept: list[SemanticUnit] = []
    rejected = 0
    rejected_records: list[dict[str, Any]] = []
    for unit in units:
        outcome = _validate_unit_core_evidence(unit, lines)
        quality = dict(unit.quality or {})
        quality["evidenceOutcome"] = outcome
        quality["validatedEvidenceIds"] = list(unit.evidenceIds or [])
        unit.quality = quality
        if outcome == CORE_EVIDENCE_INVALID:
            rejected += 1
            rejected_records.append(
                {
                    "semanticKey": unit.semanticKey,
                    "kind": unit.kind,
                    "citedIds": list(getattr(unit, "evidenceIds", None) or []),
                    "spanRanges": [
                        [span.sequenceStart, span.sequenceEnd] for span in (unit.evidence or [])
                    ],
                    "evidenceOutcome": outcome,
                }
            )
            continue
        kept.append(unit)
    trace = LAST_EXTRACTION_PARSE_TRACE.get() or empty_parse_trace()
    trace["evidenceRejectedUnitCount"] = rejected
    trace["evidenceRejectedUnits"] = rejected_records
    LAST_EXTRACTION_PARSE_TRACE.set(trace)
    return kept, rejected


def hydrate_synthesized_artifacts(
    result: WindowExtractionResult,
    units: list[SemanticUnit] | list[dict[str, Any]] | None,
    transcript: str,
) -> WindowExtractionResult:
    """Restore exact transcript evidence and semantic-unit provenance onto final artifacts."""
    normalized_units = _coerce_units(units)
    indexed_units = [(semantic_unit_id(unit, index), unit) for index, unit in enumerate(normalized_units)]
    result.tasks = [
        _hydrate_task(task, indexed_units, transcript) for task in result.tasks
    ]
    result.notes = [
        _hydrate_note(note, indexed_units, transcript) for note in result.notes
    ]
    return result


def semantic_unit_id(unit: SemanticUnit, index: int) -> str:
    return str(unit.semanticKey or f"semantic-unit-{index}")


def normalize_evidence_spans(evidence: list[EvidenceSpan] | None, transcript: str) -> list[EvidenceSpan]:
    lines = _sequence_map(transcript)
    normalized: list[EvidenceSpan] = []
    for span in evidence or []:
        sequences = _intersected_span_sequences(span, lines)
        if sequences:
            for sequence in sequences:
                text = lines[sequence].strip()
                if text:
                    normalized.append(EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text))
            continue
        if span.text:
            normalized.append(span)
    return _unique_spans(normalized)


def upstream_has_grounded_evidence(diagnostics: dict[str, Any] | None) -> bool:
    diagnostics = diagnostics or {}
    if int(diagnostics.get("taskCandidatesGenerated") or 0) > 0:
        return True
    if int(diagnostics.get("noteCandidatesGenerated") or 0) > 0:
        return True
    facts = diagnostics.get("factsExtracted") or []
    if isinstance(facts, list):
        for fact in facts:
            kind = str((fact or {}).get("kind") or "")
            if kind in _MEANINGFUL_ROLES:
                return True
    for thread in diagnostics.get("discussionThreads") or []:
        roles = {str(role) for role in (thread or {}).get("roles") or []}
        confidence = float((thread or {}).get("semanticEvidenceConfidence") or 0)
        if roles & _MEANINGFUL_ROLES and confidence >= _HIGH_CONFIDENCE:
            return True
    return False


def classify_extraction_outcome(
    *,
    has_units: bool,
    technical_failure: bool,
    upstream_evidence: bool,
    explicit_empty_verdict: bool,
    recovery_attempted: bool,
    semantic_input_assembly_failed: bool = False,
) -> ExtractionOutcome:
    if semantic_input_assembly_failed:
        return ExtractionOutcome.SEMANTIC_INPUT_ASSEMBLY_FAILED
    if technical_failure:
        return ExtractionOutcome.EXTRACTION_FAILED
    if has_units:
        return ExtractionOutcome.SUCCESS
    if not upstream_evidence:
        return ExtractionOutcome.VALID_EMPTY_EXTRACTION
    if explicit_empty_verdict:
        return ExtractionOutcome.VALID_EMPTY_EXTRACTION
    if recovery_attempted:
        return ExtractionOutcome.EXTRACTION_FAILED
    return ExtractionOutcome.EXTRACTION_FAILED


def suspicious_empty_retry_instruction(diagnostics: dict[str, Any] | None) -> str:
    diagnostics = diagnostics or {}
    role_counts: dict[str, int] = {}
    for thread in diagnostics.get("discussionThreads") or []:
        for role in (thread or {}).get("roles") or []:
            role_counts[str(role)] = role_counts.get(str(role), 0) + 1
    return (
        "SUSPICIOUS EMPTY EXTRACTION RETRY.\n"
        "Upstream semantic understanding identified grounded semantic material, "
        "but extraction returned zero semanticUnits.\n"
        f"taskCandidatesGenerated={int(diagnostics.get('taskCandidatesGenerated') or 0)} "
        f"noteCandidatesGenerated={int(diagnostics.get('noteCandidatesGenerated') or 0)} "
        f"discussionThreadCount={int(diagnostics.get('discussionThreadCount') or 0)} "
        f"roleCounts={role_counts}.\n"
        "Re-evaluate the same grounded evidence packets. Return only supported semantic "
        "units. Do not invent facts, owners, dates, or new evidence IDs. Preserve exact "
        "evidence IDs from the packets.\n"
        "Populate semanticUnits (not units). Each unit must include evidence spans with "
        "sequenceStart, sequenceEnd, and text copied from the cited evidence IDs.\n"
        "If after re-evaluation nothing should be published, return empty lists and set "
        "supportedUnitVerdict to no_supported_units, listing rejectedCandidates with reasons. "
        "If units are supported, set supportedUnitVerdict to has_supported_units."
    )


def drop_stage_for(trace: dict[str, Any], has_units: bool) -> str:
    if has_units:
        return "none"
    if trace.get("schemaEchoDetected") or trace.get("structuredOutputOutcome") == STRUCTURED_SCHEMA_ECHO:
        return "structured_schema_echo"
    if trace.get("structuredOutputOutcome") == MALFORMED_STRUCTURED_OUTPUT:
        return "malformed_structured_output"
    if (
        int(trace.get("rawSemanticUnitCount") or 0) == 0
        and int(trace.get("rawTaskCount") or 0) == 0
        and int(trace.get("rawNoteCount") or 0) == 0
    ):
        return "raw_llm_zero_units"
    if (
        int(trace.get("schemaRejectedUnitCount") or 0)
        or int(trace.get("schemaRejectedTaskCount") or 0)
        or int(trace.get("schemaRejectedNoteCount") or 0)
    ):
        return "schema_parse_rejected"
    if int(trace.get("evidenceRejectedUnitCount") or 0):
        return "evidence_validation_rejected"
    if int(trace.get("qualityRejectedTaskCount") or 0) or int(trace.get("qualityRejectedNoteCount") or 0):
        return "quality_confidence_rejected"
    return "raw_llm_zero_units"


def _alias_unit(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    unit = dict(item)
    if not unit.get("meaning"):
        for key in ("normalizedMeaning", "text", "statement"):
            if unit.get(key):
                unit["meaning"] = unit[key]
                break
    if not unit.get("semanticKey"):
        for key in ("semanticArtifactKey", "semantic_key", "key"):
            if unit.get(key):
                unit["semanticKey"] = unit[key]
                break
    if unit.get("evidenceIds") is not None:
        unit["evidenceIds"] = _alias_evidence_ids(unit.get("evidenceIds"))
    if not unit.get("evidence") and unit.get("evidenceIds"):
        unit.setdefault("evidence", [])
    if isinstance(unit.get("evidence"), list):
        unit["evidence"] = [_alias_span(span) for span in unit["evidence"]]
    return unit


def _alias_task(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    task = _alias_title_body(dict(item))
    if task.get("title") or task.get("body"):
        if not task.get("operation"):
            task["operation"] = "CREATE"
        if task.get("confidence") is None:
            task["confidence"] = 0.5
        if not task.get("origin"):
            task["origin"] = "unknown"
        if task.get("evidence") is None:
            task["evidence"] = []
    if isinstance(task.get("evidence"), list):
        task["evidence"] = [_alias_span(span) for span in task["evidence"]]
    return task


def _alias_note(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    note = _alias_title_body(dict(item))
    if (note.get("title") or note.get("body")) and note.get("confidence") is None:
        note["confidence"] = 0.5
    if isinstance(note.get("evidence"), list):
        note["evidence"] = [_alias_span(span) for span in note["evidence"]]
    elif note.get("evidence") is None and (note.get("title") or note.get("body")):
        note["evidence"] = []
    return note


def _alias_title_body(item: dict[str, Any]) -> dict[str, Any]:
    if not item.get("title"):
        for key in ("heading", "name", "summary", "description", "text", "content"):
            if str(item.get(key) or "").strip():
                item["title"] = _first_line_title(str(item[key]))
                break
    if not item.get("body"):
        for key in ("content", "description", "text", "summary", "details"):
            if str(item.get(key) or "").strip():
                item["body"] = str(item[key]).strip()
                break
    if not str(item.get("body") or "").strip() and str(item.get("title") or "").strip():
        item["body"] = str(item["title"]).strip()
    return item


def _first_line_title(value: str, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    for separator in (". ", "? ", "! "):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
            break
    if len(text) > limit:
        return f"{text[: limit - 1].rstrip()}…"
    return text


_ISSUE_KIND_ALIASES = {
    "blocker": "blocker",
    "risk": "risk",
    "open_question": "open_question",
    "open question": "open_question",
    "question": "open_question",
    "unresolved": "open_question",
    "missing_information": "missing_information",
    "missing information": "missing_information",
    "missing": "missing_information",
}

_DECISION_STATUS_ALIASES = {
    "confirmed_decision": "confirmed_decision",
    "confirmed": "confirmed_decision",
    "decision": "confirmed_decision",
    "proposal": "proposal",
    "idea": "idea",
    "unresolved_discussion": "unresolved_discussion",
    "unresolved": "unresolved_discussion",
    "discussion": "unresolved_discussion",
}


def _alias_decision(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    decision = dict(item)
    if not decision.get("title"):
        for key in ("description", "text", "summary", "name"):
            if decision.get(key):
                decision["title"] = decision[key]
                break
    if not decision.get("status"):
        raw = str(decision.get("kind") or decision.get("type") or decision.get("state") or "").strip().casefold()
        if raw in _DECISION_STATUS_ALIASES:
            decision["status"] = _DECISION_STATUS_ALIASES[raw]
    if decision.get("confidence") is None and decision.get("title"):
        decision["confidence"] = 0.5
    if isinstance(decision.get("evidence"), list):
        decision["evidence"] = [_alias_span(span) for span in decision["evidence"]]
    return decision


def _alias_issue(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    issue = dict(item)
    if not issue.get("title"):
        for key in ("description", "text", "summary", "name"):
            if issue.get(key):
                issue["title"] = issue[key]
                break
    if not issue.get("kind"):
        raw = str(issue.get("type") or issue.get("category") or issue.get("status") or "").strip().casefold()
        issue["kind"] = _ISSUE_KIND_ALIASES.get(raw, "open_question") if issue.get("title") else issue.get("kind")
        if issue.get("title") and not issue.get("kind"):
            issue["kind"] = "open_question"
    if issue.get("confidence") is None and issue.get("title"):
        issue["confidence"] = 0.5
    if isinstance(issue.get("evidence"), list):
        issue["evidence"] = [_alias_span(span) for span in issue["evidence"]]
    return issue


def alias_synthesis_payload(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    payload = dict(value)
    payload["tasks"] = [_alias_task(item) for item in _as_list(payload.get("tasks"))]
    payload["notes"] = [_alias_note(item) for item in _as_list(payload.get("notes"))]
    return payload


def _alias_span(item: Any) -> Any:
    if isinstance(item, bool):
        return item
    if isinstance(item, int):
        return {"sequenceStart": item, "sequenceEnd": item, "text": f"sequence:{item}"}
    if isinstance(item, str) and item.strip().isdigit():
        sequence = int(item.strip())
        return {"sequenceStart": sequence, "sequenceEnd": sequence, "text": f"sequence:{sequence}"}
    if not isinstance(item, dict):
        return item
    span = dict(item)
    if "sequenceStart" not in span:
        for key in ("id", "sequenceId", "sequence", "evidenceId", "start", "from"):
            if key in span and _as_sequence(span[key]) is not None:
                sequence = _as_sequence(span[key])
                span["sequenceStart"] = sequence
                break
    if "sequenceEnd" not in span and span.get("sequenceStart") is not None:
        for key in ("end", "to", "sequenceEnd"):
            if key in span and _as_sequence(span[key]) is not None:
                span["sequenceEnd"] = _as_sequence(span[key])
                break
        else:
            span["sequenceEnd"] = span["sequenceStart"]
    if not str(span.get("text") or "").strip() and span.get("sequenceStart") is not None:
        span["text"] = f"sequence:{span['sequenceStart']}"
    return span


def _hydrate_task(
    task: ExtractedTask,
    indexed_units: list[tuple[str, SemanticUnit]],
    transcript: str,
) -> ExtractedTask:
    matched_ids, matched_units = _match_source_units(task, indexed_units, getattr(task, "changes", {}) or {})
    task.evidence = _restore_evidence(task.evidence, matched_units, transcript)
    task.changes = {
        **(task.changes or {}),
        "sourceSemanticUnitIds": matched_ids,
        "provenanceRestored": bool(matched_ids),
        "supportedContext": bool(matched_ids) or bool((task.changes or {}).get("supportedContext")),
        "validatedUnitCatalogSize": len(indexed_units),
    }
    return task


def _hydrate_note(
    note: ExtractedNote,
    indexed_units: list[tuple[str, SemanticUnit]],
    transcript: str,
) -> ExtractedNote:
    matched_ids, matched_units = _match_source_units(note, indexed_units, getattr(note, "debug", {}) or {})
    note.evidence = _restore_evidence(note.evidence, matched_units, transcript)
    note.debug = {
        **(note.debug or {}),
        "sourceSemanticUnitIds": matched_ids,
        "provenanceRestored": bool(matched_ids),
        "supportedContext": bool(matched_ids) or bool((note.debug or {}).get("supportedContext")),
        "validatedUnitCatalogSize": len(indexed_units),
    }
    return note


def _restore_evidence(
    evidence: list[EvidenceSpan] | None,
    matched_units: list[SemanticUnit],
    transcript: str,
) -> list[EvidenceSpan]:
    normalized = normalize_evidence_spans(evidence, transcript)
    if normalized:
        return normalized
    restored: list[EvidenceSpan] = []
    for unit in matched_units:
        restored.extend(normalize_evidence_spans(unit.evidence, transcript))
    return _unique_spans(restored)


def _match_source_units(
    artifact: ExtractedTask | ExtractedNote,
    indexed_units: list[tuple[str, SemanticUnit]],
    metadata: dict[str, Any],
) -> tuple[list[str], list[SemanticUnit]]:
    by_id = {unit_id: unit for unit_id, unit in indexed_units}
    requested = [
        str(item)
        for item in (
            metadata.get("sourceSemanticUnitIds")
            or getattr(artifact, "sourceSemanticUnitIds", None)
            or []
        )
        if str(item).strip()
    ]
    matched_ids: list[str] = []
    matched_units: list[SemanticUnit] = []
    for unit_id in requested:
        unit = by_id.get(unit_id)
        if unit is not None:
            matched_ids.append(unit_id)
            matched_units.append(unit)
    if matched_units:
        return _dedupe_unit_matches(matched_ids, matched_units)

    key = str(metadata.get("semanticArtifactKey") or "").strip()
    if key:
        for unit_id, unit in indexed_units:
            if unit.semanticKey == key:
                matched_ids.append(unit_id)
                matched_units.append(unit)
        if matched_units:
            return _dedupe_unit_matches(matched_ids, matched_units)

    artifact_sequences = _evidence_sequences(getattr(artifact, "evidence", None))
    if not artifact_sequences:
        return [], []
    for unit_id, unit in indexed_units:
        if artifact_sequences & _evidence_sequences(unit.evidence):
            matched_ids.append(unit_id)
            matched_units.append(unit)
    return _dedupe_unit_matches(matched_ids, matched_units)


def _evidence_sequences(evidence: list[EvidenceSpan] | None) -> set[int]:
    sequences: set[int] = set()
    for span in evidence or []:
        sequences.update(range(span.sequenceStart, span.sequenceEnd + 1))
    return sequences


def _dedupe_unit_matches(ids: list[str], units: list[SemanticUnit]) -> tuple[list[str], list[SemanticUnit]]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    unique_units: list[SemanticUnit] = []
    for unit_id, unit in zip(ids, units):
        if unit_id in seen:
            continue
        seen.add(unit_id)
        unique_ids.append(unit_id)
        unique_units.append(unit)
    return unique_ids, unique_units


def _unique_spans(evidence: list[EvidenceSpan]) -> list[EvidenceSpan]:
    seen: set[tuple[int, int, str]] = set()
    unique: list[EvidenceSpan] = []
    for span in evidence:
        key = (span.sequenceStart, span.sequenceEnd, span.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)
    return unique


def _coerce_units(units: list[SemanticUnit] | list[dict[str, Any]] | None) -> list[SemanticUnit]:
    coerced: list[SemanticUnit] = []
    for item in units or []:
        if isinstance(item, SemanticUnit):
            coerced.append(item)
            continue
        if isinstance(item, dict):
            try:
                coerced.append(SemanticUnit.model_validate(_alias_unit(item)))
            except (ValidationError, TypeError, ValueError):
                continue
    return coerced


def _coerce_items(items: Any, schema) -> tuple[list[Any], int, list[str]]:
    kept: list[Any] = []
    rejected = 0
    reasons: list[str] = []
    for item in _as_list(items):
        try:
            kept.append(schema.model_validate(item) if not isinstance(item, schema) else item)
        except (ValidationError, TypeError, ValueError) as error:
            rejected += 1
            reasons.append(str(error).split("\n")[0][:180])
    return kept, rejected, reasons


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _validate_unit_core_evidence(unit: SemanticUnit, lines: dict[int, str]) -> str:
    ids = _cited_sequence_ids(unit, lines)
    if not ids or not str(unit.meaning or "").strip():
        return CORE_EVIDENCE_INVALID
    union = " ".join(lines[sequence] for sequence in ids)
    unit.evidenceIds = ids
    unit.evidence = [
        EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=lines[sequence])
        for sequence in ids
        if lines.get(sequence, "").strip()
    ]
    if not unit.evidence:
        return CORE_EVIDENCE_INVALID
    optional_invalid = False
    union_norm = _fold_ws(union)
    if unit.ownerText and _fold_ws(unit.ownerText) not in union_norm:
        unit.ownerText = None
        optional_invalid = True
    if unit.dueDateText and _fold_ws(unit.dueDateText) not in union_norm:
        unit.dueDateText = None
        optional_invalid = True
    return OPTIONAL_METADATA_INVALID if optional_invalid else EVIDENCE_VALID


def _cited_sequence_ids(unit: SemanticUnit, lines: dict[int, str]) -> list[int]:
    ids: set[int] = set()
    for value in getattr(unit, "evidenceIds", None) or []:
        ids.update(_sequences_from_value(value, lines))
    for span in unit.evidence or []:
        span_ids = _intersected_span_sequences(span, lines)
        ids.update(span_ids)
        if span_ids:
            continue
        cited = _fold_ws(getattr(span, "text", "") or "")
        if cited and not cited.startswith("sequence:"):
            ids.update(_sequences_supported_by_text(cited, lines))
    if not ids:
        ids.update(_sequences_supported_by_text(_fold_ws(unit.meaning or ""), lines))
    return sorted(sequence for sequence in ids if sequence in lines)


def _intersected_span_sequences(span: EvidenceSpan, lines: dict[int, str]) -> list[int]:
    try:
        start = int(span.sequenceStart)
        end = int(span.sequenceEnd)
    except (TypeError, ValueError):
        return []
    if start > end:
        start, end = end, start
    if not lines or not _span_range_is_sequence_space(start, end, lines):
        return []
    return [sequence for sequence in range(start, end + 1) if sequence in lines]


def _span_range_is_sequence_space(start: int, end: int, lines: dict[int, str]) -> bool:
    min_seq = min(lines)
    max_seq = max(lines)
    if start > max_seq and end > max_seq:
        return False
    if start < min_seq and end < min_seq:
        return False
    return True


def _sequences_from_value(value: Any, lines: dict[int, str]) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, dict):
        collected: list[int] = []
        for key in ("sequenceStart", "sequenceEnd", "sequenceId", "id", "sequence", "evidenceId", "start", "end"):
            collected.extend(_sequences_from_value(value.get(key), lines))
        return collected
    if isinstance(value, (list, tuple)):
        collected = []
        for item in value:
            collected.extend(_sequences_from_value(item, lines))
        return collected
    sequence = _as_sequence(value)
    if sequence is not None:
        return [sequence] if sequence in lines else []
    text = str(value or "").strip()
    match = re.fullmatch(r"\[?(\d+)\]?\s*[-–:]\s*\[?(\d+)\]?", text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if start > end:
            start, end = end, start
        if _span_range_is_sequence_space(start, end, lines):
            return [sequence for sequence in range(start, end + 1) if sequence in lines]
    return []


def _alias_evidence_ids(value: Any) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for item in _as_list(value):
        sequence = _as_sequence(item)
        candidates = [sequence] if sequence is not None else []
        if sequence is None:
            text = str(item or "").strip()
            match = re.fullmatch(r"\[?(\d+)\]?\s*[-–:]\s*\[?(\d+)\]?", text)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                if start > end:
                    start, end = end, start
                if 0 <= start and end - start <= 64:
                    candidates = list(range(start, end + 1))
        for candidate in candidates:
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            ids.append(candidate)
    return ids


def _sequences_supported_by_text(cited: str, lines: dict[int, str]) -> list[int]:
    if not cited:
        return []
    supported: list[int] = []
    for sequence, text in lines.items():
        line = _fold_ws(text)
        if not line:
            continue
        if cited in line or (len(line) >= 12 and line in cited) or _token_overlap_supports(line, cited):
            supported.append(sequence)
    return supported


_FUNCTION_WORDS = {
    "a", "an", "the", "to", "for", "of", "and", "or", "in", "on", "at", "by", "with",
    "is", "are", "was", "were", "be", "been", "will", "would", "should", "shall",
    "please", "also", "that", "this", "those", "these", "it", "we", "they", "i",
    "you", "he", "she", "them", "us", "our", "before", "after", "from", "into",
}


def _token_overlap_supports(line: str, cited: str) -> bool:
    line_tokens = _content_tokens(line)
    cited_tokens = _content_tokens(cited)
    if len(line_tokens) < 3 or len(cited_tokens) < 3:
        return False
    overlap = line_tokens & cited_tokens
    needed = max(3, (len(line_tokens) * 3 + 4) // 5)
    return len(overlap) >= needed


def _content_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9\u0900-\u097f]+", value) if token not in _FUNCTION_WORDS and len(token) > 1}


def _as_sequence(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        for key in ("sequenceStart", "sequenceId", "id", "sequence", "evidenceId"):
            if key in value:
                return _as_sequence(value[key])
        return None
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        try:
            return int(text)
        except ValueError:
            return None
    match = re.fullmatch(r"\[(\d+)\]", text) or re.fullmatch(r"(?:seq|sequence)[:\-\s]+(\d+)", text, re.I)
    if match:
        return int(match.group(1))
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return None


def _fold_ws(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _sequence_map(transcript: str) -> dict[int, str]:
    lines: dict[int, str] = {}
    for raw in (transcript or "").splitlines():
        text = raw.strip()
        if text.startswith("[") and "]" in text:
            marker, rest = text.split("]", 1)
            try:
                lines[int(marker[1:])] = rest.strip()
            except ValueError:
                continue
    return lines


def _evidence_matches_transcript(evidence: list[EvidenceSpan], lines: dict[int, str], transcript: str) -> bool:
    haystack = " ".join((transcript or "").casefold().split())
    for span in evidence:
        cited = " ".join((span.text or "").casefold().split())
        if not cited:
            return False
        expected = " ".join(
            lines.get(sequence, "")
            for sequence in range(span.sequenceStart, span.sequenceEnd + 1)
        ).casefold()
        expected = " ".join(expected.split())
        if expected and cited in expected:
            continue
        if cited in haystack:
            continue
        return False
    return True
