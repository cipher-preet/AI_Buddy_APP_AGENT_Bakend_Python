from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from apps.api_gateway.config.setting import settings
from services.conversation.models import EvidenceSpan, ExtractedDecision, ExtractedIssue, ExtractedNote, ExtractedTask


Origin = Literal["explicit", "strongly_inferred", "unknown"]


@dataclass(frozen=True)
class ConfidencePolicy:
    # Thresholds come from configured policy. A provider self-score is never
    # permission to publish, and there is no hidden 0.70 floor here.
    publish_threshold: float = 0.55
    suggestion_threshold: float = 0.45
    evidence_weight: float = 0.28
    explicitness_weight: float = 0.2
    completeness_weight: float = 0.16
    context_weight: float = 0.12
    validation_weight: float = 0.16
    conflict_penalty: float = 0.22
    shallow_penalty: float = 0.12
    speculation_penalty: float = 0.22


def default_confidence_policy() -> ConfidencePolicy:
    return ConfidencePolicy(
        publish_threshold=settings.INTELLIGENCE_CONFIDENCE_PUBLISH_THRESHOLD,
        suggestion_threshold=settings.INTELLIGENCE_CONFIDENCE_SUGGESTION_THRESHOLD,
        evidence_weight=settings.INTELLIGENCE_CONFIDENCE_EVIDENCE_WEIGHT,
        explicitness_weight=settings.INTELLIGENCE_CONFIDENCE_EXPLICITNESS_WEIGHT,
        completeness_weight=settings.INTELLIGENCE_CONFIDENCE_COMPLETENESS_WEIGHT,
        context_weight=settings.INTELLIGENCE_CONFIDENCE_CONTEXT_WEIGHT,
        validation_weight=settings.INTELLIGENCE_CONFIDENCE_VALIDATION_WEIGHT,
        conflict_penalty=settings.INTELLIGENCE_CONFIDENCE_CONFLICT_PENALTY,
        shallow_penalty=settings.INTELLIGENCE_CONFIDENCE_SHALLOW_PENALTY,
        speculation_penalty=settings.INTELLIGENCE_CONFIDENCE_SPECULATION_PENALTY,
    )


def score_and_filter_result(result, transcript: str, policy: ConfidencePolicy | None = None, diagnostics: dict | None = None):
    policy = policy or default_confidence_policy()
    task_count = len(result.tasks)
    note_count = len(result.notes)
    kept_tasks, rejected_tasks, task_records = _filter_tasks(result.tasks, transcript, policy)
    kept_notes, rejected_notes, note_records = _filter_notes(result.notes, transcript, policy)
    result.tasks = _dedupe_semantic_items(kept_tasks)
    result.notes = _dedupe_semantic_items(kept_notes)
    result.decisions = _filter_decisions(result.decisions, transcript, policy)
    result.issues = _filter_issues(result.issues, transcript, policy)
    if diagnostics is not None:
        diagnostics["qualityRejectedTaskCount"] = max(0, task_count - len(result.tasks))
        diagnostics["qualityRejectedNoteCount"] = max(0, note_count - len(result.notes))
        diagnostics["requiredConfidence"] = policy.publish_threshold
        diagnostics["qualityArtifactDiagnostics"] = task_records + note_records
        diagnostics["qualityRejectedTaskItems"] = rejected_tasks
        diagnostics["qualityRejectedNoteItems"] = rejected_notes
    return result


def _dedupe_semantic_items(items: list[Any]) -> list[Any]:
    """Merge only candidates with the same supported meaning, before storage."""
    merged: list[Any] = []
    for item in sorted(items, key=lambda value: float(getattr(value, "confidence", 0.0)), reverse=True):
        match = next((existing for existing in merged if _same_supported_meaning(existing, item)), None)
        if match is None:
            merged.append(item)
            continue
        match.evidence = _unique_evidence([*match.evidence, *item.evidence])
        match.confidence = max(match.confidence, item.confidence)
        if _word_count(getattr(item, "body", "")) > _word_count(getattr(match, "body", "")):
            match.title = item.title
            match.body = item.body
    return merged


def _same_supported_meaning(left: Any, right: Any) -> bool:
    # A token overlap is not semantic equivalence: it can collapse distinct
    # actions that share a subject.  The semantic synthesizer may provide an
    # opaque artifact key after comparing complete evidence packets.  Without
    # it, only byte-for-byte duplicate output is safe to merge here.
    left_key = _semantic_artifact_key(left)
    right_key = _semantic_artifact_key(right)
    if left_key and right_key:
        return left_key == right_key
    return _normalize(f"{getattr(left, 'title', '')}\n{getattr(left, 'body', '')}") == _normalize(
        f"{getattr(right, 'title', '')}\n{getattr(right, 'body', '')}"
    )


def _semantic_artifact_key(item: Any) -> str:
    metadata = getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}
    return _normalize(str(metadata.get("semanticArtifactKey") or ""))


def _unique_evidence(evidence: list[EvidenceSpan]) -> list[EvidenceSpan]:
    seen: set[tuple[int, int, str]] = set()
    unique: list[EvidenceSpan] = []
    for span in evidence:
        key = (span.sequenceStart, span.sequenceEnd, span.text)
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def score_task(task: ExtractedTask, transcript: str, policy: ConfidencePolicy | None = None) -> tuple[float, dict[str, Any]]:
    policy = policy or default_confidence_policy()
    evidence_score = _evidence_score(task.evidence, transcript)
    origin = _origin(task)
    explicit_score = _task_explicitness(task, origin)
    completeness_score = _task_completeness(task)
    context_score = _context_score(task.evidence, task)
    corroboration_score = _corroboration_score(task)
    validation_score = 1.0 if task.operation != "NO_ACTION" and task.evidence else 0.0
    conflict = _has_conflict(task)
    speculative = _has_speculation(task) and origin != "explicit"
    shallow = _is_shallow(task.title, task.body)
    stt_ambiguity_penalty = _stt_ambiguity_penalty(task.evidence)
    score = (
        evidence_score * policy.evidence_weight
        + explicit_score * policy.explicitness_weight
        + completeness_score * policy.completeness_weight
        + context_score * policy.context_weight
        + validation_score * policy.validation_weight
    )
    if conflict:
        score -= policy.conflict_penalty
    if speculative:
        score -= policy.speculation_penalty
    if shallow and origin == "unknown" and not _has_validated_provenance(task):
        score -= policy.shallow_penalty
    score -= stt_ambiguity_penalty
    score = _clamp(score)
    trace = {
        "engine": "deterministic-confidence-v1",
        "llmConfidence": task.confidence,
        "evidence": round(evidence_score, 4),
        "explicitness": round(explicit_score, 4),
        "completeness": round(completeness_score, 4),
        "context": round(context_score, 4),
        "corroboration": round(corroboration_score, 4),
        "validation": round(validation_score, 4),
        "conflictPenalty": policy.conflict_penalty if conflict else 0.0,
        "speculationPenalty": policy.speculation_penalty if speculative else 0.0,
        "shallowPenalty": policy.shallow_penalty if shallow and origin == "unknown" and not _has_validated_provenance(task) else 0.0,
        "sttAmbiguityPenalty": round(stt_ambiguity_penalty, 4),
        "origin": origin,
        "score": score,
    }
    return score, trace


def score_note(note: ExtractedNote, transcript: str, policy: ConfidencePolicy | None = None) -> tuple[float, dict[str, Any]]:
    policy = policy or default_confidence_policy()
    evidence_score = _evidence_score(note.evidence, transcript)
    durable_score = _durability_score(note.title, note.body, note.evidence)
    if _has_validated_provenance(note) and evidence_score > 0:
        durable_score = max(durable_score, 0.6)
    context_score = _context_score(note.evidence, note)
    validation_score = 1.0 if note.evidence else 0.0
    conflict = _has_conflict(note)
    shallow = _is_shallow(note.title, note.body)
    stt_ambiguity_penalty = _stt_ambiguity_penalty(note.evidence)
    corroboration_score = _corroboration_score(note)
    score = (
        evidence_score * policy.evidence_weight
        + durable_score * policy.explicitness_weight
        + min(1.0, _word_count(note.body) / 42) * policy.completeness_weight
        + context_score * policy.context_weight
        + validation_score * policy.validation_weight
    )
    if conflict:
        score -= policy.conflict_penalty
    if shallow and _word_count(note.body) < 12:
        score -= policy.shallow_penalty
    score -= stt_ambiguity_penalty
    score = _clamp(score)
    trace = {
        "engine": "deterministic-confidence-v1",
        "llmConfidence": note.confidence,
        "evidence": round(evidence_score, 4),
        "durability": round(durable_score, 4),
        "context": round(context_score, 4),
        "corroboration": round(corroboration_score, 4),
        "validation": round(validation_score, 4),
        "conflictPenalty": policy.conflict_penalty if conflict else 0.0,
        "shallowPenalty": policy.shallow_penalty if shallow else 0.0,
        "sttAmbiguityPenalty": round(stt_ambiguity_penalty, 4),
        "score": score,
    }
    return score, trace


def score_simple_item(item: ExtractedDecision | ExtractedIssue, transcript: str, policy: ConfidencePolicy | None = None) -> tuple[float, dict[str, Any]]:
    policy = policy or default_confidence_policy()
    evidence_score = _evidence_score(item.evidence, transcript)
    explicit_score = _explicitness_from_evidence(item.evidence)
    validation_score = 1.0 if item.evidence else 0.0
    conflict = _has_conflict(item)
    score = evidence_score * 0.44 + explicit_score * 0.28 + validation_score * 0.28
    if conflict and not isinstance(item, ExtractedDecision):
        score -= policy.conflict_penalty
    score = _clamp(score)
    return score, {
        "engine": "deterministic-confidence-v1",
        "llmConfidence": item.confidence,
        "evidence": round(evidence_score, 4),
        "explicitness": round(explicit_score, 4),
        "validation": validation_score,
        "conflictPenalty": policy.conflict_penalty if conflict else 0.0,
        "score": score,
    }


def validation_decision_for_task(task: ExtractedTask, transcript: str, policy: ConfidencePolicy | None = None) -> tuple[bool, str]:
    policy = policy or default_confidence_policy()
    reasons = task_rejection_reasons(task, transcript, policy)
    if reasons:
        return False, reasons[0]
    return True, "accepted"


def validation_decision_for_note(note: ExtractedNote, transcript: str, policy: ConfidencePolicy | None = None) -> tuple[bool, str]:
    policy = policy or default_confidence_policy()
    reasons = note_rejection_reasons(note, transcript, policy)
    if reasons:
        return False, reasons[0]
    return True, "accepted"


def task_rejection_reasons(task: ExtractedTask, transcript: str, policy: ConfidencePolicy | None = None) -> list[str]:
    policy = policy or default_confidence_policy()
    reasons: list[str] = []
    if not task.evidence:
        reasons.append("missing_evidence")
    elif not _all_evidence_has_matching_sequence(task.evidence, transcript):
        reasons.append("evidence_sequence_mismatch")
    stripped = _strip_unsupported_task_metadata(task)
    if stripped:
        task.changes = {
            **(task.changes or {}),
            "optionalMetadataInvalid": stripped,
            "optionalMetadataOutcome": "OPTIONAL_METADATA_INVALID",
        }
    if not _has_independent_content(task.title, task.body):
        reasons.append("generic_or_non_actionable_task")
    ungrounded = _ungrounded_quality_reason(task)
    if ungrounded:
        reasons.append(ungrounded)
    if _disconnected_from_validated_units(task):
        reasons.append("disconnected_from_validated_units")
    if _has_speculation(task) and _origin(task) != "explicit":
        reasons.append("speculative_inference")
    score, _ = score_task(task, transcript, policy)
    if score < policy.publish_threshold and "missing_evidence" not in reasons:
        reasons.append("low_confidence")
    return reasons


def note_rejection_reasons(note: ExtractedNote, transcript: str, policy: ConfidencePolicy | None = None) -> list[str]:
    policy = policy or default_confidence_policy()
    reasons: list[str] = []
    if not note.evidence:
        reasons.append("missing_evidence")
    elif not _all_evidence_has_matching_sequence(note.evidence, transcript):
        reasons.append("evidence_sequence_mismatch")
    if not _has_independent_content(note.title, note.body):
        reasons.append("generic_or_template_note")
    ungrounded = _ungrounded_quality_reason(note)
    if ungrounded:
        reasons.append("generic_or_template_note" if ungrounded == "ungrounded_quality_verdict" else ungrounded)
    if _disconnected_from_validated_units(note):
        reasons.append("disconnected_from_validated_units")
    score, _ = score_note(note, transcript, policy)
    if score < policy.publish_threshold and "missing_evidence" not in reasons:
        reasons.append("low_value_or_low_confidence")
    return reasons


def _filter_tasks(
    tasks: list[ExtractedTask], transcript: str, policy: ConfidencePolicy
) -> tuple[list[ExtractedTask], list[ExtractedTask], list[dict[str, Any]]]:
    kept: list[ExtractedTask] = []
    rejected: list[ExtractedTask] = []
    records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        reasons = task_rejection_reasons(task, transcript, policy)
        keep = not reasons
        score, trace = score_task(task, transcript, policy)
        task.confidence = score
        task.changes = {
            **task.changes,
            "confidenceTrace": trace,
            "validatorDecision": "accepted" if keep else reasons[0],
            "qualityRejectionReasons": reasons,
        }
        records.append(_artifact_quality_record(task, "task", index, keep, reasons, score, trace, policy))
        if keep:
            kept.append(task)
        else:
            rejected.append(task)
    return kept, rejected, records


def _filter_notes(
    notes: list[ExtractedNote], transcript: str, policy: ConfidencePolicy
) -> tuple[list[ExtractedNote], list[ExtractedNote], list[dict[str, Any]]]:
    kept: list[ExtractedNote] = []
    rejected: list[ExtractedNote] = []
    records: list[dict[str, Any]] = []
    for index, note in enumerate(notes):
        reasons = note_rejection_reasons(note, transcript, policy)
        keep = not reasons
        score, trace = score_note(note, transcript, policy)
        note.confidence = score
        if hasattr(note, "debug"):
            note.debug = {
                **(note.debug or {}),
                "confidenceTrace": trace,
                "validatorDecision": "accepted" if keep else reasons[0],
                "qualityRejectionReasons": reasons,
            }
        records.append(_artifact_quality_record(note, "note", index, keep, reasons, score, trace, policy))
        if keep:
            kept.append(note)
        else:
            rejected.append(note)
    return kept, rejected, records


def _filter_decisions(items: list[ExtractedDecision], transcript: str, policy: ConfidencePolicy) -> list[ExtractedDecision]:
    kept: list[ExtractedDecision] = []
    for item in items:
        score, _ = score_simple_item(item, transcript, policy)
        item.confidence = score
        if item.evidence and _all_evidence_has_matching_sequence(item.evidence, transcript) and score >= policy.publish_threshold:
            kept.append(item)
    return kept


def _filter_issues(items: list[ExtractedIssue], transcript: str, policy: ConfidencePolicy) -> list[ExtractedIssue]:
    kept: list[ExtractedIssue] = []
    for item in items:
        score, _ = score_simple_item(item, transcript, policy)
        item.confidence = score
        if item.evidence and _all_evidence_has_matching_sequence(item.evidence, transcript) and score >= policy.publish_threshold:
            kept.append(item)
    return kept


def _origin(item: Any) -> Origin:
    value = getattr(item, "origin", None) or getattr(item, "changes", {}).get("origin")
    if value in {"explicit", "strongly_inferred"}:
        return value
    return "unknown"


def _task_explicitness(task: ExtractedTask, origin: Origin) -> float:
    if origin == "explicit":
        return 1.0
    if origin == "strongly_inferred":
        return 0.72
    if _has_validated_provenance(task):
        return 0.72
    return _explicitness_from_evidence(task.evidence)


def _has_validated_provenance(item: Any) -> bool:
    metadata = getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}
    if metadata.get("sourceSemanticUnitIds") or metadata.get("provenanceRestored"):
        return True
    return False


def _disconnected_from_validated_units(item: Any) -> bool:
    metadata = getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}
    if metadata.get("synthesisSource") != "llm":
        return False
    if int(metadata.get("validatedUnitCatalogSize") or 0) <= 0:
        return False
    return not bool(metadata.get("sourceSemanticUnitIds"))


def _corroboration_score(item: Any) -> float:
    evidence = getattr(item, "evidence", []) or []
    metadata = getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}
    unit_count = len(metadata.get("sourceSemanticUnitIds") or [])
    return _clamp(min(1.0, (len(evidence) / 3) + (0.2 * unit_count)))


def _artifact_quality_record(
    item: Any,
    artifact_type: str,
    index: int,
    keep: bool,
    reasons: list[str],
    score: float,
    trace: dict[str, Any],
    policy: ConfidencePolicy,
) -> dict[str, Any]:
    metadata = getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}
    evidence = getattr(item, "evidence", []) or []
    return {
        "artifactId": getattr(item, "artifactId", None) or getattr(item, "fingerprint", None) or f"{artifact_type}-{index}",
        "artifactType": artifact_type,
        "sourceSemanticUnitIds": list(metadata.get("sourceSemanticUnitIds") or []),
        "evidenceSpanCount": len(evidence),
        "evidenceIntegrityScore": round(float(trace.get("evidence") or 0.0), 4),
        "groundingScore": round(float(trace.get("validation") or 0.0), 4),
        "contextCompletenessScore": round(float(trace.get("completeness") or trace.get("durability") or 0.0), 4),
        "corroborationScore": round(float(trace.get("corroboration") or _corroboration_score(item)), 4),
        "ambiguityScore": round(float(trace.get("sttAmbiguityPenalty") or 0.0), 4),
        "consistencyScore": round(1.0 - float(trace.get("conflictPenalty") or 0.0), 4),
        "computedConfidence": score,
        "requiredConfidence": policy.publish_threshold,
        "qualityVerdict": "accepted" if keep else "rejected",
        "qualityRejectionReasons": reasons,
    }


def _evidence_score(evidence: list[EvidenceSpan], transcript: str) -> float:
    if not evidence:
        return 0.0
    exact = sum(1 for span in evidence if _evidence_matches_span(span, transcript))
    count_score = min(1.0, len(evidence) / 3)
    exact_score = exact / len(evidence)
    length_score = min(1.0, sum(_word_count(span.text) for span in evidence) / 36)
    return 0.45 * exact_score + 0.35 * count_score + 0.2 * length_score


def _task_completeness(task: ExtractedTask) -> float:
    score = 0.35
    if _word_count(task.body) >= 12:
        score += 0.25
    if len(task.evidence) >= 2:
        score += 0.2
    if task.changes.get("supportedContext"):
        score += 0.1
    if task.ownerText:
        score += 0.08
    if task.dueDateText or task.dueDateResolved:
        score += 0.08
    if task.existingTaskId or task.artifactId:
        score += 0.06
    return min(1.0, score)


def _durability_score(title: str, body: str, evidence: list[EvidenceSpan]) -> float:
    # Durability is about the amount and structure of grounded information,
    # never the subject being discussed.
    return min(1.0, min(0.70, _word_count(body) / 50) + min(0.30, 0.15 * len(evidence)))


def _context_score(evidence: list[EvidenceSpan], item: Any | None = None) -> float:
    if not evidence:
        return 0.0
    starts = [span.sequenceStart for span in evidence]
    ends = [span.sequenceEnd for span in evidence]
    span_width = max(ends) - min(starts)
    if len(evidence) >= 2 and span_width >= 1:
        score = 1.0
    elif len(evidence) >= 2:
        score = 0.8
    else:
        score = 0.45
    if item is not None and _has_validated_provenance(item):
        score = max(score, 0.8)
    return score


def _explicitness_from_evidence(evidence: list[EvidenceSpan]) -> float:
    # Explicitness is supplied by the semantic classifier as task origin; raw
    # words are not a cross-language proxy for semantic intent.
    return 0.5 if evidence else 0.0


def _has_conflict(item: Any) -> bool:
    # Conflict detection is semantic-model work. Raw-language marker lists
    # break for new languages and should never change persistence decisions.
    return _semantic_flag(item, "semanticConflict")


def _has_speculation(item: Any) -> bool:
    return _semantic_flag(item, "semanticSpeculation")


def _semantic_flag(item: Any, name: str) -> bool:
    metadata = getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}
    return bool(metadata.get(name))


def _stt_ambiguity_penalty(evidence: list[EvidenceSpan]) -> float:
    """Small quality penalty; corroborated semantics are scored separately."""
    text = " ".join(span.text for span in evidence)
    tokens = max(1, _word_count(text))
    corrupted = text.count("Ã") + text.count("�")
    very_short = sum(1 for span in evidence if _word_count(span.text) <= 2)
    ambiguity = min(1.0, corrupted / tokens + 0.15 * very_short)
    return round(0.08 * ambiguity, 4)


def _is_shallow(title: str, body: str) -> bool:
    return _word_count(title) <= 4 and _word_count(body) <= 8


def _evidence_matches(evidence_text: str, transcript: str) -> bool:
    evidence = _normalize(evidence_text)
    haystack = _normalize(transcript)
    if not evidence:
        return False
    if evidence in haystack:
        return True
    words = evidence.split()
    return len(words) >= 5 and any(" ".join(words[index:index + 5]) in haystack for index in range(len(words) - 4))


def _evidence_matches_span(span: EvidenceSpan, transcript: str) -> bool:
    if _evidence_matches_sequence(span, transcript):
        return True
    return False


def _all_evidence_has_matching_sequence(evidence: list[EvidenceSpan], transcript: str) -> bool:
    return bool(evidence) and all(_evidence_matches_sequence(span, transcript) for span in evidence)


def _evidence_matches_sequence(span: EvidenceSpan, transcript: str) -> bool:
    lines = _sequence_texts(transcript)
    if not lines:
        return _evidence_matches(span.text, transcript)
    combined = " ".join(lines.get(sequence, "") for sequence in range(span.sequenceStart, span.sequenceEnd + 1)).strip()
    return _evidence_matches(span.text, combined)


def _sequence_texts(transcript: str) -> dict[int, str]:
    pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\s*\[\d+\]|\Z)", re.DOTALL)
    return {int(match.group(1)): re.sub(r"\s+", " ", match.group(2)).strip() for match in pattern.finditer(transcript or "")}


def _strip_unsupported_task_metadata(task: ExtractedTask) -> list[str]:
    evidence_text = _evidence_blob(task.evidence)
    stripped: list[str] = []
    if task.ownerText and _normalize(task.ownerText) not in _normalize(evidence_text):
        task.ownerText = None
        task.ownerUserId = None
        stripped.append("owner")
    elif task.ownerUserId and not task.ownerText:
        task.ownerUserId = None
        stripped.append("owner")
    if task.dueDateText and _normalize(task.dueDateText) not in _normalize(evidence_text):
        task.dueDateText = None
        task.dueDateResolved = None
        if task.dueDateStatus != "none":
            task.dueDateStatus = "none"
        stripped.append("deadline")
    elif task.dueDateResolved and not task.dueDateText and _normalize(task.dueDateResolved) not in _normalize(evidence_text):
        task.dueDateResolved = None
        stripped.append("deadline")
    priority = str((task.changes or {}).get("priority") or "").strip()
    if priority and _normalize(priority) not in _normalize(evidence_text):
        changes = dict(task.changes or {})
        changes.pop("priority", None)
        task.changes = changes
        stripped.append("priority")
    return stripped


def _unsupported_task_metadata(task: ExtractedTask) -> str | None:
    # Compatibility wrapper: optional metadata is now stripped, not used as a
    # unit-destroying rejection reason.
    stripped = _strip_unsupported_task_metadata(task)
    if "owner" in stripped:
        return "invented_owner"
    if "deadline" in stripped:
        return "invented_deadline"
    if "priority" in stripped:
        return "invented_priority"
    return None


def _is_publish_quality_note(note: ExtractedNote) -> bool:
    return _has_independent_content(note.title, note.body) and _has_grounded_structure(note, minimum_evidence=1)


def _is_publish_quality_task(task: ExtractedTask) -> bool:
    # Task vs Note is a synthesis decision. Quality only checks grounding,
    # independent content, and evidence identity — not a legacy origin label.
    return _has_independent_content(task.title, task.body) and _has_grounded_structure(task, minimum_evidence=1)


def _is_generic_or_classifier_output(title: str, body: str, evidence: list[EvidenceSpan], require_body_detail: bool = True) -> bool:
    # Retained as a compatibility helper for callers outside this module.  It
    # is structural only; quality meaning is evaluated by the semantic model.
    class Candidate:
        pass
    candidate = Candidate()
    candidate.title, candidate.body, candidate.evidence = title, body, evidence
    return not (_has_independent_content(title, body) and _has_grounded_structure(candidate, minimum_evidence=1))


def _has_independent_content(title: str, body: str) -> bool:
    title_text = _normalize(title)
    body_text = _normalize(body)
    if not title_text or not body_text:
        return False
    # A body which merely repeats its heading is circular regardless of the
    # language in which it was written.
    return body_text != title_text and len(body_text) > len(title_text)


def _has_grounded_structure(item: Any, minimum_evidence: int) -> bool:
    evidence = getattr(item, "evidence", []) or []
    if len(evidence) < minimum_evidence:
        return False
    return _ungrounded_quality_reason(item) is None


def _ungrounded_quality_reason(item: Any) -> str | None:
    metadata = getattr(item, "changes", {}) or getattr(item, "debug", {}) or {}
    quality = metadata.get("quality")
    # Optional model quality flags are authoritative only when present. Missing
    # metadata must not zero-out an artifact that already has exact evidence.
    if isinstance(quality, dict) and quality:
        if quality.get("grounded") is False or quality.get("independentlyUseful") is False:
            return "ungrounded_quality_verdict"
    return None


def _evidence_blob(evidence: list[EvidenceSpan]) -> str:
    return " ".join(span.text for span in evidence).casefold()


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)
