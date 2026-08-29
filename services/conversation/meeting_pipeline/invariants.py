"""Deterministic persistence gates. No semantic reconstruction."""

from __future__ import annotations

from services.conversation.meeting_pipeline.schemas import VerifiedArtifact, VerifierVerdict
from services.conversation.meeting_pipeline.ledger import CandidateLedger


def persistence_ready(
    artifact: VerifiedArtifact,
    *,
    ledger: CandidateLedger,
    sequence_text: dict[int, str],
    conversation_id: str,
) -> tuple[bool, str]:
    if artifact.verdict != VerifierVerdict.SUPPORTED:
        return False, "verifier_not_supported"
    if not artifact.title or not str(artifact.title).strip():
        return False, "missing_title"
    if artifact.kind == "note" and not str(artifact.body or "").strip():
        return False, "missing_note_body"
    evidence = list(artifact.evidenceSequences or [])
    if not evidence:
        return False, "missing_evidence"
    known_sequences = set(sequence_text)
    if any(sequence not in known_sequences for sequence in evidence):
        return False, "fabricated_or_foreign_sequence"
    known_ids = set(ledger.by_id())
    if not artifact.sourceCandidateIds:
        return False, "missing_source_candidate_ids"
    if any(candidate_id not in known_ids for candidate_id in artifact.sourceCandidateIds):
        return False, "unknown_source_candidate"
    allowed = set(ledger.evidence_union(list(artifact.sourceCandidateIds)))
    if any(sequence not in allowed for sequence in evidence):
        return False, "evidence_not_from_source_candidates"
    if conversation_id and any(sequence < 0 for sequence in evidence):
        return False, "invalid_sequence"
    return True, "ok"


def apply_invariant_gate(
    artifacts: list[VerifiedArtifact],
    *,
    ledger: CandidateLedger,
    sequence_text: dict[int, str],
    conversation_id: str,
) -> tuple[list[VerifiedArtifact], list[VerifiedArtifact]]:
    accepted: list[VerifiedArtifact] = []
    rejected: list[VerifiedArtifact] = []
    for artifact in artifacts:
        ok, reason = persistence_ready(
            artifact,
            ledger=ledger,
            sequence_text=sequence_text,
            conversation_id=conversation_id,
        )
        if ok:
            accepted.append(artifact)
            continue
        rejected.append(artifact.model_copy(update={"reason": artifact.reason or reason, "verdict": VerifierVerdict.UNSUPPORTED}))
    return accepted, rejected
