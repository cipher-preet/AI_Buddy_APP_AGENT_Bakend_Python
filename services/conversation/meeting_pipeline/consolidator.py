"""Global consolidation of the candidate ledger into tasks and notes."""

from __future__ import annotations

from services.conversation.event_pipeline.textutil import token_jaccard
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.llm import generate_structured
from services.conversation.meeting_pipeline.schemas import ArtifactClaim, CandidateKind, MeetingCandidate, MeetingConsolidatorResponse
from services.llm.router import LLMCapability, LLMRouter

_NOTE_KINDS = frozenset(
    {
        CandidateKind.REQUIREMENT,
        CandidateKind.DECISION,
        CandidateKind.FACT,
        CandidateKind.RATIONALE,
        CandidateKind.ISSUE,
        CandidateKind.IDEA,
        CandidateKind.QUESTION,
    }
)
_TITLE_PARAPHRASE = 0.72
_NOTE_PARAPHRASE = 0.55


class GlobalArtifactConsolidator:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.last_provider = "none"
        self.last_model = "none"

    async def consolidate(
        self,
        ledger: CandidateLedger,
        sequence_text: dict[int, str],
    ) -> tuple[list[ArtifactClaim], str, list[str]]:
        self.calls += 1
        if not ledger.candidates:
            return [], "", []
        lookup = {
            str(sequence): sequence_text[sequence]
            for candidate in ledger.candidates
            for sequence in candidate.evidenceSequences
            if sequence in sequence_text
        }
        payload = {
            "candidates": ledger.compact_payload(),
            "citedTranscript": lookup,
        }
        response, provider, model = await generate_structured(
            self.router,
            LLMCapability.FINAL_SYNTHESIS,
            "meeting-artifact-consolidator-v1",
            MeetingConsolidatorResponse,
            payload,
            stage="consolidator",
        )
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        known_ids = set(ledger.by_id())
        meeting_sequences = set(sequence_text)
        artifacts: list[ArtifactClaim] = []
        for index, item in enumerate(response.tasks or []):
            claim = _task_claim(item, index, known_ids, ledger, meeting_sequences)
            if claim is not None:
                artifacts.append(claim)
        for index, item in enumerate(response.notes or []):
            claim = _note_claim(item, index, known_ids, ledger, meeting_sequences)
            if claim is not None:
                artifacts.append(claim)
        topics = [str(topic).strip() for topic in (response.topics or []) if str(topic).strip()]
        return artifacts, str(response.summary or "").strip(), topics


def recover_unpublished_notes(
    artifacts: list[ArtifactClaim],
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> tuple[list[ArtifactClaim], int]:
    """Publish leftover memory as notes when consolidation folded it into tasks."""
    recovered = list(artifacts)
    cited_by_notes = {
        str(candidate_id)
        for item in recovered
        if item.kind == "note"
        for candidate_id in item.sourceCandidateIds
    }
    index = ledger.by_id()
    pending = [
        candidate
        for candidate in ledger.candidates
        if _is_note_kind(candidate) and candidate.candidateId not in cited_by_notes
    ]
    if not any(item.kind == "note" for item in recovered) and not pending:
        pending = _supporting_candidates_from_tasks(recovered, index, cited_by_notes)

    added = 0
    for offset, candidate in enumerate(pending):
        if candidate.candidateId in cited_by_notes:
            continue
        if _memory_already_published(candidate.meaning, recovered):
            continue
        claim = _note_from_candidate(candidate, offset, ledger, meeting_sequences)
        if claim is None:
            continue
        recovered.append(claim)
        cited_by_notes.add(candidate.candidateId)
        added += 1
    return recovered, added


def _task_claim(item, index: int, known_ids: set[str], ledger: CandidateLedger, meeting_sequences: set[int]) -> ArtifactClaim | None:
    title = " ".join(str(item.title or "").split())
    body = " ".join(str(item.description or "").split())
    if not title:
        return None
    source_ids = [str(value) for value in (item.sourceCandidateIds or []) if str(value) in known_ids]
    if not source_ids:
        return None
    evidence = _clip_evidence(item.evidenceSequences, source_ids, ledger, meeting_sequences)
    if not evidence:
        return None
    return ArtifactClaim(
        artifactKey=f"task:{index}:{title}",
        kind="task",
        title=title,
        body=body,
        owner=_optional_text(item.owner),
        dueDate=_optional_text(item.dueDate),
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _note_claim(item, index: int, known_ids: set[str], ledger: CandidateLedger, meeting_sequences: set[int]) -> ArtifactClaim | None:
    title = " ".join(str(item.title or "").split())
    body = " ".join(str(item.body or "").split()) or title
    if not title or not body:
        return None
    source_ids = [str(value) for value in (item.sourceCandidateIds or []) if str(value) in known_ids]
    if not source_ids:
        return None
    evidence = _clip_evidence(item.evidenceSequences, source_ids, ledger, meeting_sequences)
    if not evidence:
        return None
    return ArtifactClaim(
        artifactKey=f"note:{index}:{title}",
        kind="note",
        title=title,
        body=body,
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _note_from_candidate(
    candidate: MeetingCandidate,
    offset: int,
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> ArtifactClaim | None:
    meaning = " ".join(str(candidate.meaning or "").split())
    title = _title_from_meaning(meaning)
    if not title or not meaning:
        return None
    source_ids = [candidate.candidateId]
    evidence = _clip_evidence(candidate.evidenceSequences, source_ids, ledger, meeting_sequences)
    if not evidence:
        return None
    return ArtifactClaim(
        artifactKey=f"note:recovered:{offset}:{title}",
        kind="note",
        title=title,
        body=meaning,
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _clip_evidence(values, source_ids: list[str], ledger: CandidateLedger, meeting_sequences: set[int]) -> list[int]:
    if not source_ids:
        return []
    allowed = set(ledger.evidence_union(source_ids))
    provided = list(values or [])
    if not provided:
        return [sequence for sequence in ledger.evidence_union(source_ids) if sequence in meeting_sequences]
    sequences: list[int] = []
    seen: set[int] = set()
    for value in provided:
        try:
            sequence = int(value)
        except (TypeError, ValueError):
            continue
        if sequence not in meeting_sequences or sequence not in allowed or sequence in seen:
            continue
        seen.add(sequence)
        sequences.append(sequence)
    return sequences


def _supporting_candidates_from_tasks(
    artifacts: list[ArtifactClaim],
    index: dict[str, MeetingCandidate],
    cited_by_notes: set[str],
) -> list[MeetingCandidate]:
    pending: list[MeetingCandidate] = []
    seen: set[str] = set()
    for item in artifacts:
        if item.kind != "task":
            continue
        cited = [index[candidate_id] for candidate_id in item.sourceCandidateIds if candidate_id in index]
        if len(cited) < 2:
            continue
        primary = max(cited, key=lambda candidate: token_jaccard(candidate.meaning, item.title))
        for candidate in cited:
            if candidate.candidateId == primary.candidateId:
                continue
            if candidate.candidateId in cited_by_notes or candidate.candidateId in seen:
                continue
            seen.add(candidate.candidateId)
            pending.append(candidate)
    return pending


def _memory_already_published(meaning: str, artifacts: list[ArtifactClaim]) -> bool:
    for item in artifacts:
        if item.kind == "task":
            if token_jaccard(meaning, item.title) >= _TITLE_PARAPHRASE:
                return True
            continue
        blob = f"{item.title} {item.body}".strip()
        if token_jaccard(meaning, item.title) >= _TITLE_PARAPHRASE or token_jaccard(meaning, blob) >= _NOTE_PARAPHRASE:
            return True
    return False


def _is_note_kind(candidate: MeetingCandidate) -> bool:
    kind = candidate.kind
    if isinstance(kind, CandidateKind):
        return kind in _NOTE_KINDS
    try:
        return CandidateKind(str(kind)) in _NOTE_KINDS
    except ValueError:
        return False


def _title_from_meaning(meaning: str) -> str:
    text = " ".join(str(meaning or "").split())
    if not text:
        return ""
    if len(text) <= 80:
        return text
    for separator in (". ", "। ", "? ", "! "):
        index = text.find(separator)
        if 12 <= index <= 80:
            return text[:index]
    clipped = text[:80]
    if " " in clipped:
        return clipped.rsplit(" ", 1)[0]
    return clipped


def _optional_text(value: str | None) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None
