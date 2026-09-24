"""LLM-decided recovery of unpublished intended work as Tasks.

Language-agnostic: no keyword lists. The extractor LLM assigns candidate kinds;
this reviewer LLM decides which unpublished candidates are independently usable
Tasks, and merges paraphrases so the same work is not published twice.

Notes recovery is unchanged and runs after this stage.
"""

from __future__ import annotations

import re

from services.conversation.meeting_pipeline.consolidator import _clip_evidence, _optional_text
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.llm import generate_structured
from services.conversation.meeting_pipeline.observability import log_pipeline
from services.conversation.meeting_pipeline.schemas import (
    ArtifactClaim,
    CandidateKind,
    MeetingCandidate,
    MeetingTaskEligibilityResponse,
)
from services.llm.async_runtime import reraise_if_hard_runtime
from services.llm.router import LLMCapability, LLMRouter

# Kinds the extractor LLM may use for executable work. Not a language word list.
_REVIEW_KINDS = frozenset(
    {
        CandidateKind.ACTION,
        CandidateKind.REQUIREMENT,
        CandidateKind.ISSUE,
    }
)
_TITLE_OVERLAP = 0.72
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_BATCH_SIZE = 16


class TaskEligibilityReviewer:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.last_provider = "none"
        self.last_model = "none"

    async def review(
        self,
        pending: list[MeetingCandidate],
        existing_tasks: list[ArtifactClaim],
        ledger: CandidateLedger,
        sequence_text: dict[int, str],
    ) -> list[ArtifactClaim]:
        if not pending:
            return []
        approved: list[ArtifactClaim] = []
        existing = [item for item in existing_tasks if item.kind == "task"]
        for offset in range(0, len(pending), _BATCH_SIZE):
            batch = pending[offset : offset + _BATCH_SIZE]
            try:
                batch_approved = await self._review_batch(batch, existing + approved, ledger, sequence_text)
            except Exception as error:
                reraise_if_hard_runtime(error)
                log_pipeline(
                    {
                        "event": "task_eligibility_batch_failed",
                        "batchStart": offset,
                        "batchSize": len(batch),
                        "error": str(error)[:400],
                    }
                )
                continue
            for claim in batch_approved:
                if _task_already_published(claim, existing + approved):
                    continue
                approved.append(claim)
        return approved

    async def _review_batch(
        self,
        pending: list[MeetingCandidate],
        existing_tasks: list[ArtifactClaim],
        ledger: CandidateLedger,
        sequence_text: dict[int, str],
    ) -> list[ArtifactClaim]:
        self.calls += 1
        known_ids = {candidate.candidateId for candidate in pending}
        meeting_sequences = set(sequence_text)
        lookup = {
            str(sequence): sequence_text[sequence]
            for candidate in pending
            for sequence in candidate.evidenceSequences
            if sequence in sequence_text
        }
        payload = {
            "existingTasks": [
                {
                    "title": item.title,
                    "description": item.body,
                    "sourceCandidateIds": list(item.sourceCandidateIds),
                }
                for item in existing_tasks
            ],
            "unpublishedCandidates": [
                {
                    "candidateId": candidate.candidateId,
                    "kind": candidate.kind.value if hasattr(candidate.kind, "value") else str(candidate.kind),
                    "meaning": candidate.meaning,
                    "owner": candidate.owner,
                    "dueDate": candidate.dueDate,
                    "evidenceSequences": list(candidate.evidenceSequences),
                }
                for candidate in pending
            ],
            "citedTranscript": lookup,
            "writingContract": {
                "taskTitle": "One concrete action with a specific object.",
                "taskDescription": "2-4 grounded sentences. Never copy the title.",
                "language": "Judge meaning in whatever language the transcript uses. Do not require English.",
                "dedupe": "Do not emit a task that repeats an existing task or another approved task.",
            },
        }
        # HIGH_ACCURACY_REASONING → Krutrim gemma-4-31b-it (31B, temperature 0, json_schema).
        response, provider, model = await generate_structured(
            self.router,
            LLMCapability.HIGH_ACCURACY_REASONING,
            "meeting-task-eligibility-v1",
            MeetingTaskEligibilityResponse,
            payload,
            stage="task_eligibility",
        )
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        approved: list[ArtifactClaim] = []
        for index, item in enumerate(response.tasks or []):
            claim = _eligibility_task_claim(item, index, known_ids, ledger, meeting_sequences)
            if claim is None:
                continue
            if _task_already_published(claim, existing_tasks + approved):
                continue
            approved.append(claim)
        return approved


def unpublished_review_candidates(
    artifacts: list[ArtifactClaim],
    ledger: CandidateLedger,
) -> list[MeetingCandidate]:
    cited_by_tasks = {
        str(candidate_id)
        for item in artifacts
        if item.kind == "task"
        for candidate_id in item.sourceCandidateIds
    }
    pending: list[MeetingCandidate] = []
    for candidate in ledger.candidates:
        if candidate.candidateId in cited_by_tasks:
            continue
        if not _is_review_kind(candidate):
            continue
        if _meaning_already_a_task(candidate.meaning, artifacts):
            continue
        pending.append(candidate)
    return pending


def unique_task_claims(artifacts: list[ArtifactClaim]) -> list[ArtifactClaim]:
    """Drop repeated Tasks only. Notes keep their original order and contents."""
    kept: list[ArtifactClaim] = []
    seen_tasks: list[ArtifactClaim] = []
    for item in artifacts:
        if item.kind == "task":
            if _task_already_published(item, seen_tasks):
                continue
            seen_tasks.append(item)
        kept.append(item)
    return kept


async def recover_unpublished_actions(
    artifacts: list[ArtifactClaim],
    ledger: CandidateLedger,
    sequence_text: dict[int, str],
    *,
    reviewer: TaskEligibilityReviewer | None = None,
) -> tuple[list[ArtifactClaim], dict[str, int]]:
    """Recover intended work the consolidator under-published.

    The reviewer LLM is the only publisher of recovered tasks. No keyword
    fallback and no blind ACTION→Task conversion. Does not modify notes.
    """
    pending = unpublished_review_candidates(artifacts, ledger)
    stats = {
        "pendingActionCandidates": len(pending),
        "recoveredTaskCount": 0,
        "eligibilityReviewed": 0,
        "eligibilityApproved": 0,
        "deterministicFallback": 0,
        "eligibilityFailed": 0,
    }
    if not pending or reviewer is None:
        return list(artifacts), stats

    recovered = list(artifacts)
    approved: list[ArtifactClaim] = []
    stats["eligibilityReviewed"] = 1
    try:
        approved = await reviewer.review(pending, recovered, ledger, sequence_text)
        stats["eligibilityApproved"] = len(approved)
    except Exception as error:
        reraise_if_hard_runtime(error)
        stats["eligibilityFailed"] = 1
        log_pipeline(
            {
                "event": "task_eligibility_failed",
                "pendingActionCandidates": len(pending),
                "error": str(error)[:400],
            }
        )
        return recovered, stats

    for claim in approved:
        if _task_already_published(claim, recovered):
            continue
        recovered.append(claim)
        stats["recoveredTaskCount"] += 1
    return recovered, stats


def _eligibility_task_claim(
    item,
    index: int,
    known_ids: set[str],
    ledger: CandidateLedger,
    meeting_sequences: set[int],
) -> ArtifactClaim | None:
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
    if not body or body == title:
        index_map = ledger.by_id()
        meanings = [
            index_map[candidate_id].meaning
            for candidate_id in source_ids
            if candidate_id in index_map and index_map[candidate_id].meaning
        ]
        body = " ".join(meanings) if meanings else title
    return ArtifactClaim(
        artifactKey=f"task:eligible:{index}:{title}",
        kind="task",
        title=title,
        body=body,
        owner=_optional_text(item.owner),
        dueDate=_optional_text(item.dueDate),
        sourceCandidateIds=source_ids,
        evidenceSequences=evidence,
    )


def _task_already_published(claim: ArtifactClaim, artifacts: list[ArtifactClaim]) -> bool:
    source = set(claim.sourceCandidateIds)
    title = _norm(claim.title)
    blob = _norm(f"{claim.title} {claim.body}")
    for item in artifacts:
        if item.kind != "task":
            continue
        if source and source & set(item.sourceCandidateIds):
            return True
        if title and title == _norm(item.title):
            return True
        other = _norm(f"{item.title} {item.body}")
        if _token_overlap(title, _norm(item.title)) >= _TITLE_OVERLAP:
            return True
        if _token_overlap(blob, other) >= _TITLE_OVERLAP:
            return True
    return False


def _meaning_already_a_task(meaning: str, artifacts: list[ArtifactClaim]) -> bool:
    seed = _norm(meaning)
    if not seed:
        return False
    for item in artifacts:
        if item.kind != "task":
            continue
        if _token_overlap(seed, _norm(item.title)) >= _TITLE_OVERLAP:
            return True
        if _token_overlap(seed, _norm(f"{item.title} {item.body}")) >= _TITLE_OVERLAP:
            return True
    return False


def _is_review_kind(candidate: MeetingCandidate) -> bool:
    kind = candidate.kind
    if isinstance(kind, CandidateKind):
        return kind in _REVIEW_KINDS
    try:
        return CandidateKind(str(kind)) in _REVIEW_KINDS
    except ValueError:
        return False


def _norm(text: str | None) -> str:
    return " ".join(str(text or "").split()).casefold()


def _token_overlap(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokens(text: str | None) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(text or ""):
        folded = token.casefold()
        if not folded:
            continue
        # Keep non-Latin unigrams (CJK and similar). Drop short ASCII particles only.
        if len(folded) > 1 or not folded.isascii():
            tokens.add(folded)
    return tokens
