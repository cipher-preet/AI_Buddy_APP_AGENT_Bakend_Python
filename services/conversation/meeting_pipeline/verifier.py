"""Narrow evidence verifier. Does not discover or expand artifacts."""

from __future__ import annotations

from services.conversation.meeting_pipeline.llm import generate_structured
from services.conversation.meeting_pipeline.schemas import (
    ArtifactClaim,
    FieldSupport,
    MeetingArtifactRepairResponse,
    MeetingVerifierResponse,
    VerifiedArtifact,
    VerifierVerdict,
)
from services.llm.router import LLMCapability, LLMRouter

_BATCH_SIZE = 8
_METADATA_FIELDS = {"owner", "dueDate", "ownerText", "dueDateText"}


class ArtifactEvidenceVerifier:
    def __init__(self, router: LLMRouter):
        self.router = router
        self.calls = 0
        self.last_provider = "none"
        self.last_model = "none"
        self.supported = 0
        self.partial = 0
        self.unsupported = 0
        self.repair_calls = 0

    async def verify(
        self,
        artifacts: list[ArtifactClaim],
        sequence_text: dict[int, str],
        meeting_at=None,
    ) -> list[VerifiedArtifact]:
        self._meeting_at = meeting_at
        verified: list[VerifiedArtifact] = []
        for offset in range(0, len(artifacts), _BATCH_SIZE):
            batch = artifacts[offset : offset + _BATCH_SIZE]
            items = await self._review_batch(batch, sequence_text)
            by_key = {item.artifactKey: item for item in items}
            for claim in batch:
                row = by_key.get(claim.artifactKey)
                verified.append(apply_field_support(_apply_verdict(claim, row)))
        repaired: list[VerifiedArtifact] = []
        for item in verified:
            if item.verdict == VerifierVerdict.SUPPORTED:
                self.supported += 1
                repaired.append(item)
                continue
            if item.verdict == VerifierVerdict.UNSUPPORTED:
                self.unsupported += 1
                repaired.append(item)
                continue
            self.partial += 1
            once = await self._repair_once(item, sequence_text)
            if once.verdict == VerifierVerdict.SUPPORTED:
                self.supported += 1
            elif once.verdict == VerifierVerdict.UNSUPPORTED:
                self.unsupported += 1
            repaired.append(once)
        return repaired

    async def _review_batch(self, batch: list[ArtifactClaim], sequence_text: dict[int, str]) -> list:
        self.calls += 1
        meeting = getattr(self, "_meeting_at", None)
        payload = {
            "meetingTimestamp": meeting.isoformat() if meeting is not None else "",
            "artifacts": [
                {
                    "artifactKey": item.artifactKey,
                    "kind": item.kind,
                    "title": item.title,
                    "body": item.body,
                    "owner": item.owner,
                    "dueDate": item.dueDate,
                    "evidence": {
                        str(sequence): sequence_text.get(sequence, "")
                        for sequence in item.evidenceSequences
                    },
                }
                for item in batch
            ]
        }
        response, provider, model = await generate_structured(
            self.router,
            LLMCapability.VALIDATION,
            "meeting-evidence-verifier-v1",
            MeetingVerifierResponse,
            payload,
            stage="verifier",
        )
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        return list(response.items or [])

    async def _repair_once(self, item: VerifiedArtifact, sequence_text: dict[int, str]) -> VerifiedArtifact:
        stripped = apply_field_support(_strip_unsupported_metadata(item))
        needs_rewrite = any(field not in _METADATA_FIELDS for field in (item.unsupportedFields or []))
        needs_grounding = item.verdict == VerifierVerdict.PARTIALLY_SUPPORTED
        repaired = stripped
        if needs_rewrite or needs_grounding:
            try:
                # Keep repaired owner/dueDate for re-review. fieldSupport from the
                # first pass must not wipe values the repair just grounded.
                repaired = await self._llm_repair(stripped, sequence_text)
            except Exception:
                repaired = stripped
        claim = ArtifactClaim(
            artifactKey=repaired.artifactKey,
            kind=repaired.kind,
            title=repaired.title,
            body=repaired.body,
            owner=repaired.owner,
            dueDate=repaired.dueDate,
            sourceCandidateIds=repaired.sourceCandidateIds,
            evidenceSequences=repaired.evidenceSequences,
        )
        rows = await self._review_batch([claim], sequence_text)
        row = rows[0] if rows else None
        result = apply_field_support(_apply_verdict(claim, row))
        result.repaired = True
        if result.verdict != VerifierVerdict.SUPPORTED:
            result.verdict = VerifierVerdict.UNSUPPORTED
            result.reason = result.reason or "repair_not_supported"
        return result

    async def _llm_repair(self, item: VerifiedArtifact, sequence_text: dict[int, str]) -> VerifiedArtifact:
        self.calls += 1
        self.repair_calls += 1
        payload = {
            "title": item.title,
            "body": item.body,
            "owner": item.owner,
            "dueDate": item.dueDate,
            "unsupportedFields": item.unsupportedFields,
            "fieldSupport": item.fieldSupport.model_dump() if item.fieldSupport else {},
            "evidence": {str(sequence): sequence_text.get(sequence, "") for sequence in item.evidenceSequences},
        }
        response, provider, model = await generate_structured(
            self.router,
            LLMCapability.VALIDATION,
            "meeting-artifact-repair-v1",
            MeetingArtifactRepairResponse,
            payload,
            stage="repair",
        )
        self.last_provider = str(getattr(provider, "name", None) or provider or "unknown")
        self.last_model = str(model or "unknown")
        title = " ".join(str(response.title or item.title).split()) or item.title
        body = " ".join(str(response.body or item.body).split())
        return item.model_copy(
            update={
                "title": title,
                "body": body or item.body,
                "owner": _optional_text(response.owner),
                "dueDate": _optional_text(response.dueDate),
                "repaired": True,
            }
        )


def apply_field_support(item: VerifiedArtifact) -> VerifiedArtifact:
    """Null owner/dueDate unless the verifier explicitly supported those fields."""
    support = item.fieldSupport or FieldSupport()
    owner = item.owner if support.owner is True else None
    due = item.dueDate if support.dueDate is True else None
    return item.model_copy(update={"owner": owner, "dueDate": due, "fieldSupport": support})


def _apply_verdict(claim: ArtifactClaim, row) -> VerifiedArtifact:
    verdict = VerifierVerdict.UNSUPPORTED
    unsupported: list[str] = []
    reason = "missing_verifier_verdict"
    support = FieldSupport()
    if row is not None:
        raw = getattr(row, "verdict", None)
        try:
            verdict = raw if isinstance(raw, VerifierVerdict) else VerifierVerdict(str(raw).upper())
        except ValueError:
            verdict = VerifierVerdict.UNSUPPORTED
        unsupported = [str(field) for field in (getattr(row, "unsupportedFields", None) or []) if str(field).strip()]
        reason = str(getattr(row, "reason", "") or "")
        raw_support = getattr(row, "fieldSupport", None)
        if raw_support is None and isinstance(row, dict):
            raw_support = row.get("fieldSupport")
        if isinstance(raw_support, FieldSupport):
            support = raw_support
        elif isinstance(raw_support, dict):
            support = FieldSupport.model_validate(raw_support)
        else:
            dumped = getattr(raw_support, "model_dump", None)
            if callable(dumped):
                support = FieldSupport.model_validate(dumped())
        if "owner" in {field.casefold() for field in unsupported}:
            support = support.model_copy(update={"owner": False})
        if "duedate" in {field.casefold() for field in unsupported} or "duedatetext" in {field.casefold() for field in unsupported}:
            support = support.model_copy(update={"dueDate": False})
    return VerifiedArtifact(
        kind=claim.kind,
        title=claim.title,
        body=claim.body,
        owner=claim.owner,
        dueDate=claim.dueDate,
        sourceCandidateIds=list(claim.sourceCandidateIds),
        evidenceSequences=list(claim.evidenceSequences),
        verdict=verdict,
        unsupportedFields=unsupported,
        fieldSupport=support,
        reason=reason,
        artifactKey=claim.artifactKey,
    )


def _strip_unsupported_metadata(item: VerifiedArtifact) -> VerifiedArtifact:
    owner = item.owner
    due = item.dueDate
    fields = {field.casefold() for field in (item.unsupportedFields or [])}
    if "owner" in fields or "ownertext" in fields:
        owner = None
    if "duedate" in fields or "duedatetext" in fields:
        due = None
    return item.model_copy(update={"owner": owner, "dueDate": due, "repaired": True})


def _optional_text(value: str | None) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None
