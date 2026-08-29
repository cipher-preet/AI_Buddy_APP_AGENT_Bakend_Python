"""Candidate ledger. Merge-only; no semantic deduplication."""

from __future__ import annotations

from services.conversation.meeting_pipeline.schemas import ExtractionWindow, MeetingCandidate


class CandidateLedger:
    def __init__(self) -> None:
        self._candidates: list[MeetingCandidate] = []
        self._ids: set[str] = set()
        self._window_ids: list[str] = []

    @property
    def candidates(self) -> list[MeetingCandidate]:
        return list(self._candidates)

    def replace_window(self, window: ExtractionWindow, candidates: list[MeetingCandidate]) -> None:
        """Idempotent window write: retries replace that window's previous rows."""
        window_id = window.window_id
        self._candidates = [item for item in self._candidates if item.sourceWindowId != window_id]
        self._ids = {item.candidateId for item in self._candidates}
        if window_id not in self._window_ids:
            self._window_ids.append(window_id)
        for candidate in candidates:
            self._append(candidate)

    def add(self, candidate: MeetingCandidate) -> None:
        self._append(candidate)

    def compact_payload(self) -> list[dict]:
        return [
            {
                "candidateId": item.candidateId,
                "kind": item.kind.value if hasattr(item.kind, "value") else str(item.kind),
                "meaning": item.meaning,
                "evidenceSequences": list(item.evidenceSequences),
                "owner": item.owner,
                "dueDate": item.dueDate,
                "sourceWindowId": item.sourceWindowId,
                "sourceWindowIndex": item.sourceWindowIndex,
            }
            for item in self._candidates
        ]

    def by_id(self) -> dict[str, MeetingCandidate]:
        return {item.candidateId: item for item in self._candidates}

    def evidence_union(self, candidate_ids: list[str]) -> list[int]:
        allowed: list[int] = []
        seen: set[int] = set()
        index = self.by_id()
        for candidate_id in candidate_ids:
            candidate = index.get(candidate_id)
            if candidate is None:
                continue
            for sequence in candidate.evidenceSequences:
                if sequence in seen:
                    continue
                seen.add(sequence)
                allowed.append(sequence)
        return allowed

    def _append(self, candidate: MeetingCandidate) -> None:
        if not candidate.candidateId or candidate.candidateId in self._ids:
            return
        self._ids.add(candidate.candidateId)
        self._candidates.append(candidate)
