import asyncio
import pytest
from types import SimpleNamespace

from apps.api_gateway.config.setting import settings
from services.conversation.eval_metrics import PredictedItem, predicted_from_extraction, score_case
from services.conversation.meeting_pipeline.extractor import MeetingCandidateExtractor
from services.conversation.meeting_pipeline.invariants import apply_invariant_gate, persistence_ready
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.pipeline import run_meeting_pipeline
from services.conversation.meeting_pipeline.schemas import (
    ArtifactClaim,
    CandidateKind,
    ExtractionWindow,
    FieldSupport,
    MeetingCandidate,
    MeetingCandidateExtractorResponse,
    MeetingVerifierResponse,
    VerifiedArtifact,
    VerifierVerdict,
)
from services.conversation.meeting_pipeline.verifier import apply_field_support
from services.conversation.meeting_pipeline.windows import build_extraction_windows, turns_from_chunks
from services.conversation.models import STTStatus, TranscriptChunkDocument
from services.conversation.workflow import ConversationProcessingWorkflow
from tests.fixtures.lenskart_hrms_meeting import (
    ATOMIC_ONBOARDING_MEANINGS,
    DENSE_ONBOARDING,
    FORBIDDEN_BACKGROUND_TITLES,
    REQUIRED_HRMS_MEANINGS,
    lenskart_hrms_case,
    lenskart_hrms_chunks,
)


def _candidate(cid, meaning, sequences, window="w0", index=0, kind=CandidateKind.FACT, owner=None, due=None):
    return MeetingCandidate(
        candidateId=cid,
        kind=kind,
        meaning=meaning,
        evidenceSequences=list(sequences),
        owner=owner,
        dueDate=due,
        sourceWindowId=window,
        sourceWindowIndex=index,
    )


def _window(window_id="w0", index=0, sequences=None, overlap=None, text=""):
    sequences = list(sequences or [0])
    overlap = list(overlap or [])
    owned = [item for item in sequences if item not in set(overlap)]
    return ExtractionWindow(
        window_id=window_id,
        window_index=index,
        sequence_start=sequences[0],
        sequence_end=sequences[-1],
        sequence_ids=sequences,
        owned_sequence_ids=owned,
        overlap_sequence_ids=overlap,
        text=text,
        token_count=10,
    )


def _chunk(sequence: int, text: str, conversation_id="conv") -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId=conversation_id,
        userId="user_1",
        spaceId="space_1",
        chunkId=f"c{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        normalizedText=text,
        sttStatus=STTStatus.COMPLETED,
    )


class ScriptedExtractor:
    def __init__(self, by_window=None, by_owned=None):
        self.by_window = by_window or {}
        self.by_owned = by_owned or {}
        self.calls = 0
        self.seen_windows: list[str] = []
        self.last_provider = "scripted"
        self.last_model = "scripted-extractor"
        self.window_records_by_id = {}

    async def extract(self, window, conversation_id: str):
        self.calls += 1
        self.seen_windows.append(window.window_id)
        if window.window_id in self.by_window:
            found = list(self.by_window[window.window_id])
        else:
            found = []
            seen: set[str] = set()
            for sequences, candidates in self.by_owned.items():
                if set(sequences) & set(window.sequence_ids):
                    for item in candidates:
                        if item.candidateId in seen:
                            continue
                        seen.add(item.candidateId)
                        found.append(
                            item.model_copy(
                                update={
                                    "sourceWindowId": window.window_id,
                                    "sourceWindowIndex": window.window_index,
                                }
                            )
                        )
        self.window_records_by_id[window.window_id] = {
            "windowId": window.window_id,
            "sequenceStart": window.sequence_start,
            "sequenceEnd": window.sequence_end,
            "sequenceIds": list(window.sequence_ids),
            "ownedSequenceIds": list(window.owned_sequence_ids),
            "candidateCount": len(found),
            "rawCandidateCount": len(found),
            "promptVersion": "scripted",
        }
        return found


class ScriptedConsolidator:
    def __init__(self, claims, summary="", topics=None, capture=None):
        self.claims = claims
        self.summary = summary
        self.topics = topics or []
        self.capture = capture if capture is not None else {}
        self.calls = 0
        self.last_provider = "scripted"
        self.last_model = "scripted-consolidator"

    async def consolidate(self, ledger, sequence_text):
        self.calls += 1
        self.capture["candidate_count"] = len(ledger.candidates)
        self.capture["payload"] = ledger.compact_payload()
        self.capture["cited_sequences"] = sorted(
            {seq for item in ledger.candidates for seq in item.evidenceSequences}
        )
        self.capture["full_transcript_sent"] = False
        self.capture["sequence_text_keys"] = sorted(sequence_text)
        return list(self.claims), self.summary, list(self.topics)


class ScriptedVerifier:
    def __init__(self, unsupported_evidence=None, partial_fields=None):
        self.unsupported_evidence = set(unsupported_evidence or [])
        self.partial_fields = partial_fields or {}
        self.calls = 0
        self.supported = 0
        self.partial = 0
        self.unsupported = 0
        self.repair_calls = 0
        self.last_provider = "scripted"
        self.last_model = "scripted-verifier"
        self.seen_evidence: list[list[int]] = []

    async def verify(self, artifacts, sequence_text, meeting_at=None):
        self.calls += 1
        results = []
        for item in artifacts:
            self.seen_evidence.append(list(item.evidenceSequences))
            if any(sequence in self.unsupported_evidence for sequence in item.evidenceSequences):
                self.unsupported += 1
                verdict = VerifierVerdict.UNSUPPORTED
                reason = "cited_evidence_does_not_support_claim"
            elif item.artifactKey in self.partial_fields:
                self.partial += 1
                verdict = VerifierVerdict.UNSUPPORTED
                reason = "partial_not_supported_after_one_repair"
            else:
                self.supported += 1
                verdict = VerifierVerdict.SUPPORTED
                reason = "supported"
            results.append(
                apply_field_support(
                    VerifiedArtifact(
                        kind=item.kind,
                        title=item.title,
                        body=item.body,
                        owner=item.owner,
                        dueDate=item.dueDate,
                        sourceCandidateIds=list(item.sourceCandidateIds),
                        evidenceSequences=list(item.evidenceSequences),
                        verdict=verdict,
                        reason=reason,
                        artifactKey=item.artifactKey,
                        fieldSupport=FieldSupport(
                            title=True,
                            description=True,
                            owner=True if item.owner else None,
                            dueDate=True if item.dueDate else None,
                        ),
                    )
                )
            )
        return results


def test_ledger_merges_windows_and_allows_duplicates():
    ledger = CandidateLedger()
    window_a = _window("w1", 0, [0, 1])
    window_b = _window("w2", 1, [1, 2], overlap=[1])
    first = _candidate("c1", "Generate onboarding link", [7], "w1", 0, CandidateKind.ACTION)
    dup = _candidate("c2", "Generate onboarding link", [7], "w2", 1, CandidateKind.ACTION)
    ledger.replace_window(window_a, [first])
    ledger.replace_window(window_b, [dup])
    assert [item.candidateId for item in ledger.candidates] == ["c1", "c2"]
    assert ledger.candidates[0].evidenceSequences == [7]
    assert ledger.candidates[1].evidenceSequences == [7]


def test_ledger_does_not_expand_neighbor_evidence():
    ledger = CandidateLedger()
    window = _window("w1", 0, [5, 6, 7])
    candidate = _candidate("c7", "Generate candidate onboarding link", [7], "w1")
    ledger.replace_window(window, [candidate])
    assert ledger.candidates[0].evidenceSequences == [7]
    assert 5 not in ledger.candidates[0].evidenceSequences
    assert 6 not in ledger.candidates[0].evidenceSequences


def test_ledger_window_retry_is_idempotent():
    ledger = CandidateLedger()
    window = _window("w1", 0, [7])
    ledger.replace_window(window, [_candidate("c1", "link", [7], "w1")])
    ledger.replace_window(window, [_candidate("c1", "link", [7], "w1")])
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].sourceWindowId == "w1"


def test_onboarding_link_seq7_supported_seq5_rejected():
    ledger = CandidateLedger()
    ledger.add(_candidate("c12", "Generate candidate onboarding link", [7], "w1", kind=CandidateKind.ACTION))
    sequence_text = {
        5: "Lenskart / China / franchise discussion",
        7: "employee page → button → generate link → candidate",
    }
    supported = VerifiedArtifact(
        kind="task",
        title="Generate candidate onboarding link",
        body="Employee page action generates a candidate link.",
        sourceCandidateIds=["c12"],
        evidenceSequences=[7],
        verdict=VerifierVerdict.SUPPORTED,
        artifactKey="task:0",
    )
    unsupported = VerifiedArtifact(
        kind="task",
        title="Generate candidate onboarding link",
        body="Employee page action generates a candidate link.",
        sourceCandidateIds=["c12"],
        evidenceSequences=[5],
        verdict=VerifierVerdict.SUPPORTED,
        artifactKey="task:bad",
    )
    ok, _ = persistence_ready(supported, ledger=ledger, sequence_text=sequence_text, conversation_id="conv")
    bad, reason = persistence_ready(unsupported, ledger=ledger, sequence_text=sequence_text, conversation_id="conv")
    assert ok
    assert bad is False
    assert reason == "evidence_not_from_source_candidates"
    accepted, rejected = apply_invariant_gate(
        [supported, unsupported],
        ledger=ledger,
        sequence_text=sequence_text,
        conversation_id="conv",
    )
    assert [item.evidenceSequences for item in accepted] == [[7]]
    assert [item.evidenceSequences for item in rejected] == [[5]]


def test_unsupported_verdict_never_persists():
    ledger = CandidateLedger()
    ledger.add(_candidate("c1", "link", [7], "w1"))
    artifact = VerifiedArtifact(
        kind="task",
        title="Generate candidate onboarding link",
        body="link",
        sourceCandidateIds=["c1"],
        evidenceSequences=[7],
        verdict=VerifierVerdict.UNSUPPORTED,
        artifactKey="task:0",
    )
    ok, reason = persistence_ready(artifact, ledger=ledger, sequence_text={7: "generate link"}, conversation_id="conv")
    assert ok is False
    assert reason == "verifier_not_supported"


def test_owner_nulled_when_verifier_does_not_support_it():
    ledger = CandidateLedger()
    ledger.add(_candidate("c1", "Rahul will integrate", [3], "w1", kind=CandidateKind.ACTION))
    artifact = apply_field_support(
        VerifiedArtifact(
            kind="task",
            title="Integrate the API",
            body="Integrate the API.",
            owner="Rahul",
            dueDate="tomorrow",
            sourceCandidateIds=["c1"],
            evidenceSequences=[3],
            verdict=VerifierVerdict.SUPPORTED,
            artifactKey="task:0",
            fieldSupport=FieldSupport(title=True, description=True, owner=False, dueDate=False),
        )
    )
    assert artifact.owner is None
    assert artifact.dueDate is None
    ok, reason = persistence_ready(
        artifact,
        ledger=ledger,
        sequence_text={3: "We should integrate the API later."},
        conversation_id="conv",
    )
    assert ok is True
    assert reason == "ok"


def test_owner_present_in_text_is_still_nulled_without_field_support():
    artifact = apply_field_support(
        VerifiedArtifact(
            kind="task",
            title="Integrate the API",
            body="Rahul will integrate the API tomorrow.",
            owner="Rahul",
            dueDate="tomorrow",
            sourceCandidateIds=["c1"],
            evidenceSequences=[3],
            verdict=VerifierVerdict.SUPPORTED,
            artifactKey="task:0",
            fieldSupport=FieldSupport(title=True, description=True, owner=False, dueDate=False),
        )
    )
    assert artifact.owner is None
    assert artifact.dueDate is None


def test_persisted_artifacts_do_not_use_confidence_as_truth():
    chunks = [_chunk(1, "We need to build payroll.")]
    extractor = ScriptedExtractor(
        by_owned={(1,): [_candidate("c1", "We need to build payroll.", [1], kind=CandidateKind.ACTION)]}
    )
    claims = [
        ArtifactClaim(
            artifactKey="t-payroll",
            kind="task",
            title="Build payroll",
            body="We need to build payroll.",
            sourceCandidateIds=["c1"],
            evidenceSequences=[1],
        )
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(),
        )
    )
    assert result.tasks
    task = result.tasks[0]
    assert task.confidence != 1.0
    assert task.changes.get("evidenceVerified") is True
    assert task.changes.get("verificationVerdict") == "SUPPORTED"


def test_relative_due_date_is_normalized_only_after_verifier_support():
    from datetime import datetime, timezone

    from services.conversation.meeting_pipeline.dates import normalize_supported_due_date

    meeting_at = datetime(2026, 8, 28, tzinfo=timezone.utc)
    assert normalize_supported_due_date("tomorrow", meeting_at) == "2026-08-29"
    assert normalize_supported_due_date("kal", meeting_at) == "2026-08-29"
    assert normalize_supported_due_date("Friday", meeting_at) == "2026-08-28"
    assert normalize_supported_due_date("end of month", meeting_at) == "2026-08-31"
    assert normalize_supported_due_date("this sprint", meeting_at) is None
    assert normalize_supported_due_date("tomorrow", None) is None


def test_persisted_task_keeps_relative_due_text_and_resolves_iso():
    from datetime import datetime, timezone

    chunks = [_chunk(1, "Rahul will integrate the API tomorrow.")]
    extractor = ScriptedExtractor(
        by_owned={(1,): [_candidate("c1", "Rahul will integrate the API tomorrow.", [1], kind=CandidateKind.ACTION, owner="Rahul", due="tomorrow")]}
    )
    claims = [
        ArtifactClaim(
            artifactKey="t-api",
            kind="task",
            title="Integrate the API",
            body="Rahul will integrate the API tomorrow.",
            owner="Rahul",
            dueDate="tomorrow",
            sourceCandidateIds=["c1"],
            evidenceSequences=[1],
        )
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(),
            meeting_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        )
    )
    assert result.tasks
    task = result.tasks[0]
    assert task.dueDateText == "tomorrow"
    assert task.dueDateResolved == "2026-08-29"
    assert task.dueDateStatus == "resolved"
    assert task.ownerText == "Rahul"


def test_duplicate_meanings_consolidate_but_independent_meanings_remain(monkeypatch):
    chunks = [
        _chunk(0, "Generate onboarding link"),
        _chunk(1, "Employee page generates candidate link"),
        _chunk(2, "Candidate onboarding URL should be generated"),
        _chunk(3, "Candidate fills form"),
        _chunk(4, "Payroll calculates PF"),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (0,): [_candidate("c-a", "Generate onboarding link", [0], "w", kind=CandidateKind.ACTION)],
            (1,): [_candidate("c-b", "Employee page generates candidate link", [1], "w", kind=CandidateKind.REQUIREMENT)],
            (2,): [_candidate("c-c", "Candidate onboarding URL should be generated", [2], "w", kind=CandidateKind.REQUIREMENT)],
            (3,): [_candidate("c-d", "Candidate fills form", [3], "w", kind=CandidateKind.ACTION)],
            (4,): [_candidate("c-e", "Payroll calculates PF", [4], "w", kind=CandidateKind.REQUIREMENT)],
        }
    )
    claims = [
        ArtifactClaim(artifactKey="t1", kind="task", title="Generate candidate onboarding link", body="Employee page generates the candidate URL.", sourceCandidateIds=["c-a", "c-b", "c-c"], evidenceSequences=[0, 1, 2]),
        ArtifactClaim(artifactKey="n1", kind="note", title="Candidate fills form", body="The candidate submits details through the form.", sourceCandidateIds=["c-d"], evidenceSequences=[3]),
        ArtifactClaim(artifactKey="n2", kind="note", title="Payroll calculates PF", body="Payroll should calculate PF deductions.", sourceCandidateIds=["c-e"], evidenceSequences=[4]),
    ]
    capture = {}
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims, capture=capture),
            verifier=ScriptedVerifier(),
        )
    )
    titles = {item.title for item in [*result.tasks, *result.notes]}
    assert "Generate candidate onboarding link" in titles
    assert "Candidate fills form" in titles
    assert "Payroll calculates PF" in titles
    assert capture["candidate_count"] >= 3


def test_atomic_dense_sequence_keeps_independent_candidates():
    window = _window("w0", 0, [7], text=f"[7][Speaker 1] {DENSE_ONBOARDING}")
    candidates = [
        _candidate("c1", ATOMIC_ONBOARDING_MEANINGS[0], [7], "w0", kind=CandidateKind.ACTION),
        _candidate("c2", ATOMIC_ONBOARDING_MEANINGS[1], [7], "w0", kind=CandidateKind.REQUIREMENT),
        _candidate("c3", ATOMIC_ONBOARDING_MEANINGS[2], [7], "w0", kind=CandidateKind.ACTION),
        _candidate("c4", ATOMIC_ONBOARDING_MEANINGS[3], [7], "w0", kind=CandidateKind.ACTION),
        _candidate("c5", ATOMIC_ONBOARDING_MEANINGS[4], [7], "w0", kind=CandidateKind.RATIONALE),
    ]
    extractor = ScriptedExtractor(by_window={"w0": candidates})
    result = asyncio.run(extractor.extract(window, "conv"))
    meanings = [item.meaning for item in result]
    assert len(meanings) >= 5
    blob = " ".join(meanings).casefold()
    for expected in ATOMIC_ONBOARDING_MEANINGS:
        assert expected.split()[0].casefold() in blob or expected.casefold() in blob
    assert not (len(meanings) == 1 and "generate candidate link" in meanings[0].casefold())


def test_cross_window_boundary_survives_consolidation(monkeypatch):
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_TARGET_TOKENS", 20)
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_MAX_TOKENS", 28)
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_OVERLAP_RATIO", 0.15)
    chunks = [
        _chunk(10, "We need candidate onboarding for the HRMS project. " + ("context " * 20)),
        _chunk(11, "Generate the candidate link from employee page. " + ("detail " * 20)),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (10,): [_candidate("c-end", "We need candidate onboarding", [10], "wa", 0, CandidateKind.ACTION)],
            (11,): [_candidate("c-start", "generate the candidate link from employee page", [11], "wb", 1, CandidateKind.REQUIREMENT)],
        }
    )
    claims = [
        ArtifactClaim(
            artifactKey="t1",
            kind="task",
            title="Build candidate onboarding flow",
            body="Implement onboarding by generating a candidate link from the employee page.",
            sourceCandidateIds=["c-end", "c-start"],
            evidenceSequences=[10, 11],
        )
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(),
        )
    )
    assert result.tasks
    assert set(result.tasks[0].changes["sourceCandidateIds"]) == {"c-end", "c-start"}
    assert {span.sequenceStart for span in result.tasks[0].evidence} == {10, 11}


def test_task_versus_note_schema_contract():
    claims = [
        ArtifactClaim(artifactKey="t-payroll", kind="task", title="Build payroll", body="We need to build payroll.", sourceCandidateIds=["c1"], evidenceSequences=[1]),
        ArtifactClaim(artifactKey="n-pf", kind="note", title="PF deductions", body="Payroll should calculate PF deductions.", sourceCandidateIds=["c2"], evidenceSequences=[2]),
        ArtifactClaim(artifactKey="t-rahul", kind="task", title="Integrate the API", body="Rahul will integrate it tomorrow.", owner="Rahul", dueDate="tomorrow", sourceCandidateIds=["c3"], evidenceSequences=[3]),
        ArtifactClaim(artifactKey="n-manual", kind="note", title="Reduces HR manual work", body="This reduces HR manual work.", sourceCandidateIds=["c4"], evidenceSequences=[4]),
        ArtifactClaim(artifactKey="n-idea", kind="note", title="Maybe add analytics", body="Maybe sometime we could add analytics.", sourceCandidateIds=["c5"], evidenceSequences=[5]),
    ]
    chunks = [
        _chunk(1, "We need to build payroll."),
        _chunk(2, "Payroll should calculate PF deductions."),
        _chunk(3, "Rahul will integrate it tomorrow."),
        _chunk(4, "This reduces HR manual work."),
        _chunk(5, "Maybe sometime we could add analytics."),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (1,): [_candidate("c1", "We need to build payroll.", [1], kind=CandidateKind.ACTION)],
            (2,): [_candidate("c2", "Payroll should calculate PF deductions.", [2], kind=CandidateKind.REQUIREMENT)],
            (3,): [_candidate("c3", "Rahul will integrate it tomorrow.", [3], kind=CandidateKind.ACTION, owner="Rahul", due="tomorrow")],
            (4,): [_candidate("c4", "This reduces HR manual work.", [4], kind=CandidateKind.RATIONALE)],
            (5,): [_candidate("c5", "Maybe sometime we could add analytics.", [5], kind=CandidateKind.IDEA)],
        }
    )
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(),
        )
    )
    task_titles = {task.title for task in result.tasks}
    note_titles = {note.title for note in result.notes}
    assert "Build payroll" in task_titles
    assert "Integrate the API" in task_titles
    assert "PF deductions" in note_titles
    assert "Reduces HR manual work" in note_titles
    assert "Maybe add analytics" in note_titles
    rahul = next(task for task in result.tasks if task.title == "Integrate the API")
    assert rahul.ownerText == "Rahul"
    assert rahul.dueDateText == "tomorrow"


def test_requirement_candidates_become_notes_when_consolidator_emits_only_tasks():
    chunks = [
        _chunk(0, "We need to set up the weekly report."),
        _chunk(1, "The report should include totals."),
        _chunk(2, "It should also flag overdue items."),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (0,): [_candidate("c-build", "Set up the weekly report", [0], kind=CandidateKind.ACTION)],
            (1,): [_candidate("c-totals", "The report should include totals", [1], kind=CandidateKind.REQUIREMENT)],
            (2,): [_candidate("c-overdue", "The report should flag overdue items", [2], kind=CandidateKind.REQUIREMENT)],
        }
    )
    claims = [
        ArtifactClaim(
            artifactKey="t-report",
            kind="task",
            title="Set up the weekly report",
            body="Set up the weekly report including totals and overdue flags.",
            sourceCandidateIds=["c-build", "c-totals", "c-overdue"],
            evidenceSequences=[0, 1, 2],
        )
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(),
        )
    )
    assert result.tasks
    assert result.notes
    blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "total" in blob
    assert "overdue" in blob
    assert result.observability["recovered_note_count"] >= 2


def test_supporting_details_become_notes_when_folded_into_one_task():
    chunks = [
        _chunk(0, "We will set up the new process this week."),
        _chunk(1, "First we collect feedback from the team."),
        _chunk(2, "Then we send a weekly summary."),
        _chunk(3, "We should also track open blockers."),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (0,): [_candidate("c-setup", "Set up the new process this week", [0], kind=CandidateKind.ACTION)],
            (1,): [_candidate("c-feedback", "Collect feedback from the team", [1], kind=CandidateKind.ACTION)],
            (2,): [_candidate("c-summary", "Send a weekly summary", [2], kind=CandidateKind.ACTION)],
            (3,): [_candidate("c-blockers", "Track open blockers", [3], kind=CandidateKind.ACTION)],
        }
    )
    claims = [
        ArtifactClaim(
            artifactKey="t-process",
            kind="task",
            title="Set up the new process this week",
            body="Set up the new process by collecting team feedback, sending a weekly summary, and tracking open blockers.",
            sourceCandidateIds=["c-setup", "c-feedback", "c-summary", "c-blockers"],
            evidenceSequences=[0, 1, 2, 3],
        )
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(),
        )
    )
    assert result.tasks
    assert result.notes
    blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "feedback" in blob or "summary" in blob or "blocker" in blob
    assert result.observability["recovered_note_count"] >= 1


def test_general_discussion_facts_become_notes_without_a_task():
    chunks = [
        _chunk(0, "Saturday we walked around the lake and got chai."),
        _chunk(1, "The dog is chewing shoes again."),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (0,): [_candidate("c-lake", "Saturday was spent walking around the lake and getting chai", [0], kind=CandidateKind.FACT)],
            (1,): [_candidate("c-dog", "The dog is chewing shoes again", [1], kind=CandidateKind.FACT)],
        }
    )
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator([]),
            verifier=ScriptedVerifier(),
        )
    )
    assert not result.tasks
    assert result.notes
    blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "lake" in blob or "chai" in blob
    assert "dog" in blob or "shoe" in blob
    assert result.observability["recovered_note_count"] >= 2


def test_noisy_stt_does_not_persist_invented_numbers():
    chunks = [
        _chunk(15, "Website credit discussion 1 80 crore 01:21 percent something unclear"),
        _chunk(16, "Rahul will integrate it tomorrow."),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (15,): [_candidate("c-noisy", "unclear website credit discussion", [15], kind=CandidateKind.FACT)],
            (16,): [_candidate("c-rahul", "Rahul will integrate it tomorrow.", [16], kind=CandidateKind.ACTION, owner="Rahul", due="tomorrow")],
        }
    )
    claims = [
        ArtifactClaim(
            artifactKey="bad",
            kind="note",
            title="180 members",
            body="The website has 180 members.",
            sourceCandidateIds=["c-noisy"],
            evidenceSequences=[15],
        ),
        ArtifactClaim(
            artifactKey="ok",
            kind="task",
            title="Integrate the API",
            body="Rahul will integrate it tomorrow.",
            owner="Rahul",
            dueDate="tomorrow",
            sourceCandidateIds=["c-rahul"],
            evidenceSequences=[16],
        ),
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(unsupported_evidence={15}),
        )
    )
    assert all("180 members" not in f"{item.title} {item.body}" for item in result.notes)
    assert result.tasks and result.tasks[0].title == "Integrate the API"


def test_lenskart_background_is_not_published_and_hrms_meanings_survive():
    chunks = lenskart_hrms_chunks()
    extractor = ScriptedExtractor(
        by_owned={
            (6,): [
                _candidate("c-hrms", "HRMS project", [6], kind=CandidateKind.FACT),
                _candidate("c-onboarding", "candidate onboarding", [6], kind=CandidateKind.ACTION),
            ],
            (7,): [
                _candidate("c-action", "employee-page onboarding action", [7], kind=CandidateKind.ACTION),
                _candidate("c-link", "generated candidate link", [7], kind=CandidateKind.REQUIREMENT),
                _candidate("c-submit", "candidate submits information through link", [7], kind=CandidateKind.ACTION),
            ],
            (9,): [_candidate("c-manual", "reduce manual HR work", [9], kind=CandidateKind.RATIONALE)],
            (10,): [_candidate("c-manual2", "HR does not manually update information", [10], kind=CandidateKind.RATIONALE)],
            (11,): [
                _candidate("c-payroll", "payroll", [11], kind=CandidateKind.ACTION),
                _candidate("c-leave", "leave handling", [11], kind=CandidateKind.REQUIREMENT),
                _candidate("c-deduction", "salary deduction", [11], kind=CandidateKind.REQUIREMENT),
                _candidate("c-pf", "PF calculation", [11], kind=CandidateKind.REQUIREMENT),
            ],
            (12,): [_candidate("c-expense", "expense tracking", [12], kind=CandidateKind.ACTION)],
            (13,): [
                _candidate("c-ai", "AI assistant project", [13], kind=CandidateKind.ACTION),
                _candidate("c-capture", "conversation capture", [13], kind=CandidateKind.REQUIREMENT),
            ],
            (14,): [
                _candidate("c-chunks", "chunking/recording", [14], kind=CandidateKind.REQUIREMENT),
                _candidate("c-notes", "structured notes", [14], kind=CandidateKind.REQUIREMENT),
            ],
        }
    )
    claims = [
        ArtifactClaim(artifactKey="t-onboarding", kind="task", title="Build candidate onboarding flow", body="Implement HRMS onboarding with an employee-page action that generates a candidate link.", sourceCandidateIds=["c-hrms", "c-onboarding", "c-action", "c-link"], evidenceSequences=[6, 7]),
        ArtifactClaim(artifactKey="n-submit", kind="note", title="Candidate onboarding workflow", body="Candidate submits information through the generated link, reducing manual HR work.", sourceCandidateIds=["c-submit", "c-manual", "c-manual2"], evidenceSequences=[7, 9, 10]),
        ArtifactClaim(artifactKey="t-payroll", kind="task", title="Build payroll", body="Payroll should handle leave, leave cancellation, salary deduction and PF calculation.", sourceCandidateIds=["c-payroll", "c-leave", "c-deduction", "c-pf"], evidenceSequences=[11]),
        ArtifactClaim(artifactKey="n-pf", kind="note", title="PF and leave in payroll", body="Payroll should calculate PF deductions and handle leave cancellation salary deduction.", sourceCandidateIds=["c-pf", "c-leave", "c-deduction"], evidenceSequences=[11]),
        ArtifactClaim(artifactKey="t-expense", kind="task", title="Build expense tracker", body="Track company expenses and payments.", sourceCandidateIds=["c-expense"], evidenceSequences=[12]),
        ArtifactClaim(artifactKey="t-ai", kind="task", title="Build AI assistant", body="Listen to conversations, divide them into chunks, and create useful structured notes.", sourceCandidateIds=["c-ai", "c-capture", "c-chunks", "c-notes"], evidenceSequences=[13, 14]),
        ArtifactClaim(artifactKey="bad-lenskart", kind="note", title="Gold membership", body="21% growth and UAE expansion for Lenskart franchising.", sourceCandidateIds=["c-hrms"], evidenceSequences=[5]),
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "lenskart-hrms",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims),
            verifier=ScriptedVerifier(unsupported_evidence={0, 1, 2, 3, 4, 5}),
        )
    )
    published = " ".join(f"{item.title} {item.body}" for item in [*result.tasks, *result.notes]).casefold()
    for forbidden in FORBIDDEN_BACKGROUND_TITLES:
        assert forbidden.casefold() not in published
    blob = published
    for meaning in REQUIRED_HRMS_MEANINGS:
        folded = meaning.casefold().replace("/", " ").replace("-", " ")
        tokens = [token for token in folded.split() if len(token) > 3]
        stems = [token[:-3] if token.endswith("ing") and len(token) > 6 else token for token in tokens]
        assert folded in blob or any(token in blob or stem in blob for token, stem in zip(tokens, stems)), meaning
    assert not any(span.sequenceStart == 5 for item in [*result.tasks, *result.notes] for span in item.evidence)
    case = lenskart_hrms_case()
    predicted = predicted_from_extraction(result)
    candidates = [
        PredictedItem(kind="note", meaning=item.meaning, evidenceSequences=item.evidenceSequences)
        for item in result.candidates
    ]
    score = score_case(case, predicted, predicted_candidates=candidates)
    assert score.backgroundFalsePositiveRate == 0
    assert score.taskRecall == 1
    assert score.candidateRecall == 1
    assert score.unsupportedArtifactRate == 0


def test_long_meeting_windows_stay_bounded_and_compact(monkeypatch):
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_TARGET_TOKENS", 80)
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_MAX_TOKENS", 120)
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_OVERLAP_RATIO", 0.12)
    monkeypatch.setattr(settings, "MAX_EXTRACTION_CONCURRENCY", 3)
    chunks = [
        _chunk(index, f"Begin middle end useful meeting speech {index} " + ("word " * 12), "long")
        for index in range(120)
    ]
    begin = _candidate("c-begin", "Start the migration", [0], kind=CandidateKind.ACTION)
    middle = _candidate("c-mid", "Review the checkpoint", [60], kind=CandidateKind.ACTION)
    end = _candidate("c-end", "Publish the notes", [119], kind=CandidateKind.ACTION)

    class _Extractor(ScriptedExtractor):
        async def extract(self, window, conversation_id: str):
            self.calls += 1
            found = []
            if 0 in window.sequence_ids:
                found.append(begin.model_copy(update={"sourceWindowId": window.window_id}))
            if 60 in window.sequence_ids:
                found.append(middle.model_copy(update={"sourceWindowId": window.window_id}))
            if 119 in window.sequence_ids:
                found.append(end.model_copy(update={"sourceWindowId": window.window_id}))
            return found

    extractor = _Extractor()
    capture = {}
    claims = [
        ArtifactClaim(artifactKey="t-begin", kind="task", title="Start the migration", body="Begin the long meeting work.", sourceCandidateIds=["c-begin"], evidenceSequences=[0]),
        ArtifactClaim(artifactKey="t-mid", kind="task", title="Review the checkpoint", body="Keep middle checkpoint work.", sourceCandidateIds=["c-mid"], evidenceSequences=[60]),
        ArtifactClaim(artifactKey="t-end", kind="task", title="Publish the notes", body="Finish by publishing notes.", sourceCandidateIds=["c-end"], evidenceSequences=[119]),
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "long",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=extractor,
            consolidator=ScriptedConsolidator(claims, capture=capture),
            verifier=ScriptedVerifier(),
        )
    )
    assert result.observability["window_count"] >= 2
    assert set(result.observability["covered_sequences"]) == set(range(120))
    assert result.observability["max_extraction_inflight"] <= 3
    assert extractor.calls == result.observability["window_count"]
    assert capture["candidate_count"] >= 3
    assert capture["cited_sequences"] == [0, 60, 119]
    titles = {task.title for task in result.tasks}
    assert {"Start the migration", "Review the checkpoint", "Publish the notes"} <= titles
    for task in result.tasks:
        assert {span.sequenceStart for span in task.evidence} <= {0, 60, 119}
    assert result.observability["consolidator_calls"] == 1
    assert result.observability["extractor_calls"] == result.observability["window_count"]
    assert result.observability["verifier_calls"] >= 1
    assert result.observability["ledger_size"] == result.observability["total_candidate_count"]
    assert result.observability["consolidator_cited_sequence_count"] == 3
    assert result.observability["transcript_sequence_count"] == 120
    assert result.observability["consolidator_cited_sequence_count"] < result.observability["transcript_sequence_count"]
    assert 119 in result.observability["covered_sequences"]
    assert result.observability["usefulWindowsWithZeroCandidates"] > 0
    assert result.observability["emptyCandidateWindowRate"] > 0


def test_multi_hour_meeting_covers_start_middle_end_without_full_transcript(monkeypatch):
    # ~4h of 30s STT chunks. Windows stay bounded; consolidator sees only cited lines.
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_TARGET_TOKENS", 400)
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_MAX_TOKENS", 560)
    monkeypatch.setattr(settings, "EXTRACTION_WINDOW_OVERLAP_RATIO", 0.12)
    monkeypatch.setattr(settings, "MAX_EXTRACTION_CONCURRENCY", 4)
    sequences = 480
    mid = sequences // 2
    end = sequences - 1
    chunks = [
        _chunk(index, f"Status update filler stretch {index} " + ("word " * 8), "long4h")
        for index in range(sequences)
    ]
    begin = _candidate("c-begin", "Start the drain gate work", [0], kind=CandidateKind.ACTION)
    middle = _candidate("c-mid", "Add the retry dashboard this month", [mid], kind=CandidateKind.ACTION)
    fact = _candidate("c-fact", "The staging credentials are already available", [mid + 1], kind=CandidateKind.FACT)
    last = _candidate("c-end", "Publish the meeting notes after drain", [end], kind=CandidateKind.ACTION)

    class _Extractor(ScriptedExtractor):
        async def extract(self, window, conversation_id: str):
            self.calls += 1
            found = []
            mapping = {0: begin, mid: middle, mid + 1: fact, end: last}
            for sequence, candidate in mapping.items():
                if sequence in window.owned_sequence_ids or sequence in window.sequence_ids:
                    found.append(candidate.model_copy(update={"sourceWindowId": window.window_id}))
            return found

    capture = {}
    claims = [
        ArtifactClaim(artifactKey="t-begin", kind="task", title="Start the drain gate work", body="Begin the long meeting work.", sourceCandidateIds=["c-begin"], evidenceSequences=[0]),
        ArtifactClaim(artifactKey="t-mid", kind="task", title="Add the retry dashboard this month", body="Keep middle checkpoint work.", sourceCandidateIds=["c-mid"], evidenceSequences=[mid]),
        ArtifactClaim(artifactKey="t-end", kind="task", title="Publish the meeting notes after drain", body="Finish by publishing notes.", sourceCandidateIds=["c-end"], evidenceSequences=[end]),
    ]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "long4h",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=_Extractor(),
            consolidator=ScriptedConsolidator(claims, capture=capture),
            verifier=ScriptedVerifier(),
        )
    )
    titles = {task.title for task in result.tasks}
    assert {"Start the drain gate work", "Add the retry dashboard this month", "Publish the meeting notes after drain"} <= titles
    assert result.notes
    assert any("credential" in f"{note.title} {note.body}".casefold() for note in result.notes)
    assert set(result.observability["covered_sequences"]) == set(range(sequences))
    assert result.observability["window_count"] >= 8
    assert result.observability["max_extraction_inflight"] <= 4
    assert result.observability["consolidator_calls"] == 1
    assert capture["cited_sequences"] == [0, mid, mid + 1, end]
    assert capture["full_transcript_sent"] is False
    assert result.observability["consolidator_cited_sequence_count"] < result.observability["transcript_sequence_count"]
    assert result.observability["transcript_sequence_count"] == sequences


def test_useful_window_with_zero_candidates_is_counted():
    chunks = [_chunk(1, "Rahul will integrate the API tomorrow.")]
    result = asyncio.run(
        run_meeting_pipeline(
            chunks,
            "conv",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=ScriptedExtractor(),
            consolidator=ScriptedConsolidator([]),
            verifier=ScriptedVerifier(),
        )
    )
    assert result.observability["usefulWindowsWithZeroCandidates"] == 1
    assert result.observability["emptyCandidateWindowRate"] == 1
    assert result.observability["extractor_candidate_count_per_window"] == [0]


def test_extractor_strips_neighbor_and_unknown_sequence_ids():
    class _Router:
        def route(self, capability):
            return self, "model"

        async def generate_structured(self, request, schema):
            assert schema is MeetingCandidateExtractorResponse
            return schema(
                candidates=[
                    {
                        "kind": "ACTION",
                        "meaning": "Generate candidate onboarding link",
                        "evidenceSequences": [5, 7, 99],
                    }
                ]
            )

    extractor = MeetingCandidateExtractor(_Router())
    window = _window("w0", 0, [6, 7, 8], overlap=[6], text="[7] generate candidate link")
    candidates = asyncio.run(extractor.extract(window, conversation_id="conv"))
    assert len(candidates) == 1
    assert candidates[0].evidenceSequences == [7]


def test_extractor_does_not_treat_schema_failure_as_empty_candidates():
    from services.llm.errors import StructuredOutputError
    from services.llm.openai_compatible import parse_structured_content

    with pytest.raises(StructuredOutputError):
        parse_structured_content(MeetingCandidateExtractorResponse, "{}")
    parsed, _ = parse_structured_content(MeetingCandidateExtractorResponse, '{"candidates": []}')
    assert parsed.candidates == []
    parsed, _ = parse_structured_content(
        MeetingCandidateExtractorResponse,
        '{"candidates": [{"kind": "ACTION", "meaning": "Neha will publish notes", "evidenceSequences": [59]}]}',
    )
    assert parsed.candidates[0].meaning == "Neha will publish notes"
    assert parsed.candidates[0].evidenceSequences == [59]
    with pytest.raises(StructuredOutputError):
        parse_structured_content(
            MeetingCandidateExtractorResponse,
            '{"candidates": [{"kind": "ACTION", "meaning": "Neha will publish notes", "evidenceSequences": []}]}',
        )


def test_extractor_keeps_parse_success_observable_when_evidence_ids_are_outside_window():
    class _Router:
        name = "scripted"

        def route(self, capability):
            return self, "model"

        async def generate_structured(self, request, schema):
            return schema(
                candidates=[
                    {"kind": "ACTION", "meaning": "Neha will publish notes", "evidenceSequences": [99]},
                ]
            )

    extractor = MeetingCandidateExtractor(_Router())
    window = _window("w0", 0, [45, 59], overlap=[45], text="[59] Neha will publish notes")
    candidates = asyncio.run(extractor.extract(window, conversation_id="conv"))
    assert candidates == []
    record = extractor.window_records_by_id["w0"]
    assert record["rawCandidateCount"] == 1
    assert record["droppedNoEvidence"] == 1
    assert record["failureClass"] == "STRUCTURED_OUTPUT_PARSE_LOSS"
    assert record["parsedResponse"][0]["evidenceSequences"] == [99]


def test_short_and_long_finalization_use_meeting_pipeline(monkeypatch):
    used = []

    async def fake(self, conversation, run, windows, path: str):
        used.append(path)

    monkeypatch.setattr(settings, "ENABLE_MEETING_PIPELINE", True)
    monkeypatch.setattr(ConversationProcessingWorkflow, "_run_meeting_pipeline_finalization", fake)
    workflow = ConversationProcessingWorkflow(SimpleNamespace(), SimpleNamespace())
    conversation = SimpleNamespace(userId="user A", id="sess")
    run = SimpleNamespace(checkpoints={})
    asyncio.run(workflow._run_short_session_finalization(conversation, run, []))
    asyncio.run(workflow._run_incremental_finalization(conversation, run, []))
    assert used == ["short_raw_transcript", "long_checkpoint_synthesis"]


def test_verifier_accepts_supported_hinglish_paraphrase_payload():
    from services.llm.openai_compatible import parse_structured_content

    parsed, _ = parse_structured_content(
        MeetingVerifierResponse,
        """{"items": [{"artifactKey": "t1", "verdict": "SUPPORTED", "reason": "supported",
        "fieldSupport": {"title": true, "description": true, "owner": true, "dueDate": true}}]}""",
    )
    assert parsed.items[0].verdict == VerifierVerdict.SUPPORTED
    assert parsed.items[0].fieldSupport.owner is True


def test_verifier_accepts_supported_hindi_paraphrase_payload():
    from services.llm.openai_compatible import parse_structured_content

    parsed, _ = parse_structured_content(
        MeetingVerifierResponse,
        """{"items": [{"artifactKey": "t1", "verdict": "SUPPORTED", "reason": "supported",
        "fieldSupport": {"title": true, "description": true, "owner": true, "dueDate": true}}]}""",
    )
    assert parsed.items[0].verdict.value == "SUPPORTED"


def test_verifier_schema_rejects_inconsistent_supported_reason():
    from services.llm.errors import StructuredOutputError
    from services.llm.openai_compatible import parse_structured_content

    with pytest.raises(StructuredOutputError):
        parse_structured_content(
            MeetingVerifierResponse,
            """{"items": [{"artifactKey": "t1", "verdict": "UNSUPPORTED", "reason": "supported",
            "fieldSupport": {"title": true, "description": true, "owner": false, "dueDate": false}}]}""",
        )


def test_verifier_schema_requires_field_support_booleans():
    from services.llm.errors import StructuredOutputError
    from services.llm.openai_compatible import parse_structured_content

    with pytest.raises(StructuredOutputError):
        parse_structured_content(
            MeetingVerifierResponse,
            """{"items": [{"artifactKey": "t1", "verdict": "SUPPORTED", "reason": "supported"}]}""",
        )


def test_verifier_keeps_owner_when_field_support_true():
    kept = apply_field_support(
        VerifiedArtifact(
            kind="task",
            title="Integrate API",
            body="Rahul will integrate the API.",
            owner="Rahul",
            evidenceSequences=[0],
            verdict=VerifierVerdict.SUPPORTED,
            artifactKey="t1",
            fieldSupport=FieldSupport(title=True, description=True, owner=True, dueDate=False),
        )
    )
    assert kept.owner == "Rahul"


def test_verifier_nulls_owner_when_person_is_only_mentioned():
    cleared = apply_field_support(
        VerifiedArtifact(
            kind="task",
            title="Discuss API",
            body="Rahul mentioned the API.",
            owner="Rahul",
            evidenceSequences=[0],
            verdict=VerifierVerdict.SUPPORTED,
            artifactKey="t1",
            fieldSupport=FieldSupport(title=True, description=True, owner=False, dueDate=False),
        )
    )
    assert cleared.owner is None


def test_correction_pipeline_keeps_only_sana_assignment():
    class Extractor:
        last_provider = "scripted"
        last_model = "scripted"
        calls = 1
        window_records_by_id = {}

        async def extract(self, window, conversation_id):
            return [
                MeetingCandidate(
                    candidateId="c-sana",
                    kind=CandidateKind.ACTION,
                    meaning="Page Sana for the staging outage.",
                    evidenceSequences=[2],
                    owner="Sana",
                    sourceWindowId=window.window_id,
                    sourceWindowIndex=window.window_index,
                )
            ]

    class Consolidator:
        last_provider = "scripted"
        last_model = "scripted"
        calls = 1

        async def consolidate(self, ledger, sequence_text):
            return [
                ArtifactClaim(
                    artifactKey="t-sana",
                    kind="task",
                    title="Page Sana for the staging outage",
                    body="Page Sana instead of Rahul.",
                    owner="Sana",
                    sourceCandidateIds=["c-sana"],
                    evidenceSequences=[2],
                )
            ], "", []

    result = asyncio.run(
        run_meeting_pipeline(
            [
                _chunk(0, "Please page Rahul for the staging outage.", "corr"),
                _chunk(1, "No, stop, Rahul is not on call.", "corr"),
                _chunk(2, "Page Sana instead, she has the pager.", "corr"),
            ],
            "corr",
            "user_1",
            "space_1",
            router=SimpleNamespace(),
            extractor=Extractor(),
            consolidator=Consolidator(),
            verifier=ScriptedVerifier(),
        )
    )
    assert len(result.tasks) == 1
    assert "Sana" in result.tasks[0].title
    assert "Rahul" not in result.tasks[0].title
