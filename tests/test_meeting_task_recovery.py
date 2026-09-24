"""Task recovery / eligibility for the meeting pipeline (notes path untouched)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.pipeline import run_meeting_pipeline
from services.conversation.meeting_pipeline.schemas import ArtifactClaim, CandidateKind, ConsolidatedTaskItem
from services.conversation.meeting_pipeline.task_eligibility import (
    recover_unpublished_actions,
    unpublished_review_candidates,
)
from tests.test_meeting_pipeline import ScriptedConsolidator, ScriptedExtractor, ScriptedVerifier, _candidate, _chunk


class ScriptedEligibility:
    def __init__(self, tasks=None, raise_error: bool = False):
        self.tasks = list(tasks or [])
        self.raise_error = raise_error
        self.calls = 0
        self.last_provider = "scripted"
        self.last_model = "scripted-eligibility"
        self.router = SimpleNamespace()

    async def review(self, pending, existing_tasks, ledger, sequence_text):
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("eligibility unavailable")
        meeting_sequences = set(sequence_text)
        known = {item.candidateId for item in pending}
        approved: list[ArtifactClaim] = []
        for index, item in enumerate(self.tasks):
            source_ids = [cid for cid in item.sourceCandidateIds if cid in known]
            if not source_ids:
                continue
            evidence = []
            for cid in source_ids:
                for candidate in pending:
                    if candidate.candidateId == cid:
                        evidence.extend(candidate.evidenceSequences)
            evidence = [seq for seq in evidence if seq in meeting_sequences]
            if not evidence:
                continue
            approved.append(
                ArtifactClaim(
                    artifactKey=f"task:eligible:{index}:{item.title}",
                    kind="task",
                    title=item.title,
                    body=item.description or item.title,
                    owner=item.owner,
                    dueDate=item.dueDate,
                    sourceCandidateIds=source_ids,
                    evidenceSequences=sorted(set(evidence)),
                )
            )
        return approved


def test_unpublished_candidates_skip_already_cited_tasks():
    ledger = CandidateLedger()
    ledger.add(_candidate("c1", "Create the shared export script", [1], kind=CandidateKind.ACTION))
    ledger.add(_candidate("c2", "Add a model preview in the product", [2], kind=CandidateKind.ACTION))
    ledger.add(_candidate("c3", "Output files must stay lossless", [3], kind=CandidateKind.REQUIREMENT))
    ledger.add(_candidate("c4", "The lake walk was on Saturday", [4], kind=CandidateKind.FACT))
    artifacts = [
        ArtifactClaim(
            artifactKey="t1",
            kind="task",
            title="Create the shared export script",
            body="Create one script that runs all required exports.",
            sourceCandidateIds=["c1"],
            evidenceSequences=[1],
        )
    ]
    pending = unpublished_review_candidates(artifacts, ledger)
    ids = {item.candidateId for item in pending}
    assert "c1" not in ids
    assert "c2" in ids
    assert "c3" in ids
    assert "c4" not in ids


def test_no_reviewer_does_not_invent_tasks():
    ledger = CandidateLedger()
    ledger.add(_candidate("c-queue", "Put generation jobs on a server queue", [4], kind=CandidateKind.ACTION))
    recovered, stats = asyncio.run(
        recover_unpublished_actions([], ledger, {4: "queue"}, reviewer=None)
    )
    assert recovered == []
    assert stats["recoveredTaskCount"] == 0
    assert stats["deterministicFallback"] == 0
    assert stats["eligibilityReviewed"] == 0


def test_eligibility_ai_rejection_is_respected():
    ledger = CandidateLedger()
    ledger.add(_candidate("c-idea", "Maybe later try a new approach", [6], kind=CandidateKind.ACTION))
    reviewer = ScriptedEligibility(tasks=[])
    recovered, stats = asyncio.run(
        recover_unpublished_actions([], ledger, {6: "Maybe later try a new approach"}, reviewer=reviewer)
    )
    assert stats["eligibilityReviewed"] == 1
    assert stats["eligibilityApproved"] == 0
    assert stats["recoveredTaskCount"] == 0
    assert recovered == []


def test_eligibility_approves_independent_work_and_skips_duplicates():
    ledger = CandidateLedger()
    ledger.add(
        _candidate(
            "c-master",
            "Create one script that runs every required export view",
            [10],
            kind=CandidateKind.ACTION,
        )
    )
    ledger.add(
        _candidate(
            "c-queue",
            "Put generation jobs on a server-side queue",
            [11],
            kind=CandidateKind.ACTION,
        )
    )
    reviewer = ScriptedEligibility(
        tasks=[
            ConsolidatedTaskItem(
                title="Create a master export script for every view",
                description="Build one script that executes all required views and writes the outputs.",
                sourceCandidateIds=["c-master"],
                evidenceSequences=[10],
            ),
            ConsolidatedTaskItem(
                title="Queue generation jobs on the server",
                description="Process multiple projects asynchronously so work continues after the client disconnects.",
                sourceCandidateIds=["c-queue"],
                evidenceSequences=[11],
            ),
            ConsolidatedTaskItem(
                title="Create a master export script for every view",
                description="Duplicate of the first task.",
                sourceCandidateIds=["c-master"],
                evidenceSequences=[10],
            ),
        ]
    )
    recovered, stats = asyncio.run(
        recover_unpublished_actions(
            [],
            ledger,
            {10: "we need one script for all views", 11: "jobs should continue on the server"},
            reviewer=reviewer,
        )
    )
    assert stats["eligibilityApproved"] == 3
    assert stats["recoveredTaskCount"] == 2
    titles = [item.title for item in recovered if item.kind == "task"]
    assert len(titles) == 2
    assert len(set(titles)) == 2


def test_pipeline_recovers_tasks_only_when_llm_approves():
    chunks = [
        _chunk(0, "Next step: create a master export script for all views."),
        _chunk(1, "Also put jobs on a server queue so generation continues offline."),
        _chunk(2, "Output files must stay high contrast."),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (0,): [
                _candidate(
                    "c-script",
                    "Create a master export script for all views",
                    [0],
                    kind=CandidateKind.ACTION,
                )
            ],
            (1,): [
                _candidate(
                    "c-queue",
                    "Put jobs on a server queue so generation continues offline",
                    [1],
                    kind=CandidateKind.ACTION,
                )
            ],
            (2,): [
                _candidate(
                    "c-png",
                    "Output files must stay high contrast",
                    [2],
                    kind=CandidateKind.REQUIREMENT,
                )
            ],
        }
    )
    reviewer = ScriptedEligibility(
        tasks=[
            ConsolidatedTaskItem(
                title="Create a master export script for all views",
                description="Generate one script that runs every required view export.",
                sourceCandidateIds=["c-script"],
                evidenceSequences=[0],
            ),
            ConsolidatedTaskItem(
                title="Queue generation jobs on the server",
                description="Keep processing on the server after the client disconnects.",
                sourceCandidateIds=["c-queue"],
                evidenceSequences=[1],
            ),
        ]
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
            task_eligibility=reviewer,
        )
    )
    assert len(result.tasks) == 2
    blob = " ".join(f"{task.title} {task.body}" for task in result.tasks).casefold()
    assert "script" in blob
    assert "queue" in blob
    assert result.notes
    assert result.observability["recovered_task_count"] == 2
    assert result.observability["recovered_note_count"] >= 1


def test_unique_task_claims_drops_repeated_tasks_and_keeps_notes():
    from services.conversation.meeting_pipeline.task_eligibility import unique_task_claims

    artifacts = [
        ArtifactClaim(
            artifactKey="t1",
            kind="task",
            title="Create the shared export script",
            body="Write one script that runs every required export.",
            sourceCandidateIds=["c1"],
            evidenceSequences=[1],
        ),
        ArtifactClaim(
            artifactKey="t1-dup",
            kind="task",
            title="Create the shared export script",
            body="Write one script that runs every required export.",
            sourceCandidateIds=["c1"],
            evidenceSequences=[1],
        ),
        ArtifactClaim(
            artifactKey="n1",
            kind="note",
            title="Output must stay lossless",
            body="Exported files must stay lossless.",
            sourceCandidateIds=["c3"],
            evidenceSequences=[3],
        ),
    ]
    kept = unique_task_claims(artifacts)
    assert [item.kind for item in kept] == ["task", "note"]
    assert kept[0].artifactKey == "t1"
    assert kept[1].kind == "note"


def test_eligibility_failure_does_not_block_note_recovery():
    chunks = [
        _chunk(0, "We will ship the weekly report."),
        _chunk(1, "Saturday we walked around the lake."),
    ]
    extractor = ScriptedExtractor(
        by_owned={
            (0,): [_candidate("c-action", "Ship the weekly report", [0], kind=CandidateKind.ACTION)],
            (1,): [_candidate("c-lake", "Saturday was spent walking around the lake", [1], kind=CandidateKind.FACT)],
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
            task_eligibility=ScriptedEligibility(raise_error=True),
        )
    )
    assert not result.tasks
    assert result.notes
    assert result.observability["task_eligibility_failed"] == 1
    assert result.observability["recovered_note_count"] >= 1
    blob = " ".join(f"{note.title} {note.body}" for note in result.notes).casefold()
    assert "lake" in blob


def test_hard_runtime_eligibility_error_is_not_swallowed():
    from services.conversation.event_pipeline.budget import PipelineBudgetExceeded

    class HardFailEligibility(ScriptedEligibility):
        async def review(self, pending, existing_tasks, ledger, sequence_text):
            raise PipelineBudgetExceeded("budget")

    ledger = CandidateLedger()
    ledger.add(_candidate("c1", "Ship the weekly report", [0], kind=CandidateKind.ACTION))
    try:
        asyncio.run(
            recover_unpublished_actions(
                [],
                ledger,
                {0: "Ship the weekly report"},
                reviewer=HardFailEligibility(),
            )
        )
    except PipelineBudgetExceeded:
        return
    raise AssertionError("hard runtime error should propagate")


def test_pipeline_notes_path_unchanged_for_fact_only_meeting():
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
    assert result.observability["recovered_task_count"] == 0
    assert result.observability["recovered_note_count"] >= 2
