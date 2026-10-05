import asyncio
from types import SimpleNamespace

from services.conversation.meeting_pipeline.composer import compose_artifacts
from services.conversation.meeting_pipeline.consolidator import (
    GlobalArtifactConsolidator,
    partition_candidates,
    recover_unpublished_notes,
)
from services.conversation.meeting_pipeline.harness import (
    TokenProfile,
    artifact_confidence,
    clean_multiline,
    normalize_priority,
    same_topic,
    strip_internal_references,
)
from services.conversation.meeting_pipeline.ledger import CandidateLedger
from services.conversation.meeting_pipeline.pipeline import _to_task
from services.conversation.meeting_pipeline.schemas import (
    ArtifactClaim,
    CandidateKind,
    ExtractionWindow,
    FieldSupport,
    MeetingCandidate,
    MeetingConsolidatorResponse,
    VerifiedArtifact,
    VerifierFieldSupport,
    VerifierItem,
    VerifierVerdict,
)
from services.conversation.meeting_pipeline.task_eligibility import unpublished_review_candidates
from services.conversation.meeting_pipeline.verifier import ArtifactEvidenceVerifier
from services.conversation.meeting_pipeline.windows import format_window_line, turns_from_chunks
from services.conversation.models import ExtractedTask, STTStatus, TranscriptChunkDocument


def _chunk(sequence: int, text: str) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="conv",
        userId="user_1",
        spaceId="space_1",
        chunkId=f"c{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        normalizedText=text,
        sttStatus=STTStatus.COMPLETED,
    )


def _candidate(cid, meaning, sequences, kind=CandidateKind.REQUIREMENT, topic=None, window=0):
    return MeetingCandidate(
        candidateId=cid,
        kind=kind,
        meaning=meaning,
        evidenceSequences=list(sequences),
        topic=topic,
        sourceWindowId=f"w{window}",
        sourceWindowIndex=window,
    )


def _ledger(*candidates):
    ledger = CandidateLedger()
    for candidate in candidates:
        ledger.add(candidate)
    return ledger


def test_bracketed_speaker_label_without_colon_is_parsed():
    turns = turns_from_chunks([_chunk(3, "[Speaker 0] I will send the deck tomorrow")])
    assert turns[0].speaker == "Speaker 0"
    assert turns[0].raw_text == "I will send the deck tomorrow"
    assert format_window_line(turns[0]) == "[3][Speaker 0] I will send the deck tomorrow"


def test_multi_speaker_chunk_keeps_per_line_speaker_and_sequence():
    turns = turns_from_chunks([_chunk(7, "[Speaker 0] Grace period is ten minutes.\n[Speaker 1] And after that?\n[Speaker 0] Then delayed.")])
    assert turns[0].speaker is None
    lines = format_window_line(turns[0]).splitlines()
    assert lines == [
        "[7][Speaker 0] Grace period is ten minutes.",
        "[7][Speaker 1] And after that?",
        "[7][Speaker 0] Then delayed.",
    ]


def test_single_speaker_multiline_chunk_drops_repeated_labels():
    turns = turns_from_chunks([_chunk(2, "[Speaker 0] Hello.\n[Speaker 0] How are you?")])
    assert turns[0].speaker == "Speaker 0"
    assert turns[0].raw_text == "Hello.\nHow are you?"


def test_rich_task_description_is_not_padded_with_unrelated_ledger_text():
    ledger = _ledger(
        _candidate("c-task", "Speaker 0 will send the pricing deck to the client", [1], kind=CandidateKind.ACTION),
        _candidate("c-noise", "Speaker 1 mentioned their daughter has exams next week", [2], kind=CandidateKind.FACT),
        _candidate("c-lunch", "Speaker 0 said the team lunch is on Friday", [30], kind=CandidateKind.FACT),
    )
    task = ArtifactClaim(
        artifactKey="t",
        kind="task",
        title="Send pricing deck to the client",
        body="Share the updated pricing deck with the client so they can review the revised tiers before the renewal call.",
        sourceCandidateIds=["c-task"],
        evidenceSequences=[1],
    )
    note = ArtifactClaim(
        artifactKey="n",
        kind="note",
        title="Family",
        body="Speaker 1 mentioned their daughter has exams next week",
        sourceCandidateIds=["c-noise"],
        evidenceSequences=[2],
    )
    composed, stats = compose_artifacts([task, note], ledger, {1, 2, 30})
    out = next(item for item in composed if item.kind == "task")
    assert out.body == task.body
    assert out.sourceCandidateIds == ["c-task"]
    assert stats["tasksEnriched"] == 0


def test_notes_with_different_topics_are_not_merged_by_adjacency():
    ledger = _ledger(
        _candidate("a", "Shift supports fixed and flexible types", [1], topic="Shift types"),
        _candidate("b", "Leave can be paid or unpaid", [2], topic="Leave master"),
    )
    notes = [
        ArtifactClaim(artifactKey="n1", kind="note", title="Shift types", body="Shift supports fixed and flexible types.", topic="Shift types", sourceCandidateIds=["a"], evidenceSequences=[1]),
        ArtifactClaim(artifactKey="n2", kind="note", title="Leave master", body="Leave can be paid or unpaid.", topic="Leave master", sourceCandidateIds=["b"], evidenceSequences=[2]),
    ]
    composed, stats = compose_artifacts(notes, ledger, {1, 2})
    assert len(composed) == 2
    assert stats["notesClustered"] == 0


def test_same_topic_notes_merge_into_structured_body():
    ledger = _ledger(
        _candidate("a", "Grace period is configurable per shift", [1], topic="Grace period logic"),
        _candidate("b", "After the grace period the employee is present with delay", [2], topic="Grace period logic"),
    )
    notes = [
        ArtifactClaim(artifactKey="n1", kind="note", title="Grace period", body="Grace period is configurable per shift.", topic="Grace period logic", sourceCandidateIds=["a"], evidenceSequences=[1]),
        ArtifactClaim(artifactKey="n2", kind="note", title="Delay status", body="After the grace period the employee is present with delay.", topic="Grace period logic", sourceCandidateIds=["b"], evidenceSequences=[2]),
    ]
    composed, _ = compose_artifacts(notes, ledger, {1, 2})
    assert len(composed) == 1
    merged = composed[0]
    assert merged.title == "Grace period logic"
    assert merged.body.splitlines()[0] == "Grace period is configurable per shift."
    assert merged.body.splitlines()[1].startswith("- After the grace period")
    assert set(merged.sourceCandidateIds) == {"a", "b"}


def test_recovery_skips_discarded_noise_and_groups_leftovers_by_topic():
    ledger = _ledger(
        _candidate("q", "One participant did not understand the explanation", [1], kind=CandidateKind.FACT),
        _candidate("r1", "Holiday master is a separate screen", [2], topic="Holiday master"),
        _candidate("r2", "Holidays must feed attendance calculation", [3], topic="Holiday master"),
    )
    recovered, added = recover_unpublished_notes([], ledger, {1, 2, 3}, discarded={"q"})
    assert added == 2
    assert len(recovered) == 1
    note = recovered[0]
    assert note.title == "Holiday master"
    assert "separate screen" in note.body and "attendance calculation" in note.body
    assert "understand" not in note.body


def test_recovery_attaches_leftover_to_existing_topic_note():
    ledger = _ledger(
        _candidate("a", "Leave has a name, type and description", [1], topic="Leave master"),
        _candidate("b", "Leave type is paid or unpaid", [2], topic="Leave master"),
    )
    existing = ArtifactClaim(artifactKey="n", kind="note", title="Leave master", body="Leave has a name, type and description.", topic="Leave master", sourceCandidateIds=["a"], evidenceSequences=[1])
    recovered, added = recover_unpublished_notes([existing], ledger, {1, 2})
    assert added == 1 and len(recovered) == 1
    assert recovered[0].body.endswith("- Leave type is paid or unpaid")
    assert recovered[0].evidenceSequences == [1, 2]


def test_eligibility_review_excludes_discarded_candidates():
    ledger = _ledger(
        _candidate("x", "Delete it from there", [0], kind=CandidateKind.ACTION),
        _candidate("y", "Build the overtime module", [5], kind=CandidateKind.ACTION),
    )
    pending = unpublished_review_candidates([], ledger, discarded={"x"})
    assert [item.candidateId for item in pending] == ["y"]


def test_partition_keeps_topics_whole_under_budget():
    candidates = []
    for index in range(40):
        topic = ["Shift setup", "Leave rules", "Loan module", "Holiday master"][index % 4]
        candidates.append(_candidate(f"c{index}", f"{topic} detail number {index} " + "word " * 30, [index], topic=topic, window=index // 10))
    text = {index: "line " * 20 for index in range(40)}
    parts = partition_candidates(candidates, text, budget_tokens=2500)
    assert len(parts) > 1
    assert sorted(c.candidateId for part in parts for c in part) == sorted(c.candidateId for c in candidates)
    homes: dict[str, set[int]] = {}
    for position, part in enumerate(parts):
        for candidate in part:
            homes.setdefault(candidate.topic, set()).add(position)
    assert all(len(positions) == 1 for positions in homes.values())


def test_partition_single_call_when_small():
    candidates = [_candidate("a", "small", [1], topic="T")]
    assert len(partition_candidates(candidates, {1: "x"}, budget_tokens=10000)) == 1


def test_consolidator_collects_discarded_ids_and_task_priority(monkeypatch):
    ledger = _ledger(
        _candidate("act", "Build the overtime module", [1], kind=CandidateKind.ACTION, topic="Overtime"),
        _candidate("noise", "Hello, how are you", [2], kind=CandidateKind.FACT),
    )

    async def fake_generate(router, capability, prompt, schema, payload, **kwargs):
        return (
            MeetingConsolidatorResponse(
                tasks=[{"title": "Complete overtime module", "description": "Build overtime rules linked to shifts.", "acceptanceCriteria": "Overtime is calculated from shift rules.", "priority": "medium", "sourceCandidateIds": ["act"], "evidenceSequences": [1]}],
                discardedCandidateIds=["noise", "unknown"],
            ),
            "fake",
            "fake-model",
        )

    monkeypatch.setattr("services.conversation.meeting_pipeline.consolidator.generate_structured", fake_generate)
    consolidator = GlobalArtifactConsolidator(SimpleNamespace())
    claims, _, _ = asyncio.run(consolidator.consolidate(ledger, {1: "build overtime", 2: "hello"}))
    assert consolidator.last_discarded_ids == {"noise"}
    task = claims[0]
    assert task.priority == "Medium"
    assert task.acceptanceCriteria == "Overtime is calculated from shift rules."
    assert task.topic == "Overtime"


def test_consolidator_repartitions_when_single_call_fails(monkeypatch):
    from services.conversation.meeting_pipeline.consolidator import _payload_tokens

    candidates = [
        _candidate(f"c{i}", f"{['Alpha', 'Beta'][i % 2]} requirement {i} " + "word " * 160, [i], topic=["Alpha", "Beta"][i % 2])
        for i in range(30)
    ]
    text = {i: "line " * 10 for i in range(30)}
    total = _payload_tokens(candidates, text)
    assert total >= 4200
    monkeypatch.setattr("services.conversation.meeting_pipeline.consolidator.consolidation_partition_tokens", lambda: total)
    ledger = _ledger(*candidates)
    calls: list[int] = []

    async def fake_generate(router, capability, prompt, schema, payload, **kwargs):
        calls.append(len(payload["candidates"]))
        if len(calls) == 1 and payload.get("partition") is None:
            raise RuntimeError("output truncated")
        first = payload["candidates"][0]
        return (
            MeetingConsolidatorResponse(notes=[{"title": first["topic"], "body": "Grounded.", "sourceCandidateIds": [first["candidateId"]], "evidenceSequences": first["evidenceSequences"]}]),
            "fake",
            "fake-model",
        )

    monkeypatch.setattr("services.conversation.meeting_pipeline.consolidator.generate_structured", fake_generate)
    consolidator = GlobalArtifactConsolidator(SimpleNamespace())
    claims, _, _ = asyncio.run(consolidator.consolidate(ledger, text))
    assert consolidator.last_partition_count > 1
    assert {claim.title for claim in claims} == {"Alpha", "Beta"}


def test_coverage_pass_attaches_or_discards_every_uncovered_candidate(monkeypatch):
    from services.conversation.meeting_pipeline.schemas import MeetingCoverageResponse

    ledger = _ledger(
        _candidate("base", "Shift supports fixed and flexible types", [1], topic="Shift setup"),
        _candidate("extra", "Flexible shifts show a start and end window", [2], topic="Shift setup"),
        _candidate("noise", "Is the phone ringing", [3], kind=CandidateKind.FACT),
    )
    seen: dict = {}

    async def fake_generate(router, capability, prompt, schema, payload, **kwargs):
        if prompt == "meeting-artifact-coverage-v1":
            seen["payload"] = payload
            alias = {c["meaning"]: c["candidateId"] for c in payload["uncoveredCandidates"]}
            return (
                MeetingCoverageResponse(
                    attachments=[{"artifactId": "A1", "candidateIds": [alias["Flexible shifts show a start and end window"]], "addition": "Flexible shifts show a start/end window."}],
                    discardedCandidateIds=[alias["Is the phone ringing"]],
                ),
                "fake",
                "fake-model",
            )
        return (
            MeetingConsolidatorResponse(notes=[{"title": "Shift setup", "body": "Shifts are fixed or flexible.", "sourceCandidateIds": ["C1"], "evidenceSequences": [1]}]),
            "fake",
            "fake-model",
        )

    monkeypatch.setattr("services.conversation.meeting_pipeline.consolidator.generate_structured", fake_generate)
    consolidator = GlobalArtifactConsolidator(SimpleNamespace())
    claims, _, _ = asyncio.run(consolidator.consolidate(ledger, {1: "a", 2: "b", 3: "c"}))
    assert len(seen["payload"]["uncoveredCandidates"]) == 2
    note = claims[0]
    assert note.body == "Shifts are fixed or flexible.\n- Flexible shifts show a start/end window."
    assert set(note.sourceCandidateIds) == {"base", "extra"}
    assert note.evidenceSequences == [1, 2]
    assert consolidator.last_discarded_ids == {"noise"}
    assert consolidator.coverage_attached == 1


class _FlakyVerifier(ArtifactEvidenceVerifier):
    def __init__(self, fail_over: int, drop_keys=()):
        super().__init__(SimpleNamespace())
        self.fail_over = fail_over
        self.drop_keys = set(drop_keys)
        self.batch_sizes: list[int] = []

    async def _review_batch(self, batch, sequence_text):
        self.calls += 1
        self.batch_sizes.append(len(batch))
        if len(batch) > self.fail_over:
            raise RuntimeError("truncated")
        rows = []
        for claim in batch:
            if claim.artifactKey in self.drop_keys and len(batch) > 1:
                continue
            rows.append(
                VerifierItem(
                    artifactKey=claim.artifactKey,
                    verdict=VerifierVerdict.SUPPORTED,
                    fieldSupport=VerifierFieldSupport(title=True, description=True, owner=False, dueDate=False),
                    reason="supported",
                )
            )
        return rows


def _claims(count):
    return [
        ArtifactClaim(artifactKey=f"k{i}", kind="note", title=f"Topic {i}", body="Body.", sourceCandidateIds=[f"c{i}"], evidenceSequences=[i])
        for i in range(count)
    ]


def test_verifier_splits_failed_batch_instead_of_losing_artifacts():
    verifier = _FlakyVerifier(fail_over=2)
    result = asyncio.run(verifier.verify(_claims(6), {i: "x" for i in range(6)}))
    assert all(item.verdict == VerifierVerdict.SUPPORTED for item in result)
    assert len(result) == 6


def test_verifier_rechecks_rows_the_model_omitted():
    verifier = _FlakyVerifier(fail_over=99, drop_keys={"k1"})
    result = asyncio.run(verifier.verify(_claims(3), {i: "x" for i in range(3)}))
    assert [item.verdict for item in result] == [VerifierVerdict.SUPPORTED] * 3


def test_task_output_carries_priority_expected_result_and_calibrated_confidence():
    item = VerifiedArtifact(
        kind="task",
        title="Add configurable grace period",
        body="Shifts need a configurable grace period.",
        acceptanceCriteria="Arrivals within the grace period are marked on time.",
        priority="High",
        topic="Attendance",
        sourceCandidateIds=["a", "b"],
        evidenceSequences=[1, 2],
        verdict=VerifierVerdict.SUPPORTED,
        fieldSupport=FieldSupport(title=True, description=True, owner=False, dueDate=False),
        artifactKey="t",
    )
    task = _to_task(item, "conv", "space", {1: "one", 2: "two"})
    assert task.priority == "High"
    assert task.body.endswith("Expected result: Arrivals within the grace period are marked on time.")
    assert 0.5 < task.confidence < 1.0
    assert task.changes["topic"] == "Attendance"


def test_extracted_task_priority_is_lenient():
    base = dict(title="t", operation="CREATE", confidence=0.5, sourceConversationId="c", evidence=[])
    assert ExtractedTask(**base, priority="high").priority == "High"
    assert ExtractedTask(**base, priority="whatever").priority is None
    assert ExtractedTask(**base).priority is None


def test_harness_helpers():
    assert normalize_priority("P1") == "High"
    assert normalize_priority("later") == "Low"
    assert normalize_priority("") is None
    assert clean_multiline("Lead.\n•  one\n* two\n\n\n3) three") == "Lead.\n- one\n- two\n\n- three"
    assert same_topic("Leave master", "leave  Master")
    assert not same_topic("Leave master", "Shift master")
    profile = TokenProfile(["Speaker 0 talks shift", "Speaker 1 talks leave", "Speaker 0 talks loan", "speaker talks holiday"])
    assert "speaker" not in profile.distinctive("Speaker 0 shift")
    assert artifact_confidence(supported=False, field_support={}, repaired=False, source_count=1, evidence_count=1) == 0.2
    assert artifact_confidence(supported=True, field_support={"title": True, "description": True}, repaired=False, source_count=9, evidence_count=9) <= 0.95


def test_outline_merges_sections_tasks_and_ranks_priority():
    from services.conversation.meeting_pipeline.organizer import MeetingOutlineOrganizer
    from services.conversation.meeting_pipeline.schemas import MeetingOutlineResponse

    artifacts = [
        ArtifactClaim(artifactKey="t1", kind="task", title="Build export form", body="Form exports reports.", priority="High", sourceCandidateIds=["a"], evidenceSequences=[1]),
        ArtifactClaim(artifactKey="t2", kind="task", title="Add date filter to export form", body="Filter by date range.", priority="High", sourceCandidateIds=["b"], evidenceSequences=[2]),
        ArtifactClaim(artifactKey="t3", kind="task", title="Call the plumber", body="Personal call.", sourceCandidateIds=["c"], evidenceSequences=[3]),
        ArtifactClaim(artifactKey="n1", kind="note", title="Export format", body="Exports are CSV.", topic="Export format", sourceCandidateIds=["d"], evidenceSequences=[4]),
        ArtifactClaim(artifactKey="n2", kind="note", title="Export limits", body="Max 10k rows.", topic="Export limits", sourceCandidateIds=["e"], evidenceSequences=[5]),
    ]

    async def fake_generate(router, capability, prompt, schema, payload, **kwargs):
        assert prompt == "meeting-outline-organizer-v1"
        assert [row["id"] for row in payload["tasks"]] == ["T1", "T2", "T3"]
        return (
            MeetingOutlineResponse(
                sections=[{"title": "Report Export", "noteIds": ["N1", "n2", "N9"]}],
                taskMerges=[{"title": "Build export form with date filter", "taskIds": ["T1", "T2"]}],
                priorities=[{"taskId": "T1", "priority": "medium"}],
                offTopicIds=["T3"],
            ),
            "fake",
            "fake-model",
        )

    import services.conversation.meeting_pipeline.organizer as organizer_module

    original = organizer_module.generate_structured
    organizer_module.generate_structured = fake_generate
    try:
        out, stats = asyncio.run(MeetingOutlineOrganizer(SimpleNamespace()).organize(artifacts))
    finally:
        organizer_module.generate_structured = original
    tasks = [item for item in out if item.kind == "task"]
    notes = [item for item in out if item.kind == "note"]
    assert len(tasks) == 1 and tasks[0].title == "Build export form with date filter"
    assert tasks[0].priority == "Medium"
    assert tasks[0].sourceCandidateIds == ["a", "b"] and tasks[0].evidenceSequences == [1, 2]
    assert len(notes) == 1 and notes[0].title == "Report Export"
    assert "Max 10k rows." in notes[0].body and notes[0].evidenceSequences == [4, 5]
    assert stats["tasksMerged"] == 1 and stats["offTopicDropped"] == 1 and stats["prioritiesSet"] == 1
    assert stats["prioritiesBanded"] == 0 and stats["failed"] == 0


def test_organizer_judges_off_topic_from_cited_transcript():
    from services.conversation.meeting_pipeline.organizer import MeetingOutlineOrganizer
    from services.conversation.meeting_pipeline.schemas import MeetingOutlineResponse

    artifacts = [
        ArtifactClaim(artifactKey="t1", kind="task", title="Send the updated build", body="Distribute the build.", sourceCandidateIds=["a"], evidenceSequences=[1, 9]),
        ArtifactClaim(artifactKey="t2", kind="task", title="Fix the edit button", body="Edit changes records unintentionally.", sourceCandidateIds=["b"], evidenceSequences=[2]),
        ArtifactClaim(artifactKey="n1", kind="note", title="Empty evidence", body="No lines.", sourceCandidateIds=["c"], evidenceSequences=[8]),
    ]
    seen: dict = {}

    async def fake_generate(router, capability, prompt, schema, payload, **kwargs):
        if prompt != "meeting-outline-organizer-v1":
            from services.conversation.meeting_pipeline.schemas import MeetingLogisticsResponse
            return MeetingLogisticsResponse(), "fake", "fake-model"
        seen["tasks"] = payload["tasks"]
        seen["notes"] = payload["notes"]
        return MeetingOutlineResponse(), "fake", "fake-model"

    import services.conversation.meeting_pipeline.organizer as organizer_module

    original = organizer_module.generate_structured
    organizer_module.generate_structured = fake_generate
    try:
        asyncio.run(
            MeetingOutlineOrganizer(SimpleNamespace()).organize(
                artifacts,
                {1: "[Speaker 0] I will mail the build from my laptop after this call", 2: "[Speaker 1] the edit button changes saved attendance"},
            )
        )
    finally:
        organizer_module.generate_structured = original
    assert seen["tasks"][0]["evidence"] == ["[Speaker 0] I will mail the build from my laptop after this call"]
    assert seen["tasks"][1]["evidence"] == [
        "[Speaker 0] I will mail the build from my laptop after this call",
        "[Speaker 1] the edit button changes saved attendance",
    ]
    assert "evidence" not in seen["notes"][0]


def test_logistics_review_drops_side_tasks_and_keeps_core_work():
    from services.conversation.meeting_pipeline.organizer import MeetingOutlineOrganizer
    from services.conversation.meeting_pipeline.schemas import MeetingLogisticsResponse, MeetingOutlineResponse

    artifacts = [
        ArtifactClaim(artifactKey="core", kind="task", title="Fix the edit button", body="The edit button changes saved records.", priority="High", sourceCandidateIds=["a"], evidenceSequences=[2]),
        ArtifactClaim(artifactKey="mail", kind="task", title="Send the updated build", body="Mail the build after this call.", sourceCandidateIds=["b"], evidenceSequences=[4]),
        ArtifactClaim(artifactKey="rule", kind="task", title="Set the grace period", body="Grace is 10 minutes.", sourceCandidateIds=["c"], evidenceSequences=[5]),
        ArtifactClaim(artifactKey="note", kind="note", title="Attendance rules", body="Grace is 10 minutes after shift start.", sourceCandidateIds=["d"], evidenceSequences=[5]),
    ]
    seen: dict = {}

    async def fake_generate(router, capability, prompt, schema, payload, **kwargs):
        if prompt == "meeting-logistics-review-v1":
            seen["candidates"] = payload["candidates"]
            seen["subject"] = payload["meetingSubject"]
            assert all(row["priority"] != "High" for row in payload["candidates"])
            mail = next(row["id"] for row in payload["candidates"] if "build" in row["title"])
            return MeetingLogisticsResponse(offTopicIds=[mail, "L9", "L1", "L2", "L3"]), "fake", "fake-model"
        return (
            MeetingOutlineResponse(
                rankedTaskIds=["T1", "T2", "T3"],
                priorities=[{"taskId": "T1", "priority": "High"}, {"taskId": "T2", "priority": "Medium"}, {"taskId": "T3", "priority": "Medium"}],
            ),
            "fake",
            "fake-model",
        )

    import services.conversation.meeting_pipeline.organizer as organizer_module

    original = organizer_module.generate_structured
    organizer_module.generate_structured = fake_generate
    try:
        out, stats = asyncio.run(
            MeetingOutlineOrganizer(SimpleNamespace()).organize(
                artifacts,
                {
                    1: "[Speaker 0] let me share my screen",
                    2: "[Speaker 1] the edit button changes saved attendance",
                    3: "[Speaker 0] I will mail the build from my laptop",
                    4: "[Speaker 0] sending it on gmail after we hang up",
                    5: "[Speaker 1] grace period is 10 minutes",
                },
            )
        )
    finally:
        organizer_module.generate_structured = original
    titles = [item.title for item in out if item.kind == "task"]
    assert "Send the updated build" not in titles
    assert "Fix the edit button" in titles
    assert "Set the grace period" in titles
    assert seen["subject"] == ["Attendance rules"]
    assert seen["candidates"][0]["evidence"][0] == "[Speaker 0] I will mail the build from my laptop"
    assert stats["logisticsDropped"] == 1


def test_near_duplicate_tasks_collapse_without_umbrella_merges():
    from services.conversation.meeting_pipeline.organizer import apply_outline
    from services.conversation.meeting_pipeline.schemas import MeetingOutlineResponse

    def task(key, title, body, sequence):
        return ArtifactClaim(
            artifactKey=key,
            kind="task",
            title=title,
            body=body,
            sourceCandidateIds=[key],
            evidenceSequences=[sequence],
        )

    same = [
        task("t1", "Add payment frequency dropdown", "Users select a monthly or quarterly payment frequency, and late payments are blocked.", 1),
        task("t2", "Enforce payment frequency limits", "Payment frequency selection must block late payments for monthly and quarterly schedules.", 2),
    ]
    other = [
        task("t3", "Fix the attendance edit button", "The edit button allows unintended changes to saved attendance records.", 3),
        task("t4", "Create the holiday calendar", "Build a holiday master that stores the company holiday dates.", 4),
    ]
    copies = [
        task(
            f"c{i}",
            "Configure leave accrual",
            "Configure the leave accrual period as days, weeks, or months using the same accrual rules.",
            i,
        )
        for i in range(1, 5)
    ]
    different_values = [
        task("g1", "Set the grace period", "Employees stay present for 10 minutes after the shift starts.", 1),
        task("g2", "Set the grace period", "Employees stay present for 20 minutes after the shift starts.", 2),
    ]
    merged, stats = apply_outline(MeetingOutlineResponse(), {}, {f"T{i}": item for i, item in enumerate([*same, *other], start=1)})
    titles = [item.title for item in merged]
    assert len(merged) == 3
    assert any("payment frequency" in title.casefold() for title in titles)
    assert stats["nearDuplicatesMerged"] == 1
    kept, wide = apply_outline(MeetingOutlineResponse(), {}, {f"T{i}": item for i, item in enumerate(copies, start=1)})
    assert len(kept) >= 2
    assert wide["umbrellaMergesRejected"] >= 1
    separate, _stats = apply_outline(
        MeetingOutlineResponse(),
        {},
        {f"T{i}": item for i, item in enumerate(different_values, start=1)},
    )
    assert len(separate) == 2


def test_priority_bands_follow_meeting_rank_and_keep_deferred_low():
    from services.conversation.meeting_pipeline.organizer import apply_outline
    from services.conversation.meeting_pipeline.schemas import MeetingOutlineResponse

    tasks = {
        f"T{i}": ArtifactClaim(
            artifactKey=f"t{i}",
            kind="task",
            title=f"Deliverable {i}",
            body=f"Distinct work item number {i} for this meeting.",
            priority="High",
            sourceCandidateIds=[f"s{i}"],
            evidenceSequences=[i],
        )
        for i in range(1, 9)
    }
    response = MeetingOutlineResponse(
        rankedTaskIds=[f"T{i}" for i in range(1, 9)],
        priorities=[{"taskId": "T2", "priority": "Low"}, *({"taskId": f"T{i}", "priority": "High"} for i in range(1, 9) if i != 2)],
    )
    out, stats = apply_outline(response, {}, tasks)
    by_title = {item.title: item.priority for item in out}
    assert by_title["Deliverable 1"] == "High"
    assert by_title["Deliverable 2"] == "Low"
    assert by_title["Deliverable 3"] == "High"
    assert by_title["Deliverable 4"] == "Medium"
    assert sum(priority == "High" for priority in by_title.values()) == 2
    assert stats["prioritiesBanded"] > 0

    small = {
        f"T{i}": ArtifactClaim(artifactKey=f"t{i}", kind="task", title=f"Item {i}", body=f"Work {i}.", priority="High", sourceCandidateIds=[str(i)], evidenceSequences=[i])
        for i in range(1, 4)
    }
    kept, _stats = apply_outline(MeetingOutlineResponse(priorities=[{"taskId": f"T{i}", "priority": "High"} for i in range(1, 4)]), {}, small)
    assert {item.priority for item in kept} == {"High"}


def test_outline_rejects_umbrella_task_merges():
    from services.conversation.meeting_pipeline.organizer import apply_outline
    from services.conversation.meeting_pipeline.schemas import MeetingOutlineResponse

    tasks = {f"T{i}": ArtifactClaim(artifactKey=f"t{i}", kind="task", title=f"Task {i}", sourceCandidateIds=[str(i)], evidenceSequences=[i]) for i in range(1, 6)}
    response = MeetingOutlineResponse(taskMerges=[{"title": "Implement everything", "taskIds": list(tasks)}])
    out, stats = apply_outline(response, {}, tasks)
    assert len(out) == 5 and stats["umbrellaMergesRejected"] == 1


def test_alias_references_are_stripped_from_text():
    assert strip_internal_references("Shift is blue (C8).\n- Grace is 10 min [C7, C9]") == "Shift is blue.\n- Grace is 10 min"
    assert strip_internal_references("Use model C3PO") == "Use model C3PO"


def test_outline_failure_and_excess_drops_are_safe():
    from services.conversation.meeting_pipeline.organizer import apply_outline
    from services.conversation.meeting_pipeline.schemas import MeetingOutlineResponse

    tasks = {f"T{i}": ArtifactClaim(artifactKey=f"t{i}", kind="task", title=f"Task {i}", sourceCandidateIds=[str(i)], evidenceSequences=[i]) for i in range(1, 5)}
    out, stats = apply_outline(MeetingOutlineResponse(offTopicIds=["T1", "T2", "T3"]), {}, tasks, {"sectionsMerged": 0, "tasksMerged": 0, "offTopicDropped": 0, "prioritiesSet": 0, "failed": 0})
    assert len(out) == 4 and stats["offTopicDropped"] == 0


def test_merged_note_drops_restated_bullets_and_kind_labels():
    from services.conversation.meeting_pipeline.composer import _merge_note_group

    group = [
        ArtifactClaim(artifactKey="a", kind="note", title="Shifts", body="Each shift is fixed or flexible, with fixed shifts using 10-minute slots.\n- Shift type: fixed or flexible.", evidenceSequences=[1]),
        ArtifactClaim(artifactKey="b", kind="note", title="Grace", body="Requirement: Grace period is 15 minutes after shift start.\n- Open decision: whether grace applies to night shifts.", evidenceSequences=[2]),
    ]
    merged = _merge_note_group(group)
    assert merged.body == (
        "Each shift is fixed or flexible, with fixed shifts using 10-minute slots.\n"
        "- Grace period is 15 minutes after shift start.\n"
        "- Open decision: whether grace applies to night shifts."
    )


def test_restated_detection_keeps_lines_with_new_numbers():
    from services.conversation.meeting_pipeline.composer import _restated

    prev = ["Users can add multiple shifts via an Add button and later edit or view each shift."]
    assert _restated("Add button allows adding multiple shifts.", prev)
    assert not _restated("Grace period is 15 minutes after start.", ["Grace period is 10 minutes after start."])
    assert not _restated("Overtime requires manager approval.", prev)


def test_meeting_pipeline_publishes_tasks_and_notes_to_their_collections():
    from bson import ObjectId

    from services.conversation.meeting_pipeline.pipeline import _to_note, to_window_result
    from services.conversation.meeting_pipeline.schemas import MeetingPipelineResult
    from services.conversation.models import ExtractionRunDocument
    from services.conversation.repository import ConversationRepository

    conversation_id = "6ac39465781a7382692e580c"
    user_id = "6ac39465781a7382692e580d"
    space_id = "6ac39465781a7382692e580e"
    task = _to_task(
        VerifiedArtifact(
            kind="task",
            title="Add configurable grace period",
            body="Shifts need a configurable grace period.",
            acceptanceCriteria="Arrivals inside the grace period are on time.",
            priority="High",
            evidenceSequences=[4],
            verdict=VerifierVerdict.SUPPORTED,
            fieldSupport=FieldSupport(title=True, description=True),
            artifactKey="t",
        ),
        conversation_id,
        space_id,
        {4: "grace is ten minutes"},
    )
    note = _to_note(
        VerifiedArtifact(
            kind="note",
            title="Attendance timing",
            body="Grace is ten minutes after the shift starts.",
            evidenceSequences=[4],
            verdict=VerifierVerdict.SUPPORTED,
            fieldSupport=FieldSupport(title=True, description=True),
            artifactKey="n",
        ),
        conversation_id,
        space_id,
        {4: "grace is ten minutes"},
    )
    window = to_window_result(MeetingPipelineResult(tasks=[task], notes=[note], summary="Attendance timing was defined."))
    assert [item.title for item in window.tasks] == ["Add configurable grace period"]
    assert [item.title for item in window.notes] == ["Attendance timing"]

    class RecordingCollection:
        def __init__(self, name, store):
            self.name = name
            self.store = store

        async def delete_many(self, query):
            self.store.calls.append(("delete", self.name))

        async def insert_many(self, docs, ordered=False):
            self.store.docs.setdefault(self.name, []).extend(docs)
            self.store.calls.append(("insert", self.name))

        async def update_one(self, query, update, upsert=False):
            self.store.calls.append(("update", self.name))

        async def find_one_and_update(self, query, update, upsert=False, return_document=None):
            doc = {**update.get("$setOnInsert", {}), **update["$set"]}
            self.store.docs.setdefault(self.name, []).append(doc)
            self.store.calls.append(("upsert", self.name))
            return {"_id": doc.get("_id"), **doc}

    class RecordingDB:
        def __init__(self):
            self.calls = []
            self.docs = {}

        def __getattr__(self, name):
            return RecordingCollection(name, self)

        def __getitem__(self, name):
            return RecordingCollection(name, self)

    run = ExtractionRunDocument(
        conversationId=conversation_id,
        userId=user_id,
        spaceId=space_id,
        processingVersion=1,
        provider="krutrim",
        model="gpt-oss-120b",
        stagedTasks=window.tasks,
        stagedNotes=window.notes,
    )
    store = RecordingDB()
    asyncio.run(ConversationRepository(store).save_extraction_run(run))

    inserted = {name for action, name in store.calls if action == "insert"}
    upserted = {name for action, name in store.calls if action == "upsert"}
    deleted = {name for action, name in store.calls if action == "delete"}
    assert inserted == {"stagedTasks", "stagedNotes"}
    assert deleted == {"stagedTasks", "stagedNotes", "stagedDecisions", "stagedIssues"}
    assert upserted == {"tasks", "notes"}
    saved_task = store.docs["tasks"][0]
    saved_note = store.docs["notes"][0]
    assert saved_task["title"] == "Add configurable grace period"
    assert saved_task["priority"] == "High"
    assert saved_task["status"] == "pending"
    assert saved_task["source"] == "meeting"
    assert saved_task["operation"] == "CREATE"
    assert isinstance(saved_task["conversationId"], ObjectId)
    assert "Expected result:" in saved_task["body"]
    assert saved_note["title"] == "Attendance timing"
    assert saved_note["body"] == "Grace is ten minutes after the shift starts."
    assert "priority" not in saved_note
    assert {item["title"] for item in store.docs["stagedTasks"]} == {"Add configurable grace period"}
    assert {item["title"] for item in store.docs["stagedNotes"]} == {"Attendance timing"}
