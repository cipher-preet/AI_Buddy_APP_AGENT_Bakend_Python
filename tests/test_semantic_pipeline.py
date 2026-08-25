import asyncio
from types import SimpleNamespace

from services.conversation import agents
from services.conversation.artifact_resolver import (
    reconcile_incoming_artifacts,
    resolve_incoming_artifacts,
)
from services.conversation.artifacts import artifacts_from_window
from services.conversation.budget import semantic_window_token_target, semantic_window_useful_duration_ms
from services.conversation.intelligence import validation_decision_for_note, validation_decision_for_task
from services.conversation.models import (
    ArtifactReconcileDecision,
    ConversationDocument,
    ConversationWindowDocument,
    EvidenceSpan,
    ExtractionOutcome,
    ExtractedTask,
    ReconcileAction,
    SemanticUnit,
    TranscriptChunkDocument,
    WindowExtractionResult,
    WindowProcessingStatus,
)
from services.conversation.transcript import estimate_tokens
from services.conversation.windowing import (
    CLOSE_REASON_DURATION_MAX,
    CLOSE_REASON_FORCED_FINAL,
    CLOSE_REASON_TOKEN_MAX,
    CLOSE_REASON_TOKEN_TARGET,
    build_ready_windows,
)
from services.conversation.workflow import _is_completed_checkpoint
from tests.fixtures.conversation_meetings import (
    COMPLEX_MEETING_TRANSCRIPT,
    complex_meeting_result,
    evidence,
    failing_router,
    scripted_note,
    scripted_router,
    scripted_task,
)


def _conversation():
    return ConversationDocument(_id="conv_1", userId="user_1", spaceId="space_1")


def _chunk(sequence: int, text: str, duration_ms: int = 30_000) -> TranscriptChunkDocument:
    return TranscriptChunkDocument(
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        chunkId=f"chunk_{sequence}",
        sequenceNumber=sequence,
        rawText=text,
        endTimeMs=duration_ms,
    )


def _window(text: str, **kwargs):
    return SimpleNamespace(
        text=text,
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        id=kwargs.get("id", "window"),
        sequenceStart=kwargs.get("sequenceStart", 0),
        sequenceEnd=kwargs.get("sequenceEnd", 10),
        windowIndex=kwargs.get("windowIndex", 0),
        isFinalPartial=kwargs.get("isFinalPartial", False),
        extractionSkipped=kwargs.get("extractionSkipped", False),
        status=kwargs.get("status", WindowProcessingStatus.COMPLETED),
        result=kwargs.get("result"),
    )


def test_short_sessions_do_not_close_before_stop(monkeypatch):
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_TRANSCRIPT_TOKENS", 22000)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 22000)
    conversation = _conversation()
    chunks = [_chunk(0, "A two minute useful conversation about the drain-safe STOP lifecycle.")]
    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=False)
    assert windows == []
    final = build_ready_windows(conversation, chunks, start_index=0, force_final=True)
    assert len(final) == 1
    assert final[0].window.isFinalPartial is True
    assert final[0].window.closeReason == CLOSE_REASON_FORCED_FINAL


def test_thirty_and_fifty_nine_minute_sessions_stay_raw(monkeypatch):
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_USEFUL_MINUTES", 60)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_MAX_DURATION_MS", 60 * 60 * 1000)
    conversation = _conversation()
    chunks = [
        _chunk(0, "Discussed the pending STT drain.", duration_ms=29 * 60 * 1000),
        _chunk(1, "STOP after a fifty-nine minute working session.", duration_ms=30 * 60 * 1000),
    ]
    live = build_ready_windows(conversation, chunks, start_index=0, force_final=False)
    assert live == []
    stopped = build_ready_windows(conversation, chunks, start_index=0, force_final=True)
    assert len(stopped) == 1
    assert stopped[0].window.isFinalPartial is True


def test_useful_hour_closes_a_semantic_checkpoint_window(monkeypatch):
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_USEFUL_MINUTES", 60)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_MAX_DURATION_MS", 60 * 60 * 1000)
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_TRANSCRIPT_TOKENS", 50000)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 50000)
    conversation = _conversation()
    chunks = [
        _chunk(0, "Hour one technical discussion continues with useful speech.", duration_ms=40 * 60 * 1000),
        _chunk(1, "Still in the first hour of useful transcribed speech.", duration_ms=25 * 60 * 1000),
    ]
    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=False)
    assert len(windows) == 1
    assert windows[0].window.closeReason == CLOSE_REASON_DURATION_MAX
    assert windows[0].window.isFinalPartial is False


def test_token_budget_closes_before_sixty_minutes(monkeypatch):
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_TRANSCRIPT_TOKENS", 40)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 40)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_MAX_TOKENS", 50)
    conversation = _conversation()
    chunks = [
        _chunk(index, " ".join(f"token{n}" for n in range(20)), duration_ms=60_000)
        for index in range(6)
    ]
    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=False)
    assert len(windows) >= 1
    assert windows[0].window.closeReason in {CLOSE_REASON_TOKEN_TARGET, CLOSE_REASON_TOKEN_MAX}


def test_long_silence_does_not_close_a_window(monkeypatch):
    monkeypatch.setattr("services.conversation.windowing.settings.SPARSE_WINDOW_MAX_WALL_CLOCK_MS", 0)
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_TRANSCRIPT_TOKENS", 22000)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 22000)
    conversation = _conversation()
    chunks = [
        _chunk(0, "A little useful speech.", duration_ms=5_000),
        _chunk(1, "", duration_ms=20 * 60 * 1000),
        _chunk(2, "", duration_ms=20 * 60 * 1000),
    ]
    windows = build_ready_windows(conversation, chunks, start_index=0, force_final=False)
    assert windows == []


def test_final_partial_is_not_a_completed_checkpoint():
    window = _window("leftover raw", isFinalPartial=True, extractionSkipped=True, result=WindowExtractionResult())
    assert _is_completed_checkpoint(window) is False


def test_completed_checkpoint_window_is_detected():
    result = WindowExtractionResult(isCheckpoint=True, semanticUnits=[
        SemanticUnit(semanticKey="k1", kind="commitment", meaning="Rahul will prepare the proposal.", evidence=[evidence("Rahul will prepare the proposal.", 1)])
    ])
    window = _window("checkpoint text", result=result)
    assert _is_completed_checkpoint(window) is True


def test_no_model_abstains_instead_of_keyword_reconstruction():
    window = _window("[1] Please arrange the impossible thing tomorrow.")
    result, _, _ = asyncio.run(agents.extract_window(failing_router(), window, context={}, meeting_context={}, mode="final"))
    assert result.extractionOutcome == ExtractionOutcome.EXTRACTION_FAILED
    assert not result.tasks and not result.notes


def test_short_raw_extraction_uses_scripted_semantics_not_exact_prose():
    result, provider, _ = asyncio.run(
        agents.extract_from_raw_transcript(
            scripted_router(complex_meeting_result()),
            "conv_1",
            "user_1",
            "space_1",
            COMPLEX_MEETING_TRANSCRIPT,
            {},
        )
    )
    assert provider == "scripted-test-provider"
    assert any("sequence" in task.body.casefold() or "drain" in task.body.casefold() for task in result.tasks)
    assert any("raw transcript" in note.body.casefold() for note in result.notes)
    for item in [*result.tasks, *result.notes]:
        assert item.evidence
        assert all(span.text in COMPLEX_MEETING_TRANSCRIPT for span in item.evidence)


def test_invented_evidence_ids_fail_validation():
    transcript = "[1] Mira will wait for every expected sequence ID."
    task = scripted_task(
        "Wait for sequences",
        "Mira will wait for every expected sequence ID.",
        9,
        "Mira will wait for every expected sequence ID.",
    )
    keep, reason = validation_decision_for_task(task, transcript)
    assert keep is False
    assert reason == "evidence_sequence_mismatch"


def test_grounded_task_and_note_survive_with_real_evidence():
    transcript = COMPLEX_MEETING_TRANSCRIPT
    task = complex_meeting_result().tasks[0]
    note = complex_meeting_result().notes[0]
    assert validation_decision_for_task(task, transcript)[0] is True
    assert validation_decision_for_note(note, transcript)[0] is True


def _decision(action: ReconcileAction, target_id: str | None, evidence: list[EvidenceSpan], reason: str = "") -> ArtifactReconcileDecision:
    return ArtifactReconcileDecision(
        incomingIndex=0,
        action=action,
        targetArtifactId=target_id,
        evidence=evidence,
        reason=reason,
    )


def _reconcile_router(decisions: list[ArtifactReconcileDecision]):
    from services.llm.router import LLMCapability

    class _Provider:
        name = "scripted-reconcile-provider"

        async def generate_structured(self, request, schema):
            name = getattr(schema, "__name__", "")
            if name == "ArtifactReconcileResponse":
                return schema(decisions=decisions)
            return schema()

    class _Router:
        def route(self, capability: LLMCapability):
            return _Provider(), "scripted-reconcile-model"

    return _Router()


def test_paraphrased_commitments_share_semantic_identity():
    text = "[1] Rahul needs to deploy the backend tomorrow.\n[2] The backend release is Rahul's responsibility tomorrow."
    first = artifacts_from_window(
        _window(text),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="window-1-opaque-deploy",
                    kind="commitment",
                    meaning="Rahul needs to deploy the backend tomorrow.",
                    evidence=[evidence("Rahul needs to deploy the backend tomorrow.", 1)],
                )
            ],
        ),
    )
    second = artifacts_from_window(
        _window(text, id="window-2"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="window-2-different-opaque-key",
                    kind="commitment",
                    meaning="The backend release is Rahul's responsibility tomorrow.",
                    evidence=[evidence("The backend release is Rahul's responsibility tomorrow.", 2)],
                )
            ],
        ),
    )
    assert first[0].semanticHint != second[0].semanticHint
    assert first[0].identityKey != second[0].identityKey
    merged = resolve_incoming_artifacts(
        first,
        second,
        [_decision(ReconcileAction.UPDATE_EXISTING, str(first[0].id), second[0].evidence, "same commitment restated")],
    )
    assert len(merged) == 1
    assert len(merged[0].evidence) >= 2
    assert merged[0].history[-1].evidence[0].text == "The backend release is Rahul's responsibility tomorrow."


def test_divergent_semantic_keys_for_same_task_reconcile_to_one_artifact():
    first = artifacts_from_window(
        _window("[1] Mira will file the drain-safe finalizer notes.", id="w1"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="opaque-key-hour-1",
                    kind="commitment",
                    meaning="Mira will file the drain-safe finalizer notes.",
                    evidence=[evidence("Mira will file the drain-safe finalizer notes.", 1)],
                )
            ],
        ),
    )
    second = artifacts_from_window(
        _window("[40] Mira still owns writing those drain-safe finalizer notes.", id="w2"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="opaque-key-hour-2",
                    kind="commitment",
                    meaning="Mira still owns writing those drain-safe finalizer notes.",
                    evidence=[evidence("Mira still owns writing those drain-safe finalizer notes.", 40)],
                )
            ],
        ),
    )
    assert first[0].semanticHint != second[0].semanticHint
    without_llm = resolve_incoming_artifacts(first, second)
    assert len(without_llm) == 2
    merged = asyncio.run(
        reconcile_incoming_artifacts(
            _reconcile_router(
                [_decision(ReconcileAction.UPDATE_EXISTING, str(first[0].id), second[0].evidence, "same task, different keys")]
            ),
            first,
            second,
            "[40] Mira still owns writing those drain-safe finalizer notes.",
        )
    )
    assert len(merged) == 1
    assert {span.sequenceStart for span in merged[0].evidence} == {1, 40}
    assert merged[0].history[-1].evidence[0].text == "Mira still owns writing those drain-safe finalizer notes."


def test_similar_vocabulary_keeps_distinct_actions():
    text = "[1] Rahul needs to deploy the backend.\n[2] Rahul needs to test the backend."
    first = artifacts_from_window(
        _window(text, id="w1"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="deploy-backend",
                    kind="action_candidate",
                    meaning="Rahul needs to deploy the backend.",
                    evidence=[evidence("Rahul needs to deploy the backend.", 1)],
                )
            ],
        ),
    )
    second = artifacts_from_window(
        _window(text, id="w2"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="test-backend",
                    kind="action_candidate",
                    meaning="Rahul needs to test the backend.",
                    evidence=[evidence("Rahul needs to test the backend.", 2)],
                )
            ],
        ),
    )
    merged = asyncio.run(
        reconcile_incoming_artifacts(
            _reconcile_router(
                [_decision(ReconcileAction.RELATED_BUT_DISTINCT, str(first[0].id), second[0].evidence, "different actions")]
            ),
            first,
            second,
            text,
        )
    )
    assert len(merged) == 2
    meanings = {item.content for item in merged}
    assert "Rahul needs to deploy the backend." in meanings
    assert "Rahul needs to test the backend." in meanings


def test_later_window_updates_earlier_artifact_state():
    hour1 = artifacts_from_window(
        _window("[1] Rahul will prepare the pricing proposal.", id="h1"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="pricing-hour-1",
                    kind="commitment",
                    meaning="Rahul will prepare the pricing proposal.",
                    state="proposed",
                    evidence=[evidence("Rahul will prepare the pricing proposal.", 1)],
                )
            ],
        ),
    )
    hour3 = artifacts_from_window(
        _window("[80] Rahul already sent the proposal to the customer.", id="h3"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="pricing-hour-3-complete",
                    kind="completion",
                    meaning="Rahul already sent the proposal to the customer.",
                    state="completed",
                    evidence=[evidence("Rahul already sent the proposal to the customer.", 80)],
                )
            ],
        ),
    )
    merged = resolve_incoming_artifacts(
        hour1,
        hour3,
        [_decision(ReconcileAction.COMPLETE_EXISTING, str(hour1[0].id), hour3[0].evidence, "commitment completed")],
    )
    assert len(merged) == 1
    assert merged[0].status.value == "completed"
    assert {span.sequenceStart for span in merged[0].evidence} == {1, 80}


def test_hindi_hinglish_and_invented_vocabulary_use_semantic_keys():
    text = "\n".join(
        [
            "[1] Kal zafran-lens ko andhere mein rakhna hai.",
            "[2] Keep the zafran-lens away from bright lamps overnight.",
            "[3] Vireli pollen changes the dusk glass.",
        ]
    )
    result = WindowExtractionResult(
        isCheckpoint=True,
        semanticUnits=[
            SemanticUnit(
                semanticKey="zafran-lens-dark",
                kind="requirement",
                meaning="Keep the zafran lens in darkness overnight.",
                evidence=[
                    evidence("Kal zafran-lens ko andhere mein rakhna hai.", 1),
                    evidence("Keep the zafran-lens away from bright lamps overnight.", 2),
                ],
            ),
            SemanticUnit(
                semanticKey="vireli-pollen",
                kind="fact",
                meaning="Vireli pollen changes dusk glass color.",
                evidence=[evidence("Vireli pollen changes the dusk glass.", 3)],
            ),
        ],
    )
    artifacts = artifacts_from_window(_window(text), result)
    assert len(artifacts) == 2
    assert len({item.identityKey for item in artifacts}) == 2


def test_eight_hour_final_payload_stays_bounded():
    checkpoints = []
    for index in range(8):
        checkpoints.append(
            {
                "windowIndex": index,
                "narrative": f"Semantic checkpoint {index} with bounded meaning and evidence ids.",
                "semanticUnits": [{"semanticKey": f"k{index}", "meaning": "A compact unit.", "evidence": [{"sequenceStart": index, "sequenceEnd": index, "text": "evidence"}]}],
            }
        )
    leftover = "[800] Remaining thirty five minutes of raw speech about the unfinished window."
    payload = {"semanticCheckpoints": checkpoints, "leftoverRawTranscript": leftover, "artifacts": [{"title": "one", "content": "compact"}]}
    assert estimate_tokens(str(payload)) < 20000


def test_discussion_without_action_does_not_force_a_task():
    text = "[1] A nural weave absorbs vibration across its outer rings."
    result = WindowExtractionResult(
        isCheckpoint=True,
        semanticUnits=[
            SemanticUnit(
                semanticKey="nural-weave",
                kind="fact",
                meaning="A nural weave absorbs vibration across its outer rings.",
                evidence=[evidence("A nural weave absorbs vibration across its outer rings.", 1)],
            )
        ],
    )
    artifacts = artifacts_from_window(_window(text), result)
    assert artifacts
    assert all(item.artifactType.value != "task" for item in artifacts)


def test_finalization_does_not_requeue_raw_passthrough_windows():
    from services.conversation.finalization import _should_publish_window_job
    from datetime import datetime, timezone

    window = ConversationWindowDocument(
        conversationId="conv_1",
        userId="user_1",
        spaceId="space_1",
        windowIndex=0,
        sequenceStart=0,
        sequenceEnd=1,
        text="leftover",
        tokenCount=4,
        status=WindowProcessingStatus.PENDING,
        isFinalPartial=True,
        extractionSkipped=True,
    )
    assert _should_publish_window_job(window, datetime.now(timezone.utc)) is False


def test_two_hour_thirty_session_creates_hourly_checkpoints_and_raw_remainder(monkeypatch):
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_USEFUL_MINUTES", 60)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_MAX_DURATION_MS", 60 * 60 * 1000)
    monkeypatch.setattr("services.conversation.budget.settings.SEMANTIC_WINDOW_MAX_TRANSCRIPT_TOKENS", 100000)
    monkeypatch.setattr("services.conversation.budget.settings.INCREMENTAL_WINDOW_TARGET_TOKENS", 100000)
    conversation = _conversation()
    chunks = [
        _chunk(0, "Hour one useful discussion.", duration_ms=60 * 60 * 1000),
        _chunk(1, "Hour two useful discussion.", duration_ms=60 * 60 * 1000),
        _chunk(2, "Final thirty five minutes remain raw.", duration_ms=35 * 60 * 1000),
    ]
    live = build_ready_windows(conversation, chunks[:2], start_index=0, force_final=False)
    assert len(live) == 1
    remainder = build_ready_windows(conversation, [chunks[1], chunks[2]], start_index=1, force_final=True)
    assert remainder
    assert remainder[-1].window.isFinalPartial is True
    assert remainder[-1].window.closeReason == CLOSE_REASON_FORCED_FINAL


def test_cancelled_and_completed_states_are_model_driven():
    text = "[1] Rahul will prepare the pricing proposal.\n[2] Cancel the pricing proposal; we will not send it."
    first = artifacts_from_window(
        _window(text, id="h1"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="pricing-create-key",
                    kind="commitment",
                    meaning="Rahul will prepare the pricing proposal.",
                    state="proposed",
                    evidence=[evidence("Rahul will prepare the pricing proposal.", 1)],
                )
            ],
        ),
    )
    later = artifacts_from_window(
        _window(text, id="h2"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="pricing-cancel-key",
                    kind="cancellation",
                    meaning="Cancel the pricing proposal; we will not send it.",
                    state="cancelled",
                    evidence=[evidence("Cancel the pricing proposal; we will not send it.", 2)],
                )
            ],
        ),
    )
    merged = resolve_incoming_artifacts(
        first,
        later,
        [_decision(ReconcileAction.CANCEL_EXISTING, str(first[0].id), later[0].evidence, "commitment cancelled")],
    )
    assert len(merged) == 1
    assert merged[0].status.value == "cancelled"
    assert merged[0].history[-1].evidence[0].text == "Cancel the pricing proposal; we will not send it."


def test_semantic_hint_is_not_authoritative_identity():
    first = artifacts_from_window(
        _window("[1] File the notes.", id="a"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="same-hint",
                    kind="commitment",
                    meaning="File the notes.",
                    evidence=[evidence("File the notes.", 1)],
                )
            ],
        ),
    )
    second = artifacts_from_window(
        _window("[2] File the notes.", id="b"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="same-hint",
                    kind="commitment",
                    meaning="File the notes.",
                    evidence=[evidence("File the notes.", 2)],
                )
            ],
        ),
    )
    assert first[0].semanticHint == second[0].semanticHint == "same-hint"
    assert first[0].identityKey != second[0].identityKey
    assert len(resolve_incoming_artifacts(first, second)) == 2


def test_modifying_action_without_target_id_does_not_create_a_duplicate():
    first = artifacts_from_window(
        _window("[1] File the notes.", id="a"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="k1",
                    kind="commitment",
                    meaning="File the notes.",
                    evidence=[evidence("File the notes.", 1)],
                )
            ],
        ),
    )
    second = artifacts_from_window(
        _window("[2] File the notes before Friday.", id="b"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="k2",
                    kind="commitment",
                    meaning="File the notes before Friday.",
                    evidence=[evidence("File the notes before Friday.", 2)],
                )
            ],
        ),
    )
    merged = resolve_incoming_artifacts(
        first,
        second,
        [_decision(ReconcileAction.UPDATE_EXISTING, None, second[0].evidence)],
    )
    assert len(merged) == 1
    assert str(merged[0].id) == str(first[0].id)
    assert merged[0].content == "File the notes."


def test_unknown_target_id_does_not_create_a_duplicate():
    first = artifacts_from_window(
        _window("[1] File the notes.", id="a"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="k1",
                    kind="commitment",
                    meaning="File the notes.",
                    evidence=[evidence("File the notes.", 1)],
                )
            ],
        ),
    )
    second = artifacts_from_window(
        _window("[2] File the notes before Friday.", id="b"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="k2",
                    kind="commitment",
                    meaning="File the notes before Friday.",
                    evidence=[evidence("File the notes before Friday.", 2)],
                )
            ],
        ),
    )
    merged = resolve_incoming_artifacts(
        first,
        second,
        [_decision(ReconcileAction.COMPLETE_EXISTING, "missing-target-id", second[0].evidence)],
    )
    assert len(merged) == 1
    assert merged[0].status.value != "completed"


def test_missing_target_recovery_can_supply_the_artifact_id():
    first = artifacts_from_window(
        _window("[1] Mira will file the drain-safe finalizer notes.", id="w1"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="opaque-key-hour-1",
                    kind="commitment",
                    meaning="Mira will file the drain-safe finalizer notes.",
                    evidence=[evidence("Mira will file the drain-safe finalizer notes.", 1)],
                )
            ],
        ),
    )
    second = artifacts_from_window(
        _window("[40] Mira still owns writing those drain-safe finalizer notes.", id="w2"),
        WindowExtractionResult(
            isCheckpoint=True,
            semanticUnits=[
                SemanticUnit(
                    semanticKey="opaque-key-hour-2",
                    kind="commitment",
                    meaning="Mira still owns writing those drain-safe finalizer notes.",
                    evidence=[evidence("Mira still owns writing those drain-safe finalizer notes.", 40)],
                )
            ],
        ),
    )
    calls = {"count": 0}

    def _repair_router():
        from services.llm.router import LLMCapability

        class _Provider:
            name = "scripted-reconcile-repair-provider"

            async def generate_structured(self, request, schema):
                name = getattr(schema, "__name__", "")
                if name != "ArtifactReconcileResponse":
                    return schema()
                calls["count"] += 1
                if calls["count"] == 1:
                    return schema(decisions=[_decision(ReconcileAction.UPDATE_EXISTING, None, second[0].evidence)])
                return schema(
                    decisions=[
                        _decision(ReconcileAction.UPDATE_EXISTING, str(first[0].id), second[0].evidence, "repaired target")
                    ]
                )

        class _Router:
            def route(self, capability: LLMCapability):
                return _Provider(), "scripted-reconcile-repair-model"

        return _Router()

    merged = asyncio.run(
        reconcile_incoming_artifacts(
            _repair_router(),
            first,
            second,
            "[40] Mira still owns writing those drain-safe finalizer notes.",
        )
    )
    assert calls["count"] == 2
    assert len(merged) == 1
    assert {span.sequenceStart for span in merged[0].evidence} == {1, 40}


def test_successful_empty_extraction_is_valid_empty():
    window = _window("[1] The soral membrane was mentioned without any request.")
    result, _, _ = asyncio.run(
        agents.extract_window(
            scripted_router(WindowExtractionResult(summary="No actionable units.")),
            window,
            context={},
            meeting_context={},
            mode="final",
        )
    )
    assert result.extractionOutcome == ExtractionOutcome.VALID_EMPTY_EXTRACTION
    assert not result.tasks and not result.notes and not result.semanticUnits


def test_malformed_structured_output_uses_eligible_model_fallback():
    from services.llm.errors import LLMProviderError
    from services.llm.router import LLMCapability

    class _FailingPrimary:
        name = "primary-malformed"

        async def generate_structured(self, request, schema):
            raise LLMProviderError("malformed structured json", retryable=True, status_code=422)

    class _FallbackProvider:
        name = "eligible-fallback"

        async def generate_structured(self, request, schema):
            name = getattr(schema, "__name__", "")
            if name == "WindowExtractionLLMResponse":
                return schema(
                    summary="Recovered extraction.",
                    semanticUnits=[
                        {
                            "semanticKey": "recovered",
                            "kind": "fact",
                            "meaning": "The soral membrane was mentioned without any request.",
                            "evidence": [evidence("The soral membrane was mentioned without any request.", 1).model_dump()],
                        }
                    ],
                )
            return schema()

    class _Router:
        def route(self, capability: LLMCapability):
            if capability == LLMCapability.FALLBACK:
                return _FallbackProvider(), "fallback-model"
            return _FailingPrimary(), "primary-model"

    window = _window("[1] The soral membrane was mentioned without any request.")
    result, provider, _ = asyncio.run(
        agents.extract_window(_Router(), window, context={}, meeting_context={}, mode="checkpoint")
    )
    assert provider == "eligible-fallback"
    assert result.extractionOutcome == ExtractionOutcome.SUCCESS
    assert result.semanticUnits


def test_configured_cost_optimized_primary_is_groq_not_xai_grok():
    from apps.api_gateway.config.setting import settings
    from services.llm.router import LLMCapability, LLMRouter

    assert "api.groq.com" in settings.GROQ_BASE_URL
    assert "x.ai" not in settings.GROQ_BASE_URL.lower()
    assert "grok" not in (settings.GROQ_FREE_MODEL or "").lower()
    router = LLMRouter(
        {
            "groq": SimpleNamespace(name="groq", configured=True),
            "gemini": SimpleNamespace(name="gemini", configured=True),
            "mistral": SimpleNamespace(name="mistral", configured=True),
            "sarvam": SimpleNamespace(name="sarvam", configured=True),
        }
    )
    candidates = router._cost_optimized_candidates(LLMCapability.HIGH_ACCURACY_REASONING)
    assert [item.provider.name for item in candidates] == ["groq", "gemini", "mistral", "sarvam"]
    assert candidates[0].provider.name == "groq"
    assert all(item.provider.name != "xai" for item in candidates)
    assert "xai" not in router.providers
