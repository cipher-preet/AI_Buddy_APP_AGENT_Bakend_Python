"""Real Gemma action-object extraction. Skipped without credentials."""

from __future__ import annotations

import asyncio

import pytest

from services.conversation.event_pipeline.channels import event_is_actionable, is_generic_task_text, unresolved_action_object
from services.conversation.event_pipeline.embeddings import default_embedder
from services.conversation.event_pipeline.gold_scoring import score_action_signals
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.conversation.event_pipeline.schemas import ActionSignal, AtomicEvent, MemorySignal
from services.conversation.models import EvidenceSpan, STTStatus, TranscriptChunkDocument
from services.llm.router import get_llm_router
from tests.fixtures.action_object_gold import CASES
from tests.integration.conftest import requires_real_models


pytestmark = [pytest.mark.integration, pytest.mark.real_models, requires_real_models]


def test_action_object_real_gemma():
    chunks = []
    gold_events = []
    for index, case in enumerate(CASES):
        sequence = index * 20
        for offset, text in enumerate([case["text"], case["text"]]):
            chunks.append(
                TranscriptChunkDocument(
                    conversationId="action-object-real",
                    userId="u",
                    spaceId="s",
                    chunkId=f"chunk_{sequence + offset}",
                    sequenceNumber=sequence + offset,
                    rawText=text,
                    sttStatus=STTStatus.COMPLETED,
                )
            )
        gold_events.append(
            AtomicEvent(
                eventId=case["id"],
                topicId="T1",
                kind=case["kind"],
                meaning=case["text"],
                object=case.get("object"),
                actor=case.get("actor"),
                timeExpression=case.get("deadline"),
                evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=case["text"])],
                sequenceIds=[sequence],
                actionSignal=ActionSignal(
                    isActionable=bool(case["actionable"]),
                    verb=case.get("verb"),
                    object=case.get("object"),
                    actor=case.get("actor"),
                    deadline=case.get("deadline"),
                ),
                memorySignal=MemorySignal(isMemoryWorthy=not case["actionable"] or case["kind"].value in {"REQUIREMENT", "ISSUE", "STATE"}),
            )
        )
    result = asyncio.run(
        run_event_pipeline(
            chunks,
            "action-object-real",
            "u",
            "s",
            router=get_llm_router(),
            embedder=default_embedder(prefer_provider=True, lexical_fallback=False),
            polish_with_llm=False,
        )
    )
    metrics = score_action_signals(result.events, gold_events)
    print("ACTION_OBJECT_GOLD_REAL", metrics)
    print(
        "ACTION_OBJECT_EVENTS",
        [
            {
                "kind": event.kind.value,
                "meaning": event.meaning,
                "actionable": event_is_actionable(event),
                "verb": getattr(event.actionSignal, "verb", None) if event.actionSignal else None,
                "object": event.object,
                "status": getattr(event.actionSignal, "artifactStatus", None) if event.actionSignal else None,
                "sequences": list(event.sequenceIds),
            }
            for event in result.events
            if event.kind.value != "NOISE"
        ],
    )
    print("ACTION_OBJECT_TASKS", [task.title for task in result.tasks])
    assert result.observability.gemmaCalls >= 1
    assert metrics["genericActionRate"] == 0 or metrics["genericActionRate"] <= 0.15
    assert not any(is_generic_task_text(task.title, task.body) for task in result.tasks)
    unresolved = [event for event in result.events if event_is_actionable(event) and unresolved_action_object(event)]
    for event in unresolved:
        related = [task for task in result.tasks if event.eventId in ((task.changes or {}).get("sourceSemanticUnitIds") or [])]
        assert related == []
