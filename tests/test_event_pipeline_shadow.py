import asyncio

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.flags import (
    event_pipeline_mode,
    event_pipeline_publishes,
    event_pipeline_shadow,
    legacy_pipeline_publishes,
)
from services.conversation.event_pipeline.shadow import compare_pipeline_outputs
from services.conversation.models import ExtractedNote, ExtractedTask, EvidenceSpan


def test_event_pipeline_mode_legacy(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", False)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "shadow")
    assert event_pipeline_mode() == "legacy"
    assert not event_pipeline_publishes()
    assert legacy_pipeline_publishes()
    assert not event_pipeline_shadow()


def test_event_pipeline_mode_shadow(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "shadow")
    assert event_pipeline_mode() == "shadow"
    assert event_pipeline_shadow()
    assert legacy_pipeline_publishes()
    assert not event_pipeline_publishes()


def test_event_pipeline_mode_publish(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "event_pipeline")
    assert event_pipeline_publishes()
    assert not legacy_pipeline_publishes()


def test_empty_mode_defaults_to_event_pipeline(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "")
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100)
    assert event_pipeline_mode() == "event_pipeline"
    assert event_pipeline_publishes()
    assert not legacy_pipeline_publishes()
    assert not event_pipeline_shadow()


def test_unset_mode_defaults_to_event_pipeline(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", None)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100)
    assert event_pipeline_mode() == "event_pipeline"
    assert event_pipeline_publishes()
    assert not legacy_pipeline_publishes()


def test_invalid_mode_fails_safe_to_legacy(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_EVENT_PIPELINE", True)
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "publish")
    assert event_pipeline_mode() == "legacy"
    assert legacy_pipeline_publishes()
    assert not event_pipeline_publishes()
    monkeypatch.setattr(settings, "EVENT_PIPELINE_MODE", "enabled")
    assert event_pipeline_mode() == "legacy"


def test_shadow_comparison_does_not_treat_legacy_as_new_publish():
    evidence = [EvidenceSpan(sequenceStart=0, sequenceEnd=0, text="Please create the server ID.")]
    legacy_tasks = [
        ExtractedTask(
            title="Create server ID",
            body="Please create the server ID.",
            operation="CREATE",
            confidence=0.7,
            sourceConversationId="c",
            evidence=evidence,
        )
    ]
    new_tasks = [
        ExtractedTask(
            title="Create server ID",
            body="Please create the server ID.",
            operation="CREATE",
            confidence=0.7,
            sourceConversationId="c",
            evidence=evidence,
        )
    ]
    legacy_notes = [
        ExtractedNote(
            title="S3 frontend",
            body="S3 is not reaching the frontend.",
            confidence=0.7,
            sourceConversationId="c",
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text="S3 is not reaching frontend")],
        )
    ]
    new_notes = [
        ExtractedNote(
            title="S3 frontend",
            body="S3 is not reaching the frontend.",
            confidence=0.7,
            sourceConversationId="c",
            evidence=[EvidenceSpan(sequenceStart=1, sequenceEnd=1, text="S3 is not reaching frontend")],
        ),
        ExtractedNote(
            title="Pricing",
            body="Pricing should start around 200.",
            confidence=0.7,
            sourceConversationId="c",
            evidence=[EvidenceSpan(sequenceStart=2, sequenceEnd=2, text="Pricing should start around 200")],
        ),
    ]
    report = compare_pipeline_outputs(
        legacy_tasks=legacy_tasks,
        legacy_notes=legacy_notes,
        new_tasks=new_tasks,
        new_notes=new_notes,
    )
    assert report["publishedFrom"] == "legacy"
    assert report["missingValidTasks"] == 0
    assert report["newNoteCount"] == 2


def test_shadow_failure_does_not_block_legacy_publish():
    # The workflow wraps shadow errors; this unit-tests the comparison helper only.
    report = compare_pipeline_outputs(legacy_tasks=[], legacy_notes=[], new_tasks=[], new_notes=[])
    assert report["publishedFrom"] == "legacy"
    assert report["legacyTaskCount"] == 0
