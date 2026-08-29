"""Thread-linking gold: interleaved topics plus confusing shared entities."""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import AtomicEvent, EventKind
from services.conversation.models import EvidenceSpan


def _event(event_id: str, kind: EventKind, meaning: str, sequence: int, text: str, **kwargs) -> AtomicEvent:
    return AtomicEvent(
        eventId=event_id,
        topicId=kwargs.get("topicId", "T"),
        kind=kind,
        meaning=meaning,
        object=kwargs.get("object"),
        entities=kwargs.get("entities") or [],
        evidence=[EvidenceSpan(sequenceStart=sequence, sequenceEnd=sequence, text=text)],
        sequenceIds=[sequence],
        sourceIds=[f"chunk_{sequence}"],
        conversationId="thread-gold",
        userId="u",
        spaceId="s",
    )


def interleaved_topic_events() -> dict:
    events = [
        _event("e-s3-1", EventKind.ISSUE, "S3 has a problem.", 10, "S3 has a problem.", object="S3", entities=["S3"]),
        _event("e-price-1", EventKind.PROPOSAL, "Pricing should start around 200.", 20, "Pricing should start around 200.", object="pricing", entities=["Pricing"]),
        _event("e-db-1", EventKind.ISSUE, "Database connection failed.", 30, "database connection failed", object="database connection", entities=["database"]),
        _event("e-s3-2", EventKind.FACT, "S3 configuration was changed.", 40, "S3 configuration was changed", object="S3 configuration", entities=["S3"]),
        _event("e-store-1", EventKind.ISSUE, "Play Store issue exists.", 50, "Play Store issue exist karta hai", object="Play Store issue", entities=["Play", "Store"]),
        _event("e-s3-3", EventKind.STATE, "S3 still does not reach frontend.", 60, "S3 still does not reach frontend", object="S3 frontend", entities=["S3", "frontend"]),
    ]
    return {
        "events": events,
        "goldClusters": [
            ["e-s3-1", "e-s3-2", "e-s3-3"],
            ["e-price-1"],
            ["e-db-1"],
            ["e-store-1"],
        ],
    }


def confusing_server_events() -> dict:
    events = [
        _event("e-id", EventKind.REQUEST, "Create server ID.", 1, "Server ID create karna hai", object="server ID", entities=["Server", "ID"]),
        _event("e-conn", EventKind.ISSUE, "Server connection failed.", 2, "server connection failure", object="server connection", entities=["server", "connection"]),
        _event("e-price", EventKind.PROPOSAL, "Server pricing starts at 200.", 3, "server pricing plan 200", object="server pricing", entities=["server", "pricing"]),
    ]
    return {
        "events": events,
        "goldClusters": [["e-id"], ["e-conn"], ["e-price"]],
        "mustNotMerge": True,
    }
