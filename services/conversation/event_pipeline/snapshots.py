"""Compact per-stage snapshots so a long transcript can be traced end-to-end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.flags import debug_snapshots_enabled
from services.conversation.event_pipeline.textutil import evidence_sequence_ids


def build_snapshots(
    *,
    cleaning,
    blocks,
    topics,
    events,
    threads,
    actions,
    memory,
    tasks,
    notes,
    coverage,
) -> dict[str, Any]:
    sequence_to_block = {}
    for block in blocks or []:
        for sequence in block.sequenceIds:
            sequence_to_block[sequence] = block.microBlockId
    block_to_topic = {}
    for topic in topics or []:
        for block_id in topic.microBlockIds:
            block_to_topic[block_id] = topic.topicId
    event_to_thread = {event.eventId: event.threadId for event in events or []}
    traces = []
    for task in tasks or []:
        traces.append(_artifact_trace(task, "task", sequence_to_block, block_to_topic, events, event_to_thread))
    for note in notes or []:
        traces.append(_artifact_trace(note, "note", sequence_to_block, block_to_topic, events, event_to_thread))
    return {
        "CLEANED_SEQUENCES": _cleaned(cleaning),
        "MICRO_BLOCKS": [
            {
                "microBlockId": block.microBlockId,
                "sequenceIds": list(block.sequenceIds),
                "tokenCount": block.tokenCount,
                "text": (block.text or "")[:240],
            }
            for block in blocks or []
        ],
        "TOPICS": [
            {
                "topicId": topic.topicId,
                "label": topic.label,
                "microBlockIds": list(topic.microBlockIds),
                "sequenceStart": topic.sequenceStart,
                "sequenceEnd": topic.sequenceEnd,
                "entities": list(topic.entities or []),
                "coherence": getattr(topic, "coherence", None),
                "boundaryReason": getattr(topic, "boundaryReason", None),
                "tokenCount": getattr(topic, "tokenCount", None),
            }
            for topic in topics or []
        ],
        "ATOMIC_EVENTS": [
            {
                "eventId": event.eventId,
                "kind": event.kind.value if hasattr(event.kind, "value") else event.kind,
                "meaning": event.meaning,
                "object": event.object,
                "threadId": event.threadId,
                "channel": event.channel,
                "disposition": event.disposition.value if event.disposition else None,
                "dispositionReason": event.dispositionReason,
                "sequenceIds": list(event.sequenceIds),
                "microBlockIds": list(event.microBlockIds),
                "actionable": bool(getattr(event.actionSignal, "isActionable", False)),
                "actionStrength": getattr(event.actionSignal, "actionStrength", None) if event.actionSignal else None,
                "memoryWorthy": bool(getattr(event.memorySignal, "isMemoryWorthy", False)),
                "actionVerb": getattr(event.actionSignal, "verb", None) if event.actionSignal else None,
                "actionObject": getattr(event.actionSignal, "object", None) if event.actionSignal else event.object,
                "objectGroundingType": getattr(event.actionSignal, "objectGroundingType", None) if event.actionSignal else None,
                "artifactStatus": getattr(event.actionSignal, "artifactStatus", None) if event.actionSignal else None,
            }
            for event in events or []
        ],
        "GLOBAL_THREADS": [
            {
                "threadId": thread.threadId,
                "label": thread.label,
                "eventIds": list(thread.eventIds),
                "entities": list(thread.entities or []),
                "sequenceStart": thread.sequenceStart,
                "sequenceEnd": thread.sequenceEnd,
            }
            for thread in threads or []
        ],
        "ACTION_EVENTS": [event.eventId for event in actions or []],
        "MEMORY_EVENTS": [event.eventId for event in memory or []],
        "TASK_CANDIDATES": [_artifact_card(item) for item in tasks or []],
        "NOTE_CANDIDATES": [_artifact_card(item) for item in notes or []],
        "VALIDATED_ARTIFACTS": {
            "tasks": [_artifact_card(item) for item in tasks or []],
            "notes": [_artifact_card(item) for item in notes or []],
        },
        "COVERAGE_LEDGER": coverage.as_metrics() if coverage is not None else {},
        "TRACES": traces,
    }


def persist_snapshots(conversation_id: str, snapshots: dict[str, Any]) -> str | None:
    if not debug_snapshots_enabled():
        return None
    directory = str(getattr(settings, "EVENT_PIPELINE_DEBUG_SNAPSHOT_DIR", "") or "").strip() or "tmp/event_pipeline_snapshots"
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{conversation_id}.json"
    target.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(target)


def format_traces(snapshots: dict[str, Any]) -> str:
    lines = []
    for trace in snapshots.get("TRACES") or []:
        parts = [
            f"sequence {trace.get('sequence')}",
            str(trace.get("microBlockId") or "MB_?"),
            f"topic {trace.get('topicId') or '?'}",
            f"event {trace.get('eventId') or '?'}",
            f"thread {trace.get('threadId') or '?'}",
            f"{trace.get('kind')} {json.dumps(trace.get('title') or '', ensure_ascii=False)}",
        ]
        lines.append(" → ".join(parts))
    return "\n".join(lines)


def _cleaned(cleaning) -> list[dict[str, Any]]:
    if cleaning is None:
        return []
    rows = []
    for record in getattr(cleaning, "records", []) or []:
        rows.append(
            {
                "sequenceId": record.sequenceId,
                "excluded": record.excluded,
                "reason": record.exclusionReason,
                "text": (record.rawText or "")[:180],
            }
        )
    return rows


def _artifact_card(item) -> dict[str, Any]:
    metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
    return {
        "title": getattr(item, "title", ""),
        "body": (getattr(item, "body", "") or "")[:240],
        "evidenceSequences": evidence_sequence_ids(getattr(item, "evidence", [])),
        "eventId": metadata.get("eventId"),
        "threadId": metadata.get("threadId"),
        "threadContextEvents": metadata.get("threadContextEvents") or [],
        "artifactEvidence": metadata.get("artifactEvidence") or evidence_sequence_ids(getattr(item, "evidence", [])),
    }


def _artifact_trace(item, kind: str, sequence_to_block, block_to_topic, events, event_to_thread) -> dict[str, Any]:
    metadata = getattr(item, "changes", None) or getattr(item, "debug", None) or {}
    sequences = evidence_sequence_ids(getattr(item, "evidence", []))
    sequence = sequences[0] if sequences else None
    event_id = metadata.get("eventId") or (metadata.get("sourceSemanticUnitIds") or [None])[0]
    micro_id = metadata.get("microBlockId") or sequence_to_block.get(sequence)
    topic_id = metadata.get("topicId") or block_to_topic.get(micro_id)
    thread_id = metadata.get("threadId") or event_to_thread.get(event_id)
    return {
        "kind": kind,
        "title": getattr(item, "title", ""),
        "sequence": sequence,
        "microBlockId": micro_id,
        "topicId": topic_id,
        "eventId": event_id,
        "threadId": thread_id,
    }
