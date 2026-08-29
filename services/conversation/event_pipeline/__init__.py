"""Evidence-preserving hierarchical conversation intelligence pipeline.

Raw STT → clean → micro-blocks → local topics → atomic events →
global threads → task/note synthesis → validation → coverage → persist.

This package is inserted into the existing conversation-intelligence layer.
It does not replace queue, session, STOP/drain, provider routing, or persistence.
"""

from services.conversation.event_pipeline.flags import (
    event_pipeline_enabled,
    event_pipeline_mode,
    event_pipeline_publishes,
    event_pipeline_selected_for,
    event_pipeline_shadow,
    legacy_pipeline_publishes,
)
from services.conversation.event_pipeline.pipeline import (
    EventPipelineResult,
    extract_window_events,
    run_event_pipeline,
)

__all__ = [
    "EventPipelineResult",
    "event_pipeline_enabled",
    "event_pipeline_mode",
    "event_pipeline_publishes",
    "event_pipeline_selected_for",
    "event_pipeline_shadow",
    "extract_window_events",
    "legacy_pipeline_publishes",
    "run_event_pipeline",
]
