"""Mechanical-window recall pipeline for meeting Tasks/Notes.

Transcript → mechanical windows → recall extraction → candidate ledger →
global consolidation → composition harness → evidence verification →
invariant gate → persist.

Python owns transport, provenance, schemas, composition, and persistence safety.
The LLM owns language understanding. This package does not hardcode domain
semantics, keywords, or meeting-specific templates.
"""

from services.conversation.meeting_pipeline.flags import meeting_pipeline_enabled
from services.conversation.meeting_pipeline.pipeline import run_meeting_pipeline
from services.conversation.meeting_pipeline.schemas import MeetingPipelineResult

__all__ = [
    "MeetingPipelineResult",
    "meeting_pipeline_enabled",
    "run_meeting_pipeline",
]
