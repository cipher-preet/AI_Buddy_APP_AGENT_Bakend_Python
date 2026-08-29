"""Internal event-pipeline version metadata. Does not change public APIs."""

from __future__ import annotations

PIPELINE_VERSION = "event-hierarchical-v1"
EVENT_SCHEMA_VERSION = "atomic-event-v1"
ARTIFACT_PIPELINE_VERSION = "event-hierarchical-v1"
LEGACY_PIPELINE_VERSION = "legacy-final-synthesis-v1"

PROMPT_VERSIONS = {
    "atomic-event-extractor-v1": "atomic-event-extractor-v1",
    "semantic-completeness-reviewer-v1": "semantic-completeness-reviewer-v1",
    "semantic-completeness-repair-v1": "semantic-completeness-repair-v1",
    "topic-label-v1": "topic-label-v1",
    "thread-membership-v1": "thread-membership-v1",
    "task-synthesizer-v1": "task-synthesizer-v1",
    "note-synthesizer-v1": "note-synthesizer-v1",
    "event-artifact-validator-v1": "event-artifact-validator-v1",
    "factual-nli-v1": "factual-nli-v1",
}


def version_metadata() -> dict[str, str]:
    return {
        "pipelineVersion": PIPELINE_VERSION,
        "eventSchemaVersion": EVENT_SCHEMA_VERSION,
        "promptVersion": ",".join(sorted(set(PROMPT_VERSIONS.values()))),
        "artifactPipelineVersion": ARTIFACT_PIPELINE_VERSION,
    }


def artifact_provenance(pipeline_mode: str = "event_pipeline") -> dict[str, str]:
    versions = version_metadata()
    return {
        "artifactPipelineVersion": versions["artifactPipelineVersion"],
        "pipelineMode": pipeline_mode or "event_pipeline",
        "eventSchemaVersion": versions["eventSchemaVersion"],
        "promptVersion": versions["promptVersion"],
    }


def stamp_artifact_provenance(tasks, notes, *, pipeline_mode: str = "event_pipeline") -> None:
    """Record internal pipeline provenance on Tasks/Notes. Does not change public fields."""
    stamp = artifact_provenance(pipeline_mode)
    for task in tasks or []:
        changes = dict(getattr(task, "changes", None) or {})
        changes.update(stamp)
        task.changes = changes
    for note in notes or []:
        debug = dict(getattr(note, "debug", None) or {})
        debug.update(stamp)
        note.debug = debug
