from __future__ import annotations

import hashlib

from apps.api_gateway.config.setting import settings

_VALID_MODES = frozenset({"legacy", "shadow", "event_pipeline"})
_ROLLOUT_PHASES = {
    0: {"mode": "shadow", "percent": 0, "label": "Phase 0 shadow"},
    1: {"mode": "event_pipeline", "percent": 5, "label": "Phase 1 5%"},
    2: {"mode": "event_pipeline", "percent": 10, "label": "Phase 2 10%"},
    3: {"mode": "event_pipeline", "percent": 25, "label": "Phase 3 25%"},
    4: {"mode": "event_pipeline", "percent": 50, "label": "Phase 4 50%"},
    5: {"mode": "event_pipeline", "percent": 100, "label": "Phase 5 100%"},
}


def event_pipeline_mode() -> str:
    """Resolve rollout mode.

    ``legacy``         — old pipeline publishes; new pipeline does not run
    ``shadow``         — old pipeline publishes; new pipeline runs for comparison only
    ``event_pipeline`` — new pipeline publishes

    ``ENABLE_EVENT_PIPELINE=False`` is a hard off switch (always legacy).
    Unset/empty ``EVENT_PIPELINE_MODE`` resolves to ``event_pipeline`` so
    production defaults to the new path. Explicit ``legacy`` still wins
    immediately. Invalid values fail safe to legacy. Only ``event_pipeline``
    may publish new Tasks/Notes.
    """
    if not bool(getattr(settings, "ENABLE_EVENT_PIPELINE", True)):
        return "legacy"
    raw = str(getattr(settings, "EVENT_PIPELINE_MODE", "") or "").strip().lower()
    if not raw:
        return "event_pipeline"
    if raw in _VALID_MODES:
        return raw
    return "legacy"


def event_pipeline_enabled() -> bool:
    """True when the hierarchical event pipeline should execute (shadow or publish)."""
    return event_pipeline_mode() in {"shadow", "event_pipeline"}


def event_pipeline_publishes() -> bool:
    """Mode allows event-pipeline publishing. Percent=0 disables publishing."""
    return event_pipeline_mode() == "event_pipeline" and rollout_percent() > 0


def legacy_pipeline_publishes() -> bool:
    return event_pipeline_mode() in {"legacy", "shadow"}


def event_pipeline_shadow() -> bool:
    return event_pipeline_mode() == "shadow"


def rollout_percent() -> int:
    try:
        value = int(getattr(settings, "EVENT_PIPELINE_ROLLOUT_PERCENT", 100) or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, value))


def rollout_bucket(identifier: str) -> int:
    digest = hashlib.sha256(f"buddy:event-pipeline:rollout:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100


def event_pipeline_selected_for(user_id: str | None = None, session_id: str | None = None) -> bool:
    """Deterministic canary selection. Same user stays on the same pipeline."""
    if not event_pipeline_publishes():
        return False
    percent = rollout_percent()
    if percent >= 100:
        return True
    identifier = str(user_id or "").strip() or str(session_id or "").strip()
    if not identifier:
        return False
    return rollout_bucket(identifier) < percent


def legacy_pipeline_publishes_for(user_id: str | None = None, session_id: str | None = None) -> bool:
    if event_pipeline_mode() in {"legacy", "shadow"}:
        return True
    return not event_pipeline_selected_for(user_id, session_id)


def rollout_phase_config(phase: int | None = None) -> dict:
    if phase is None:
        raw = str(getattr(settings, "EVENT_PIPELINE_PHASE", "") or "").strip()
        if raw.isdigit():
            phase = int(raw)
        else:
            return {
                "mode": event_pipeline_mode(),
                "percent": rollout_percent(),
                "label": "configured",
            }
    return dict(_ROLLOUT_PHASES.get(int(phase), _ROLLOUT_PHASES[0]))


def semantic_microblocks_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_SEMANTIC_MICROBLOCKS", True))


def global_thread_graph_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_GLOBAL_THREAD_GRAPH", True))


def factual_validation_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_FACTUAL_VALIDATION", False))


def coverage_ledger_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_COVERAGE_LEDGER", True))


def debug_snapshots_enabled() -> bool:
    return bool(getattr(settings, "ENABLE_EVENT_PIPELINE_DEBUG_SNAPSHOTS", False))


def microblock_similarity_threshold() -> float:
    return float(getattr(settings, "MICROBLOCK_SIMILARITY_THRESHOLD", 0.34))


def thread_candidate_similarity_threshold() -> float:
    return float(getattr(settings, "THREAD_CANDIDATE_SIMILARITY_THRESHOLD", 0.15))


def thread_entityless_min_similarity() -> float:
    return float(getattr(settings, "THREAD_ENTITYLESS_MIN_SIMILARITY", 0.72))


def topic_continue_similarity_threshold() -> float:
    return float(getattr(settings, "TOPIC_CONTINUE_SIMILARITY_THRESHOLD", 0.56))


def topic_coherence_drop_threshold() -> float:
    return float(getattr(settings, "TOPIC_COHERENCE_DROP_THRESHOLD", 0.10))


def topic_object_discontinuity_max_overlap() -> float:
    return float(getattr(settings, "TOPIC_OBJECT_DISCONTINUITY_MAX_OVERLAP", 0.12))


def topic_object_discontinuity_max_similarity() -> float:
    return float(getattr(settings, "TOPIC_OBJECT_DISCONTINUITY_MAX_SIMILARITY", 0.66))


def topic_safety_max_micro_blocks() -> int:
    return int(getattr(settings, "TOPIC_SAFETY_MAX_MICRO_BLOCKS", 6))


def topic_safety_max_tokens() -> int:
    return int(getattr(settings, "TOPIC_SAFETY_MAX_TOKENS", 1600))


def topic_safety_continue_similarity() -> float:
    return float(getattr(settings, "TOPIC_SAFETY_CONTINUE_SIMILARITY", 0.78))


def topic_filler_density_threshold() -> float:
    return float(getattr(settings, "TOPIC_FILLER_DENSITY_THRESHOLD", 0.40))
