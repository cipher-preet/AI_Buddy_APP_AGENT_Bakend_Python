from __future__ import annotations

from apps.api_gateway.config.setting import settings


def meeting_pipeline_enabled() -> bool:
    """True when the extract → ledger → consolidate → verify path publishes."""
    return bool(getattr(settings, "ENABLE_MEETING_PIPELINE", True))


def extraction_window_target_tokens() -> int:
    return max(200, int(getattr(settings, "EXTRACTION_WINDOW_TARGET_TOKENS", 5000) or 5000))


def extraction_window_max_tokens() -> int:
    target = extraction_window_target_tokens()
    maximum = int(getattr(settings, "EXTRACTION_WINDOW_MAX_TOKENS", 7000) or 7000)
    return max(target, maximum)


def extraction_window_overlap_ratio() -> float:
    ratio = float(getattr(settings, "EXTRACTION_WINDOW_OVERLAP_RATIO", 0.12) or 0.12)
    return min(0.5, max(0.0, ratio))


def output_language() -> str:
    return str(getattr(settings, "MEETING_OUTPUT_LANGUAGE", "") or "English").strip() or "English"


def consolidation_partition_tokens() -> int:
    """Ledger payload size above which consolidation is partitioned by topic."""
    return max(2000, int(getattr(settings, "MEETING_CONSOLIDATION_PARTITION_TOKENS", 24000) or 24000))


def consolidation_max_candidates() -> int:
    """Candidate count above which one consolidation call loses coverage."""
    return max(8, int(getattr(settings, "MEETING_CONSOLIDATION_MAX_CANDIDATES", 40) or 40))


def coverage_batch_candidates() -> int:
    return max(4, int(getattr(settings, "MEETING_COVERAGE_BATCH_CANDIDATES", 30) or 30))


def outline_organizer_enabled() -> bool:
    return bool(getattr(settings, "MEETING_OUTLINE_ORGANIZER", True))


def verifier_batch_chars() -> int:
    return max(2000, int(getattr(settings, "MEETING_VERIFIER_BATCH_CHARS", 14000) or 14000))


def max_extraction_concurrency() -> int:
    return max(1, min(16, int(getattr(settings, "MAX_EXTRACTION_CONCURRENCY", 4) or 4)))
