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


def max_extraction_concurrency() -> int:
    return max(1, min(16, int(getattr(settings, "MAX_EXTRACTION_CONCURRENCY", 4) or 4)))
