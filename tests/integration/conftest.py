"""Real production-model integration tests.

Excluded from default CI when provider credentials are missing.

Run manually / in staging:

    pytest tests/integration/test_event_pipeline_real_models.py -v
"""

from __future__ import annotations

import os

import pytest

from apps.api_gateway.config.setting import settings


def _secret(value) -> str:
    return settings.secret_value(value) if value is not None else ""


def real_embeddings_available() -> bool:
    if os.environ.get("EVENT_PIPELINE_REAL_MODELS") == "0":
        return False
    return bool(_secret(getattr(settings, "OPENAI_API_KEY", None)))


def real_models_available() -> bool:
    if os.environ.get("EVENT_PIPELINE_REAL_MODELS") == "0":
        return False
    krutrim = _secret(getattr(settings, "KRUTRIM_API_KEY", None))
    openai = _secret(getattr(settings, "OPENAI_API_KEY", None))
    return bool(krutrim) and bool(openai)


requires_real_embeddings = pytest.mark.skipif(
    not real_embeddings_available(),
    reason="SKIPPED — credentials unavailable (OPENAI_API_KEY required for real embeddings)",
)

requires_real_models = pytest.mark.skipif(
    not real_models_available(),
    reason="SKIPPED — credentials unavailable (KRUTRIM_API_KEY and OPENAI_API_KEY required for real-model integration)",
)
