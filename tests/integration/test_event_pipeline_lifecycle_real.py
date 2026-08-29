"""Repeated real-model lifecycle checks. Skipped without credentials.

These tests prove Event loop is closed does not return after worker-style
asyncio.run restarts. They are expensive; set EVENT_PIPELINE_LIFECYCLE_REAL=1
to opt in beyond the default skip when keys exist.

    pytest tests/integration/test_event_pipeline_lifecycle_real.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

from services.conversation.event_pipeline.embeddings import default_embedder
from services.conversation.event_pipeline.pipeline import run_event_pipeline
from services.llm.router import get_llm_router
from tests.fixtures.reviewed_meetings import all_reviewed_meetings
from tests.integration.conftest import requires_real_models


pytestmark = [pytest.mark.integration, pytest.mark.real_models, requires_real_models]


def _opted_in() -> bool:
    return os.environ.get("EVENT_PIPELINE_LIFECYCLE_REAL") == "1"


@pytest.mark.skipif(not _opted_in(), reason="set EVENT_PIPELINE_LIFECYCLE_REAL=1 to run expensive lifecycle repeats")
def test_repeated_real_meetings_have_zero_async_lifecycle_errors():
    router = get_llm_router()
    meetings = list(all_reviewed_meetings())[:2]
    errors = 0
    runs = 0
    for meeting in meetings:
        for _ in range(2):
            result = asyncio.run(
                run_event_pipeline(
                    meeting["chunks"],
                    f"{meeting['id']}-lifecycle",
                    "user_1",
                    "space_1",
                    router=router,
                    embedder=default_embedder(prefer_provider=True, lexical_fallback=False),
                    polish_with_llm=True,
                )
            )
            runs += 1
            errors += int(result.observability.asyncLifecycleErrors or 0)
    assert runs >= 4
    assert errors == 0
