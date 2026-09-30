from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

from apps.api_gateway.config.setting import settings
from services.chat import actions
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.reminders import pipeline as reminder_pipeline
from services.reminders.schemas import ReminderCollected


class RecordingProvider:
    name = "krutrim"
    configured = True

    def __init__(self, result=None, error: Exception | None = None):
        self.requests: list[StructuredLLMRequest] = []
        self.result = result
        self.error = error

    async def generate_structured(self, request, schema):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result if self.result is not None else schema()


def _router(provider):
    return SimpleNamespace(providers={"krutrim": provider})


def _write_request() -> StructuredLLMRequest:
    return StructuredLLMRequest(
        model="gemma-4-31b-it",
        max_tokens=700,
        schema_name="ChatWriteSpec",
        messages=[LLMMessage(role="user", content="add a task to call Rahul")],
    )


def test_chat_write_spec_uses_gpt_oss_120b_with_low_reasoning(monkeypatch):
    provider = RecordingProvider(result=actions.ChatWriteSpec(action="create_task", title="Call Rahul"))
    monkeypatch.setattr(actions, "get_llm_router", lambda: _router(provider))
    spec = asyncio.run(actions._strong_write_spec(_write_request()))
    request = provider.requests[0]
    assert spec is not None and spec.title == "Call Rahul"
    assert request.model == "gpt-oss-120b"
    assert request.max_tokens >= 2000
    assert request.metadata["extra_body"] == {"reasoning_effort": settings.CHAT_WRITE_REASONING_EFFORT}


def test_chat_write_spec_falls_back_when_strong_model_fails(monkeypatch):
    provider = RecordingProvider(error=RuntimeError("krutrim down"))
    monkeypatch.setattr(actions, "get_llm_router", lambda: _router(provider))
    assert asyncio.run(actions._strong_write_spec(_write_request())) is None


def test_reminder_voice_extraction_stays_on_fast_model(monkeypatch):
    provider = RecordingProvider()
    monkeypatch.setattr(reminder_pipeline, "get_llm_router", lambda: _router(provider))
    asyncio.run(
        reminder_pipeline.extract_with_krutrim("remind me tomorrow", ReminderCollected(), datetime.now().astimezone())
    )
    assert provider.requests[0].model == settings.REMINDER_EXTRACTION_MODEL == "gemma-4-31b-it"
