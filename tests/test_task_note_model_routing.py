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

    def __init__(self, result=None, error: Exception | None = None, name: str = "krutrim"):
        self.requests: list[StructuredLLMRequest] = []
        self.result = result
        self.error = error
        self.name = name

    async def generate_structured(self, request, schema):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result if self.result is not None else schema()


def _router(*providers):
    return SimpleNamespace(providers={provider.name: provider for provider in providers})


def _write_request() -> StructuredLLMRequest:
    return StructuredLLMRequest(
        model="gemma-4-31b-it",
        max_tokens=700,
        schema_name="ChatWriteSpec",
        messages=[LLMMessage(role="user", content="add a task to call Rahul")],
    )


def test_chat_write_spec_uses_mistral_glm_flash(monkeypatch):
    provider = RecordingProvider(result=actions.ChatWriteSpec(action="create_task", title="Call Rahul"), name="mistral")
    fallback = RecordingProvider(name="krutrim")
    monkeypatch.setattr(actions, "get_llm_router", lambda: _router(provider, fallback))
    spec = asyncio.run(actions._strong_write_spec(_write_request()))
    request = provider.requests[0]
    assert spec is not None and spec.title == "Call Rahul"
    assert settings.CHAT_WRITE_PROVIDER == "mistral"
    assert request.model == "zai-glm-5-3-flash"
    assert request.max_tokens >= 2000
    assert request.metadata["extra_body"] == {}
    assert fallback.requests == []


def test_chat_write_spec_falls_back_to_krutrim_gpt_oss(monkeypatch):
    primary = RecordingProvider(error=RuntimeError("mistral down"), name="mistral")
    fallback = RecordingProvider(result=actions.ChatWriteSpec(action="create_note", title="Server notes"), name="krutrim")
    monkeypatch.setattr(actions, "get_llm_router", lambda: _router(primary, fallback))
    spec = asyncio.run(actions._strong_write_spec(_write_request()))
    assert spec is not None and spec.title == "Server notes"
    assert fallback.requests[0].model == "gpt-oss-120b"
    assert fallback.requests[0].metadata["extra_body"] == {"reasoning_effort": settings.CHAT_WRITE_REASONING_EFFORT}


def test_chat_write_spec_falls_back_when_strong_model_fails(monkeypatch):
    provider = RecordingProvider(error=RuntimeError("provider down"), name="mistral")
    fallback = RecordingProvider(error=RuntimeError("krutrim down"), name="krutrim")
    monkeypatch.setattr(actions, "get_llm_router", lambda: _router(provider, fallback))
    assert asyncio.run(actions._strong_write_spec(_write_request())) is None


def test_reminder_voice_extraction_stays_on_fast_model(monkeypatch):
    provider = RecordingProvider(name="krutrim")
    monkeypatch.setattr(reminder_pipeline, "get_llm_router", lambda: _router(provider))
    asyncio.run(
        reminder_pipeline.extract_with_krutrim("remind me tomorrow", ReminderCollected(), datetime.now().astimezone())
    )
    assert provider.requests[0].model == settings.REMINDER_EXTRACTION_MODEL == "gemma-4-31b-it"
