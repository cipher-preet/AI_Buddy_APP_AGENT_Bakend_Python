from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from services.reminders.occurrence import occurrence_id, to_occurrence_key
from services.reminders.redis_client import (
    ReminderRedisConfigError,
    describe_redis_url,
    format_redis_target,
    redact_redis_secrets,
)
from services.reminders.schedule_store import ReminderRedisKeys


SAMPLE = "redis://default:super-secret-password@redis-18535.example.redislabs.com:18535"


def test_reminder_redis_target_omits_credentials():
    target = describe_redis_url(SAMPLE)
    formatted = format_redis_target(target)
    assert target["host"] == "redis-18535.example.redislabs.com"
    assert target["port"] == 18535
    assert target["tls"] is False
    assert "super-secret-password" not in formatted
    assert "super-secret-password" not in str(target)
    assert formatted == "host=redis-18535.example.redislabs.com port=18535 tls=false"


def test_redact_redis_secrets_strips_url():
    redacted = redact_redis_secrets(
        "connect failed redis://default:super-secret-password@host:18535 extra"
    )
    assert "super-secret-password" not in redacted
    assert "redis://<redacted>" in redacted


def test_missing_reminder_redis_url_is_explicit(monkeypatch):
    import services.reminders.redis_client as redis_client
    from types import SimpleNamespace

    monkeypatch.setattr(
        redis_client,
        "settings",
        SimpleNamespace(REMINDER_REDIS_URL=""),
    )
    with pytest.raises(ReminderRedisConfigError, match="REMINDER_REDIS_URL is not configured"):
        redis_client.get_reminder_redis_url()


def test_speech_queue_still_uses_redis_url():
    import services.queue.redis_queue as redis_queue
    from apps.api_gateway.workers import conversation_workers, speech_worker, vector_worker

    source = inspect.getsource(redis_queue)
    assert "settings.REDIS_URL" in source
    assert "REMINDER_REDIS_URL" not in source
    for module in (speech_worker, conversation_workers, vector_worker):
        worker_source = inspect.getsource(module)
        assert "REMINDER_REDIS_URL" not in worker_source
        assert "get_reminder_redis_client" not in worker_source


def test_reminder_worker_uses_reminder_redis_client():
    import apps.api_gateway.workers.reminder_worker as reminder_worker
    import services.reminders.schedule_store as schedule_store

    source = inspect.getsource(reminder_worker)
    assert "get_reminder_redis_client" in source
    assert "from services.queue.redis_queue import redis_client" not in source
    store_source = inspect.getsource(schedule_store)
    assert "settings.REDIS_URL" not in store_source
    assert "from services.queue.redis_queue import redis_client" not in store_source


def test_node_and_python_occurrence_member_format_matches():
    instant = datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc)
    assert to_occurrence_key(instant) == "2026-08-30T10:00:00Z"
    assert occurrence_id("67abc", instant) == "67abc:2026-08-30T10:00:00Z"
    keys = ReminderRedisKeys()
    assert keys.schedule == "buddy:reminder:schedule"
    assert keys.processing == "buddy:reminder:processing"
    assert keys.retry == "buddy:reminder:retry"
    assert keys.dead_letter == "buddy:reminder:dead-letter"
    assert keys.payload("67abc:2026-08-30T10:00:00Z") == (
        "buddy:reminder:payload:67abc:2026-08-30T10:00:00Z"
    )
