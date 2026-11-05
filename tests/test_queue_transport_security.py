"""Queue API transport policy. No request is sent to a real endpoint."""

from __future__ import annotations

import time

import pytest
from fastapi import Request

from apps.api_gateway.config.setting import settings
from apps.api_gateway.queue_api import _verify_request
from services.queue.http_queue import QueueApiError, QueueApiPublisher
from services.queue.transport_security import queue_api_transport_error


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://queue-api.example.com", True),
        ("https://queue-api.example.com/base", True),
        ("http://127.0.0.1:8080", True),
        ("http://localhost:8080", True),
        ("http://queue-api:8080", True),
        ("http://10.0.0.8:8080", True),
        ("http://192.168.1.9", True),
        ("http://172.16.0.4", True),
        ("http://203.0.113.10", False),
        ("http://queue.example.com", False),
        ("http://user:secret@127.0.0.1:8080", False),
        ("ftp://queue-api.example.com", False),
    ],
)
def test_public_http_is_rejected_and_private_http_is_allowed(url, allowed):
    error = queue_api_transport_error(url)
    assert (error is None) is allowed
    if error:
        assert "secret" not in error


def test_queue_provider_rejects_a_public_http_url():
    with pytest.raises(ValueError, match="https"):
        settings.__class__(
            _env_file=None,
            SERVICE_ROLE="api",
            QUEUE_PROVIDER="queue_api",
            QUEUE_API_BASE_URL="http://203.0.113.10",
            QUEUE_API_SERVICE_TOKEN="test-token",
        )


def test_direct_redis_provider_does_not_reject_an_unused_url():
    loaded = settings.__class__(
        _env_file=None,
        SERVICE_ROLE="api",
        QUEUE_PROVIDER="redis",
        QUEUE_API_BASE_URL="http://203.0.113.10",
    )
    assert loaded.QUEUE_PROVIDER == "redis"


def test_public_http_is_allowed_only_when_the_operator_opts_in():
    assert queue_api_transport_error("http://203.0.113.10", allow_public_http=True) is None
    loaded = settings.__class__(
        _env_file=None,
        SERVICE_ROLE="api",
        QUEUE_PROVIDER="queue_api",
        QUEUE_API_BASE_URL="http://203.0.113.10",
        QUEUE_API_SERVICE_TOKEN="test-token",
        QUEUE_API_ALLOW_PUBLIC_HTTP=True,
    )
    assert loaded.QUEUE_API_ALLOW_PUBLIC_HTTP is True


def test_https_queue_provider_is_accepted():
    loaded = settings.__class__(
        _env_file=None,
        SERVICE_ROLE="api",
        QUEUE_PROVIDER="queue_api",
        QUEUE_API_BASE_URL="https://queue-api.example.com",
        QUEUE_API_SERVICE_TOKEN="test-token",
    )
    assert loaded.QUEUE_API_BASE_URL.startswith("https://")


@pytest.mark.asyncio
async def test_publisher_does_not_disable_tls_or_follow_redirects(monkeypatch):
    captured: dict = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, content, headers):
            captured["url"] = url
            captured["authorization"] = headers.get("authorization", "")

            class Response:
                status_code = 202

                def json(self):
                    return {"accepted": True, "duplicate": False, "eventId": "e", "correlationId": "c"}

            return Response()

    monkeypatch.setattr("services.queue.http_queue.httpx.AsyncClient", Client)
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "")
    result = await QueueApiPublisher("https://queue-api.example.com").publish({"eventId": "e", "correlationId": "c"})
    assert result.accepted is True
    assert captured["verify"] is True
    assert captured["follow_redirects"] is False
    assert captured["url"] == "https://queue-api.example.com/internal/events"
    assert captured["authorization"] == "Bearer test-service-token"


@pytest.mark.asyncio
async def test_publisher_rejects_public_http_before_opening_a_client(monkeypatch):
    def explode(**kwargs):
        raise AssertionError("client was constructed")

    monkeypatch.setattr("services.queue.http_queue.httpx.AsyncClient", explode)
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "test-service-token")
    with pytest.raises(QueueApiError, match="https") as raised:
        await QueueApiPublisher("http://203.0.113.10").publish({"eventId": "e"})
    assert "test-service-token" not in str(raised.value)


def _request(headers: dict[str, str]) -> Request:
    raw = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
    return Request({"type": "http", "headers": raw, "method": "POST", "path": "/internal/events"})


def test_user_bearer_is_not_a_queue_service_credential(monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "")
    with pytest.raises(PermissionError):
        _verify_request(_request({"authorization": "Bearer user-hmac-token"}), b"{}")
    _verify_request(_request({"authorization": "Bearer test-service-token"}), b"{}")


def test_queue_hmac_rejects_a_bad_signature(monkeypatch):
    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "test-hmac-secret")
    with pytest.raises(PermissionError):
        _verify_request(
            _request(
                {
                    "x-buddy-timestamp": str(int(time.time())),
                    "x-buddy-signature": "sha256=deadbeef",
                }
            ),
            b"{}",
        )
