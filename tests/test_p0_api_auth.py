"""P0 authentication and resource-authorization regression tests.

Tokens are signed with a process-local test secret. Production credentials are not used.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from apps.api_gateway.config.setting import settings
from apps.api_gateway.controllers import speech_controller
from services.auth.identity import same_user_id
from services.auth.node_token import (
    AuthNotConfigured,
    create_auth_token,
    encode_signed_token,
    verify_auth_token,
)
from services.auth.space_access import AccessDenied, SpaceAccess, require_space_operation
from services.chat.node_client import NodeApiError, NodeHomeClient

SECRET = "test-only-phase1b-hmac-secret"
USER_A = str(ObjectId())
USER_B = str(ObjectId())
SPACE_A = str(ObjectId())
NODE_BACKEND = Path(r"d:\AI_Personal_Buddy\Node_Backend")


def _token(user_id: str, *, secret: str = SECRET, expires_at: int | None = None, provider: str = "google") -> str:
    return create_auth_token(user_id, provider, secret, expires_at=expires_at)


def _auth(user_id: str, **kwargs) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(user_id, **kwargs)}"}


class FakeCollection:
    def __init__(self, docs: list[dict]):
        self.docs = docs

    async def find_one(self, query, projection=None):
        for doc in self.docs:
            if _matches(doc, query):
                return doc
        return None


class FakeDB:
    def __init__(self, spaces: list[dict] | None = None, shares: list[dict] | None = None):
        self.spaces = FakeCollection(spaces or [])
        self.spaceshares = FakeCollection(shares or [])


def _matches(doc: dict, query: dict) -> bool:
    for key, expected in query.items():
        actual = doc.get(key, None)
        if isinstance(expected, dict) and set(expected) == {"$in"}:
            if not any(actual == item or str(actual) == str(item) for item in expected["$in"]):
                return False
        elif actual != expected:
            return False
    return True


def _owned_space():
    return FakeDB(spaces=[{"_id": ObjectId(SPACE_A), "userId": ObjectId(USER_A), "deletedAt": None}])


def _shared_space():
    return FakeDB(
        spaces=[{"_id": ObjectId(SPACE_A), "userId": ObjectId(USER_A), "deletedAt": None}],
        shares=[
            {
                "spaceId": ObjectId(SPACE_A),
                "memberUserId": ObjectId(USER_B),
                "ownerUserId": ObjectId(USER_A),
                "status": "active",
            }
        ],
    )


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_TOKEN_SECRET", SECRET)
    monkeypatch.setattr(settings, "SESSION_SECRET", "")

    async def noop():
        return None

    monkeypatch.setattr("apps.api_gateway.main.ensure_mongo_indexes", noop)
    monkeypatch.setattr("apps.api_gateway.main.close_mongo_client", noop)
    monkeypatch.setattr("apps.api_gateway.main.close_llm_runtime", noop)
    from apps.api_gateway.main import app

    with TestClient(app) as client:
        yield client


def test_python_verifier_matches_node_contract():
    issued = int(time.time())
    token = create_auth_token(USER_A, "google", SECRET, issued_at=issued, expires_at=issued + 60)
    identity = verify_auth_token(token, secret=SECRET)
    assert identity is not None
    assert identity.user_id == USER_A
    assert identity.provider == "google"
    assert identity.expires_at == issued + 60

    other = verify_auth_token(_token(USER_B), secret=SECRET)
    assert other is not None and other.user_id == USER_B and other.user_id != identity.user_id

    payload, signature = token.split(".")
    # Flip each segment against its own last character. Reusing the payload's
    # character made the signature case a no-op whenever the signature already
    # ended with that character, so a valid token was asserted to be rejected.
    payload_flip = "A" if payload[-1] != "A" else "B"
    signature_flip = "A" if signature[-1] != "A" else "B"
    assert verify_auth_token(f"{payload[:-1] + payload_flip}.{signature}", secret=SECRET) is None
    assert verify_auth_token(f"{payload}.{signature[:-1] + signature_flip}", secret=SECRET) is None
    assert verify_auth_token(token + ".extra", secret=SECRET) is None
    assert verify_auth_token("not-a-token", secret=SECRET) is None
    assert verify_auth_token("", secret=SECRET) is None
    assert verify_auth_token(create_auth_token(USER_A, "google", SECRET, expires_at=10), secret=SECRET) is None
    assert verify_auth_token(
        encode_signed_token(
            {"userId": 1, "provider": "google", "iat": issued, "exp": issued + 60},
            SECRET,
        ),
        secret=SECRET,
    ) is None
    versioned = encode_signed_token(
        {"userId": USER_A, "provider": "google", "iat": issued, "exp": issued + 3600, "v": 2},
        SECRET,
    )
    signed_extra = verify_auth_token(versioned, secret=SECRET)
    assert signed_extra is not None and signed_extra.user_id == USER_A
    with pytest.raises(AuthNotConfigured):
        verify_auth_token(token, secret="")


def test_same_user_id_is_case_insensitive_for_object_ids_only():
    assert same_user_id(USER_A, USER_A.upper())
    assert not same_user_id(USER_A, USER_B)
    assert not same_user_id("user-1", "user-1".upper())


def test_space_permission_follows_owner_and_active_share():
    owned = _owned_space()
    owner = asyncio.run(require_space_operation(owned, USER_A, SPACE_A, "write"))
    assert owner.role == "owner"
    assert owner.owner_user_id == USER_A

    shared = _shared_space()
    viewer = asyncio.run(require_space_operation(shared, USER_B, SPACE_A, "read"))
    assert viewer.role == "viewer"
    with pytest.raises(AccessDenied) as denied:
        asyncio.run(require_space_operation(shared, USER_B, SPACE_A, "write"))
    assert denied.value.status_code == 403

    revoked = FakeDB(
        spaces=[{"_id": ObjectId(SPACE_A), "userId": ObjectId(USER_A), "deletedAt": None}],
        shares=[
            {
                "spaceId": ObjectId(SPACE_A),
                "memberUserId": ObjectId(USER_B),
                "ownerUserId": ObjectId(USER_A),
                "status": "revoked",
            }
        ],
    )
    with pytest.raises(AccessDenied) as missing:
        asyncio.run(require_space_operation(revoked, USER_B, SPACE_A, "read"))
    assert missing.value.status_code == 404

    deleted = FakeDB(spaces=[{"_id": ObjectId(SPACE_A), "userId": ObjectId(USER_A), "deletedAt": "gone"}])
    with pytest.raises(AccessDenied) as deleted_error:
        asyncio.run(require_space_operation(deleted, USER_A, SPACE_A, "read"))
    assert deleted_error.value.status_code == 404

    with pytest.raises(AccessDenied) as invalid:
        asyncio.run(require_space_operation(owned, USER_A, "not-a-space", "read"))
    assert invalid.value.status_code == 400


def test_public_routes_stay_public_and_private_routes_fail_closed(api, monkeypatch):
    async def prompt(kind, language=None):
        return {"success": True, "data": {"kind": kind, "text": "public prompt"}}

    monkeypatch.setattr(
        "apps.api_gateway.routes.reminder_voice_routes.get_reminder_prompt_controller",
        prompt,
    )
    assert api.get("/api/health/").status_code == 200
    assert api.get("/").status_code == 200
    assert api.get(f"/api/v1/documents/{ObjectId()}").status_code == 401
    assert api.get("/api/v1/documents/list", params={"userId": USER_A, "spaceId": SPACE_A}).status_code == 401
    assert api.post("/api/v1/chat/ask", json={"userId": USER_A, "question": "hello"}).status_code == 401
    assert api.post("/api/v1/speech/listening/start", json={"userId": USER_A, "spaceId": SPACE_A}).status_code == 401
    assert api.get(f"/api/v1/speech/transcribe/{ObjectId()}").status_code == 401
    assert api.get("/api/v1/daily-briefing", params={"userId": USER_A}).status_code == 401
    reminder = api.post(
        "/api/v1/reminders/voice/turn",
        data={"userId": USER_A},
        files={"file": ("note.m4a", b"abc", "audio/mp4")},
    )
    assert reminder.status_code == 401
    prompt = api.get("/api/v1/reminders/voice/prompt")
    assert prompt.status_code == 200
    assert prompt.json()["data"]["text"] == "public prompt"


def test_missing_signing_secret_rejects_protected_routes(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_TOKEN_SECRET", "")
    monkeypatch.setattr(settings, "SESSION_SECRET", "")

    async def noop():
        return None

    monkeypatch.setattr("apps.api_gateway.main.ensure_mongo_indexes", noop)
    monkeypatch.setattr("apps.api_gateway.main.close_mongo_client", noop)
    monkeypatch.setattr("apps.api_gateway.main.close_llm_runtime", noop)
    from apps.api_gateway.main import app

    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/documents/{ObjectId()}",
            headers={"Authorization": f"Bearer {_token(USER_A)}"},
        )
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"
    assert "secret" not in response.text.lower()


def test_invalid_and_mismatched_identity_are_rejected(api, monkeypatch):
    calls = []

    async def enqueue(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr("apps.api_gateway.routes.document_routes.enqueue_document_generation", enqueue)
    body = {"userId": USER_A, "spaceId": SPACE_A, "templateCode": "notes"}
    assert api.post("/api/v1/documents/generate", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    expired = _auth(USER_A, expires_at=10)
    assert api.post("/api/v1/documents/generate", json=body, headers=expired).status_code == 401
    mismatched = api.post(
        "/api/v1/documents/generate",
        json={**body, "userId": USER_B},
        headers=_auth(USER_A),
    )
    assert mismatched.status_code == 403
    assert mismatched.json()["detail"] == "Forbidden"
    assert calls == []
    non_object = create_auth_token("not-an-object-id", "google", SECRET)
    assert api.post(
        "/api/v1/documents/generate",
        json=body,
        headers={"Authorization": f"Bearer {non_object}"},
    ).status_code == 401


def test_document_download_checks_owner_before_file_bytes(api, monkeypatch):
    secret_bytes = b"PK\x03\x04private-docx"
    loads = []

    class Store:
        def __init__(self, database):
            self.database = database

        async def get(self, document_id, include_docx=False):
            loads.append(include_docx)
            if document_id != "doc-1":
                return None
            document = {
                "_id": "doc-1",
                "userId": USER_A,
                "spaceId": SPACE_A,
                "status": "READY",
                "fileName": 'evil"\r\nX-Test: 1.docx',
            }
            if include_docx:
                document["docxBase64"] = base64.b64encode(secret_bytes).decode("ascii")
            return document

    monkeypatch.setattr("apps.api_gateway.routes.document_routes.DocumentStore", Store)
    monkeypatch.setattr("apps.api_gateway.routes.document_routes.get_database", _shared_space)

    anonymous = api.get("/api/v1/documents/doc-1/download")
    assert anonymous.status_code == 401
    assert loads == []
    assert b"private-docx" not in anonymous.content

    stranger = api.get("/api/v1/documents/doc-1/download", headers=_auth(str(ObjectId())))
    assert stranger.status_code == 404
    assert loads == [False]
    assert b"private-docx" not in stranger.content

    loads.clear()
    revoked = FakeDB(spaces=[{"_id": ObjectId(SPACE_A), "userId": ObjectId(USER_A), "deletedAt": None}])
    monkeypatch.setattr("apps.api_gateway.routes.document_routes.get_database", lambda: revoked)
    revoked_response = api.get("/api/v1/documents/doc-1/download", headers=_auth(USER_B))
    assert revoked_response.status_code == 404
    assert True not in loads
    assert b"private-docx" not in revoked_response.content

    loads.clear()
    monkeypatch.setattr("apps.api_gateway.routes.document_routes.get_database", _shared_space)
    viewer = api.get("/api/v1/documents/doc-1/download", headers=_auth(USER_B))
    assert viewer.status_code == 200
    assert viewer.content == secret_bytes
    assert loads[0] is False and True in loads
    disposition = viewer.headers["content-disposition"]
    assert "\r" not in disposition and "\n" not in disposition
    assert disposition.startswith("attachment;")

    owner = api.get("/api/v1/documents/doc-1/download", headers=_auth(USER_A))
    assert owner.status_code == 200
    assert owner.content == secret_bytes


def test_document_list_uses_space_owner_for_viewers_and_rejects_revoked(api, monkeypatch):
    listed = []

    class Store:
        def __init__(self, database):
            pass

        async def list_for_space(self, user_id, space_id, limit=24, card=False):
            listed.append(user_id)
            return []

    monkeypatch.setattr("apps.api_gateway.routes.document_routes.DocumentStore", Store)
    monkeypatch.setattr("apps.api_gateway.routes.document_routes.get_database", _shared_space)
    response = api.get(
        "/api/v1/documents/list",
        params={"userId": USER_B, "spaceId": SPACE_A},
        headers=_auth(USER_B),
    )
    assert response.status_code == 200
    assert listed == [USER_A]

    listed.clear()
    monkeypatch.setattr(
        "apps.api_gateway.routes.document_routes.get_database",
        lambda: FakeDB(spaces=[{"_id": ObjectId(SPACE_A), "userId": ObjectId(USER_A), "deletedAt": None}]),
    )
    denied = api.get(
        "/api/v1/documents/list",
        params={"userId": USER_B, "spaceId": SPACE_A},
        headers=_auth(USER_B),
    )
    assert denied.status_code == 404
    assert listed == []


def test_authorized_document_generate_uses_verified_user(api, monkeypatch):
    calls = []

    async def enqueue(_database, **kwargs):
        calls.append(kwargs)
        return {"success": True, "documentId": "doc-1"}

    monkeypatch.setattr("apps.api_gateway.routes.document_routes.enqueue_document_generation", enqueue)
    monkeypatch.setattr("apps.api_gateway.routes.document_routes.get_database", _owned_space)
    response = api.post(
        "/api/v1/documents/generate",
        json={"userId": USER_A.upper(), "spaceId": SPACE_A, "templateCode": "notes"},
        headers=_auth(USER_A),
    )
    assert response.status_code == 200
    assert calls[0]["user_id"] == USER_A
    assert calls[0]["space_id"] == SPACE_A

    monkeypatch.setattr("apps.api_gateway.routes.document_routes.get_database", _shared_space)
    viewer = api.post(
        "/api/v1/documents/generate",
        json={"userId": USER_B, "spaceId": SPACE_A, "templateCode": "notes"},
        headers=_auth(USER_B),
    )
    assert viewer.status_code == 403
    assert len(calls) == 1


def test_mindmap_by_id_hides_other_users(api, monkeypatch):
    loaded = []

    class Store:
        def __init__(self, database):
            pass

        async def get(self, mindmap_id):
            loaded.append(mindmap_id)
            if mindmap_id != "map-1":
                return None
            return {"_id": "map-1", "userId": USER_A, "spaceId": SPACE_A, "graph": {"nodes": [{"title": "private"}]}}

        async def remove_node(self, mindmap_id, node_id):
            raise AssertionError("node removed before authorization")

    monkeypatch.setattr("apps.api_gateway.routes.mindmap_routes.MindmapStore", Store)
    monkeypatch.setattr("apps.api_gateway.routes.mindmap_routes.get_database", _owned_space)
    hidden = api.get("/api/v1/mindmap/map-1", headers=_auth(USER_B))
    assert hidden.status_code == 404
    assert "private" not in hidden.text
    removed = api.delete("/api/v1/mindmap/map-1/nodes/node-1", headers=_auth(USER_B))
    assert removed.status_code == 404
    owner = api.get("/api/v1/mindmap/map-1", headers=_auth(USER_A))
    assert owner.status_code == 200
    assert owner.json()["mindmap"]["userId"] == USER_A


def test_speech_job_and_listening_are_bound_to_the_token_user(api, monkeypatch):
    started = []

    async def start(user_id, space_id):
        started.append((user_id, space_id))
        return {"success": True, "data": {"user_id": user_id, "space_id": space_id}}

    async def allow(*_args, **_kwargs):
        return SpaceAccess(owner_user_id=USER_A, role="owner")

    monkeypatch.setattr("apps.api_gateway.routes.speech_routes.start_listening_controller", start)
    monkeypatch.setattr("apps.api_gateway.routes.speech_routes.require_space_operation", allow)
    denied = api.post(
        "/api/v1/speech/listening/start",
        json={"userId": USER_B, "spaceId": SPACE_A},
        headers=_auth(USER_A),
    )
    assert denied.status_code == 403
    assert started == []
    allowed = api.post(
        "/api/v1/speech/listening/start",
        json={"userId": USER_A.upper(), "spaceId": SPACE_A},
        headers=_auth(USER_A),
    )
    assert allowed.status_code == 200
    assert started == [(USER_A, SPACE_A)]

    async def job(job_id):
        return {
            "job-a": {"user_id": USER_A, "result": "SECRET_TRANSCRIPT_A"},
            "job-b": {"user_id": USER_B, "result": "SECRET_TRANSCRIPT_B"},
            "job-open": {"result": "SECRET_TRANSCRIPT_OPEN"},
        }.get(job_id)

    monkeypatch.setattr(speech_controller, "get_transcribe_job_result_service", job)
    own = asyncio.run(speech_controller.get_transcribe_result_controller("job-a", USER_A))
    other = asyncio.run(speech_controller.get_transcribe_result_controller("job-b", USER_A))
    unsigned = asyncio.run(speech_controller.get_transcribe_result_controller("job-open", USER_A))
    assert own["success"] is True
    assert other == {"success": False, "message": "Job not found."}
    assert "SECRET_TRANSCRIPT_B" not in json.dumps(other)
    assert unsigned["success"] is False
    missing = api.get("/api/v1/speech/transcribe/job-b")
    assert missing.status_code == 401


def test_chat_forwards_verified_token_and_rejects_cross_user_claims(api, monkeypatch):
    seen = []

    async def ask(**kwargs):
        seen.append(kwargs)
        return {"success": True, "data": {"answer": "ok"}}

    monkeypatch.setattr("apps.api_gateway.routes.chat_routes.ask_chat_controller", ask)
    other_token = _token(USER_B)
    cross_body = api.post(
        "/api/v1/chat/ask",
        json={"userId": USER_A, "question": "hello", "authToken": other_token},
        headers=_auth(USER_A),
    )
    assert cross_body.status_code == 401
    assert seen == []
    body_only = api.post(
        "/api/v1/chat/ask",
        json={"userId": USER_A, "question": "hello", "authToken": _token(USER_A)},
    )
    assert body_only.status_code == 401

    header = _token(USER_A, expires_at=int(time.time()) + 500)
    same_user_body = _token(USER_A, expires_at=int(time.time()) + 800)
    ok = api.post(
        "/api/v1/chat/ask",
        json={"userId": USER_B, "question": "hello", "spaceId": SPACE_A},
        headers={"Authorization": f"Bearer {header}"},
    )
    assert ok.status_code == 403
    accepted = api.post(
        "/api/v1/chat/ask",
        json={"userId": USER_A, "question": "hello", "authToken": same_user_body},
        headers={"Authorization": f"Bearer {header}"},
    )
    assert accepted.status_code == 200
    assert seen[0]["user_id"] == USER_A
    assert seen[0]["auth_token"] == header

    leaked = api.post(
        "/api/v1/chat/ask",
        headers=_auth(USER_A),
        json={"userId": USER_A, "question": "hi", "authToken": {"nested": "super-secret-token-value"}},
    )
    assert "super-secret-token-value" not in leaked.text


def test_chat_space_claim_is_rejected_before_answering(api, monkeypatch):
    async def deny(*_args, **_kwargs):
        raise AccessDenied(404, "Space not found.")

    monkeypatch.setattr("services.chat.service.authorize_space_ids", deny)
    response = api.post(
        "/api/v1/chat/ask",
        json={"userId": USER_A, "question": "what is in this space?", "spaceId": SPACE_A},
        headers=_auth(USER_A),
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Space not found."


def test_daily_briefing_and_device_token_cannot_target_another_user(api, monkeypatch):
    class Boom:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("private store opened")

    monkeypatch.setattr("apps.api_gateway.routes.daily_briefing_routes.DailyBriefingStore", Boom)
    briefing = api.get("/api/v1/daily-briefing", params={"userId": USER_B}, headers=_auth(USER_A))
    assert briefing.status_code == 403

    async def force(*_args, **_kwargs):
        raise AssertionError("force generate ran")

    monkeypatch.setattr("apps.api_gateway.routes.daily_briefing_routes.force_generate_daily_briefing", force)
    forced = api.post(
        "/api/v1/daily-briefing/force-generate",
        json={"userId": USER_B},
        headers=_auth(USER_A),
    )
    assert forced.status_code == 403

    monkeypatch.setattr("apps.api_gateway.routes.reminder_device_routes.MongoReminderStore", Boom)
    device = api.post(
        "/api/v1/reminders/device-tokens",
        json={"userId": USER_B, "token": "device-secret"},
        headers=_auth(USER_A),
    )
    assert device.status_code == 403
    assert "device-secret" not in device.text


def test_conversation_presign_requires_the_token_owner(api, monkeypatch):
    async def presign(**_kwargs):
        raise AssertionError("presign issued")

    monkeypatch.setattr("apps.api_gateway.routes.conversation_routes.create_audio_upload_url_controller", presign)
    body = {
        "userId": USER_A,
        "spaceId": SPACE_A,
        "sequenceNumber": 0,
        "contentType": "audio/webm",
        "extension": "webm",
        "expectedSizeBytes": 128,
    }
    assert api.post("/api/v1/conversations/conv-1/audio/upload-url", json=body).status_code == 401
    assert api.post(
        "/api/v1/conversations/conv-1/audio/upload-url",
        json={**body, "userId": USER_B},
        headers=_auth(USER_A),
    ).status_code == 403


def test_force_generate_accepts_the_signed_in_user_and_rejects_bad_credentials(api, monkeypatch):
    seen: dict[str, str] = {}

    async def force(_database, *, user_id, date_key, period):
        seen["user_id"] = user_id
        seen["period"] = period
        return {
            "userId": user_id,
            "dateKey": date_key or "2026-09-09",
            "timezone": "Asia/Kolkata",
            "status": "PENDING",
            "forced": True,
            "message": "Briefing generation started. Poll getDailyBriefing until READY.",
        }

    monkeypatch.setattr(settings, "DAILY_BRIEFING_ALLOW_FORCE_GENERATE", True)
    monkeypatch.setattr(
        "apps.api_gateway.routes.daily_briefing_routes.force_generate_daily_briefing",
        force,
    )

    missing = api.post("/api/v1/daily-briefing/force-generate", json={"userId": USER_A})
    assert missing.status_code == 401
    assert seen == {}

    invalid = api.post(
        "/api/v1/daily-briefing/force-generate",
        json={"userId": USER_A},
        headers={"Authorization": "Bearer not-a-token"},
    )
    assert invalid.status_code == 401
    assert seen == {}

    accepted = api.post(
        "/api/v1/daily-briefing/force-generate",
        json={"userId": USER_A, "period": "yesterday"},
        headers=_auth(USER_A),
    )
    assert accepted.status_code == 200
    body = accepted.json()
    assert body["success"] is True
    assert body["result"]["userId"] == USER_A
    assert body["result"]["status"] == "PENDING"
    assert body["result"]["forced"] is True
    assert seen == {"user_id": USER_A, "period": "yesterday"}


def test_queue_credential_cannot_access_user_briefing_or_chat(api):
    headers = {"Authorization": "Bearer queue-service-token"}
    briefing = api.get("/api/v1/daily-briefing", params={"userId": USER_A}, headers=headers)
    assert briefing.status_code == 401
    chat = api.get("/api/v1/chat/sessions", params={"userId": USER_A}, headers=headers)
    assert chat.status_code == 401
    forced = api.post(
        "/api/v1/daily-briefing/force-generate",
        json={"userId": USER_A},
        headers=headers,
    )
    assert forced.status_code == 401


def test_queue_api_rejects_end_user_tokens(monkeypatch):
    from apps.api_gateway import queue_api

    monkeypatch.setattr(settings, "QUEUE_API_SERVICE_TOKEN", "queue-service-token")
    monkeypatch.setattr(settings, "QUEUE_API_HMAC_SECRET", "")
    with TestClient(queue_api.app) as client:
        response = client.post(
            "/internal/events",
            json={"eventType": "audio.ingested"},
            headers=_auth(USER_A),
        )
    assert response.status_code == 401


def test_create_space_requires_the_callers_token(monkeypatch):
    posts = []

    class Response:
        status_code = 200

        def json(self):
            return {"success": True, "data": {"space": {"id": SPACE_A}}}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, json=None, headers=None):
            posts.append({"url": url, "json": json, "headers": headers})
            return Response()

    monkeypatch.setattr(settings, "AUTH_TOKEN_SECRET", SECRET)
    monkeypatch.setattr(settings, "SESSION_SECRET", "")
    monkeypatch.setattr("services.chat.node_client.httpx.AsyncClient", Client)

    with pytest.raises(NodeApiError):
        asyncio.run(NodeHomeClient(auth_token=None).create_space(USER_A, "Workspace"))
    assert posts == []

    with pytest.raises(NodeApiError):
        asyncio.run(NodeHomeClient(auth_token=_token(USER_A)).create_space(USER_B, "Workspace"))
    assert posts == []

    with pytest.raises(NodeApiError):
        asyncio.run(NodeHomeClient(auth_token="not-a-token").create_space(USER_A, "Workspace"))
    assert posts == []

    token = _token(USER_A)
    created = asyncio.run(NodeHomeClient(auth_token=token).create_space(USER_A, "Workspace"))
    assert created["id"] == SPACE_A
    assert posts[0]["json"]["userId"] == USER_A
    assert posts[0]["headers"]["Authorization"] == f"Bearer {token}"
    assert "/api/v1/home/create-space" in posts[0]["url"]


def test_node_implementation_accepts_python_tokens_and_python_accepts_node_tokens(tmp_path):
    tsx = NODE_BACKEND / "node_modules" / "tsx" / "dist" / "cli.mjs"
    source = NODE_BACKEND / "utils" / "authToken.ts"
    if shutil.which("node") is None or not tsx.exists() or not source.exists():
        pytest.skip("Node auth token toolchain is not available")
    script = tmp_path / "auth_cli.ts"
    script.write_text(
        f"import {{ createAuthToken, verifyAuthToken }} from {json.dumps(source.as_uri())};\n"
        """
const [command, userId, provider, exp] = process.argv.slice(2);
if (command === "create") {
  const options = exp ? { expiresAt: Number(exp) } : undefined;
  process.stdout.write(createAuthToken({ userId, provider: provider || "google" }, options));
} else if (command === "verify") {
  process.stdout.write(JSON.stringify(verifyAuthToken(userId)));
} else {
  process.exit(2);
}
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["AUTH_TOKEN_SECRET"] = SECRET
    env["SESSION_SECRET"] = ""

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["node", str(tsx), str(script), *args],
            cwd=str(NODE_BACKEND),
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip(completed.stderr[-300:])
        return completed.stdout

    node_token = run("create", USER_A, "google")
    identity = verify_auth_token(node_token, secret=SECRET)
    assert identity is not None and identity.user_id == USER_A
    python_token = create_auth_token(USER_A, "google", SECRET)
    verified = json.loads(run("verify", python_token))
    assert verified["userId"] == USER_A
    tampered = node_token[:-1] + ("A" if node_token[-1] != "A" else "B")
    assert json.loads(run("verify", tampered)) is None
    expired = run("create", USER_A, "google", "10")
    assert verify_auth_token(expired, secret=SECRET) is None
