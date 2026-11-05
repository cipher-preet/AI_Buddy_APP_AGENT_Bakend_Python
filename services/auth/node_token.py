"""Verify the Node.js custom HMAC bearer token.

Node issues ``base64url(json).base64url(hmac_sha256)`` from ``utils/authToken.ts``.
This is not a JWT. The signed bytes are the encoded payload segment, not the raw JSON.
Python does not issue tokens to clients. ``create_auth_token`` reproduces Node's
signer so tests and local interoperability checks can use the same contract.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

from apps.api_gateway.config.setting import settings

_LOG = logging.getLogger("auth")
_B64URL_RE = re.compile(r"[A-Za-z0-9_-]+")
_MAX_TOKEN_LENGTH = 4096
_missing_secret_logged = False


class AuthNotConfigured(Exception):
    """The HMAC secret is missing. Callers must fail closed."""


@dataclass(frozen=True)
class AuthIdentity:
    user_id: str
    provider: str
    expires_at: int


def create_auth_token(
    user_id: str,
    provider: str,
    secret: str,
    *,
    issued_at: int | None = None,
    expires_at: int | None = None,
) -> str:
    """Build a token with Node's payload order: userId, provider, iat, exp."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("userId is required")
    if not isinstance(provider, str):
        raise ValueError("provider is required")
    issued = int(time.time()) if issued_at is None else int(issued_at)
    expires = issued + 60 * 60 * 24 * 30 if expires_at is None else int(expires_at)
    payload = {
        "userId": user_id,
        "provider": provider,
        "iat": issued,
        "exp": expires,
    }
    return encode_signed_token(payload, secret)


def encode_signed_token(payload: dict[str, Any], secret: str) -> str:
    if not isinstance(secret, str) or secret == "":
        raise AuthNotConfigured("Auth token secret is not configured")
    encoded = _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return f"{encoded}.{_sign(encoded, secret)}"


def verify_auth_token(token: str, *, secret: str | None = None) -> AuthIdentity | None:
    """Return the signed identity, or None when the token is not acceptable.

    A missing secret raises ``AuthNotConfigured`` so protected routes fail closed
    instead of treating an unverified payload as a user.
    """
    key = settings.resolved_auth_token_secret if secret is None else secret
    if not isinstance(key, str) or key == "":
        _note_missing_secret()
        raise AuthNotConfigured("Auth token secret is not configured")
    if not isinstance(token, str):
        return None
    presented = token.strip()
    if not presented or len(presented) > _MAX_TOKEN_LENGTH:
        return None
    parts = presented.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    encoded_payload, signature = parts
    if not _B64URL_RE.fullmatch(encoded_payload) or not _B64URL_RE.fullmatch(signature):
        return None
    expected = _sign(encoded_payload, key)
    if len(expected) != len(signature) or not hmac.compare_digest(expected, signature):
        return None
    try:
        decoded = _b64url_decode(encoded_payload)
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    user_id = payload.get("userId")
    expires_at = payload.get("exp")
    if not isinstance(user_id, str) or not user_id.strip():
        return None
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return None
    if isinstance(expires_at, float) and not math.isfinite(expires_at):
        return None
    if expires_at < math.floor(time.time()):
        return None
    provider = payload.get("provider")
    return AuthIdentity(
        user_id=user_id,
        provider=provider if isinstance(provider, str) else "",
        expires_at=int(expires_at),
    )


def _sign(encoded_payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    return _b64url(digest)


def _b64url(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii").rstrip("=").replace("+", "-").replace("/", "_")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.b64decode(value + padding, validate=True)


def _note_missing_secret() -> None:
    global _missing_secret_logged
    if _missing_secret_logged:
        return
    _missing_secret_logged = True
    _LOG.warning("auth.token_secret_missing")
