"""Authenticated identity. The verified token owner is authoritative."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bson import ObjectId


class IdentityMismatch(Exception):
    """The caller supplied a user id that is not the authenticated user."""


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    provider: str
    expires_at: int
    token: str

    def __repr__(self) -> str:
        return (
            f"AuthenticatedUser(user_id={self.user_id!r}, provider={self.provider!r}, "
            f"expires_at={self.expires_at!r})"
        )


def same_user_id(left: Any, right: Any) -> bool:
    """Match Node ``sameUserId``: exact, or case-insensitive when both are ObjectIds."""
    if left is None or right is None:
        return False
    left_text = str(left).strip()
    right_text = str(right).strip()
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if ObjectId.is_valid(left_text) and ObjectId.is_valid(right_text):
        return left_text.lower() == right_text.lower()
    return False


def read_supplied_user_id(value: Any) -> str | None:
    """Read a client-supplied user id the way Node does. Objects are not ids."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return str(value)
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


def bind_user_id(actor: AuthenticatedUser, supplied: Any) -> str:
    """Return the verified user id. A supplied id is accepted only when it is the same user."""
    presented = read_supplied_user_id(supplied)
    if presented is None:
        return actor.user_id
    if not same_user_id(actor.user_id, presented):
        raise IdentityMismatch()
    return actor.user_id
