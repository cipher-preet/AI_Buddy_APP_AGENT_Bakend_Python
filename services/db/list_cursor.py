"""Opaque cursors for list pages ordered by a timestamp, a version, and _id."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def list_stamp(doc: dict[str, Any]) -> datetime:
    stamp = doc.get("updatedAt") or doc.get("createdAt")
    if isinstance(stamp, datetime):
        return _as_utc(stamp)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def encode_list_cursor(updated_at: datetime, version: int, doc_id: Any) -> str:
    payload = {
        "updatedAt": _as_utc(updated_at).isoformat(),
        "version": int(version),
        "id": str(doc_id),
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_list_cursor(cursor: str) -> tuple[datetime, int, ObjectId]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8"))
        updated_at = datetime.fromisoformat(str(payload["updatedAt"]))
        version = int(payload["version"])
        doc_id = ObjectId(str(payload["id"]))
    except Exception as error:
        raise ValueError("Invalid list cursor") from error
    if not isinstance(updated_at, datetime):
        raise ValueError("Invalid list cursor")
    return _as_utc(updated_at), version, doc_id


def cursor_match(
    updated_at: datetime,
    version: int,
    doc_id: ObjectId,
    *,
    version_first: bool,
) -> dict[str, Any]:
    """Items strictly after the cursor in the list's descending sort."""
    if version_first:
        return {
            "$or": [
                {"version": {"$lt": version}},
                {"version": version, "updatedAt": {"$lt": updated_at}},
                {"version": version, "updatedAt": updated_at, "_id": {"$lt": doc_id}},
            ]
        }
    return {
        "$or": [
            {"updatedAt": {"$lt": updated_at}},
            {"updatedAt": updated_at, "version": {"$lt": version}},
            {"updatedAt": updated_at, "version": version, "_id": {"$lt": doc_id}},
        ]
    }
