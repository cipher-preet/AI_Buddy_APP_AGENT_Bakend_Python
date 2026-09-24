from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from services.chat.node_client import NodeApiError, NodeHomeClient
from services.conversation.repository import mongo_id_candidates, to_mongo_id
from services.daily_briefing.timezones import DEFAULT_TIMEZONE
from services.db.mongo import get_database
from services.reminders.dates import format_date_label
from services.reminders.occurrence import parse_time_label

EntityKind = Literal["task", "note", "space"]


class ChatWriteStore:
    """Chat write tools that create/update/delete documents through the Node home APIs."""

    def __init__(self, database=None, auth_token: str | None = None):
        self.db = database or get_database()
        self.client = NodeHomeClient(auth_token=auth_token)

    async def create_task(
        self,
        user_id: str,
        space_id: str,
        title: str,
        description: str = "",
        due_date: str | None = None,
        priority: str = "Medium",
    ) -> dict[str, Any]:
        _ = user_id  # auth token identifies the user on Node
        try:
            return await self.client.create_task(
                space_id=space_id,
                title=title,
                description=description,
                due_date=due_date,
                priority=_normalize_priority(priority),
            )
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def create_note(
        self,
        user_id: str,
        space_id: str,
        title: str,
        body: str = "",
        date_key: str | None = None,
    ) -> dict[str, Any]:
        _ = user_id
        try:
            return await self.client.create_note(
                space_id=space_id,
                title=title,
                body=body,
                date_key=date_key,
            )
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def create_space(self, user_id: str, spacename: str, description: str = "New") -> dict[str, Any]:
        name = spacename.strip()[:80]
        if len(name) < 3:
            raise ValueError("Space name must be at least 3 characters.")
        try:
            return await self.client.create_space(
                user_id=user_id,
                spacename=name,
                description=description or "New",
            )
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def update_space(
        self,
        user_id: str,
        space_id: str,
        spacename: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        _ = user_id
        try:
            return await self.client.update_space(
                space_id=space_id,
                spacename=spacename,
                description=description,
            )
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def update_task(
        self,
        user_id: str,
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        due_date: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        _ = user_id
        try:
            return await self.client.update_task(
                task_id=task_id,
                title=title,
                description=description,
                due_date=due_date,
                priority=_normalize_priority(priority) if priority is not None else None,
            )
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def update_note(
        self,
        user_id: str,
        note_id: str,
        title: str | None = None,
        body: str | None = None,
        date_key: str | None = None,
    ) -> dict[str, Any]:
        _ = user_id
        try:
            return await self.client.update_note(
                note_id=note_id,
                title=title,
                body=body,
                date_key=date_key,
            )
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def delete_space(self, user_id: str, space_id: str) -> dict[str, Any]:
        _ = user_id
        try:
            return await self.client.delete_space(space_id=space_id)
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def delete_task(self, user_id: str, task_id: str) -> dict[str, Any]:
        _ = user_id
        try:
            return await self.client.delete_task(task_id=task_id)
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def delete_note(self, user_id: str, note_id: str) -> dict[str, Any]:
        _ = user_id
        try:
            return await self.client.delete_note(note_id=note_id)
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def resolve_entity_id(
        self,
        user_id: str,
        entity_kind: EntityKind,
        title: str,
        space_ids: list[str] | None = None,
    ) -> str | None:
        """Resolve a task/note/space id by case-insensitive title (or spacename)."""
        needle = (title or "").strip()
        if not needle:
            return None
        user_keys = mongo_id_candidates(user_id)
        title_re = {"$regex": f"^{re.escape(needle)}$", "$options": "i"}

        if entity_kind == "space":
            query: dict[str, Any] = {
                "userId": {"$in": user_keys},
                "$and": [
                    {"$or": [{"deletedAt": None}, {"deletedAt": {"$exists": False}}]},
                    {
                        "$or": [
                            {"spacename": title_re},
                            {"spaceName": title_re},
                            {"name": title_re},
                            {"title": title_re},
                        ]
                    },
                ],
            }
            for collection_name in ("spaces", "space", "Spaces"):
                try:
                    doc = await self.db[collection_name].find_one(query, {"_id": 1})
                except Exception:
                    doc = None
                if doc and doc.get("_id") is not None:
                    return str(doc["_id"])
            return None

        query = {
            "userId": {"$in": user_keys},
            "title": title_re,
        }
        preferred_spaces = [sid for sid in (space_ids or []) if sid]
        if preferred_spaces:
            space_keys: list[Any] = []
            for sid in preferred_spaces:
                space_keys.extend(mongo_id_candidates(sid))
            query["spaceId"] = {"$in": space_keys}

        collection = self.db.tasks if entity_kind == "task" else self.db.notes
        try:
            doc = await collection.find_one(query, {"_id": 1})
        except Exception:
            doc = None
        if doc and doc.get("_id") is not None:
            return str(doc["_id"])

        # Fallback: staged collections (chat creates may still be staged-only).
        staged = self.db["stagedTasks"] if entity_kind == "task" else self.db["stagedNotes"]
        try:
            staged_doc = await staged.find_one(query, {"_id": 1})
        except Exception:
            staged_doc = None
        if staged_doc and staged_doc.get("_id") is not None:
            return str(staged_doc["_id"])
        return None

    async def create_reminder(
        self,
        user_id: str,
        title: str,
        description: str = "",
        date_key: str | None = None,
        date_label: str | None = None,
        time_label: str | None = None,
        repeat: str = "once",
        timezone_name: str = DEFAULT_TIMEZONE,
    ) -> dict[str, Any]:
        _ = user_id
        if not date_key or not _is_date_key(date_key):
            raise ValueError("A valid reminder date is required (YYYY-MM-DD).")
        if not time_label or not str(time_label).strip():
            raise ValueError("A reminder time is required (for example 3:30 PM).")
        if not parse_time_label(time_label.strip()):
            raise ValueError("Reminder time must look like 3:30 PM.")

        label = date_label or format_date_label(_datetime_from_date_key(date_key) or datetime.now(timezone.utc))
        payload = {
            "title": title.strip()[:80],
            "description": (description or "").strip()[:500],
            "dateKey": date_key,
            "dateLabel": label,
            "timeLabel": time_label.strip(),
            "repeat": _normalize_repeat(repeat),
            "aiCalling": False,
            "notification": True,
            "beeping": True,
            "source": "ai",
            "timezone": timezone_name,
        }
        try:
            return await self.client.create_reminder(payload)
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def create_event(
        self,
        user_id: str,
        title: str,
        description: str = "",
        location: str = "",
        date_key: str | None = None,
        date_label: str | None = None,
        start_time_label: str | None = None,
        end_time_label: str | None = None,
    ) -> dict[str, Any]:
        _ = user_id
        if not date_key or not _is_date_key(date_key):
            raise ValueError("A valid event date is required (YYYY-MM-DD).")
        if not start_time_label or not str(start_time_label).strip():
            raise ValueError("An event start time is required.")
        start = start_time_label.strip()
        end_label = (end_time_label or "").strip() or _default_end_time(start)
        label = date_label or format_date_label(_datetime_from_date_key(date_key) or datetime.now(timezone.utc))
        payload = {
            "title": title.strip()[:80],
            "description": (description or "").strip()[:500],
            "location": (location or "").strip()[:120],
            "dateKey": date_key,
            "dateLabel": label,
            "startTimeLabel": start,
            "endTimeLabel": end_label,
            "aiReminder": False,
            "aiCalling": False,
            "notification": False,
            "beeping": False,
            "remindBeforeMinutes": 0,
        }
        try:
            return await self.client.create_event(payload)
        except NodeApiError:
            raise
        except Exception as error:
            raise ValueError(str(error)) from error

    async def resolve_space_id(
        self,
        user_id: str,
        preferred_space_ids: list[str] | None = None,
        space_name_hint: str | None = None,
    ) -> tuple[str | None, str | None, list[dict[str, Any]]]:
        """Return (space_id, space_label, available_spaces)."""
        from services.conversation.repository import ConversationRepository

        spaces = await ConversationRepository(self.db).list_user_spaces(user_id)
        if not spaces:
            return None, None, []

        preferred = [sid for sid in (preferred_space_ids or []) if sid]
        hint = (space_name_hint or "").strip().lower()

        if hint:
            for space in spaces:
                label = str(space.get("label") or "").strip().lower()
                sid = str(space.get("spaceId") or "")
                if hint == label or hint in label or hint == sid.lower():
                    return sid, str(space.get("label") or sid), spaces

        for sid in preferred:
            for space in spaces:
                space_sid = str(space.get("spaceId") or "")
                if space_sid == sid or str(to_mongo_id(space_sid)) == str(to_mongo_id(sid)):
                    return sid, str(space.get("label") or sid), spaces

        if len(preferred) == 1:
            only = preferred[0]
            label = next((str(s.get("label") or only) for s in spaces if str(s.get("spaceId")) == only), only)
            return only, label, spaces

        if len(spaces) == 1:
            only = spaces[0]
            return str(only.get("spaceId")), str(only.get("label") or only.get("spaceId")), spaces

        return None, None, spaces


def _normalize_priority(value: str | None) -> str:
    raw = str(value or "Medium").strip().lower()
    if raw in {"high", "h", "urgent"}:
        return "High"
    if raw in {"low", "l"}:
        return "Low"
    return "Medium"


def _normalize_repeat(value: str | None) -> str:
    raw = str(value or "once").strip().lower()
    if raw in {"daily", "every day", "everyday"}:
        return "daily"
    if raw in {"weekly", "every week"}:
        return "weekly"
    if raw in {"weekdays", "weekday"}:
        return "weekdays"
    if raw in {"monthly", "every month"}:
        return "monthly"
    return "once"


def _is_date_key(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return len(value) == 10
    except Exception:
        return False


def _datetime_from_date_key(date_key: str | None) -> datetime | None:
    if not date_key or not _is_date_key(date_key):
        return None
    return datetime.fromisoformat(date_key).replace(tzinfo=timezone.utc)


def _default_end_time(start_label: str) -> str:
    parsed = parse_time_label(start_label)
    if not parsed:
        return start_label
    hour, minute = parsed
    hour = (hour + 1) % 24
    period = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d} {period}"
