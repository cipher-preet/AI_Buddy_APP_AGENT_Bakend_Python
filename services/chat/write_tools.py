from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from services.chat.writes import ChatWriteStore


@dataclass(frozen=True)
class WriteToolResult:
    ok: bool
    tool_name: str
    summary: str
    data: dict[str, Any] | None = None
    error: str | None = None


ToolHandler = Callable[[ChatWriteStore, dict[str, Any]], Awaitable[dict[str, Any]]]


WRITE_TOOL_NAMES = (
    "create_task",
    "create_note",
    "create_space",
    "create_reminder",
    "create_event",
    "update_task",
    "update_note",
    "update_space",
    "delete_task",
    "delete_note",
    "delete_space",
)


class ChatWriteToolRegistry:
    """Optimized write tools: each create/update/delete maps to one Node API tool."""

    def __init__(self, store: ChatWriteStore):
        self.store = store
        self._handlers: dict[str, ToolHandler] = {
            "create_task": self._create_task,
            "create_note": self._create_note,
            "create_space": self._create_space,
            "create_reminder": self._create_reminder,
            "create_event": self._create_event,
            "update_task": self._update_task,
            "update_note": self._update_note,
            "update_space": self._update_space,
            "delete_task": self._delete_task,
            "delete_note": self._delete_note,
            "delete_space": self._delete_space,
        }

    def has(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    async def run(self, tool_name: str, args: dict[str, Any]) -> WriteToolResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return WriteToolResult(
                ok=False,
                tool_name=tool_name,
                summary=f"Unknown write tool: {tool_name}",
                error=f"Unknown write tool: {tool_name}",
            )
        try:
            data = await handler(self.store, args)
            return WriteToolResult(
                ok=True,
                tool_name=tool_name,
                summary=_success_summary(tool_name, data, args),
                data=data,
            )
        except Exception as error:
            return WriteToolResult(
                ok=False,
                tool_name=tool_name,
                summary=str(error),
                error=str(error),
            )

    @staticmethod
    async def _create_task(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.create_task(
            user_id=str(args["user_id"]),
            space_id=str(args["space_id"]),
            title=str(args["title"]),
            description=str(args.get("description") or ""),
            due_date=args.get("due_date"),
            priority=str(args.get("priority") or "Medium"),
        )

    @staticmethod
    async def _create_note(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.create_note(
            user_id=str(args["user_id"]),
            space_id=str(args["space_id"]),
            title=str(args["title"]),
            body=str(args.get("description") or args.get("body") or ""),
            date_key=args.get("date_key"),
        )

    @staticmethod
    async def _create_space(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.create_space(
            user_id=str(args["user_id"]),
            spacename=str(args["title"]),
            description=str(args.get("description") or "New"),
        )

    @staticmethod
    async def _create_reminder(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.create_reminder(
            user_id=str(args["user_id"]),
            title=str(args["title"]),
            description=str(args.get("description") or ""),
            date_key=args.get("date_key"),
            date_label=args.get("date_label"),
            time_label=args.get("time_label"),
            repeat=str(args.get("repeat") or "once"),
        )

    @staticmethod
    async def _create_event(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.create_event(
            user_id=str(args["user_id"]),
            title=str(args["title"]),
            description=str(args.get("description") or ""),
            location=str(args.get("location") or ""),
            date_key=args.get("date_key"),
            date_label=args.get("date_label"),
            start_time_label=args.get("start_time_label"),
            end_time_label=args.get("end_time_label"),
        )

    @staticmethod
    async def _update_task(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.update_task(
            user_id=str(args["user_id"]),
            task_id=str(args["entity_id"]),
            title=args.get("title"),
            description=args.get("description"),
            due_date=args.get("due_date"),
            priority=args.get("priority"),
        )

    @staticmethod
    async def _update_note(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        body = args.get("description") if args.get("description") is not None else args.get("body")
        return await store.update_note(
            user_id=str(args["user_id"]),
            note_id=str(args["entity_id"]),
            title=args.get("title"),
            body=body,
            date_key=args.get("date_key"),
        )

    @staticmethod
    async def _update_space(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.update_space(
            user_id=str(args["user_id"]),
            space_id=str(args["entity_id"]),
            spacename=args.get("title"),
            description=args.get("description"),
        )

    @staticmethod
    async def _delete_task(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.delete_task(
            user_id=str(args["user_id"]),
            task_id=str(args["entity_id"]),
        )

    @staticmethod
    async def _delete_note(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.delete_note(
            user_id=str(args["user_id"]),
            note_id=str(args["entity_id"]),
        )

    @staticmethod
    async def _delete_space(store: ChatWriteStore, args: dict[str, Any]) -> dict[str, Any]:
        return await store.delete_space(
            user_id=str(args["user_id"]),
            space_id=str(args["entity_id"]),
        )


def _success_summary(tool_name: str, data: dict[str, Any], args: dict[str, Any]) -> str:
    space_label = args.get("space_label")
    label = args.get("title") or data.get("title") or data.get("spacename") or data.get("id") or "item"
    if tool_name == "create_task":
        return (
            f"Done - I created the task **{data.get('title')}**"
            + (f" in **{space_label}**" if space_label else "")
            + (f" (due {data.get('dueDate')})" if data.get("dueDate") else "")
            + f" · priority {data.get('priority')}."
        )
    if tool_name == "create_note":
        return (
            f"Done - I saved the note **{data.get('title')}**"
            + (f" in **{space_label}**" if space_label else "")
            + "."
        )
    if tool_name == "create_space":
        return (
            f"Done - I created the space **{data.get('spacename')}** and saved it. "
            "You can add it as chat context anytime."
        )
    if tool_name == "create_reminder":
        repeat = data.get("repeat") or "once"
        return (
            f"Done - reminder set: **{data.get('title')}** on {data.get('dateLabel')} at {data.get('timeLabel')}"
            + (f" ({repeat})" if repeat != "once" else "")
            + "."
        )
    if tool_name == "create_event":
        return (
            f"Done - event **{data.get('title')}** scheduled for {data.get('dateLabel')} "
            f"from {data.get('startTimeLabel')} to {data.get('endTimeLabel')}"
            + (f" at {data.get('location')}" if data.get("location") else "")
            + "."
        )
    if tool_name == "update_task":
        return f"Done - I updated the task **{data.get('title') or label}**."
    if tool_name == "update_note":
        return f"Done - I updated the note **{data.get('title') or label}**."
    if tool_name == "update_space":
        return f"Done - I updated the space **{data.get('spacename') or label}**."
    if tool_name == "delete_task":
        return f"Done - I deleted the task **{label}**."
    if tool_name == "delete_note":
        return f"Done - I deleted the note **{label}**."
    if tool_name == "delete_space":
        return f"Done - I deleted the space **{label}**."
    return f"Done - {tool_name} completed."


def tool_catalog_for_prompt() -> str:
    return (
        "Available write tools:\n"
        "- create_task(spaceId, title, description?, dueDate?, priority?) — "
        "trackable commitment / to-do the user intends to complete\n"
        "- create_note(spaceId, title, description?, date?) — "
        "information or reference material to store (not a to-do)\n"
        "- create_space(spacename, description?) — new workspace/project container\n"
        "- create_reminder(title, dateKey, dateLabel, timeLabel, repeat?, description?) — "
        "time-based notification\n"
        "- create_event(title, dateKey, dateLabel, startTimeLabel, endTimeLabel?, location?, description?) — "
        "calendar appointment\n"
        "- update_task(entityId|title, title?, description?, dueDate?, priority?)\n"
        "- update_note(entityId|title, title?, description?, date?)\n"
        "- update_space(entityId|title, spacename?, description?)\n"
        "- delete_task(entityId|title)\n"
        "- delete_note(entityId|title)\n"
        "- delete_space(entityId|title)\n"
        "These tools call Buddy Node APIs and persist to the database. "
        "Choose the tool from user meaning and context — not from fixed keywords. "
        "For update/delete, title looks up the existing item when entityId is unknown; "
        "newTitle is the replacement name when renaming."
    )
