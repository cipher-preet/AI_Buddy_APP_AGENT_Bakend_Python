from __future__ import annotations

from typing import Any

import httpx

from apps.api_gateway.config.setting import settings


class NodeApiError(ValueError):
    """Raised when a Node home create API returns an error."""


class NodeHomeClient:
    """HTTP client for Buddy Node create APIs used as chat write tools."""

    def __init__(self, auth_token: str | None = None, base_url: str | None = None):
        self.auth_token = (auth_token or "").strip() or None
        self.base_url = (base_url or settings.NODE_API_BASE_URL or "http://127.0.0.1:5000").rstrip("/")

    def _headers(self, require_auth: bool = True) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif require_auth:
            raise NodeApiError("Please sign in again so I can save this for you.")
        return headers

    async def create_space(self, user_id: str, spacename: str, description: str = "New") -> dict[str, Any]:
        # create-space currently accepts userId in body (no requireAuth).
        payload = {
            "userId": user_id,
            "spacename": spacename.strip(),
            "description": (description or "New").strip() or "New",
        }
        data = await self._post("/api/v1/home/create-space", payload, require_auth=False)
        return {
            "id": str((data.get("space") or {}).get("id") or data.get("spaceId") or ""),
            "spacename": spacename.strip(),
            "description": payload["description"],
            "raw": data,
        }

    async def create_task(
        self,
        space_id: str,
        title: str,
        description: str = "",
        due_date: str | None = None,
        priority: str = "Medium",
    ) -> dict[str, Any]:
        body = (description or "").strip() or title.strip() or "Created from Buddy chat"
        payload: dict[str, Any] = {
            "spaceId": space_id,
            "title": title.strip(),
            "description": body,
            "priority": priority or "Medium",
        }
        if due_date:
            payload["date"] = due_date
        data = await self._post("/api/v1/home/create-staged-task", payload, require_auth=True)
        task = data.get("task") or {}
        return {
            "id": str(task.get("id") or task.get("_id") or ""),
            "title": str(task.get("title") or title),
            "description": str(task.get("description") or task.get("body") or body),
            "dueDate": task.get("dueDate") or due_date,
            "priority": str(task.get("priority") or priority or "Medium"),
            "spaceId": space_id,
            "raw": data,
        }

    async def create_note(
        self,
        space_id: str,
        title: str,
        body: str = "",
        date_key: str | None = None,
    ) -> dict[str, Any]:
        description = (body or "").strip() or title.strip() or "Created from Buddy chat"
        payload: dict[str, Any] = {
            "spaceId": space_id,
            "title": title.strip(),
            "description": description,
        }
        if date_key:
            payload["date"] = date_key
        data = await self._post("/api/v1/home/create-staged-note", payload, require_auth=True)
        note = data.get("note") or {}
        return {
            "id": str(note.get("id") or note.get("_id") or ""),
            "title": str(note.get("title") or title),
            "body": str(note.get("body") or note.get("description") or description),
            "spaceId": space_id,
            "raw": data,
        }

    async def update_space(
        self,
        space_id: str,
        spacename: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"spaceId": space_id}
        if spacename is not None:
            payload["spacename"] = spacename.strip()
        if description is not None:
            payload["description"] = (description or "").strip() or "New"
        data = await self._post("/api/v1/home/update-space", payload, require_auth=True)
        space = data.get("space") or {}
        return {
            "id": str(space.get("id") or space_id),
            "spacename": str(space.get("spacename") or spacename or ""),
            "description": str(space.get("description") or description or ""),
            "raw": data,
        }

    async def update_task(
        self,
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        due_date: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"taskId": task_id}
        if title is not None:
            payload["title"] = title.strip()
        if description is not None:
            payload["description"] = description.strip()
        if due_date is not None:
            payload["date"] = due_date
        if priority is not None:
            payload["priority"] = priority
        data = await self._post("/api/v1/home/update-staged-task", payload, require_auth=True)
        task = data.get("task") or {}
        return {
            "id": str(task.get("id") or task.get("_id") or task_id),
            "title": str(task.get("title") or title or ""),
            "description": str(task.get("description") or task.get("body") or description or ""),
            "dueDate": task.get("dueDate") or due_date,
            "priority": str(task.get("priority") or priority or ""),
            "raw": data,
        }

    async def update_note(
        self,
        note_id: str,
        title: str | None = None,
        body: str | None = None,
        date_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"noteId": note_id}
        if title is not None:
            payload["title"] = title.strip()
        if body is not None:
            payload["description"] = body.strip()
        if date_key is not None:
            payload["date"] = date_key
        data = await self._post("/api/v1/home/update-staged-note", payload, require_auth=True)
        note = data.get("note") or {}
        return {
            "id": str(note.get("id") or note.get("_id") or note_id),
            "title": str(note.get("title") or title or ""),
            "body": str(note.get("body") or note.get("description") or body or ""),
            "raw": data,
        }

    async def delete_space(self, space_id: str) -> dict[str, Any]:
        data = await self._post("/api/v1/home/delete-space", {"spaceId": space_id}, require_auth=True)
        return {
            "id": str(data.get("deletedSpaceId") or space_id),
            "raw": data,
        }

    async def delete_task(self, task_id: str) -> dict[str, Any]:
        data = await self._post("/api/v1/home/delete-staged-task", {"taskId": task_id}, require_auth=True)
        return {
            "id": str(data.get("deletedTaskId") or task_id),
            "raw": data,
        }

    async def delete_note(self, note_id: str) -> dict[str, Any]:
        data = await self._post("/api/v1/home/delete-staged-note", {"noteId": note_id}, require_auth=True)
        return {
            "id": str(data.get("deletedNoteId") or note_id),
            "raw": data,
        }

    async def create_reminder(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = await self._post("/api/v1/home/create-reminder", payload, require_auth=True)
        reminder = data.get("reminder") or {}
        return {
            "id": str(reminder.get("id") or reminder.get("_id") or ""),
            "title": str(reminder.get("title") or payload.get("title") or ""),
            "dateKey": str(reminder.get("dateKey") or payload.get("dateKey") or ""),
            "dateLabel": str(reminder.get("dateLabel") or payload.get("dateLabel") or ""),
            "timeLabel": str(reminder.get("timeLabel") or payload.get("timeLabel") or ""),
            "repeat": str(reminder.get("repeat") or payload.get("repeat") or "once"),
            "raw": data,
        }

    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = await self._post("/api/v1/home/create-calendar-event", payload, require_auth=True)
        event = data.get("event") or {}
        return {
            "id": str(event.get("id") or event.get("_id") or ""),
            "title": str(event.get("title") or payload.get("title") or ""),
            "dateKey": str(event.get("dateKey") or payload.get("dateKey") or ""),
            "dateLabel": str(event.get("dateLabel") or payload.get("dateLabel") or ""),
            "startTimeLabel": str(event.get("startTimeLabel") or payload.get("startTimeLabel") or ""),
            "endTimeLabel": str(event.get("endTimeLabel") or payload.get("endTimeLabel") or ""),
            "location": str(event.get("location") or payload.get("location") or ""),
            "raw": data,
        }

    async def _post(self, path: str, payload: dict[str, Any], require_auth: bool) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        timeout = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers(require_auth=require_auth))
        except httpx.HTTPError as error:
            raise NodeApiError(f"Could not reach Buddy API ({self.base_url}). Is Node running on :5000?") from error

        try:
            body = response.json()
        except Exception:
            body = {}

        if response.status_code >= 400 or body.get("success") is False:
            message = (
                body.get("message")
                or body.get("detail")
                or body.get("error")
                or f"Buddy API error ({response.status_code})"
            )
            raise NodeApiError(str(message))

        data = body.get("data")
        if isinstance(data, dict):
            return data
        return body if isinstance(body, dict) else {"message": str(body)}
