from __future__ import annotations

from typing import Any

from apps.api_gateway.config.setting import settings
from services.mindmap.schemas import SpaceContextBundle


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def build_context_pack(bundle: SpaceContextBundle) -> dict[str, Any]:
    """Compress space activity into a bounded payload for the LLM."""
    budget = settings.MINDMAP_CONTEXT_TOKEN_BUDGET
    pack: dict[str, Any] = {
        "spaceId": bundle.spaceId,
        "spaceName": bundle.spaceName,
        "tasks": [],
        "notes": [],
        "meetingHighlights": [],
        "sourceStats": bundle.sourceStats.model_dump(),
    }
    used = _estimate_tokens(bundle.spaceName) + 80

    for task in bundle.tasks:
        chunk = {
            "title": task.get("title"),
            "body": task.get("body") or None,
            "priority": task.get("priority"),
            "status": task.get("status"),
            "dueDate": task.get("dueDate"),
        }
        cost = _estimate_tokens(str(chunk))
        if used + cost > budget:
            pack["sourceStats"]["truncated"] = True
            break
        pack["tasks"].append(chunk)
        used += cost

    for note in bundle.notes:
        chunk = {"title": note.get("title"), "body": note.get("body") or None}
        cost = _estimate_tokens(str(chunk))
        if used + cost > budget:
            pack["sourceStats"]["truncated"] = True
            break
        pack["notes"].append(chunk)
        used += cost

    # Group transcript snippets by conversation, keep newest first.
    by_conversation: dict[str, list[str]] = {}
    for item in bundle.transcripts:
        key = str(item.get("conversationId") or item.get("id") or "unknown")
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        by_conversation.setdefault(key, []).append(text)

    for conversation_id, snippets in by_conversation.items():
        joined = " ".join(snippets)[: settings.MINDMAP_TRANSCRIPT_CHARS]
        chunk = {"conversationId": conversation_id, "text": joined}
        cost = _estimate_tokens(joined)
        if used + cost > budget:
            pack["sourceStats"]["truncated"] = True
            break
        pack["meetingHighlights"].append(chunk)
        used += cost

    pack["tokenEstimate"] = used
    return pack


def has_usable_context(pack: dict[str, Any]) -> bool:
    return bool(pack.get("tasks") or pack.get("notes") or pack.get("meetingHighlights"))
