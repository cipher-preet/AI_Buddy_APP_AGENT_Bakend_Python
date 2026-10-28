from __future__ import annotations

import json
from typing import Any

from apps.api_gateway.config.setting import settings
from services.goals.ids import clip
from services.goals.schemas import GoalBaselineResult
from services.goals.store import GoalStore, id_candidates
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import get_llm_router
from services.prompts.loader import load_prompt


async def _rows(
    database, names: set[str], collection: str, query: dict[str, Any], limit: int = 12
) -> list[dict[str, Any]]:
    if collection not in names:
        return []
    cursor = database[collection].find(
        query,
        {
            "title": 1,
            "body": 1,
            "description": 1,
            "status": 1,
            "operation": 1,
            "updatedAt": 1,
            "createdAt": 1,
        },
    ).sort("updatedAt", -1).limit(limit)
    return await cursor.to_list(length=limit)


def _item_status(row: dict[str, Any]) -> str:
    return clip(str(row.get("status") or row.get("operation") or ""), 40)


async def load_space_context(
    store: GoalStore, user_id: str, space_ids: list[str], limit: int = 12
) -> dict[str, Any]:
    database = store.db
    names = {"tasks", "stagedTasks", "notes", "stagedNotes", "spaces"}
    spaces: list[dict[str, str]] = []
    tasks: list[dict[str, str]] = []
    notes: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    seen_notes: set[str] = set()
    user_filter = {"userId": {"$in": id_candidates(user_id)}}
    per_space = max(8, min(limit, 40))
    for space_id in space_ids[:8]:
        space_filter = {"spaceId": {"$in": id_candidates(space_id)}}
        query = {**user_filter, **space_filter, "$or": [{"deletedAt": None}, {"deletedAt": {"$exists": False}}]}
        space_doc = None
        if "spaces" in names:
            space_doc = await database.spaces.find_one({"_id": {"$in": id_candidates(space_id)}}, {"spacename": 1})
        spaces.append({"id": space_id, "name": str((space_doc or {}).get("spacename") or "Space")})
        for row in await _rows(database, names, "tasks", query, per_space) + await _rows(
            database, names, "stagedTasks", query, per_space
        ):
            source_id = str(row.get("_id"))
            if source_id in seen_tasks:
                continue
            seen_tasks.add(source_id)
            tasks.append(
                {
                    "sourceType": "task",
                    "sourceId": source_id,
                    "title": clip(str(row.get("title") or "Untitled task"), 140),
                    "detail": clip(str(row.get("body") or row.get("description") or ""), 240),
                    "status": _item_status(row),
                }
            )
        for row in await _rows(database, names, "notes", query, per_space) + await _rows(
            database, names, "stagedNotes", query, per_space
        ):
            source_id = str(row.get("_id"))
            if source_id in seen_notes:
                continue
            seen_notes.add(source_id)
            notes.append(
                {
                    "sourceType": "note",
                    "sourceId": source_id,
                    "title": clip(str(row.get("title") or "Untitled note"), 140),
                    "detail": clip(str(row.get("body") or ""), 240),
                    "status": _item_status(row),
                }
            )
    cap = max(24, min(limit * 2, 48))
    return {
        "spaces": spaces,
        "tasks": tasks[:cap],
        "notes": notes[:cap],
        "transcripts": await _qdrant_snippets(user_id, space_ids, limit=min(16, max(8, limit))),
    }


def _sounds_empty(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "nothing was found",
            "didn't find",
            "did not find",
            "no prior evidence",
            "no evidence",
            "nothing in the",
            "empty space",
            "space is empty",
            "could not read",
            "not finding",
        )
    )


def _fallback_baseline(goal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    title = str(goal.get("title") or "this goal").strip()
    space_names = [str(item.get("name") or "your Space") for item in (context.get("spaces") or [])]
    home = ", ".join(space_names) if space_names else "your Space"
    findings = []
    for item in (context.get("tasks") or [])[:4]:
        findings.append(
            {
                "text": f"Keep “{item['title']}” in {home} and use it as the next piece of work for “{title}”.",
                "sourceType": "task",
                "sourceId": item.get("sourceId") or "",
                "category": "OUTPUT" if str(item.get("status")).lower() in {"done", "completed"} else "COMMITMENT",
            }
        )
    for item in (context.get("notes") or [])[:2]:
        findings.append(
            {
                "text": f"The note “{item['title']}” should shape the next recommendation for “{title}”.",
                "sourceType": "note",
                "sourceId": item.get("sourceId") or "",
                "category": "INTENT",
            }
        )
    if findings:
        summary = (
            f"{home} already has material we can use for “{title}”. "
            "The next step is to turn that into the first milestone, then confirm this picture."
        )
    else:
        summary = (
            f"Use {home} as the home for “{title}”. "
            "Write the outcome you want, add the first milestone, and create the task that proves it. "
            "That is the recommendation I’ll track from here."
        )
        if space_names:
            findings.append(
                {
                    "text": f"Organize “{title}” in {home}: outcome, first milestone, then the task that proves progress.",
                    "sourceType": "space",
                    "sourceId": str((context.get("spaces") or [{}])[0].get("id") or ""),
                    "category": "COMMITMENT",
                }
            )
    return {"summary": summary, "findings": findings[:6], "model": "fallback"}


async def _qdrant_snippets(user_id: str, space_ids: list[str], limit: int = 8) -> list[dict[str, str]]:
    if not space_ids:
        return []
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        from services.vector.qdrant_client import QDRANT_COLLECTION, qdrant_client

        points, _next = await qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="userId", match=MatchValue(value=user_id)),
                    FieldCondition(key="spaceId", match=MatchAny(any=space_ids[:8])),
                ]
            ),
            limit=max(4, min(limit, 16)),
            with_payload=True,
        )
    except Exception as error:
        print("goal qdrant scroll skipped", {"error": type(error).__name__}, flush=True)
        return []
    snippets: list[dict[str, str]] = []
    for point in points:
        payload = point.payload or {}
        text = str(payload.get("text") or payload.get("rawText") or payload.get("normalizedText") or "").strip()
        if not text:
            continue
        snippets.append(
            {
                "sourceType": "transcript",
                "sourceId": str(payload.get("chunkId") or payload.get("conversationId") or point.id),
                "title": clip(text, 140),
                "detail": clip(text, 240),
            }
        )
    return snippets


async def build_baseline(store: GoalStore, goal: dict[str, Any], user_id: str) -> dict[str, Any]:
    del store, user_id
    context = {"spaces": [], "tasks": [], "notes": [], "transcripts": []}
    router = get_llm_router()
    provider_name = settings.CONVERSATION_SEMANTIC_PROVIDER
    model = settings.CONVERSATION_SEMANTIC_MODEL
    provider = router.providers.get(provider_name)
    payload = {
        "goal": {
            "title": goal.get("title"),
            "statement": goal.get("statement"),
            "successCriteria": goal.get("successCriteria"),
            "timeframe": goal.get("timeframe"),
        },
        "context": context,
    }
    if provider and getattr(provider, "configured", True) is not False and model:
        request = StructuredLLMRequest(
            messages=[
                LLMMessage(role="system", content=load_prompt("goal-baseline-v1")),
                LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
            ],
            model=model,
            temperature=0.1,
            max_tokens=settings.GOAL_CLARIFY_MAX_TOKENS,
            metadata={"stage": "goal_baseline"},
            schema_name=GoalBaselineResult.__name__,
        )
        try:
            result = await provider.generate_structured(request, GoalBaselineResult)
            if isinstance(result, GoalBaselineResult) and result.summary.strip():
                known = {item.get("sourceId") for item in context["tasks"] + context["notes"] + context.get("spaces", [])}
                findings = [
                    finding.model_dump()
                    for finding in result.findings
                    if finding.text.strip() and not _sounds_empty(finding.text) and (not finding.sourceId or finding.sourceId in known or not known)
                ][:8]
                summary = result.summary.strip()
                if _sounds_empty(summary):
                    return _fallback_baseline(goal, context)
                return {"summary": summary, "findings": findings, "model": f"{provider_name}:{model}"}
        except Exception as error:
            print("goal baseline model failed", {"error": type(error).__name__}, flush=True)
    return _fallback_baseline(goal, context)
