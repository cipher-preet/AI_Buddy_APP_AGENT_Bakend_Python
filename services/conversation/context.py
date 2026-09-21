from __future__ import annotations

from typing import Any

from services.conversation.repository import ConversationRepository, has_space_id, to_mongo_id


def _empty_space_context(user_id: Any, space_id: Any = None) -> dict[str, Any]:
    return {
        "spaceMemory": {
            "userId": to_mongo_id(user_id),
            "spaceId": to_mongo_id(space_id) if has_space_id(space_id) else None,
            "currentSummary": "",
            "importantFacts": [],
            "importantDecisions": [],
            "openQuestionIds": [],
            "activeTaskIds": [],
            "blockerIds": [],
            "recentConversationSummaryIds": [],
            "lastUpdatedConversationId": None,
            "version": 1,
        },
        "activeTasks": [],
        "recentNotes": [],
        "recentSummaries": [],
        "openQuestions": [],
        "unresolvedBlockers": [],
        "importantDecisions": [],
    }


async def load_space_context(
    repository: ConversationRepository,
    user_id: str,
    space_id: str | None,
) -> dict[str, Any]:
    # Extension meetings are not tied to a Buddy space; skip space-scoped context.
    if not has_space_id(space_id):
        return _empty_space_context(user_id, None)
    memory = await repository.get_space_memory(user_id, space_id)
    return {
        "spaceMemory": memory.model_dump(by_alias=True),
        "activeTasks": await repository.list_active_tasks(user_id, space_id),
        "recentNotes": await repository.list_recent_notes(user_id, space_id, limit=25),
        "recentSummaries": await repository.list_recent_summaries(user_id, space_id, limit=5),
        "openQuestions": [],
        "unresolvedBlockers": [],
        "importantDecisions": memory.importantDecisions,
    }
