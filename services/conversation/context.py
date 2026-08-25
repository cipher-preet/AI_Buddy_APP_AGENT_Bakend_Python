from __future__ import annotations

from typing import Any

from services.conversation.repository import ConversationRepository


async def load_space_context(
    repository: ConversationRepository,
    user_id: str,
    space_id: str,
) -> dict[str, Any]:
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
