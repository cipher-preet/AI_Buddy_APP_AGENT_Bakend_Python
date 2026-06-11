"""MongoDB persistence service for AI-generated memory tasks and notes."""

from datetime import datetime, timezone

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, UpdateOne
from pymongo.errors import BulkWriteError

from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import (
    GeneratedNote,
    GeneratedTask,
    MemoryAnalysisOutput,
)

SOURCE = "ai_memory_analysis"
TASKS_COLLECTION = "ai_memory_tasks"
NOTES_COLLECTION = "ai_memory_notes"

mongo_client = AsyncIOMotorClient(settings.MONGO_URL)
mongo_db = mongo_client[settings.MONGO_DB_NAME]


async def ensure_memory_collections() -> None:
    """Create MongoDB indexes required for querying and duplicate prevention."""
    tasks = mongo_db[TASKS_COLLECTION]
    notes = mongo_db[NOTES_COLLECTION]

    await tasks.create_index(
        [
            ("user_id", ASCENDING),
            ("space_id", ASCENDING),
            ("source", ASCENDING),
            ("title", ASCENDING),
            ("description", ASCENDING),
        ],
        name="uq_ai_memory_task_dedupe",
        unique=True,
    )
    await tasks.create_index(
        [("user_id", ASCENDING), ("space_id", ASCENDING), ("request_id", ASCENDING)],
        name="idx_ai_memory_tasks_scope",
    )

    await notes.create_index(
        [
            ("user_id", ASCENDING),
            ("space_id", ASCENDING),
            ("source", ASCENDING),
            ("title", ASCENDING),
            ("content", ASCENDING),
        ],
        name="uq_ai_memory_note_dedupe",
        unique=True,
    )
    await notes.create_index(
        [("user_id", ASCENDING), ("space_id", ASCENDING), ("request_id", ASCENDING)],
        name="idx_ai_memory_notes_scope",
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _object_id(value: str) -> ObjectId:
    return ObjectId(value)


def _task_document(
    *,
    user_id: ObjectId,
    space_id: ObjectId,
    request_id: str | None,
    task: GeneratedTask,
) -> dict[str, object]:
    now = _now()
    due_date = task.due_date.isoformat() if task.due_date else None

    return {
        "user_id": user_id,
        "space_id": space_id,
        "request_id": request_id,
        "title": task.title,
        "description": task.description,
        "priority": task.priority.value,
        "due_date": due_date,
        "source_chunk_ids": task.source_chunk_ids,
        "confidence": task.confidence,
        "source": SOURCE,
        "created_at": now,
        "updated_at": now,
    }


def _note_document(
    *,
    user_id: ObjectId,
    space_id: ObjectId,
    request_id: str | None,
    note: GeneratedNote,
) -> dict[str, object]:
    now = _now()
    return {
        "user_id": user_id,
        "space_id": space_id,
        "request_id": request_id,
        "title": note.title,
        "content": note.content,
        "tags": note.tags,
        "source_chunk_ids": note.source_chunk_ids,
        "confidence": note.confidence,
        "source": SOURCE,
        "created_at": now,
        "updated_at": now,
    }


def _is_duplicate_bulk_error(error: BulkWriteError) -> bool:
    write_errors = error.details.get("writeErrors", [])
    if not write_errors or error.details.get("writeConcernErrors"):
        return False

    return all(
        write_error.get("code") == 11000
        for write_error in write_errors
    )


async def _bulk_insert_tasks_if_new(
    *,
    user_id: ObjectId,
    space_id: ObjectId,
    request_id: str | None,
    tasks: list[GeneratedTask],
) -> int:
    if not tasks:
        return 0

    operations = []
    for task in tasks:
        document = _task_document(
            user_id=user_id,
            space_id=space_id,
            request_id=request_id,
            task=task,
        )
        operations.append(
            UpdateOne(
                {
                    "user_id": user_id,
                    "space_id": space_id,
                    "source": SOURCE,
                    "title": task.title,
                    "description": task.description,
                },
                {"$setOnInsert": document},
                upsert=True,
            )
        )

    try:
        result = await mongo_db[TASKS_COLLECTION].bulk_write(
            operations,
            ordered=False,
        )
    except BulkWriteError as error:
        if not _is_duplicate_bulk_error(error):
            raise
        return int(error.details.get("nUpserted") or 0)

    return result.upserted_count


async def _bulk_insert_notes_if_new(
    *,
    user_id: ObjectId,
    space_id: ObjectId,
    request_id: str | None,
    notes: list[GeneratedNote],
) -> int:
    if not notes:
        return 0

    operations = []
    for note in notes:
        document = _note_document(
            user_id=user_id,
            space_id=space_id,
            request_id=request_id,
            note=note,
        )
        operations.append(
            UpdateOne(
                {
                    "user_id": user_id,
                    "space_id": space_id,
                    "source": SOURCE,
                    "title": note.title,
                    "content": note.content,
                },
                {"$setOnInsert": document},
                upsert=True,
            )
        )

    try:
        result = await mongo_db[NOTES_COLLECTION].bulk_write(
            operations,
            ordered=False,
        )
    except BulkWriteError as error:
        if not _is_duplicate_bulk_error(error):
            raise
        return int(error.details.get("nUpserted") or 0)

    return result.upserted_count


async def save_generated_tasks_and_notes(
    *,
    user_id: str,
    space_id: str,
    request_id: str | None,
    output: MemoryAnalysisOutput,
) -> dict[str, int]:
    """Persist generated records in MongoDB and skip duplicates."""
    user_object_id = _object_id(user_id)
    space_object_id = _object_id(space_id)

    task_count = await _bulk_insert_tasks_if_new(
        user_id=user_object_id,
        space_id=space_object_id,
        request_id=request_id,
        tasks=output.tasks,
    )
    note_count = await _bulk_insert_notes_if_new(
        user_id=user_object_id,
        space_id=space_object_id,
        request_id=request_id,
        notes=output.notes,
    )

    return {"tasks": task_count, "notes": note_count}
