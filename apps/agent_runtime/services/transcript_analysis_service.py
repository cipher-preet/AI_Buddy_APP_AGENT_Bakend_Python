"""Transcript-window analysis pipeline for speech memory."""

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from bson import ObjectId
from openai import AsyncOpenAI

from apps.agent_runtime.llms.openai.structured import parse_chat_completion
from apps.agent_runtime.llms.prompts.transcript_analysis_prompt import (
    TRANSCRIPT_ANALYSIS_CHAT_PROMPT,
    TRANSCRIPT_ANALYSIS_COVERAGE_REPAIR_PROMPT,
    TRANSCRIPT_ANALYSIS_REPAIR_PROMPT,
    TRANSCRIPT_ANALYSIS_TASK_REPAIR_PROMPT,
)
from apps.agent_runtime.rag.vectorstores.qdrant_store import (
    MemoryVector,
    fetch_recent_speech_chunks,
    fetch_strict_unpublished_chunks,
    mark_vectors_analysis_completed,
    search_relevant_speech_chunks,
    sort_vectors_chronologically,
)
from apps.agent_runtime.services.task_note_service import (
    NOTES_COLLECTION,
    SUMMARIES_COLLECTION,
    TASKS_COLLECTION,
    ensure_memory_collections,
    mongo_db,
)
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import AnalysisJob
from packages.schemas.transcript_analysis_schema import (
    NoteOperation,
    TaskOperation,
    TranscriptAnalysisOutput,
)
from services.queue.redis_queue import redis_client
from services.vector.embedding_service import generate_embedding

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _object_id(value: str) -> ObjectId:
    return ObjectId(str(value).strip())


def normalize_text(value: str) -> str:
    """Normalize text for deterministic duplicate checks."""
    text = value.casefold().strip()
    text = re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def task_fingerprint(
    *,
    user_id: str,
    space_id: str,
    title: str,
    description: str,
    due_at: str | None,
) -> str:
    """Build a deterministic task fingerprint scoped to one user space."""
    due_date = (due_at or "").split("T", 1)[0]
    normalized = "|".join(
        [
            user_id,
            space_id,
            normalize_text(title),
            normalize_text(description)[:160],
            due_date,
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def window_id_for_chunks(user_id: str, space_id: str, chunk_ids: list[str]) -> str:
    raw = "|".join([user_id, space_id, *chunk_ids])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _chunk_document(vector: MemoryVector) -> dict[str, Any]:
    payload = vector.payload
    return {
        "point_id": vector.point_id,
        "chunkId": vector.chunk_id,
        "text": vector.text,
        "request_id": payload.get("request_id") or payload.get("requestId"),
        "createdAt": payload.get("createdAt"),
        "chunkIndex": payload.get("chunkIndex"),
    }


def _combine_window_text(chunks: list[dict[str, Any]]) -> str:
    parts = [str(chunk.get("text") or "").strip() for chunk in chunks]
    return " ".join(part for part in parts if part).strip()


def _operation_source_ids(operation: TaskOperation | NoteOperation, fallback: list[str]) -> list[str]:
    if operation.source_chunk_ids:
        return operation.source_chunk_ids
    return fallback if len(fallback) == 1 else []


async def acquire_space_lock(user_id: str, space_id: str) -> tuple[str, str] | None:
    """Acquire a Redis lock with ownership token."""
    token = str(uuid.uuid4())
    key = f"lock:transcript-analysis:{user_id}:{space_id}"
    acquired = await redis_client.set(
        key,
        token,
        nx=True,
        ex=settings.TRANSCRIPT_ANALYSIS_LOCK_TTL_SECONDS,
    )
    return (key, token) if acquired else None


async def release_space_lock(key: str, token: str) -> None:
    """Release a Redis lock only when owned by this worker."""
    script = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    end
    return 0
    """
    await redis_client.eval(script, 1, key, token)


async def _load_unpublished_window(job: AnalysisJob) -> tuple[list[MemoryVector], list[dict[str, Any]]]:
    vectors = await fetch_strict_unpublished_chunks(
        user_id=job.user_id,
        space_id=job.space_id,
        limit=settings.TRANSCRIPT_ANALYSIS_MAX_BATCH_CHUNKS,
    )
    vectors = sort_vectors_chronologically(vectors)
    if job.chunk_ids:
        chunk_ids = set(job.chunk_ids)
        vectors = [vector for vector in vectors if vector.chunk_id in chunk_ids]
    vectors = vectors[: settings.TRANSCRIPT_ANALYSIS_MAX_BATCH_CHUNKS]
    return vectors, [_chunk_document(vector) for vector in vectors]


def _build_analysis_window(
    *,
    user_id: str,
    space_id: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk_ids = [str(chunk["chunkId"]) for chunk in chunks]
    request_ids = sorted(
        {str(chunk["request_id"]) for chunk in chunks if chunk.get("request_id")}
    )
    window_id = window_id_for_chunks(user_id, space_id, chunk_ids)
    return {
        "window_id": window_id,
        "user_id": _object_id(user_id),
        "space_id": _object_id(space_id),
        "user_id_text": user_id,
        "space_id_text": space_id,
        "chunk_ids": chunk_ids,
        "chunks": chunks,
        "point_ids": [str(chunk["point_id"]) for chunk in chunks if chunk.get("point_id")],
        "request_ids": request_ids,
        "combined_text": _combine_window_text(chunks),
        "from_created_at": chunks[0].get("createdAt"),
        "to_created_at": chunks[-1].get("createdAt"),
    }


async def _load_existing_tasks(user_id: str, space_id: str) -> list[dict[str, Any]]:
    statuses = ["open", "pending", "in_progress", "in-progress", "completed"]
    cursor = (
        mongo_db[TASKS_COLLECTION]
        .find(
            {
                "user_id": _object_id(user_id),
                "space_id": _object_id(space_id),
                "$or": [
                    {"status": {"$in": statuses}},
                    {"status": {"$exists": False}},
                ],
            }
        )
        .sort("updated_at", -1)
        .limit(settings.TRANSCRIPT_ANALYSIS_TASK_LIMIT)
    )
    return [doc async for doc in cursor]


async def _load_existing_notes(user_id: str, space_id: str) -> list[dict[str, Any]]:
    cursor = (
        mongo_db[NOTES_COLLECTION]
        .find({"user_id": _object_id(user_id), "space_id": _object_id(space_id)})
        .sort("updated_at", -1)
        .limit(settings.TRANSCRIPT_ANALYSIS_NOTE_LIMIT)
    )
    return [doc async for doc in cursor]


async def _load_summary(user_id: str, space_id: str) -> dict[str, Any] | None:
    return await mongo_db[SUMMARIES_COLLECTION].find_one(
        {"user_id": _object_id(user_id), "space_id": _object_id(space_id)}
    )


def _public_record(doc: dict[str, Any], *, body_key: str) -> dict[str, Any]:
    return {
        "id": str(doc.get("_id")),
        "title": doc.get("title", ""),
        body_key: doc.get(body_key, ""),
        "status": doc.get("status"),
        "due_at": doc.get("due_at") or doc.get("due_date"),
        "source_chunk_ids": doc.get("source_chunk_ids", []),
        "fingerprint": doc.get("fingerprint"),
        "updated_at": str(doc.get("updated_at") or ""),
    }


async def build_context_package(
    *,
    user_id: str,
    space_id: str,
    window: dict[str, Any],
) -> dict[str, Any]:
    recent_vectors = await fetch_recent_speech_chunks(
        user_id=user_id,
        space_id=space_id,
        limit=settings.TRANSCRIPT_ANALYSIS_RECENT_CHUNK_LIMIT,
    )
    recent_chunks = [_chunk_document(vector) for vector in recent_vectors]
    recent_excerpt = " ".join(chunk["text"] for chunk in recent_chunks[-5:])
    query_text = f"{window['combined_text']}\n{recent_excerpt}".strip()
    query_vector = await generate_embedding(query_text)
    relevant = await search_relevant_speech_chunks(
        user_id=user_id,
        space_id=space_id,
        query_vector=query_vector,
        exclude_chunk_ids=set(window["chunk_ids"]),
        limit=settings.TRANSCRIPT_ANALYSIS_SEMANTIC_LIMIT,
    )
    summary = await _load_summary(user_id, space_id)
    tasks = await _load_existing_tasks(user_id, space_id)
    notes = await _load_existing_notes(user_id, space_id)
    tz_name = settings.USER_TIMEZONE
    current_datetime = datetime.now(ZoneInfo(tz_name)).isoformat()

    return {
        "user_id": user_id,
        "space_id": space_id,
        "current_datetime": current_datetime,
        "timezone": tz_name,
        "analysis_window": {
            "window_id": window["window_id"],
            "combined_text": window["combined_text"],
            "chunk_ids": window["chunk_ids"],
            "chunks": [
                {
                    "chunkId": chunk["chunkId"],
                    "text": chunk["text"],
                    "createdAt": chunk.get("createdAt"),
                    "chunkIndex": chunk.get("chunkIndex"),
                }
                for chunk in window.get("chunks", [])
            ],
        },
        "recent_transcripts": recent_chunks,
        "relevant_older_context": [
            {
                "chunkId": vector.chunk_id,
                "text": vector.text,
                "createdAt": vector.payload.get("createdAt"),
                "score": score,
            }
            for vector, score in relevant
        ],
        "running_summary": (summary or {}).get("running_summary", ""),
        "existing_tasks": [_public_record(task, body_key="description") for task in tasks],
        "existing_notes": [_public_record(note, body_key="content") for note in notes],
    }


async def analyze_context_package(context_package: dict[str, Any]) -> TranscriptAnalysisOutput:
    messages = TRANSCRIPT_ANALYSIS_CHAT_PROMPT.format_messages(
        current_datetime=context_package["current_datetime"],
        timezone=context_package["timezone"],
        context_package=json.dumps(context_package, ensure_ascii=True, default=str),
    )
    return await parse_chat_completion(
        client,
        model=settings.OPENAI_CHAT_MODEL,
        temperature=0,
        response_model=TranscriptAnalysisOutput,
        messages=messages,
    )


async def _find_task_by_fingerprint(
    user_id: str,
    space_id: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    return await mongo_db[TASKS_COLLECTION].find_one(
        {
            "user_id": _object_id(user_id),
            "space_id": _object_id(space_id),
            "fingerprint": fingerprint,
        }
    )


async def _apply_task_operation(
    *,
    user_id: str,
    space_id: str,
    window: dict[str, Any],
    operation: TaskOperation,
    existing_task_ids: set[str],
) -> str:
    if operation.operation == "no_change" or operation.confidence < 0.7:
        return "ignored"
    if operation.operation == "create" and not operation.title:
        return "ignored"

    now = _now()
    source_chunk_ids = _operation_source_ids(operation, window["chunk_ids"])
    fingerprint = task_fingerprint(
        user_id=user_id,
        space_id=space_id,
        title=operation.title,
        description=operation.description,
        due_at=operation.due_at,
    )
    existing = await _find_task_by_fingerprint(user_id, space_id, fingerprint)
    target_id = operation.existing_task_id
    if existing:
        target_id = str(existing["_id"])

    if operation.operation in {"update", "complete", "cancel"} and target_id not in existing_task_ids:
        return "ignored"

    if target_id:
        set_fields: dict[str, Any] = {
            "updated_at": now,
            "source_chunk_ids": source_chunk_ids,
            "source_request_ids": window.get("request_ids", []),
            "source_window_ids": [window["window_id"]],
            "confidence": operation.confidence,
        }
        if operation.operation in {"create", "update"}:
            set_fields.update(
                {
                    "title": operation.title,
                    "description": operation.description,
                    "due_at": operation.due_at,
                    "status": operation.status or "open",
                    "fingerprint": fingerprint,
                }
            )
        elif operation.operation == "complete":
            set_fields["status"] = "completed"
            set_fields["completed_at"] = now
        elif operation.operation == "cancel":
            set_fields["status"] = "cancelled"
            set_fields["cancelled_at"] = now
        await mongo_db[TASKS_COLLECTION].update_one({"_id": ObjectId(target_id)}, {"$set": set_fields})
        return "updated"

    document = {
        "user_id": _object_id(user_id),
        "space_id": _object_id(space_id),
        "title": operation.title,
        "description": operation.description,
        "status": operation.status or "open",
        "due_at": operation.due_at,
        "source_chunk_ids": source_chunk_ids,
        "source_request_ids": window.get("request_ids", []),
        "source_window_ids": [window["window_id"]],
        "confidence": operation.confidence,
        "fingerprint": fingerprint,
        "source": "transcript_analysis",
        "created_at": now,
        "updated_at": now,
    }
    await mongo_db[TASKS_COLLECTION].update_one(
        {
            "user_id": document["user_id"],
            "space_id": document["space_id"],
            "fingerprint": fingerprint,
        },
        {"$setOnInsert": document},
        upsert=True,
    )
    return "created"


async def _apply_note_operation(
    *,
    user_id: str,
    space_id: str,
    window: dict[str, Any],
    operation: NoteOperation,
    existing_note_ids: set[str],
) -> str:
    if operation.operation == "no_change" or operation.confidence < 0.7:
        return "ignored"
    if operation.operation == "create" and (not operation.title or not operation.content):
        return "ignored"

    now = _now()
    normalized_title = normalize_text(operation.title)
    source_chunk_ids = _operation_source_ids(operation, window["chunk_ids"])
    target_id = operation.existing_note_id
    existing = await mongo_db[NOTES_COLLECTION].find_one(
        {
            "user_id": _object_id(user_id),
            "space_id": _object_id(space_id),
            "normalized_title": normalized_title,
        }
    )
    if existing:
        target_id = str(existing["_id"])

    if operation.operation in {"update", "complete", "cancel"} and target_id not in existing_note_ids:
        return "ignored"

    if target_id:
        update: dict[str, Any] = {
            "title": operation.title,
            "normalized_title": normalized_title,
            "updated_at": now,
            "source_chunk_ids": source_chunk_ids,
            "source_request_ids": window.get("request_ids", []),
            "source_window_ids": [window["window_id"]],
            "confidence": operation.confidence,
        }
        if operation.content:
            previous = await mongo_db[NOTES_COLLECTION].find_one({"_id": ObjectId(target_id)})
            old_content = str((previous or {}).get("content") or "").strip()
            new_content = operation.content.strip()
            update["content"] = (
                old_content
                if new_content and new_content in old_content
                else "\n\n".join(part for part in [old_content, new_content] if part)
            )
        if operation.operation == "cancel":
            update["status"] = "cancelled"
        await mongo_db[NOTES_COLLECTION].update_one({"_id": ObjectId(target_id)}, {"$set": update})
        return "updated"

    document = {
        "user_id": _object_id(user_id),
        "space_id": _object_id(space_id),
        "title": operation.title,
        "normalized_title": normalized_title,
        "content": operation.content,
        "status": "active",
        "source_chunk_ids": source_chunk_ids,
        "source_request_ids": window.get("request_ids", []),
        "source_window_ids": [window["window_id"]],
        "confidence": operation.confidence,
        "source": "transcript_analysis",
        "created_at": now,
        "updated_at": now,
    }
    await mongo_db[NOTES_COLLECTION].update_one(
        {
            "user_id": document["user_id"],
            "space_id": document["space_id"],
            "normalized_title": normalized_title,
        },
        {"$setOnInsert": document},
        upsert=True,
    )
    return "created"


async def _update_summary(
    *,
    user_id: str,
    space_id: str,
    window: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> None:
    if not output.summary_update.should_update:
        return
    now = _now()
    await mongo_db[SUMMARIES_COLLECTION].update_one(
        {"user_id": _object_id(user_id), "space_id": _object_id(space_id)},
        {
            "$set": {
                "running_summary": output.summary_update.updated_summary,
                "last_processed_at": now,
                "last_processed_chunk_id": window["chunk_ids"][-1] if window["chunk_ids"] else None,
                "updated_at": now,
            },
            "$setOnInsert": {
                "user_id": _object_id(user_id),
                "space_id": _object_id(space_id),
                "active_topics": [],
                "open_questions": [],
                "version": 1,
                "created_at": now,
            },
        },
        upsert=True,
    )


async def persist_analysis_output(
    *,
    user_id: str,
    space_id: str,
    window: dict[str, Any],
    context_package: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> dict[str, int]:
    existing_task_ids = {str(item["id"]) for item in context_package["existing_tasks"]}
    existing_note_ids = {str(item["id"]) for item in context_package["existing_notes"]}
    counts = {"tasks_created": 0, "tasks_updated": 0, "notes_created": 0, "notes_updated": 0}

    for operation in output.task_operations:
        result = await _apply_task_operation(
            user_id=user_id,
            space_id=space_id,
            window=window,
            operation=operation,
            existing_task_ids=existing_task_ids,
        )
        if result == "created":
            counts["tasks_created"] += 1
        elif result == "updated":
            counts["tasks_updated"] += 1

    for operation in output.note_operations:
        result = await _apply_note_operation(
            user_id=user_id,
            space_id=space_id,
            window=window,
            operation=operation,
            existing_note_ids=existing_note_ids,
        )
        if result == "created":
            counts["notes_created"] += 1
        elif result == "updated":
            counts["notes_updated"] += 1

    await _update_summary(user_id=user_id, space_id=space_id, window=window, output=output)
    return counts


def _has_persisted_effect(
    counts: dict[str, int],
    output: TranscriptAnalysisOutput,
) -> bool:
    return any(counts.values()) or output.summary_update.should_update


def _is_meaningful_window_text(text: str) -> bool:
    normalized = normalize_text(text)
    return len(normalized) >= 12


def _has_memory_operations(output: TranscriptAnalysisOutput) -> bool:
    return any(
        operation.operation != "no_change"
        for operation in [*output.task_operations, *output.note_operations]
    )


def _has_note_operations(output: TranscriptAnalysisOutput) -> bool:
    return any(
        operation.operation != "no_change"
        for operation in output.note_operations
    )


def _covered_source_chunk_ids(output: TranscriptAnalysisOutput) -> set[str]:
    covered: set[str] = set()
    for operation in [*output.task_operations, *output.note_operations]:
        if operation.operation == "no_change":
            continue
        covered.update(str(chunk_id) for chunk_id in operation.source_chunk_ids if chunk_id)
    return covered


def _coverage_ratio(*, window: dict[str, Any], output: TranscriptAnalysisOutput) -> float:
    chunk_ids = {str(chunk_id) for chunk_id in window.get("chunk_ids", []) if chunk_id}
    if not chunk_ids:
        return 1.0
    return len(chunk_ids & _covered_source_chunk_ids(output)) / len(chunk_ids)


def _needs_coverage_repair(*, window: dict[str, Any], output: TranscriptAnalysisOutput) -> bool:
    chunk_count = len(window.get("chunk_ids", []))
    if chunk_count <= 1 or not _has_memory_operations(output):
        return False
    if output.requires_more_context or not output.is_complete_enough:
        return True
    if _coverage_ratio(window=window, output=output) < 0.75:
        return True
    operation_count = sum(
        1
        for operation in [*output.task_operations, *output.note_operations]
        if operation.operation != "no_change"
    )
    if chunk_count >= 4 and operation_count <= 1:
        return True
    return False


_VAGUE_OPERATION_PHRASES = (
    "all features are functioning",
    "complete the testing",
    "current project list",
    "data handling processes",
    "enhancing the user experience",
    "existing application framework",
    "finalize the testing",
    "general project tasks",
    "improving the existing",
    "integrating new features",
    "integration of new features",
    "new features into the existing",
    "ongoing testing",
    "project is progressing",
)

_FUTURE_WORK_CUE_PATTERN = re.compile(
    r"("
    r"\b(next|then|after that|afterwards|later|first|start|begin|fix|test|qa|deploy|release|send|merge|add|connect|integrat\w*|optimi[sz]\w*|handoff|review|bug|performance|notification|dashboard|frontend|staging|client build)\b"
    r"|अब|अगला|उसके बाद|पहले|फिर|अगर|तो|करेंगे|करूँगा|करूंगा|कर लेते|कर दूँगा|जोड़|टेस्ट|डिप्लॉय|भेज|ठीक|मर्ज|शुरू|बेहतर|तेज़|ऑप्टिमाइज़|इंटीग्रेशन|नोटिफिकेशन|रिलीज़|बचे"
    r")",
    re.IGNORECASE,
)

_COMPLETED_ONLY_PATTERN = re.compile(
    r"(complete(d)?|done|finished|पूरा हो गया|पूरा कर लिया|हो गई है|तैयार है|लगभग तैयार|लगभग पूरा)",
    re.IGNORECASE,
)


def _estimate_future_work_count(text: str) -> int:
    """Estimate distinct future-work clauses without relying on one language."""
    clauses = [
        clause.strip()
        for clause in re.split(
            r"[.!?\n।]+|,|;|\bthen\b|\bafter that\b|उसके बाद|पहले|फिर",
            text,
            flags=re.IGNORECASE,
        )
        if clause.strip()
    ]
    count = 0
    for clause in clauses:
        has_future_cue = bool(_FUTURE_WORK_CUE_PATTERN.search(clause))
        completed_only = bool(_COMPLETED_ONLY_PATTERN.search(clause)) and not re.search(
            r"\b(next|then|after|later|start|begin|fix|test|qa|deploy|release|send|merge|add|connect|integrat\w*|optimi[sz]\w*)\b|अब|अगला|उसके बाद|पहले|फिर|लेकिन|बस|बचे|करेंगे|करूँगा|करूंगा|जोड़|टेस्ट|डिप्लॉय|भेज|ठीक|मर्ज|शुरू",
            clause,
            flags=re.IGNORECASE,
        )
        if has_future_cue and not completed_only:
            count += 1
    return count


def _looks_vague_operation_text(text: str) -> bool:
    normalized = _normalized_for_comparison(text)
    return any(phrase in normalized for phrase in _VAGUE_OPERATION_PHRASES)


def _needs_detail_repair(*, window: dict[str, Any], output: TranscriptAnalysisOutput) -> bool:
    """Detect outputs that are technically non-empty but too broad to be useful."""
    if not _has_memory_operations(output):
        return False

    combined_text = str(window.get("combined_text") or "")
    future_work_count = _estimate_future_work_count(combined_text)
    concrete_task_count = sum(
        1
        for operation in output.task_operations
        if operation.operation != "no_change" and operation.title.strip()
    )
    if future_work_count >= 4 and concrete_task_count < min(future_work_count, 5):
        return True

    operation_texts = [
        f"{operation.title} {operation.description}"
        for operation in output.task_operations
        if operation.operation != "no_change"
    ]
    operation_texts.extend(
        f"{operation.title} {operation.content}"
        for operation in output.note_operations
        if operation.operation != "no_change"
    )
    return future_work_count >= 2 and any(
        _looks_vague_operation_text(text) for text in operation_texts
    )


def _operation_summary_line(operation: TaskOperation | NoteOperation, *, label: str) -> str:
    title = operation.title.strip()
    if isinstance(operation, TaskOperation):
        detail = operation.description.strip()
        if operation.due_at:
            detail = f"{detail} Due: {operation.due_at}".strip()
    else:
        detail = operation.content.strip().replace("\n", " ")

    text = ": ".join(part for part in [title, detail] if part)
    return f"{label}: {text}" if text else ""


def _normalized_for_comparison(value: str) -> str:
    return re.sub(r"\s+", " ", value).casefold().strip()


def _ensure_meaningful_summary(output: TranscriptAnalysisOutput) -> TranscriptAnalysisOutput:
    """Keep the running summary useful even when the model omits it."""
    if not _has_memory_operations(output):
        return output

    existing_summary = output.summary_update.updated_summary.strip()
    note_contents = [
        _normalized_for_comparison(operation.content)
        for operation in output.note_operations
        if operation.content
    ]
    summary_duplicates_note = (
        existing_summary
        and _normalized_for_comparison(existing_summary) in note_contents
    )
    if output.summary_update.should_update and existing_summary and not summary_duplicates_note:
        return output

    lines = [
        _operation_summary_line(operation, label="Task")
        for operation in output.task_operations
        if operation.operation != "no_change"
    ]
    lines.extend(
        _operation_summary_line(operation, label="Note")
        for operation in output.note_operations
        if operation.operation != "no_change"
    )
    summary = " ".join(line for line in lines if line).strip()
    if not summary:
        return output

    output.summary_update.should_update = True
    output.summary_update.updated_summary = summary[:6000]
    return output


def _title_from_summary(summary: str) -> str:
    """Create a concise note title from an English synthesized summary."""
    normalized = re.sub(r"\s+", " ", summary).strip()
    if not normalized:
        return "Conversation memory"

    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    title = re.sub(r"^(Task|Note):\s*", "", first_sentence).strip(" .:-")
    if len(title) <= 80:
        return title or "Conversation memory"

    truncated = title[:80].rsplit(" ", 1)[0].strip(" .:-")
    return truncated or "Conversation memory"


def _promote_summary_to_note_if_needed(
    *,
    window: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> TranscriptAnalysisOutput:
    """Preserve durable analysis in the user-visible notes collection."""
    if _has_note_operations(output) or not output.summary_update.should_update:
        return output

    summary = output.summary_update.updated_summary.strip()
    if not _is_meaningful_window_text(summary):
        return output
    covered_chunk_id_set = _covered_source_chunk_ids(output)
    covered_chunk_ids = [
        str(chunk_id)
        for chunk_id in window["chunk_ids"]
        if str(chunk_id) in covered_chunk_id_set
    ]

    output.note_operations.append(
        NoteOperation(
            operation="create",
            title=_title_from_summary(summary),
            content=summary,
            confidence=0.75,
            source_chunk_ids=covered_chunk_ids or window["chunk_ids"],
        )
    )
    return output



async def _repair_empty_analysis_output_if_needed(
    *,
    window: dict[str, Any],
    context_package: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> TranscriptAnalysisOutput:
    """Ask the model for a focused English synthesis when the first pass is empty."""
    if _has_memory_operations(output):
        return output
    if not output.is_complete_enough and output.requires_more_context:
        return output
    combined_text = str(window.get("combined_text") or "").strip()
    if not _is_meaningful_window_text(combined_text):
        return output

    try:
        repaired = await parse_chat_completion(
            client,
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_model=TranscriptAnalysisOutput,
            messages=TRANSCRIPT_ANALYSIS_REPAIR_PROMPT.format_messages(
                current_datetime=context_package["current_datetime"],
                timezone=context_package["timezone"],
                context_package=json.dumps(context_package, ensure_ascii=True, default=str),
            ),
        )
    except Exception:
        logger.exception(
            "Transcript analysis repair pass failed.",
            extra={
                "user_id": context_package.get("user_id"),
                "space_id": context_package.get("space_id"),
                "window_id": window.get("window_id"),
            },
        )
        return output

    if _has_memory_operations(repaired) or repaired.summary_update.should_update:
        return _ensure_meaningful_summary(repaired)
    return output


def _normalize_generated_operations(
    *,
    window: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> TranscriptAnalysisOutput:
    """Fill persistence-required metadata that the model may omit."""
    fallback_source_ids = [str(chunk_id) for chunk_id in window.get("chunk_ids", [])]

    for operation in output.task_operations:
        if operation.operation == "update" and not operation.existing_task_id and operation.title.strip():
            operation.operation = "create"
        if operation.operation == "create" and operation.title.strip():
            if not operation.source_chunk_ids and len(fallback_source_ids) == 1:
                operation.source_chunk_ids = fallback_source_ids
            if operation.confidence < 0.7:
                operation.confidence = 0.75

    for operation in output.note_operations:
        if operation.operation == "update" and not operation.existing_note_id and operation.title.strip():
            operation.operation = "create"
        if operation.operation == "create" and operation.title.strip() and operation.content.strip():
            if not operation.source_chunk_ids and len(fallback_source_ids) == 1:
                operation.source_chunk_ids = fallback_source_ids
            if operation.confidence < 0.7:
                operation.confidence = 0.75

    return output


async def _repair_missing_tasks_if_needed(
    *,
    window: dict[str, Any],
    context_package: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> TranscriptAnalysisOutput:
    """Run a focused second pass when the first analysis missed actionable work."""
    if any(operation.operation != "no_change" for operation in output.task_operations):
        return output

    combined_text = str(window.get("combined_text") or "").strip()
    if not _is_meaningful_window_text(combined_text):
        return output

    try:
        repaired = await parse_chat_completion(
            client,
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_model=TranscriptAnalysisOutput,
            messages=TRANSCRIPT_ANALYSIS_TASK_REPAIR_PROMPT.format_messages(
                current_datetime=context_package["current_datetime"],
                timezone=context_package["timezone"],
                context_package=json.dumps(context_package, ensure_ascii=True, default=str),
            ),
        )
    except Exception:
        logger.exception(
            "Transcript analysis task repair pass failed.",
            extra={
                "user_id": context_package.get("user_id"),
                "space_id": context_package.get("space_id"),
                "window_id": window.get("window_id"),
            },
        )
        return output

    repaired_tasks = [
        operation
        for operation in repaired.task_operations
        if operation.operation != "no_change"
    ]
    if not repaired_tasks:
        return output

    output.task_operations = repaired_tasks
    if not _has_note_operations(output):
        output.note_operations = [
            operation
            for operation in repaired.note_operations
            if operation.operation != "no_change"
        ]
    if repaired.summary_update.should_update and repaired.summary_update.updated_summary.strip():
        output.summary_update = repaired.summary_update

    return _ensure_meaningful_summary(output)


async def _repair_incomplete_coverage_if_needed(
    *,
    window: dict[str, Any],
    context_package: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> TranscriptAnalysisOutput:
    """Run a complete second pass when useful parts of the window look uncovered."""
    if not _needs_coverage_repair(window=window, output=output):
        return output

    combined_text = str(window.get("combined_text") or "").strip()
    if not _is_meaningful_window_text(combined_text):
        return output

    try:
        repaired = await parse_chat_completion(
            client,
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_model=TranscriptAnalysisOutput,
            messages=TRANSCRIPT_ANALYSIS_COVERAGE_REPAIR_PROMPT.format_messages(
                current_datetime=context_package["current_datetime"],
                timezone=context_package["timezone"],
                context_package=json.dumps(context_package, ensure_ascii=True, default=str),
            ),
        )
    except Exception:
        logger.exception(
            "Transcript analysis coverage repair pass failed.",
            extra={
                "user_id": context_package.get("user_id"),
                "space_id": context_package.get("space_id"),
                "window_id": window.get("window_id"),
            },
        )
        return output

    if _has_memory_operations(repaired) or repaired.summary_update.should_update:
        return _ensure_meaningful_summary(repaired)
    return output


async def _repair_low_detail_output_if_needed(
    *,
    window: dict[str, Any],
    context_package: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> TranscriptAnalysisOutput:
    """Run a complete second pass when output collapses detailed speech into vague tasks."""
    if not _needs_detail_repair(window=window, output=output):
        return output

    combined_text = str(window.get("combined_text") or "").strip()
    if not _is_meaningful_window_text(combined_text):
        return output

    try:
        repaired = await parse_chat_completion(
            client,
            model=settings.OPENAI_CHAT_MODEL,
            temperature=0,
            response_model=TranscriptAnalysisOutput,
            messages=TRANSCRIPT_ANALYSIS_COVERAGE_REPAIR_PROMPT.format_messages(
                current_datetime=context_package["current_datetime"],
                timezone=context_package["timezone"],
                context_package=json.dumps(context_package, ensure_ascii=True, default=str),
            ),
        )
    except Exception:
        logger.exception(
            "Transcript analysis detail repair pass failed.",
            extra={
                "user_id": context_package.get("user_id"),
                "space_id": context_package.get("space_id"),
                "window_id": window.get("window_id"),
            },
        )
        return output

    if _has_memory_operations(repaired) or repaired.summary_update.should_update:
        return _ensure_meaningful_summary(repaired)
    return output


def _published_point_ids_for_output(
    *,
    vectors: list[MemoryVector],
    window: dict[str, Any],
    output: TranscriptAnalysisOutput,
) -> list[str]:
    """Choose which vector points are safe to publish after persistence."""
    coverage_ratio = _coverage_ratio(window=window, output=output)
    if (
        output.is_complete_enough
        and not output.requires_more_context
        and coverage_ratio >= 0.75
    ):
        return [vector.point_id for vector in vectors]

    covered_chunk_ids = _covered_source_chunk_ids(output)
    if not covered_chunk_ids:
        return []

    point_ids_by_chunk_id = {vector.chunk_id: vector.point_id for vector in vectors}
    return [
        point_ids_by_chunk_id[chunk_id]
        for chunk_id in window.get("chunk_ids", [])
        if chunk_id in covered_chunk_ids and chunk_id in point_ids_by_chunk_id
    ]


async def process_transcript_analysis_job(job: AnalysisJob) -> dict[str, Any]:
    """Process one user-space transcript analysis job idempotently."""
    start = time.perf_counter()
    await ensure_memory_collections()

    query_user_id = job.user_id
    query_space_id = job.space_id
    user_id = job.user_id.strip()
    space_id = job.space_id.strip()

    vectors, chunks = await _load_unpublished_window(job)
    if not chunks:
        return {"status": "skipped", "reason": "no_unpublished_chunks"}

    window = _build_analysis_window(
        user_id=user_id,
        space_id=space_id,
        chunks=chunks,
    )
    context_package = await build_context_package(
        user_id=query_user_id,
        space_id=query_space_id,
        window=window,
    )
    context_package["user_id"] = user_id
    context_package["space_id"] = space_id
    output = await analyze_context_package(context_package)
    output = await _repair_empty_analysis_output_if_needed(
        window=window,
        context_package=context_package,
        output=output,
    )
    output = await _repair_missing_tasks_if_needed(
        window=window,
        context_package=context_package,
        output=output,
    )
    output = await _repair_incomplete_coverage_if_needed(
        window=window,
        context_package=context_package,
        output=output,
    )
    output = await _repair_low_detail_output_if_needed(
        window=window,
        context_package=context_package,
        output=output,
    )
    output = _normalize_generated_operations(window=window, output=output)
    output = _ensure_meaningful_summary(output)
    output = _promote_summary_to_note_if_needed(window=window, output=output)

    if output.requires_more_context and not output.task_operations and not output.note_operations:
        return {"status": "waiting_for_context", "window_id": window["window_id"]}

    counts = await persist_analysis_output(
        user_id=user_id,
        space_id=space_id,
        window=window,
        context_package=context_package,
        output=output,
    )

    if not _has_persisted_effect(counts, output):
        logger.info(
            "Transcript analysis produced no Mongo operations; chunks left unpublished.",
            extra={
                "user_id": job.user_id,
                "space_id": job.space_id,
                "window_id": window["window_id"],
                "chunk_ids": window["chunk_ids"],
                "attempt": job.attempt,
            },
        )
        return {"status": "review_needed", "window_id": window["window_id"], **counts}

    point_ids_to_publish = _published_point_ids_for_output(
        vectors=vectors,
        window=window,
        output=output,
    )
    if not point_ids_to_publish:
        logger.info(
            "Transcript analysis saved Mongo operations but left chunks unpublished for more context.",
            extra={
                "user_id": job.user_id,
                "space_id": job.space_id,
                "window_id": window["window_id"],
                "chunk_ids": window["chunk_ids"],
                "attempt": job.attempt,
                "operation_counts": counts,
            },
        )
        return {"status": "partial_saved_waiting_for_context", "window_id": window["window_id"], **counts}

    completed_at = _now_iso()
    await mark_vectors_analysis_completed(
        point_ids_to_publish,
        window_id=window["window_id"],
        published_at=completed_at,
    )
    all_point_ids = {vector.point_id for vector in vectors}
    published_all_points = all_point_ids.issubset(set(point_ids_to_publish))
    status = "completed" if published_all_points else "partial_completed"

    logger.info(
        "Transcript analysis completed.",
        extra={
            "user_id": job.user_id,
            "space_id": job.space_id,
            "window_id": window["window_id"],
            "chunk_ids": window["chunk_ids"],
            "attempt": job.attempt,
            "processing_duration": round(time.perf_counter() - start, 4),
            "operation_counts": counts,
            "published_chunks": len(point_ids_to_publish),
            "total_window_chunks": len(vectors),
        },
    )
    return {
        "status": status,
        "window_id": window["window_id"],
        "published_chunks": len(point_ids_to_publish),
        "total_window_chunks": len(vectors),
        **counts,
    }
