"""Move large document and mind map blobs out of the records list queries read.

List responses stay the same. The bytes and graphs are stored beside the record
and joined back when a single document or mind map is opened.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from typing import Any

_offload_task: asyncio.Task[None] | None = None


def schedule_heavy_payload_offload(database: Any) -> None:
    global _offload_task
    if _offload_task is not None and not _offload_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _offload_task = loop.create_task(_offload(database))


async def _offload(database: Any) -> None:
    try:
        documents = await offload_document_payloads(database)
        mindmaps = await offload_mindmap_graphs(database)
        if documents or mindmaps:
            print(
                f"Offloaded heavy payloads documents={documents} mindmaps={mindmaps}",
                flush=True,
            )
    except Exception as error:
        print(f"Heavy payload offload failed: {type(error).__name__}: {error}", flush=True)


async def _store_if_current(collection: Any, doc_id: Any, stamp: Any, payload: dict[str, Any]) -> None:
    """Write a side record without replacing one saved by a newer generation."""
    stored = dict(payload)
    stored["inlineUpdatedAt"] = stamp
    if isinstance(stamp, datetime):
        side_filter: dict[str, Any] = {
            "_id": doc_id,
            "$or": [
                {"inlineUpdatedAt": {"$exists": False}},
                {"inlineUpdatedAt": None},
                {"inlineUpdatedAt": {"$lte": stamp}},
            ],
        }
    else:
        side_filter = {"_id": doc_id, "inlineUpdatedAt": {"$exists": False}}
    result = await collection.update_one(side_filter, {"$set": stored})
    if int(getattr(result, "matched_count", 0) or 0) > 0:
        return
    try:
        await collection.insert_one({"_id": doc_id, **stored})
    except Exception as error:
        message = str(error)
        if "E11000" in message or "duplicate" in message.lower():
            return
        raise


async def offload_document_payloads(database: Any) -> int:
    moved = 0
    cursor = database.spaceDocuments.find(
        {
            "$or": [
                {"docxBase64": {"$type": "string"}},
                {"content": {"$type": "object"}},
            ]
        },
        {"docxBase64": 1, "content": 1, "docxBytes": 1, "updatedAt": 1},
    ).batch_size(2)
    async for doc in cursor:
        encoded = doc.get("docxBase64") if isinstance(doc.get("docxBase64"), str) else None
        content = doc.get("content") if isinstance(doc.get("content"), dict) else None
        if not encoded and content is None:
            if encoded == "":
                await database.spaceDocuments.update_one(
                    {"_id": doc["_id"], "docxBase64": ""},
                    {"$unset": {"docxBase64": ""}},
                )
            continue
        payload: dict[str, Any] = {}
        main_filter: dict[str, Any] = {"_id": doc["_id"]}
        unset: dict[str, str] = {}
        if encoded:
            payload["docxBase64"] = encoded
            main_filter["docxBase64"] = {"$type": "string"}
            unset["docxBase64"] = ""
        if content is not None:
            payload["content"] = content
            main_filter["content"] = {"$type": "object"}
            unset["content"] = ""
        if isinstance(doc.get("updatedAt"), datetime):
            main_filter["updatedAt"] = doc["updatedAt"]
        await _store_if_current(
            database.spaceDocumentFiles,
            doc["_id"],
            doc.get("updatedAt"),
            payload,
        )
        update: dict[str, Any] = {"$unset": unset}
        if encoded and not doc.get("docxBytes"):
            try:
                byte_len = len(base64.b64decode(encoded))
            except Exception:
                byte_len = len(encoded)
            update["$set"] = {"docxBytes": byte_len}
        result = await database.spaceDocuments.update_one(main_filter, update)
        if int(getattr(result, "modified_count", 0) or 0) > 0:
            moved += 1
    return moved


async def offload_mindmap_graphs(database: Any) -> int:
    moved = 0
    cursor = database.spaceMindmaps.find(
        {"graph": {"$type": "object"}},
        {"graph": 1, "updatedAt": 1},
    ).batch_size(2)
    async for doc in cursor:
        graph = doc.get("graph")
        if not isinstance(graph, dict):
            continue
        await _store_if_current(
            database.spaceMindmapGraphs,
            doc["_id"],
            doc.get("updatedAt"),
            {"graph": graph},
        )
        graph_filter: dict[str, Any] = {"_id": doc["_id"], "graph": {"$type": "object"}}
        if isinstance(doc.get("updatedAt"), datetime):
            graph_filter["updatedAt"] = doc["updatedAt"]
        result = await database.spaceMindmaps.update_one(
            graph_filter,
            {"$unset": {"graph": ""}},
        )
        if int(getattr(result, "modified_count", 0) or 0) > 0:
            moved += 1
    return moved
