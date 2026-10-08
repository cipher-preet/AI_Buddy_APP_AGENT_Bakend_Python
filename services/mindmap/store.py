from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ReturnDocument

from services.conversation.models import utc_now
from services.conversation.repository import mongo_id_candidates
from services.mindmap.schemas import (
    ACTIVE_STATUSES,
    PIPELINE_VERSION,
    MindmapGraph,
    MindmapStatus,
    SourceStats,
)

CLAIM_STALE = timedelta(minutes=20)
MAX_VERSIONS_PER_SPACE = 5
PUBLIC_OMIT = {"claimedBy": 0}


def _as_utc(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def is_stale_active(doc: dict[str, Any] | None, now: datetime | None = None) -> bool:
    if not doc or doc.get("status") not in ACTIVE_STATUSES:
        return False
    stamp = doc.get("claimedAt") or doc.get("updatedAt") or doc.get("createdAt")
    parsed = _as_utc(stamp)
    if parsed is None:
        return True
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return parsed <= current - CLAIM_STALE


class MindmapStore:
    def __init__(self, database):
        self.db = database
        self.collection = database.spaceMindmaps

    async def find_active(self, user_id: str, space_id: str) -> dict[str, Any] | None:
        docs = await (
            self.collection.find(
                {
                    "userId": {"$in": mongo_id_candidates(user_id)},
                    "spaceId": {"$in": mongo_id_candidates(space_id)},
                    "status": {"$in": list(ACTIVE_STATUSES)},
                }
            )
            .sort([("updatedAt", -1)])
            .limit(5)
            .to_list(length=5)
        )
        for doc in docs:
            if not is_stale_active(doc):
                return doc
        return None

    async def get(self, mindmap_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {"_id": {"$in": mongo_id_candidates(mindmap_id)}},
            PUBLIC_OMIT,
        )

    async def get_by_job(self, job_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"jobId": str(job_id)}, PUBLIC_OMIT)

    async def get_latest_ready(self, user_id: str, space_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one(
            {
                "userId": {"$in": mongo_id_candidates(user_id)},
                "spaceId": {"$in": mongo_id_candidates(space_id)},
                "status": MindmapStatus.READY.value,
            },
            PUBLIC_OMIT,
            sort=[("version", -1), ("updatedAt", -1)],
        )

    async def next_version(self, user_id: str, space_id: str) -> int:
        latest = await self.collection.find_one(
            {
                "userId": {"$in": mongo_id_candidates(user_id)},
                "spaceId": {"$in": mongo_id_candidates(space_id)},
            },
            {"version": 1},
            sort=[("version", -1)],
        )
        return int((latest or {}).get("version") or 0) + 1

    async def reserve(
        self,
        *,
        user_id: str,
        space_id: str,
        job_id: str,
    ) -> dict[str, Any]:
        now = utc_now()
        mindmap_id = str(uuid4())
        version = await self.next_version(user_id, space_id)
        doc = {
            "_id": mindmap_id,
            "userId": str(user_id),
            "spaceId": str(space_id),
            "jobId": str(job_id),
            "status": MindmapStatus.QUEUED.value,
            "stage": "queued",
            "progress": 5,
            "message": "Queued for generation",
            "error": None,
            "graph": {"nodes": [], "edges": []},
            "sourceStats": SourceStats().model_dump(),
            "model": None,
            "pipelineVersion": PIPELINE_VERSION,
            "version": version,
            "claimedBy": None,
            "claimedAt": None,
            "createdAt": now,
            "updatedAt": now,
        }
        await self.collection.insert_one(doc)
        await self._prune_old_versions(user_id, space_id)
        return doc

    async def update_progress(
        self,
        mindmap_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: int | None = None,
        message: str | None = None,
        error: str | None = None,
        source_stats: SourceStats | dict[str, Any] | None = None,
        model: str | None = None,
        claimed_by: str | None = None,
    ) -> dict[str, Any] | None:
        updates: dict[str, Any] = {"updatedAt": utc_now()}
        if status is not None:
            updates["status"] = status
        if stage is not None:
            updates["stage"] = stage
        if progress is not None:
            updates["progress"] = max(0, min(100, int(progress)))
        if message is not None:
            updates["message"] = message
        if error is not None:
            updates["error"] = error
        if source_stats is not None:
            updates["sourceStats"] = (
                source_stats.model_dump() if isinstance(source_stats, SourceStats) else source_stats
            )
        if model is not None:
            updates["model"] = model
        if claimed_by is not None:
            updates["claimedBy"] = claimed_by
            updates["claimedAt"] = utc_now()
        return await self.collection.find_one_and_update(
            {"_id": {"$in": mongo_id_candidates(mindmap_id)}},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )

    async def claim(self, mindmap_id: str, job_id: str) -> tuple[str, dict[str, Any] | None]:
        existing = await self.get(mindmap_id)
        if existing is None:
            return "missing", None
        if existing.get("status") == MindmapStatus.READY.value:
            return "exists", existing
        if (
            existing.get("status") in ACTIVE_STATUSES
            and existing.get("claimedBy")
            and existing.get("claimedBy") != job_id
            and not is_stale_active(existing)
            and existing.get("status") != MindmapStatus.QUEUED.value
        ):
            return "busy", existing

        updated = await self.collection.find_one_and_update(
            {
                "_id": {"$in": mongo_id_candidates(mindmap_id)},
                "status": {"$in": [MindmapStatus.QUEUED.value, MindmapStatus.FAILED.value, *ACTIVE_STATUSES]},
            },
            {
                "$set": {
                    "status": MindmapStatus.GATHERING.value,
                    "stage": "gathering_context",
                    "progress": 15,
                    "message": "Gathering notes, tasks, and transcripts",
                    "claimedBy": job_id,
                    "claimedAt": utc_now(),
                    "updatedAt": utc_now(),
                    "error": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            return "busy", existing
        return "claimed", updated

    async def save_ready(
        self,
        mindmap_id: str,
        *,
        graph: MindmapGraph,
        source_stats: SourceStats,
        model: str | None,
    ) -> dict[str, Any] | None:
        graph_payload = graph.model_dump()
        return await self.collection.find_one_and_update(
            {"_id": {"$in": mongo_id_candidates(mindmap_id)}},
            {
                "$set": {
                    "status": MindmapStatus.READY.value,
                    "stage": "completed",
                    "progress": 100,
                    "message": "Mind map ready",
                    "error": None,
                    "graph": graph_payload,
                    "preview": build_mindmap_preview(graph_payload),
                    "sourceStats": source_stats.model_dump(),
                    "model": model,
                    "updatedAt": utc_now(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    async def list_for_space(
        self,
        user_id: str,
        space_id: str,
        *,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit), 50))
        docs = await (
            self.collection.find(
                {
                    "userId": {"$in": mongo_id_candidates(user_id)},
                    "spaceId": {"$in": mongo_id_candidates(space_id)},
                    "status": MindmapStatus.READY.value,
                },
                {
                    "claimedBy": 0,
                    "claimedAt": 0,
                    "graph": 0,
                },
            )
            .sort([("version", -1), ("updatedAt", -1)])
            .limit(capped)
            .to_list(length=capped)
        )
        def needs_preview_refresh(doc: dict[str, Any]) -> bool:
            preview = doc.get("preview")
            if not isinstance(preview, dict):
                return True
            nodes = preview.get("nodes")
            return not (isinstance(nodes, list) and len(nodes) > 0)

        stale = [doc for doc in docs if needs_preview_refresh(doc)]
        for doc in stale:
            full = await self.collection.find_one({"_id": doc["_id"]}, {"graph": 1})
            preview = build_mindmap_preview((full or {}).get("graph"))
            doc["preview"] = preview
            if preview:
                await self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"preview": preview}},
                )
        return docs

    async def remove_node(self, mindmap_id: str, node_id: str) -> dict[str, Any] | None:
        document = await self.get(mindmap_id)
        if document is None:
            return None

        graph = document.get("graph") if isinstance(document.get("graph"), dict) else {}
        nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
        edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
        target = str(node_id).strip()
        if not target:
            raise ValueError("nodeId is required")

        next_nodes = [
            node
            for node in nodes
            if not (isinstance(node, dict) and str(node.get("id") or "") == target)
        ]
        if len(next_nodes) == len(nodes):
            raise ValueError("Node not found on this mind map.")

        next_edges = [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and str(edge.get("source") or "") != target
            and str(edge.get("target") or "") != target
        ]
        next_graph = {"nodes": next_nodes, "edges": next_edges}
        preview = build_mindmap_preview(next_graph)

        return await self.collection.find_one_and_update(
            {"_id": {"$in": mongo_id_candidates(mindmap_id)}},
            {
                "$set": {
                    "graph": next_graph,
                    "preview": preview,
                    "updatedAt": utc_now(),
                    "message": "Mind map updated",
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    async def mark_failed(self, mindmap_id: str, error: str) -> dict[str, Any] | None:
        return await self.collection.find_one_and_update(
            {"_id": {"$in": mongo_id_candidates(mindmap_id)}},
            {
                "$set": {
                    "status": MindmapStatus.FAILED.value,
                    "stage": "failed",
                    "progress": 100,
                    "message": "Generation failed",
                    "error": str(error)[:500],
                    "updatedAt": utc_now(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    async def _prune_old_versions(self, user_id: str, space_id: str) -> None:
        docs = await (
            self.collection.find(
                {
                    "userId": {"$in": mongo_id_candidates(user_id)},
                    "spaceId": {"$in": mongo_id_candidates(space_id)},
                    "status": {"$in": [MindmapStatus.READY.value, MindmapStatus.FAILED.value]},
                },
                {"_id": 1, "version": 1},
            )
            .sort([("version", -1)])
            .to_list(length=50)
        )
        if len(docs) <= MAX_VERSIONS_PER_SPACE:
            return
        drop_ids = [doc["_id"] for doc in docs[MAX_VERSIONS_PER_SPACE:]]
        if drop_ids:
            await self.collection.delete_many({"_id": {"$in": drop_ids}})


def _clip_note(value: Any, *, max_len: int = 72) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}…"


def _preview_node_from_graph_node(node: dict[str, Any]) -> dict[str, Any] | None:
    data = node.get("data") if isinstance(node.get("data"), dict) else {}
    kind = str(data.get("kind") or "").strip()
    if kind not in {"hub", "branch", "card"}:
        return None

    title = _clip_note(data.get("title"), max_len=42) or "Untitled"
    tone = data.get("tone")
    subtitle = _clip_note(data.get("subtitle"), max_len=56) if kind == "hub" else None

    items: list[str] = []
    if kind == "card":
        raw_items = data.get("items") if isinstance(data.get("items"), list) else []
        for item in raw_items:
            clipped = _clip_note(item, max_len=48)
            if clipped:
                items.append(clipped)
            if len(items) >= 2:
                break
        if not items:
            priority_sections = (
                data.get("prioritySections")
                if isinstance(data.get("prioritySections"), list)
                else []
            )
            for section in priority_sections:
                if not isinstance(section, dict):
                    continue
                section_items = (
                    section.get("items") if isinstance(section.get("items"), list) else []
                )
                for item in section_items:
                    clipped = _clip_note(item, max_len=48)
                    if clipped:
                        items.append(clipped)
                    if len(items) >= 2:
                        break
                if len(items) >= 2:
                    break

    payload: dict[str, Any] = {
        "id": str(node.get("id") or f"{kind}-{title}"),
        "kind": kind,
        "title": title,
    }
    if tone:
        payload["tone"] = str(tone)
    if subtitle:
        payload["subtitle"] = subtitle
    if items:
        payload["items"] = items
    variant = data.get("variant")
    if kind == "card" and variant:
        payload["variant"] = str(variant)
    return payload


def build_mindmap_preview(graph: dict[str, Any] | None) -> dict[str, Any]:
    payload = graph if isinstance(graph, dict) else {}
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []

    hub: dict[str, Any] | None = None
    branches: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue
        preview_node = _preview_node_from_graph_node(node)
        if preview_node is None:
            continue
        kind = preview_node["kind"]
        if kind == "hub" and hub is None:
            hub = preview_node
        elif kind == "branch" and len(branches) < 2:
            branches.append(preview_node)
        elif kind == "card" and len(cards) < 2:
            cards.append(preview_node)

    reference_nodes: list[dict[str, Any]] = []
    if hub:
        reference_nodes.append(hub)
    reference_nodes.extend(branches)
    reference_nodes.extend(cards)

    hub_title = (hub or {}).get("title") or "Mind map"
    subtitle = (hub or {}).get("subtitle")

    return {
        "title": hub_title,
        "subtitle": subtitle,
        "nodes": reference_nodes,
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
    }


def public_mindmap_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    payload = dict(doc)
    payload["mindmapId"] = str(payload.get("_id") or "")
    payload["_id"] = str(payload.get("_id") or "")
    payload.pop("claimedBy", None)
    return payload


def public_mindmap_summary(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    payload = public_mindmap_doc(doc) or {}
    payload.pop("graph", None)
    if not payload.get("preview"):
        payload["preview"] = build_mindmap_preview(None)
    return payload
