from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from services.conversation.repository import mongo_id_candidates

PlanCountableResource = Literal["mindmaps", "documents", "goals"]

UNLIMITED = -1

DEFAULT_FREE_LIMITS: dict[str, int] = {
    "mindmaps": 10,
    "documents": 10,
    "goals": 2,
    "notesRecordingHours": 2,
    "meetingRecordingHours": 10,
}

COLLECTION_BY_RESOURCE: dict[PlanCountableResource, tuple[str, str]] = {
    "mindmaps": ("spaceMindmaps", "userId"),
    "documents": ("spaceDocuments", "userId"),
    "goals": ("goals", "ownerId"),
}


class PlanLimitExceeded(PermissionError):
    def __init__(self, message: str, *, resource: str, limit: int, used: int, plan_code: str):
        super().__init__(message)
        self.resource = resource
        self.limit = limit
        self.used = used
        self.plan_code = plan_code


@dataclass
class PlanQuota:
    allowed: bool
    resource: str
    limit: int
    used: int
    plan_code: str
    message: str | None = None


def _as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _is_paid_plan_expired(subscription: dict[str, Any] | None) -> bool:
    if not subscription:
        return False
    plan_code = str(subscription.get("planCode") or "free")
    if plan_code == "free":
        return False
    status = str(subscription.get("status") or "active").lower()
    if status in {"expired", "cancelled"}:
        return True
    period_end = _as_utc(subscription.get("currentPeriodEnd"))
    if period_end is None:
        return False
    return period_end <= datetime.now(timezone.utc)


async def _resolve_plan_limits(database, user_id: str) -> tuple[str, dict[str, Any]]:
    user_ids = mongo_id_candidates(user_id)
    subscription = await database.usersubscriptions.find_one({"userId": {"$in": user_ids}})
    if subscription is None:
        # Mongoose lowercases/pluralizes; also try explicit collection names.
        subscription = await database["usersubscriptions"].find_one({"userId": {"$in": user_ids}})
    if subscription is None:
        subscription = await database["user_subscriptions"].find_one({"userId": {"$in": user_ids}})

    # Lapsed paid plans enforce Free limits until Node demotes the subscription.
    if _is_paid_plan_expired(subscription if isinstance(subscription, dict) else None):
        free_plan = await database.plans.find_one({"code": "free", "isActive": True})
        limits = dict(DEFAULT_FREE_LIMITS)
        if isinstance(free_plan, dict) and isinstance(free_plan.get("limits"), dict):
            limits.update(free_plan["limits"])
        return "free", limits

    plan_code = str((subscription or {}).get("planCode") or "free")
    plan = None
    plan_id = (subscription or {}).get("planId")
    if plan_id is not None:
        plan = await database.plans.find_one({"_id": plan_id})
    if plan is None:
        plan = await database.plans.find_one({"code": plan_code, "isActive": True})
    if plan is None:
        plan = await database.plans.find_one({"code": "free", "isActive": True})

    limits = dict(DEFAULT_FREE_LIMITS)
    if isinstance(plan, dict) and isinstance(plan.get("limits"), dict):
        limits.update(plan["limits"])
        plan_code = str(plan.get("code") or plan_code)
    return plan_code, limits


async def _count_resource(database, user_id: str, resource: PlanCountableResource) -> int:
    collection_name, id_field = COLLECTION_BY_RESOURCE[resource]
    query: dict[str, Any] = {id_field: {"$in": mongo_id_candidates(user_id)}}
    if resource != "goals":
        query["$or"] = [{"deletedAt": None}, {"deletedAt": {"$exists": False}}]
    return int(await database[collection_name].count_documents(query))


async def assert_plan_count_allowed(
    database,
    *,
    user_id: str,
    resource: PlanCountableResource,
    next_count: int = 1,
) -> PlanQuota:
    plan_code, limits = await _resolve_plan_limits(database, user_id)
    raw_limit = limits.get(resource, DEFAULT_FREE_LIMITS[resource])
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = DEFAULT_FREE_LIMITS[resource]

    used = await _count_resource(database, user_id, resource)
    if limit != UNLIMITED and used + max(0, next_count) > limit:
        message = f"Plan limit reached. Upgrade your plan to create more {resource}."
        raise PlanLimitExceeded(
            message,
            resource=resource,
            limit=limit,
            used=used,
            plan_code=plan_code,
        )

    return PlanQuota(
        allowed=True,
        resource=resource,
        limit=limit,
        used=used,
        plan_code=plan_code,
    )
