"""Space permissions mirrored from Node ``resolveSpaceAccess`` and ``decideSpacePermission``.

Read is owner or an active viewer. Write stays with the owner. Missing access does
not reveal whether the space exists. A share document cannot elevate a viewer to owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bson import ObjectId

from services.auth.identity import same_user_id
from services.conversation.repository import mongo_id_candidates


class AccessDenied(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class SpaceAccess:
    owner_user_id: str
    role: str


async def resolve_space_access(database: Any, viewer_id: str, space_id: str) -> SpaceAccess | None:
    if not ObjectId.is_valid(viewer_id) or not ObjectId.is_valid(space_id):
        return None
    owned = await database.spaces.find_one(
        {
            "_id": {"$in": mongo_id_candidates(space_id)},
            "userId": {"$in": mongo_id_candidates(viewer_id)},
            "deletedAt": None,
        },
        {"userId": 1},
    )
    if owned and owned.get("userId") is not None:
        return SpaceAccess(owner_user_id=str(owned.get("userId")), role="owner")

    share = await database.spaceshares.find_one(
        {
            "spaceId": {"$in": mongo_id_candidates(space_id)},
            "memberUserId": {"$in": mongo_id_candidates(viewer_id)},
            "status": "active",
        },
        {"ownerUserId": 1},
    )
    if not share or share.get("ownerUserId") is None:
        return None
    space = await database.spaces.find_one(
        {
            "_id": {"$in": mongo_id_candidates(space_id)},
            "deletedAt": None,
        },
        {"_id": 1},
    )
    if not space:
        return None
    return SpaceAccess(owner_user_id=str(share.get("ownerUserId")), role="viewer")


async def require_space_operation(
    database: Any,
    viewer_id: str,
    space_id: str,
    operation: str,
) -> SpaceAccess:
    normalized_space = str(space_id or "").strip()
    if not ObjectId.is_valid(str(viewer_id or "")) or not ObjectId.is_valid(normalized_space):
        raise AccessDenied(400, "Invalid user or space.")
    if operation not in {"read", "write"}:
        raise AccessDenied(403, "You do not have permission to modify this space.")
    access = await resolve_space_access(database, str(viewer_id), normalized_space)
    if access is None or access.role not in {"owner", "viewer"} or not access.owner_user_id:
        raise AccessDenied(404, "Space not found.")
    if operation == "write" and access.role != "owner":
        raise AccessDenied(403, "You do not have permission to modify this space.")
    return access


async def authorize_space_ids(
    database: Any,
    viewer_id: str,
    space_ids: list[str] | None,
    operation: str,
) -> None:
    seen: set[str] = set()
    for raw in space_ids or []:
        space_id = str(raw or "").strip()
        if not space_id or space_id in seen:
            continue
        seen.add(space_id)
        await require_space_operation(database, viewer_id, space_id, operation)


async def require_owned_resource(
    database: Any,
    viewer_id: str,
    resource: dict[str, Any],
    *,
    operation: str,
    missing_detail: str,
) -> SpaceAccess:
    """Authorize a stored document or mind map from its own owner and space, not the caller's claim."""
    space_id = str(resource.get("spaceId") or "").strip()
    resource_owner = str(resource.get("userId") or "").strip()
    if not space_id or not resource_owner:
        raise AccessDenied(404, missing_detail)
    try:
        access = await require_space_operation(database, viewer_id, space_id, operation)
    except AccessDenied as error:
        if error.status_code == 403 and operation == "write":
            raise
        raise AccessDenied(404, missing_detail) from error
    if not same_user_id(resource_owner, access.owner_user_id):
        raise AccessDenied(404, missing_detail)
    return access
