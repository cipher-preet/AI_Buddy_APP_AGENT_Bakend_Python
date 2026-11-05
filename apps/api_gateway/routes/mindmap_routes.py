from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api_gateway.auth import raise_for_access, require_matching_user, require_user
from services.auth.identity import AuthenticatedUser
from services.auth.space_access import AccessDenied, require_owned_resource, require_space_operation
from services.mindmap.source_scope import parse_source_scope

from services.db.mongo import get_database
from services.mindmap.enqueue import enqueue_mindmap_generation
from services.mindmap.store import (
    MindmapStore,
    public_mindmap_card,
    public_mindmap_doc,
    public_mindmap_summary,
)
from services.plans.plan_limits import PlanLimitExceeded

router = APIRouter(dependencies=[Depends(require_user)])


class GenerateMindmapBody(BaseModel):
    userId: str = Field(min_length=1)
    spaceId: str = Field(min_length=1)
    scope: str = "space"
    date: str | None = None
    dates: list[str] | None = None
    timeZone: str | None = None


@router.post("/generate")
async def generate_mindmap(
    body: GenerateMindmapBody,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = require_matching_user(actor, body.userId)
    try:
        await require_space_operation(get_database(), user_id, body.spaceId.strip(), "write")
        result = await enqueue_mindmap_generation(
            get_database(),
            user_id=user_id,
            space_id=body.spaceId.strip(),
            scope=parse_source_scope(body.scope, body.date, body.timeZone, dates=body.dates),
        )
    except AccessDenied as error:
        raise_for_access(error)
    except PlanLimitExceeded as error:
        raise HTTPException(
            status_code=403,
            detail={
                "message": str(error),
                "resource": error.resource,
                "limit": error.limit,
                "used": error.used,
                "planCode": error.plan_code,
            },
        ) from error
    except PermissionError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to enqueue mind map generation: {type(error).__name__}",
        ) from error
    return result


@router.get("")
async def get_latest_mindmap(
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Query(..., alias="userId", min_length=1),
    space_id: str = Query(..., alias="spaceId", min_length=1),
):
    verified_user_id = require_matching_user(actor, user_id)
    try:
        access = await require_space_operation(get_database(), verified_user_id, space_id.strip(), "read")
    except AccessDenied as error:
        raise_for_access(error)
    store = MindmapStore(get_database())
    document = await store.get_latest_ready(access.owner_user_id, space_id.strip())
    if document is None:
        raise HTTPException(status_code=404, detail="Mind map not found for this space.")
    return {"success": True, "mindmap": public_mindmap_doc(document)}


@router.get("/list")
async def list_mindmaps(
    actor: AuthenticatedUser = Depends(require_user),
    user_id: str = Query(..., alias="userId", min_length=1),
    space_id: str = Query(..., alias="spaceId", min_length=1),
    limit: int = Query(24, ge=1, le=50),
    view: str | None = Query(default=None),
):
    verified_user_id = require_matching_user(actor, user_id)
    try:
        access = await require_space_operation(get_database(), verified_user_id, space_id.strip(), "read")
    except AccessDenied as error:
        raise_for_access(error)
    card = view == "card"
    store = MindmapStore(get_database())
    documents = await store.list_for_space(
        access.owner_user_id,
        space_id.strip(),
        limit=limit,
        card=card,
    )
    present = public_mindmap_card if card else public_mindmap_summary
    return {
        "success": True,
        "mindmaps": [present(doc) for doc in documents],
    }


@router.get("/{mindmap_id}")
async def get_mindmap_by_id(
    mindmap_id: str,
    actor: AuthenticatedUser = Depends(require_user),
):
    store = MindmapStore(get_database())
    document = await store.get(mindmap_id.strip())
    if document is None:
        raise HTTPException(status_code=404, detail="Mind map not found.")
    try:
        await require_owned_resource(
            get_database(),
            actor.user_id,
            document,
            operation="read",
            missing_detail="Mind map not found.",
        )
    except AccessDenied as error:
        raise_for_access(error)
    return {"success": True, "mindmap": public_mindmap_doc(document)}


@router.delete("/{mindmap_id}/nodes/{node_id}")
async def remove_mindmap_node(
    mindmap_id: str,
    node_id: str,
    actor: AuthenticatedUser = Depends(require_user),
):
    store = MindmapStore(get_database())
    current = await store.get(mindmap_id.strip())
    if current is None:
        raise HTTPException(status_code=404, detail="Mind map not found.")
    try:
        await require_owned_resource(
            get_database(),
            actor.user_id,
            current,
            operation="write",
            missing_detail="Mind map not found.",
        )
        document = await store.remove_node(mindmap_id.strip(), node_id.strip())
    except AccessDenied as error:
        raise_for_access(error)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to update mind map: {type(error).__name__}",
        ) from error

    if document is None:
        raise HTTPException(status_code=404, detail="Mind map not found.")
    return {"success": True, "mindmap": public_mindmap_doc(document)}
