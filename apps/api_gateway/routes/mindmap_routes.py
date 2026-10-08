from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.db.mongo import get_database
from services.mindmap.enqueue import enqueue_mindmap_generation
from services.mindmap.store import MindmapStore, public_mindmap_doc, public_mindmap_summary

router = APIRouter()


class GenerateMindmapBody(BaseModel):
    userId: str = Field(min_length=1)
    spaceId: str = Field(min_length=1)


@router.post("/generate")
async def generate_mindmap(body: GenerateMindmapBody):
    try:
        result = await enqueue_mindmap_generation(
            get_database(),
            user_id=body.userId.strip(),
            space_id=body.spaceId.strip(),
        )
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
    user_id: str = Query(..., alias="userId", min_length=1),
    space_id: str = Query(..., alias="spaceId", min_length=1),
):
    store = MindmapStore(get_database())
    document = await store.get_latest_ready(user_id.strip(), space_id.strip())
    if document is None:
        raise HTTPException(status_code=404, detail="Mind map not found for this space.")
    return {"success": True, "mindmap": public_mindmap_doc(document)}


@router.get("/list")
async def list_mindmaps(
    user_id: str = Query(..., alias="userId", min_length=1),
    space_id: str = Query(..., alias="spaceId", min_length=1),
    limit: int = Query(24, ge=1, le=50),
):
    store = MindmapStore(get_database())
    documents = await store.list_for_space(
        user_id.strip(),
        space_id.strip(),
        limit=limit,
    )
    return {
        "success": True,
        "mindmaps": [public_mindmap_summary(doc) for doc in documents],
    }


@router.get("/{mindmap_id}")
async def get_mindmap_by_id(mindmap_id: str):
    store = MindmapStore(get_database())
    document = await store.get(mindmap_id.strip())
    if document is None:
        raise HTTPException(status_code=404, detail="Mind map not found.")
    return {"success": True, "mindmap": public_mindmap_doc(document)}


@router.delete("/{mindmap_id}/nodes/{node_id}")
async def remove_mindmap_node(mindmap_id: str, node_id: str):
    store = MindmapStore(get_database())
    try:
        document = await store.remove_node(mindmap_id.strip(), node_id.strip())
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
