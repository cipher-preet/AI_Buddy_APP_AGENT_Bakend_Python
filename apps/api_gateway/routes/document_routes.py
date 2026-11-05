from __future__ import annotations

import base64
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from apps.api_gateway.auth import raise_for_access, require_matching_user, require_user
from services.auth.identity import AuthenticatedUser
from services.auth.space_access import AccessDenied, require_owned_resource, require_space_operation
from services.mindmap.source_scope import parse_source_scope

from services.db.mongo import get_database
from services.document_it.enqueue import enqueue_document_generation
from services.document_it.store import (
    DocumentStore,
    public_document_card,
    public_document_doc,
    public_document_summary,
)
from services.plans.plan_limits import PlanLimitExceeded

router = APIRouter(dependencies=[Depends(require_user)])

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class GenerateDocumentBody(BaseModel):
    userId: str = Field(min_length=1)
    spaceId: str = Field(min_length=1)
    templateCode: str = Field(min_length=1)
    scope: str = "space"
    date: str | None = None
    dates: list[str] | None = None
    timeZone: str | None = None


def _safe_download_name(file_name: str) -> str:
    base = str(file_name or "document.docx").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _FILENAME_SAFE.sub("_", base).strip("._")
    if not cleaned.lower().endswith(".docx"):
        cleaned = f"{cleaned or 'document'}.docx"
    return cleaned[:120]


@router.post("/generate")
async def generate_document(
    body: GenerateDocumentBody,
    actor: AuthenticatedUser = Depends(require_user),
):
    user_id = require_matching_user(actor, body.userId)
    try:
        await require_space_operation(get_database(), user_id, body.spaceId.strip(), "write")
        result = await enqueue_document_generation(
            get_database(),
            user_id=user_id,
            space_id=body.spaceId.strip(),
            template_code=body.templateCode.strip(),
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
            detail=f"Unable to enqueue document generation: {type(error).__name__}",
        ) from error
    return result


@router.get("/list")
async def list_documents(
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
    store = DocumentStore(get_database())
    documents = await store.list_for_space(
        access.owner_user_id,
        space_id.strip(),
        limit=limit,
        card=card,
    )
    present = public_document_card if card else public_document_summary
    return {
        "success": True,
        "documents": [present(doc) for doc in documents],
    }


@router.get("/{document_id}")
async def get_document_by_id(
    document_id: str,
    actor: AuthenticatedUser = Depends(require_user),
):
    store = DocumentStore(get_database())
    document = await store.get(document_id.strip())
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        await require_owned_resource(
            get_database(),
            actor.user_id,
            document,
            operation="read",
            missing_detail="Document not found.",
        )
    except AccessDenied as error:
        raise_for_access(error)
    return {"success": True, "document": public_document_doc(document)}


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    actor: AuthenticatedUser = Depends(require_user),
):
    store = DocumentStore(get_database())
    document = await store.get(document_id.strip(), include_docx=False)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        await require_owned_resource(
            get_database(),
            actor.user_id,
            document,
            operation="read",
            missing_detail="Document not found.",
        )
    except AccessDenied as error:
        raise_for_access(error)
    if document.get("status") != "READY":
        raise HTTPException(status_code=409, detail="Document is not ready for download yet.")

    stored = await store.get(document_id.strip(), include_docx=True)
    encoded = (stored or {}).get("docxBase64")
    if not encoded:
        raise HTTPException(status_code=404, detail="DOCX file is not available for this document.")

    try:
        payload = base64.b64decode(str(encoded))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Stored document is corrupted.") from error

    file_name = _safe_download_name(str((stored or document).get("fileName") or "document.docx"))
    return Response(
        content=payload,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{file_name}"',
            "Content-Length": str(len(payload)),
            "X-Content-Type-Options": "nosniff",
        },
    )
