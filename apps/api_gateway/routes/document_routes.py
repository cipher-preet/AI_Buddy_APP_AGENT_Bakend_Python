from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.db.mongo import get_database
from services.document_it.enqueue import enqueue_document_generation
from services.document_it.store import DocumentStore, public_document_doc, public_document_summary

router = APIRouter()


class GenerateDocumentBody(BaseModel):
    userId: str = Field(min_length=1)
    spaceId: str = Field(min_length=1)
    templateCode: str = Field(min_length=1)


@router.post("/generate")
async def generate_document(body: GenerateDocumentBody):
    try:
        result = await enqueue_document_generation(
            get_database(),
            user_id=body.userId.strip(),
            space_id=body.spaceId.strip(),
            template_code=body.templateCode.strip(),
        )
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
    user_id: str = Query(..., alias="userId", min_length=1),
    space_id: str = Query(..., alias="spaceId", min_length=1),
    limit: int = Query(24, ge=1, le=50),
):
    store = DocumentStore(get_database())
    documents = await store.list_for_space(
        user_id.strip(),
        space_id.strip(),
        limit=limit,
    )
    return {
        "success": True,
        "documents": [public_document_summary(doc) for doc in documents],
    }


@router.get("/{document_id}")
async def get_document_by_id(document_id: str):
    store = DocumentStore(get_database())
    document = await store.get(document_id.strip())
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"success": True, "document": public_document_doc(document)}


@router.get("/{document_id}/download")
async def download_document(document_id: str):
    store = DocumentStore(get_database())
    document = await store.get(document_id.strip(), include_docx=True)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if document.get("status") != "READY":
        raise HTTPException(status_code=409, detail="Document is not ready for download yet.")

    encoded = document.get("docxBase64")
    if not encoded:
        raise HTTPException(status_code=404, detail="DOCX file is not available for this document.")

    try:
        payload = base64.b64decode(str(encoded))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Stored document is corrupted.") from error

    file_name = str(document.get("fileName") or "document.docx")
    return Response(
        content=payload,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{file_name}"',
            "Content-Length": str(len(payload)),
        },
    )
