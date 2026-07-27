from datetime import datetime, timezone
from uuid import uuid4

from qdrant_client.models import PointStruct

from services.vector.qdrant_client import (
    qdrant_client,
    ensure_collection_exists,
    QDRANT_COLLECTION,
)
from services.vector.embedding_service import generate_embedding
from services.vector.chunking import chunk_text


async def store_transcript_in_vector_db(
    user_id: str,
    space_id: str,
    job_id: str,
    transcript: str,
    language_code: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
):
    if not transcript or not transcript.strip():
        return {
            "success": False,
            "message": "Transcript is empty",
        }

    user_id = user_id.strip()
    space_id = space_id.strip()

    await ensure_collection_exists()

    chunks = chunk_text(transcript)

    points: list[PointStruct] = []

    for index, chunk in enumerate(chunks):
        vector = await generate_embedding(chunk)
        chunk_id = str(uuid4())

        point = PointStruct(
            id=chunk_id,
            vector=vector,
            payload={
                "user_id": user_id,
                "space_id": space_id,
                "userId": user_id,
                "spaceId": space_id,
                "request_id": request_id,
                "session_id": session_id,
                "text": chunk,
                "source": "speech",
                "sourceType": "speech",
                "chunkIndex": index,
                "chunkId": chunk_id,
                "isPublish": False,
                "isDamaged": False,
                "isUseful": True,
                "chunkStatus": "active",
                "createdAt": datetime.now(timezone.utc).isoformat(),
            },
        )

        points.append(point)

    await qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    return {
        "success": True,
        "jobId": job_id,
        "totalChunks": len(points),
    }
