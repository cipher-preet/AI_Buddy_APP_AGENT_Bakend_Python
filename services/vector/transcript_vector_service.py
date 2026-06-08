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
):
    if not transcript or not transcript.strip():
        return {
            "success": False,
            "message": "Transcript is empty",
        }

    await ensure_collection_exists()

    chunks = chunk_text(transcript)

    points: list[PointStruct] = []

    for index, chunk in enumerate(chunks):
        vector = await generate_embedding(chunk)

        point = PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "userId": user_id,
                "spaceId": space_id,
                "text": chunk,
                "chunkIndex": index,
                "isPublish": False,
                "createdAt": datetime.now(timezone.utc).isoformat(),
            },
        )

        points.append(point)

    response = await qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )

    print("thi sis quadarnt lenlpoint ", response)

    return {
        "success": True,
        "jobId": job_id,
        "totalChunks": len(points),
    }
