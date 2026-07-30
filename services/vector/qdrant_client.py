from qdrant_client import AsyncQdrantClient
from apps.api_gateway.config.setting import settings

from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_COLLECTION = settings.QDRANT_COLLECTION
VECTOR_SIZE = settings.VECTOR_SIZE


qdrant_client = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY or None,
)


async def ensure_collection_exists():
    collections = await qdrant_client.get_collections()
    existing = [collection.name for collection in collections.collections]

    if QDRANT_COLLECTION not in existing:
        await qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
    await ensure_payload_indexes()


async def ensure_payload_indexes():
    collection_info = await qdrant_client.get_collection(QDRANT_COLLECTION)
    payload_schema = getattr(collection_info, "payload_schema", {}) or {}
    required_indexes = {
        "userId": PayloadSchemaType.KEYWORD,
        "spaceId": PayloadSchemaType.KEYWORD,
        "job_id": PayloadSchemaType.KEYWORD,
        "chunkIndex": PayloadSchemaType.INTEGER,
    }

    for field_name, field_schema in required_indexes.items():
        if field_name in payload_schema:
            continue
        try:
            await qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as error:
            message = str(error).lower()
            if "already exists" not in message and "already has" not in message:
                raise
