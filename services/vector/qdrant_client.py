from qdrant_client import AsyncQdrantClient
from apps.api_gateway.config.setting import settings

from qdrant_client.models import Distance, VectorParams

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
