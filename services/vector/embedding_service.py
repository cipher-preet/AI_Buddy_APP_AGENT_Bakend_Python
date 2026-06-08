import os
from openai import AsyncOpenAI
from apps.api_gateway.config.setting import settings

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

EMBEDDING_MODEL = settings.EMBEDDING_MODEL


async def generate_embedding(text: str) -> list[float]:
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding
