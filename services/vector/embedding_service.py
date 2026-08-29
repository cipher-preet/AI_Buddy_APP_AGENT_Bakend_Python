from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from apps.api_gateway.config.setting import settings
from services.llm.async_runtime import LoopLocalResource, LoopLocalSemaphore, current_loop_id

EMBEDDING_MODEL = settings.EMBEDDING_MODEL

_clients = LoopLocalResource(
    lambda: AsyncOpenAI(api_key=settings.secret_value(settings.OPENAI_API_KEY))
)
_semaphore = LoopLocalSemaphore(int(getattr(settings, "EVENT_PIPELINE_EMBEDDING_MAX_CONCURRENCY", 8)))


def get_embedding_client() -> AsyncOpenAI:
    return _clients.get()


async def close_embedding_client() -> None:
    await _clients.aclose_current()


async def generate_embedding(text: str) -> list[float]:
    vectors = await generate_embeddings([text])
    return vectors[0]


async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    timeout = float(getattr(settings, "EVENT_PIPELINE_EMBEDDING_TIMEOUT_SECONDS", 60) or 60)
    async with _semaphore.get():
        response = await asyncio.wait_for(
            get_embedding_client().embeddings.create(
                model=EMBEDDING_MODEL,
                input=[item or " " for item in texts],
            ),
            timeout=timeout,
        )
    by_index = {int(item.index): item.embedding for item in response.data}
    return [by_index[index] for index in range(len(texts))]


# Backward-compatible module attribute used by older imports.
client = None
_ = current_loop_id
