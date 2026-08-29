"""Embedding similarity as a local grouping / candidate-retrieval signal.

Never used as final semantic truth. Provider calls are batched and cached.
Lexical hashing is the always-available fallback so tests and provider
outages do not silently drop sequences.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Protocol

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.textutil import content_tokens, cosine_similarity, normalize_text


LEXICAL_DIMENSIONS = 256


class Embedder(Protocol):
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class LexicalEmbedder:
    """Stable hashed bag-of-tokens embedding. Local grouping signal only."""

    def __init__(self, dimensions: int = LEXICAL_DIMENSIONS):
        self.dimensions = dimensions

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = content_tokens(text) or normalize_text(text).split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.casefold().encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: dict[str, list[float]] | None = None, *, lexical_fallback: bool = True):
        self.inner = inner
        self.cache = cache if cache is not None else {}
        self.calls = 0
        self.cache_hits = 0
        self.batch_calls = 0
        self.lexical_fallback = lexical_fallback

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        missing: list[str] = []
        missing_keys: list[str] = []
        keys = [_cache_key(text) for text in texts]
        seen_missing: set[str] = set()
        for text, key in zip(texts, keys):
            if key in self.cache:
                self.cache_hits += 1
            elif key in seen_missing:
                self.cache_hits += 1
            else:
                missing.append(text)
                missing_keys.append(key)
                seen_missing.add(key)
        if missing:
            batch_size = max(1, int(getattr(settings, "EVENT_PIPELINE_EMBEDDING_BATCH_SIZE", 32)))
            produced: list[list[float]] = []
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                self.batch_calls += 1
                produced.extend(await self._embed_batch(batch))
            for key, vector in zip(missing_keys, produced):
                self.cache[key] = vector
            self.calls += len(missing)
            try:
                from services.conversation.event_pipeline.observability import record_embedding_usage

                record_embedding_usage(requests=max(1, (len(missing) + batch_size - 1) // batch_size), items=len(missing))
            except Exception:
                pass
        return [self.cache[key] for key in keys]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self.inner.embed_many(texts)
        except Exception as error:
            from services.llm.async_runtime import reraise_if_hard_runtime

            reraise_if_hard_runtime(error)
            if not self.lexical_fallback:
                raise
            fallback = LexicalEmbedder()
            return await fallback.embed_many(texts)


class ProviderEmbedder:
    def __init__(self):
        self.batch_calls = 0
        self.texts_embedded = 0

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        from services.conversation.event_pipeline.budget import charge_model_call
        from services.vector.embedding_service import generate_embeddings

        charge_model_call()
        self.batch_calls += 1
        self.texts_embedded += len(texts)
        return await generate_embeddings([text or " " for text in texts])


def default_embedder(prefer_provider: bool | None = None, *, lexical_fallback: bool | None = None) -> CachedEmbedder:
    if prefer_provider is None:
        prefer_provider = bool(getattr(settings, "EVENT_PIPELINE_PREFER_PROVIDER_EMBEDDINGS", False))
    inner: Embedder = ProviderEmbedder() if prefer_provider else LexicalEmbedder()
    allow_lexical = True if lexical_fallback is None else bool(lexical_fallback)
    if not prefer_provider:
        allow_lexical = False
    return CachedEmbedder(inner, lexical_fallback=allow_lexical)


def top_k_similar(
    query: list[float],
    candidates: Iterable[tuple[str, list[float] | None]],
    k: int = 8,
    min_score: float = 0.0,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for key, vector in candidates:
        score = cosine_similarity(query, vector)
        if score >= min_score:
            scored.append((key, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, k)]


def _cache_key(text: str) -> str:
    return normalize_text(text).casefold()
