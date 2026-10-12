from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from apps.api_gateway.config.setting import settings
from services.llm.async_runtime import LoopBoundAsyncClient, LoopLocalSemaphore
from services.search.provider import ExtractResponse, SearchRequest, SearchResponse, SearchResult
from services.search.sanitize import is_safe_http_url, normalize_url, publisher_from_url

_provider: ParallelSearchProvider | None = None


def _secret() -> str:
    return settings.secret_value(settings.PARALLEL_API_KEY)


def _cache_key(request: SearchRequest) -> str:
    payload = {
        "objective": request.objective,
        "queries": list(request.search_queries),
        "mode": request.mode,
        "location": request.location or "",
        "max_results": request.max_results,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"goal:search:cache:{digest}"


class ParallelSearchProvider:
    name = "parallel"

    def __init__(self) -> None:
        self._transport = LoopBoundAsyncClient(
            lambda: httpx.AsyncClient(
                base_url=settings.PARALLEL_BASE_URL.rstrip("/"),
                timeout=httpx.Timeout(
                    connect=10,
                    read=settings.PARALLEL_SEARCH_TIMEOUT_SECONDS,
                    write=10,
                    pool=settings.PARALLEL_SEARCH_TIMEOUT_SECONDS,
                ),
            )
        )
        self._semaphore = LoopLocalSemaphore(settings.PARALLEL_MAX_CONCURRENCY)
        self._circuit_open_until = 0.0
        self._failures = 0
        self._in_flight: dict[str, asyncio.Task[SearchResponse]] = {}

    def configured(self) -> bool:
        return bool(_secret())

    async def _client(self) -> httpx.AsyncClient:
        return await self._transport.get()

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": _secret(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= 5:
            self._circuit_open_until = time.time() + 60
            self._failures = 0

    def _record_success(self) -> None:
        self._failures = 0
        self._circuit_open_until = 0.0

    async def _budget_ok(self, request: SearchRequest) -> bool:
        if settings.GOAL_SEARCH_BUDGET_PER_GOAL <= 0:
            return False
        try:
            from services.goals.redis_client import get_goal_redis_client

            redis = await get_goal_redis_client()
        except Exception:
            return True
        if request.goal_id:
            key = f"goal:search:budget:goal:{request.goal_id}"
            used = int(await redis.incr(key))
            if used == 1:
                await redis.expire(key, 86400)
            if used > settings.GOAL_SEARCH_BUDGET_PER_GOAL:
                await redis.decr(key)
                return False
        if request.user_id:
            key = f"goal:search:budget:user:{request.user_id}"
            used = int(await redis.incr(key))
            if used == 1:
                await redis.expire(key, 86400)
            if used > settings.GOAL_SEARCH_BUDGET_PER_USER:
                await redis.decr(key)
                return False
        return True

    async def _cached(self, request: SearchRequest) -> SearchResponse | None:
        try:
            from services.goals.redis_client import get_goal_redis_client

            redis = await get_goal_redis_client()
            raw = await redis.get(_cache_key(request))
        except Exception:
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return SearchResponse(
            search_id=str(data.get("search_id") or ""),
            session_id=str(data.get("session_id") or ""),
            results=[
                SearchResult(
                    url=item.get("url") or "",
                    title=item.get("title") or "",
                    publish_date=item.get("publish_date"),
                    excerpts=list(item.get("excerpts") or []),
                    publisher=item.get("publisher") or "",
                )
                for item in data.get("results") or []
            ],
            warnings=list(data.get("warnings") or []),
            cached=True,
        )

    async def _store_cache(self, request: SearchRequest, response: SearchResponse) -> None:
        if response.degraded or not response.results:
            return
        payload = {
            "search_id": response.search_id,
            "session_id": response.session_id,
            "results": [
                {
                    "url": item.url,
                    "title": item.title,
                    "publish_date": item.publish_date,
                    "excerpts": item.excerpts,
                    "publisher": item.publisher,
                }
                for item in response.results
            ],
            "warnings": response.warnings,
        }
        try:
            from services.goals.redis_client import get_goal_redis_client

            redis = await get_goal_redis_client()
            await redis.setex(_cache_key(request), settings.GOAL_SEARCH_CACHE_TTL_SECONDS, json.dumps(payload))
        except Exception:
            return

    def _normalize_results(self, rows: list[dict[str, Any]]) -> list[SearchResult]:
        seen: set[str] = set()
        results: list[SearchResult] = []
        for row in rows:
            url = normalize_url(str(row.get("url") or ""))
            if not url or not is_safe_http_url(url) or url in seen:
                continue
            seen.add(url)
            excerpts = [str(item)[:1500] for item in (row.get("excerpts") or []) if str(item).strip()][:6]
            results.append(
                SearchResult(
                    url=url,
                    title=str(row.get("title") or urlparse(url).hostname or "")[:240],
                    publish_date=str(row.get("publish_date") or "") or None,
                    excerpts=excerpts,
                    publisher=publisher_from_url(url),
                )
            )
        return results

    async def _post(self, path: str, body: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
        client = await self._client()
        retries = settings.PARALLEL_MAX_RETRIES
        last_error: dict[str, Any] = {}
        for attempt in range(retries + 1):
            try:
                response = await client.post(path, headers=self._headers(), json=body, timeout=timeout)
            except httpx.TimeoutException:
                last_error = {"error": "timeout"}
                await asyncio.sleep(0.2 * (2**attempt) + random.random() * 0.2)
                continue
            except httpx.HTTPError as error:
                last_error = {"error": type(error).__name__}
                await asyncio.sleep(0.2 * (2**attempt) + random.random() * 0.2)
                continue
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.4 * (2**attempt)
                await asyncio.sleep(delay + random.random() * 0.2)
                continue
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"error": {"message": "invalid json"}}
            return response.status_code, data if isinstance(data, dict) else {"error": "invalid json"}
        return 599, last_error or {"error": "timeout"}

    async def search(self, request: SearchRequest) -> SearchResponse:
        queries = [item.strip() for item in request.search_queries if str(item).strip()][:4]
        if not queries:
            return SearchResponse(degraded=True, error="missing_queries")
        if not self.configured():
            return SearchResponse(degraded=True, error="missing_api_key")
        if self._circuit_open():
            cached = await self._cached(request)
            if cached:
                cached.degraded = True
                cached.error = "circuit_open"
                return cached
            return SearchResponse(degraded=True, error="circuit_open")
        cached = await self._cached(request)
        if cached:
            return cached
        if not await self._budget_ok(request):
            return SearchResponse(degraded=True, error="budget_exhausted")

        key = _cache_key(request)
        inflight = self._in_flight.get(key)
        if inflight is not None and not inflight.done():
            return await inflight

        async def _run() -> SearchResponse:
            mode = (request.mode or settings.PARALLEL_SEARCH_MODE or "fast").lower()
            if mode not in {"turbo", "fast", "basic", "advanced"}:
                mode = "fast"
            body: dict[str, Any] = {
                "objective": request.objective[:2000],
                "search_queries": queries,
                "mode": mode,
            }
            if request.session_id:
                body["session_id"] = request.session_id
            if request.max_chars_total or settings.PARALLEL_MAX_CHARS_TOTAL:
                body["max_chars_total"] = request.max_chars_total or settings.PARALLEL_MAX_CHARS_TOTAL
            advanced: dict[str, Any] = {
                "max_results": request.max_results or settings.PARALLEL_MAX_RESULTS,
            }
            if request.location:
                advanced["location"] = request.location[:2].lower()
            source_policy: dict[str, Any] = {}
            if request.include_domains:
                source_policy["include_domains"] = request.include_domains[:8]
            if request.exclude_domains:
                source_policy["exclude_domains"] = request.exclude_domains[:8]
            if source_policy:
                advanced["source_policy"] = source_policy
            body["advanced_settings"] = advanced
            async with self._semaphore.get():
                status, data = await self._post("/v1/search", body, settings.PARALLEL_SEARCH_TIMEOUT_SECONDS)
            if status != 200:
                self._record_failure()
                message = ""
                error = data.get("error") if isinstance(data.get("error"), dict) else {}
                if isinstance(error, dict):
                    message = str(error.get("message") or "")
                return SearchResponse(degraded=True, error=f"http_{status}:{message[:80]}")
            self._record_success()
            warnings = data.get("warnings") or []
            if warnings is None:
                warnings = []
            response = SearchResponse(
                search_id=str(data.get("search_id") or ""),
                session_id=str(data.get("session_id") or ""),
                results=self._normalize_results(list(data.get("results") or [])),
                warnings=[item if isinstance(item, dict) else {"message": str(item)} for item in warnings],
                usage=list(data.get("usage") or []) if isinstance(data.get("usage"), list) else [],
            )
            await self._store_cache(request, response)
            return response

        task = asyncio.create_task(_run())
        self._in_flight[key] = task
        try:
            return await task
        finally:
            self._in_flight.pop(key, None)

    async def extract(self, urls: list[str], *, objective: str = "", session_id: str | None = None) -> ExtractResponse:
        if not settings.PARALLEL_EXTRACT_ENABLED or not self.configured():
            return ExtractResponse(degraded=True, error="extract_disabled")
        safe = [normalize_url(url) for url in urls if is_safe_http_url(normalize_url(url))][:5]
        if not safe:
            return ExtractResponse(degraded=True, error="no_safe_urls")
        body: dict[str, Any] = {"urls": safe}
        if objective:
            body["objective"] = objective[:2000]
        if session_id:
            body["session_id"] = session_id
        body["advanced_settings"] = {"excerpt_settings": {"enabled": True}}
        async with self._semaphore.get():
            status, data = await self._post("/v1/extract", body, settings.PARALLEL_EXTRACT_TIMEOUT_SECONDS)
        if status != 200:
            return ExtractResponse(degraded=True, error=f"http_{status}")
        rows = data.get("results") or data.get("extractions") or []
        results: list[SearchResult] = []
        for row in rows if isinstance(rows, list) else []:
            url = normalize_url(str(row.get("url") or ""))
            excerpts = []
            if row.get("excerpts"):
                excerpts = [str(item)[:2000] for item in row.get("excerpts") if str(item).strip()]
            elif row.get("full_content") or row.get("content"):
                excerpts = [str(row.get("full_content") or row.get("content"))[:2000]]
            if url:
                results.append(
                    SearchResult(
                        url=url,
                        title=str(row.get("title") or "")[:240],
                        excerpts=excerpts,
                        publisher=publisher_from_url(url),
                    )
                )
        return ExtractResponse(results=results, session_id=str(data.get("session_id") or ""))


def get_search_provider() -> ParallelSearchProvider:
    global _provider
    if _provider is None:
        _provider = ParallelSearchProvider()
    return _provider
