from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from apps.api_gateway.config.setting import settings
from services.llm.errors import LLMProviderError, is_retryable_status
from services.llm.models import LLMRequest, LLMResponse, LLMUsage, ProviderHealth, StructuredLLMRequest


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        default_model: str,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
        max_tokens_limit: int | None = None,
    ):
        self.name = name
        self.configured = True
        self.default_model = default_model
        self.max_retries = max_retries
        self.max_tokens_limit = max_tokens_limit
        self._auth_header = auth_header
        self._auth_value = f"{auth_prefix}{api_key}" if auth_prefix else api_key
        self._semaphore = asyncio.Semaphore(max_concurrency)
        timeout = httpx.Timeout(
            connect=min(10, timeout_seconds),
            read=timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
        }
        payload.update(request.metadata.get("extra_body") or {})
        payload = _without_none_values(payload)
        max_tokens = self._bounded_max_tokens(request.max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens
        started = time.perf_counter()
        async with self._semaphore:
            response = await self._post_with_retries("/chat/completions", payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return LLMResponse(
            content=str(message.get("content") or ""),
            provider=self.name,
            model=model,
            usage=LLMUsage(
                promptTokens=int(usage.get("prompt_tokens") or 0),
                completionTokens=int(usage.get("completion_tokens") or 0),
                totalTokens=int(usage.get("total_tokens") or 0),
            ),
            latencyMs=latency_ms,
        )

    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        schema = response_schema.model_json_schema()
        schema_text = json.dumps(schema, ensure_ascii=True)
        structured_request = self._with_structured_response_format(
            request,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name or response_schema.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        try:
            return await self._generate_structured_with_validation(
                structured_request,
                response_schema,
            )
        except LLMProviderError as error:
            if error.status_code not in {400, 422}:
                raise

        fallback_request = self._with_structured_response_format(
            request,
            {"type": "json_object"},
            (
                "Return valid JSON only. The JSON must match this schema exactly: "
                f"{schema_text}"
            ),
        )
        return await self._generate_structured_with_validation(
            fallback_request,
            response_schema,
        )

    def _with_structured_response_format(
        self,
        request: StructuredLLMRequest,
        response_format: dict[str, Any],
        schema_instruction: str | None = None,
    ) -> StructuredLLMRequest:
        structured_request = request.model_copy(deep=True)
        structured_request.max_tokens = structured_request.max_tokens or settings.LLM_STRUCTURED_MAX_TOKENS
        structured_request.metadata.setdefault("extra_body", {})
        structured_request.metadata["extra_body"].update(
            {
                "response_format": response_format,
            }
        )
        if self.name == "sarvam":
            structured_request.metadata["extra_body"]["reasoning_effort"] = None
        structured_request.messages.append(
            type(request.messages[0])(
                role="system",
                content=schema_instruction
                or "Return only the JSON object requested by the response schema. Do not wrap it in markdown.",
            )
        )
        return structured_request

    async def _generate_structured_with_validation(
        self,
        structured_request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        last_error: Exception | None = None
        for _ in range(3):
            response = await self.generate(structured_request)
            try:
                return response_schema.model_validate_json(response.content)
            except ValidationError as error:
                last_error = error
                structured_request.messages.append(
                    type(structured_request.messages[0])(
                        role="user",
                        content=(
                            "Your previous response did not validate. Return corrected JSON only. "
                            f"Validation error: {error}"
                        ),
                    )
                )
        raise LLMProviderError(f"Structured response validation failed: {last_error}", retryable=False)

    async def health_check(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            await self._client.get("/")
            return ProviderHealth(
                provider=self.name,
                healthy=True,
                latencyMs=int((time.perf_counter() - started) * 1000),
            )
        except Exception as error:
            return ProviderHealth(provider=self.name, healthy=False, error=str(error))

    async def _post_with_retries(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    path,
                    headers={self._auth_header: self._auth_value},
                    json=payload,
                )
                if response.status_code < 400:
                    return response
                retryable = is_retryable_status(response.status_code)
                if not retryable:
                    body = response.text[:1000]
                    raise LLMProviderError(
                        f"{self.name} permanent error {response.status_code}: {body}",
                        retryable=False,
                        status_code=response.status_code,
                    )
                retry_after = response.headers.get("retry-after")
                await asyncio.sleep(_retry_delay(attempt, retry_after))
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout) as error:
                last_error = error
                await asyncio.sleep(_retry_delay(attempt, None))
        raise LLMProviderError(f"{self.name} request failed: {last_error}", retryable=True)

    def _bounded_max_tokens(self, max_tokens: int | None) -> int | None:
        if max_tokens is None or self.max_tokens_limit is None:
            return max_tokens
        return min(max_tokens, self.max_tokens_limit)


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 60)
        except ValueError:
            pass
    return min(60, (2**attempt) + random.uniform(0, 0.5))


def _without_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_none_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none_values(item) for item in value]
    return value
