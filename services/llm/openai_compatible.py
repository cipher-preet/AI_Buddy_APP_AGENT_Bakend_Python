from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from apps.api_gateway.config.setting import settings
from services.llm.errors import LLMProviderError, StructuredOutputError, is_retryable_status
from services.llm.models import LLMRequest, LLMResponse, LLMUsage, ProviderHealth, StructuredLLMRequest
from services.llm.schema_adapter import (
    HTTP_ERROR,
    INCOMPLETE_STRUCTURED_OUTPUT,
    MALFORMED_JSON,
    MALFORMED_STRUCTURED_OUTPUT,
    PARSED_INSTANCE,
    PROVIDER_TIMEOUT,
    RATE_LIMITED,
    SCHEMA_ECHO,
    STRUCTURED_SCHEMA_ECHO,
    STRUCTURED_SCHEMA_UNSUPPORTED,
    WIRE_REQUIRED_COLLECTIONS,
    build_structured_plan,
    classify_validation_error,
    is_schema_echo,
    provider_local_recovery_eligible,
)


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
        self.last_structured_diagnostics: dict[str, Any] = {}
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
            content=_assistant_message_text(message),
            provider=self.name,
            model=model,
            usage=LLMUsage(
                promptTokens=int(usage.get("prompt_tokens") or 0),
                completionTokens=int(usage.get("completion_tokens") or 0),
                totalTokens=int(usage.get("total_tokens") or 0),
            ),
            latencyMs=latency_ms,
            finishReason=str(choice.get("finish_reason") or "") or None,
        )

    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        schema_name = request.schema_name or response_schema.__name__
        model = request.model or self.default_model
        plan = build_structured_plan(self.name, model, response_schema, schema_name)
        last_error: Exception | None = None
        for index, attempt in enumerate(plan.attempts):
            structured_request = self._request_from_attempt(request, attempt)
            budgets = _output_budgets_for(schema_name, structured_request.max_tokens)
            for budget_index, budget in enumerate(budgets):
                structured_request.max_tokens = budget
                try:
                    return await self._generate_structured_with_validation(
                        structured_request,
                        response_schema,
                        requested_mode=attempt.mode,
                        model=model,
                    )
                except StructuredOutputError as error:
                    last_error = error
                    truncated = _output_truncated(self.last_structured_diagnostics, budget)
                    if truncated and budget_index < len(budgets) - 1:
                        print(
                            "LLM output truncated; retrying same mode with higher max_tokens:",
                            {
                                "stage": schema_name,
                                "provider": self.name,
                                "model": model,
                                "mode": attempt.mode,
                                "fromMaxTokens": budget,
                                "toMaxTokens": budgets[budget_index + 1],
                            },
                        )
                        continue
                    if truncated or index >= len(plan.attempts) - 1 or not provider_local_recovery_eligible(error):
                        raise
                    break
                except LLMProviderError as error:
                    last_error = error
                    if index >= len(plan.attempts) - 1 or not provider_local_recovery_eligible(error):
                        raise
                    break
        raise last_error or LLMProviderError("structured output failed", retryable=True, status_code=422)

    def _request_from_attempt(self, request: StructuredLLMRequest, attempt) -> StructuredLLMRequest:
        structured_request = request.model_copy(deep=True)
        structured_request.temperature = attempt.temperature
        structured_request.max_tokens = structured_request.max_tokens or settings.LLM_STRUCTURED_MAX_TOKENS
        structured_request.metadata.setdefault("extra_body", {})
        extra_body = dict(structured_request.metadata.get("extra_body") or {})
        extra_body.pop("response_format", None)
        extra_body.update(attempt.extra_body or {})
        if attempt.response_format is not None:
            extra_body["response_format"] = attempt.response_format
        if self.name == "sarvam":
            extra_body["reasoning_effort"] = None
        structured_request.metadata["extra_body"] = extra_body
        structured_request.messages.append(
            type(request.messages[0])(
                role="system",
                content=attempt.instruction
                or "Return only the JSON object requested by the response schema. Do not wrap it in markdown.",
            )
        )
        return structured_request

    def _request_for_structured_mode(
        self,
        request: StructuredLLMRequest,
        schema: dict[str, Any],
        schema_name: str,
        mode: str,
    ) -> StructuredLLMRequest:
        plan = build_structured_plan(self.name, request.model or self.default_model, None, schema_name)
        for attempt in plan.attempts:
            if attempt.mode == mode:
                return self._request_from_attempt(request, attempt)
        return self._with_structured_response_format(
            request,
            {"type": mode} if mode in {"json_schema", "json_object"} else None,
        )

    def _with_structured_response_format(
        self,
        request: StructuredLLMRequest,
        response_format: dict[str, Any] | None,
        schema_instruction: str | None = None,
    ) -> StructuredLLMRequest:
        structured_request = request.model_copy(deep=True)
        structured_request.max_tokens = structured_request.max_tokens or settings.LLM_STRUCTURED_MAX_TOKENS
        structured_request.metadata.setdefault("extra_body", {})
        extra_body = dict(structured_request.metadata.get("extra_body") or {})
        extra_body.pop("response_format", None)
        if response_format is not None:
            extra_body["response_format"] = response_format
        if self.name == "sarvam":
            extra_body["reasoning_effort"] = None
        structured_request.metadata["extra_body"] = extra_body
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
        requested_mode: str,
        model: str,
    ) -> BaseModel:
        response = await self.generate(structured_request)
        # json_schema, json_object, and plain JSON all pass through the same
        # canonical pydantic validator. Valid JSON is not sufficient by itself.
        try:
            parsed, diagnostics = parse_structured_content(response_schema, response.content)
            diagnostics.update(
                {
                    "provider": self.name,
                    "model": model,
                    "stage": structured_request.schema_name or response_schema.__name__,
                    "requestedStructuredMode": requested_mode,
                    "actualResponseFormatMode": requested_mode,
                    "structuredModeUsed": requested_mode,
                    "finishReason": response.finishReason,
                    "completionTokens": int(response.usage.completionTokens or 0),
                    "promptTokens": int(response.usage.promptTokens or 0),
                    "latencyMs": response.latencyMs,
                    "structuredOutputSuccess": True,
                }
            )
            self.last_structured_diagnostics = diagnostics
            _log_structured_attempt(diagnostics)
            return parsed
        except StructuredOutputError as error:
            diagnostics = {
                "provider": self.name,
                "model": model,
                "stage": structured_request.schema_name or response_schema.__name__,
                "requestedStructuredMode": requested_mode,
                "actualResponseFormatMode": requested_mode,
                "structuredModeUsed": requested_mode,
                "topLevelResponseKeys": _top_level_keys(response.content),
                "schemaEchoDetected": error.outcome in {STRUCTURED_SCHEMA_ECHO, SCHEMA_ECHO},
                "parsingOutcome": error.outcome,
                "finishReason": response.finishReason,
                "completionTokens": int(response.usage.completionTokens or 0),
                "promptTokens": int(response.usage.promptTokens or 0),
                "latencyMs": response.latencyMs,
                "structuredOutputSuccess": False,
                "retryReason": error.outcome,
            }
            self.last_structured_diagnostics = diagnostics
            _log_structured_attempt(diagnostics)
            raise
        except ValidationError as error:
            reason = classify_validation_error(response_schema, _load_json_payload(_sanitize_json_text(response.content))[0], error)
            diagnostics = {
                "provider": self.name,
                "model": model,
                "stage": structured_request.schema_name or response_schema.__name__,
                "requestedStructuredMode": requested_mode,
                "actualResponseFormatMode": requested_mode,
                "structuredModeUsed": requested_mode,
                "topLevelResponseKeys": _top_level_keys(response.content),
                "schemaEchoDetected": False,
                "parsingOutcome": reason,
                "finishReason": response.finishReason,
                "completionTokens": int(response.usage.completionTokens or 0),
                "promptTokens": int(response.usage.promptTokens or 0),
                "latencyMs": response.latencyMs,
                "structuredOutputSuccess": False,
                "retryReason": reason,
            }
            self.last_structured_diagnostics = diagnostics
            _log_structured_attempt(diagnostics)
            raise StructuredOutputError(reason, f"Structured response validation failed: {error}")

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
                    print(
                        "LLM HTTP call succeeded:",
                        {
                            "provider": self.name,
                            "model": payload.get("model"),
                            "statusCode": response.status_code,
                            "attempt": attempt + 1,
                        },
                    )
                    return response
                print(
                    "LLM HTTP call failed:",
                    {
                        "provider": self.name,
                        "model": payload.get("model"),
                        "statusCode": response.status_code,
                        "attempt": attempt + 1,
                        "retryable": is_retryable_status(response.status_code),
                    },
                )
                retryable = is_retryable_status(response.status_code)
                failure_reason = RATE_LIMITED if response.status_code == 429 else HTTP_ERROR
                body = response.text[:1000]
                if response.status_code in {400, 422} and any(
                    token in body.casefold() for token in ("json_schema", "response_format", "schema", "strict")
                ):
                    failure_reason = STRUCTURED_SCHEMA_UNSUPPORTED
                if not retryable:
                    raise LLMProviderError(
                        f"{self.name} permanent error {response.status_code}: {body}",
                        retryable=False,
                        status_code=response.status_code,
                        failure_reason=failure_reason,
                    )
                retry_after = response.headers.get("retry-after")
                last_error = LLMProviderError(
                    f"{self.name} error {response.status_code}: {body}",
                    retryable=True,
                    status_code=response.status_code,
                    failure_reason=failure_reason,
                )
                await asyncio.sleep(_retry_delay(attempt, retry_after))
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout) as error:
                last_error = LLMProviderError(
                    f"{self.name} request failed: {error}",
                    retryable=True,
                    failure_reason=PROVIDER_TIMEOUT,
                )
                print(
                    "LLM HTTP call failed:",
                    {
                        "provider": self.name,
                        "model": payload.get("model"),
                        "attempt": attempt + 1,
                        "error": type(error).__name__,
                    },
                )
                await asyncio.sleep(_retry_delay(attempt, None))
        raise last_error or LLMProviderError(f"{self.name} request failed", retryable=True, failure_reason=HTTP_ERROR)

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


def _validate_structured_response(response_schema: type[BaseModel], content: str) -> BaseModel:
    parsed, _ = parse_structured_content(response_schema, content)
    return parsed


def parse_structured_content(response_schema: type[BaseModel], content: str) -> tuple[BaseModel, dict[str, Any]]:
    cleaned = _sanitize_json_text(content)
    payload, top_level_keys = _load_json_payload(cleaned)
    diagnostics = {
        "topLevelResponseKeys": top_level_keys,
        "schemaEchoDetected": is_schema_echo(payload),
        "parsingOutcome": PARSED_INSTANCE,
    }
    if payload is None:
        raise StructuredOutputError(MALFORMED_JSON, MALFORMED_STRUCTURED_OUTPUT)
    if is_schema_echo(payload):
        diagnostics["parsingOutcome"] = STRUCTURED_SCHEMA_ECHO
        raise StructuredOutputError(STRUCTURED_SCHEMA_ECHO, STRUCTURED_SCHEMA_ECHO)
    required = WIRE_REQUIRED_COLLECTIONS.get(getattr(response_schema, "__name__", ""), ())
    if isinstance(payload, dict) and any(field not in payload or payload.get(field) is None for field in required):
        diagnostics["parsingOutcome"] = INCOMPLETE_STRUCTURED_OUTPUT
        raise StructuredOutputError(INCOMPLETE_STRUCTURED_OUTPUT, INCOMPLETE_STRUCTURED_OUTPUT)
    try:
        return response_schema.model_validate(payload), diagnostics
    except ValidationError as error:
        extracted = _extract_json_object(cleaned)
        if extracted and extracted != cleaned:
            nested, nested_keys = _load_json_payload(extracted)
            diagnostics["topLevelResponseKeys"] = nested_keys or top_level_keys
            if is_schema_echo(nested):
                diagnostics["schemaEchoDetected"] = True
                diagnostics["parsingOutcome"] = STRUCTURED_SCHEMA_ECHO
                raise StructuredOutputError(STRUCTURED_SCHEMA_ECHO, STRUCTURED_SCHEMA_ECHO)
            if isinstance(nested, dict) and any(field not in nested or nested.get(field) is None for field in required):
                diagnostics["parsingOutcome"] = INCOMPLETE_STRUCTURED_OUTPUT
                raise StructuredOutputError(INCOMPLETE_STRUCTURED_OUTPUT, INCOMPLETE_STRUCTURED_OUTPUT)
            if nested is not None:
                try:
                    return response_schema.model_validate(nested), diagnostics
                except ValidationError as nested_error:
                    error = nested_error
                    payload = nested
        reason = classify_validation_error(response_schema, payload, error)
        diagnostics["parsingOutcome"] = reason
        raise StructuredOutputError(reason, str(error))


def _load_json_payload(content: str) -> tuple[Any, list[str]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json_object(content) or _close_truncated_json(content)
        if not extracted:
            return None, []
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError:
            return None, []
    if isinstance(payload, dict):
        return payload, sorted(str(key) for key in payload.keys())
    return payload, []


def _top_level_keys(content: str) -> list[str]:
    _, keys = _load_json_payload(_sanitize_json_text(content))
    return keys


def _instance_instruction(schema_name: str, schema: dict[str, Any]) -> str:
    from services.llm.schema_adapter import _schema_instruction

    return _schema_instruction(schema_name, schema)


def _log_structured_attempt(diagnostics: dict[str, Any]) -> None:
    print(
        "Structured output attempt:",
        {
            "stage": diagnostics.get("stage"),
            "provider": diagnostics.get("provider"),
            "model": diagnostics.get("model"),
            "requestedStructuredMode": diagnostics.get("requestedStructuredMode"),
            "actualResponseFormatMode": diagnostics.get("actualResponseFormatMode"),
            "topLevelResponseKeys": diagnostics.get("topLevelResponseKeys") or [],
            "schemaEchoDetected": bool(diagnostics.get("schemaEchoDetected")),
            "parsingOutcome": diagnostics.get("parsingOutcome"),
            "inputTokenEstimate": diagnostics.get("promptTokens"),
            "outputTokens": diagnostics.get("completionTokens"),
            "latencyMs": diagnostics.get("latencyMs"),
            "structuredOutputSuccess": bool(diagnostics.get("structuredOutputSuccess")),
            "retryReason": diagnostics.get("retryReason"),
        },
    )


def _sanitize_json_text(content: str) -> str:
    value = str(content or "").strip()
    value = value.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Remove ASCII control characters except whitespace JSON permits.
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)


def _extract_json_object(content: str) -> str | None:
    start = content.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None


def _close_truncated_json(content: str) -> str | None:
    start = content.find("{")
    if start < 0:
        return None
    snippet = content[start:]
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in snippet:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
    repaired = snippet
    if in_string:
        if escaped:
            repaired += "\\"
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired)
    while stack:
        repaired += stack.pop()
    try:
        json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return repaired


def _assistant_message_text(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    candidates = [part for part in (_sanitize_json_text(content), _sanitize_json_text(reasoning)) if part]
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            extracted = _extract_json_object(candidate)
            if extracted:
                try:
                    json.loads(extracted)
                    return extracted
                except json.JSONDecodeError:
                    pass
    for candidate in candidates:
        repaired = _close_truncated_json(candidate)
        if repaired:
            return repaired
    return content or reasoning


def _output_truncated(diagnostics: dict[str, Any] | None, max_tokens: int | None) -> bool:
    payload = diagnostics or {}
    finish = str(payload.get("finishReason") or "").strip().casefold()
    if finish in {"length", "max_tokens", "max_completion_tokens"}:
        return True
    completion = int(payload.get("completionTokens") or 0)
    return bool(max_tokens and completion >= int(max_tokens))


def _output_budgets_for(schema_name: str, requested: int | None) -> list[int]:
    start = int(requested or settings.LLM_STRUCTURED_MAX_TOKENS)
    if schema_name != "FinalSynthesisLLMResponse":
        return [max(512, start)]
    ceiling = max(512, int(settings.LLM_SYNTHESIS_OUTPUT_MAX_TOKENS))
    start = max(512, min(start, ceiling))
    if start >= ceiling:
        return [ceiling]
    return [start, ceiling]
