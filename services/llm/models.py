from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredLLMRequest(LLMRequest):
    schema_name: str | None = None


class LLMUsage(BaseModel):
    promptTokens: int = 0
    completionTokens: int = 0
    totalTokens: int = 0


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latencyMs: int | None = None


class ProviderHealth(BaseModel):
    provider: str
    healthy: bool
    latencyMs: int | None = None
    error: str | None = None


class LLMProvider(Protocol):
    name: str

    async def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    async def generate_structured(
        self,
        request: StructuredLLMRequest,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        ...

    async def health_check(self) -> ProviderHealth:
        ...
