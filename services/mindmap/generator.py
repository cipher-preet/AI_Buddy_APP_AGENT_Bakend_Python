from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from apps.api_gateway.config.setting import settings
from services.llm.models import LLMMessage, StructuredLLMRequest
from services.llm.router import get_llm_router
from services.mindmap.layout import ensure_positions, graph_from_llm_payload
from services.mindmap.log import mindmap_log
from services.mindmap.schemas import MindmapGraph
from services.prompts.loader import load_prompt
from services.schedule_extraction.llm import parse_model_route


class MindmapBranchCard(BaseModel):
    id: str | None = None
    title: str
    items: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class MindmapBranch(BaseModel):
    id: str | None = None
    title: str
    edgeLabel: str | None = None
    cards: list[MindmapBranchCard] = Field(default_factory=list)


class MindmapGenerateSchema(BaseModel):
    spaceTitle: str
    hubSubtitle: str | None = None
    branches: list[MindmapBranch] = Field(default_factory=list)


class MindmapGenerateError(RuntimeError):
    pass


def _route() -> list[tuple[Any, str]]:
    router = get_llm_router()
    spec = settings.MINDMAP_MODELS
    route = [
        (router.providers[name], model)
        for name, model in parse_model_route(spec)
        if name in router.providers and getattr(router.providers[name], "configured", True) is not False
    ]
    if not route:
        raise MindmapGenerateError("Mind map LLM route is not configured (set OPENROUTER_API_KEY).")
    return route


async def generate_mindmap_graph(context_pack: dict[str, Any], *, space_name: str) -> tuple[MindmapGraph, str]:
    messages = [
        LLMMessage(role="system", content=load_prompt("mindmap-generate-v1")),
        LLMMessage(role="user", content=json.dumps(context_pack, ensure_ascii=False, default=str)),
    ]
    max_tokens = settings.MINDMAP_MAX_OUTPUT_TOKENS
    errors: list[str] = []

    for provider, model in _route():
        request = StructuredLLMRequest(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=max_tokens,
            metadata={"stage": "mindmap_generate", "extra_body": {}},
            schema_name=MindmapGenerateSchema.__name__,
        )
        try:
            result = await provider.generate_structured(request, MindmapGenerateSchema)
        except Exception as error:
            errors.append(f"{provider.name}:{model}: {type(error).__name__}: {str(error)[:180]}")
            mindmap_log(
                "llm_failed",
                provider=getattr(provider, "name", None),
                model=model,
                error=type(error).__name__,
            )
            continue

        payload = result.model_dump()
        graph = ensure_positions(graph_from_llm_payload(payload, space_name=space_name))
        if not graph.nodes:
            errors.append(f"{model}: empty_graph")
            continue
        mindmap_log(
            "llm_ok",
            provider=getattr(provider, "name", None),
            model=model,
            nodeCount=len(graph.nodes),
            edgeCount=len(graph.edges),
        )
        return graph, f"{getattr(provider, 'name', 'llm')}:{model}"

    raise MindmapGenerateError("; ".join(errors) or "Mind map generation failed.")
