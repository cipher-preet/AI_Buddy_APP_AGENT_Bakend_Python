from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SearchResult:
    url: str
    title: str = ""
    publish_date: str | None = None
    excerpts: list[str] = field(default_factory=list)
    publisher: str = ""


@dataclass
class SearchResponse:
    search_id: str = ""
    session_id: str = ""
    results: list[SearchResult] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    usage: list[dict[str, Any]] = field(default_factory=list)
    cached: bool = False
    degraded: bool = False
    error: str = ""


@dataclass
class SearchRequest:
    objective: str
    search_queries: list[str]
    mode: str = "fast"
    max_results: int = 8
    max_chars_total: int | None = None
    location: str | None = None
    session_id: str | None = None
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    goal_id: str = ""
    user_id: str = ""


@dataclass
class ExtractResponse:
    results: list[SearchResult] = field(default_factory=list)
    session_id: str = ""
    degraded: bool = False
    error: str = ""


class SearchProvider(Protocol):
    name: str

    async def search(self, request: SearchRequest) -> SearchResponse: ...

    async def extract(self, urls: list[str], *, objective: str = "", session_id: str | None = None) -> ExtractResponse: ...
