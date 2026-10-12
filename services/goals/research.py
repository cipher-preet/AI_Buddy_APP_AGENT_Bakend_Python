from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from apps.api_gateway.config.setting import settings
from services.goals.agents.llm import call_goal_agent
from services.goals.domains import context_profile, normalize_domain, understanding_as_dict, _text
from services.goals.schemas import EvidenceExtractionResult, ExtractedRequirement, ResearchPlanResult
from services.goals.store import utc_now
from services.search.parallel import get_search_provider
from services.search.provider import SearchRequest, SearchResult
from services.search.sanitize import publisher_from_url, wrap_untrusted

VOLATILE_TOKENS = ("job opening", "hiring", "salary", "law", "regulation", "exam date", "deadline", "current")


def _private_leaks(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("@", "phone", "ssn", "passport", "my name is"))


def heuristic_research_plan(goal: dict[str, Any]) -> ResearchPlanResult:
    understanding = understanding_as_dict(goal.get("understanding"))
    domain = normalize_domain(str(understanding.get("domain") or ""))
    statement = _text(goal.get("statement") or goal.get("title"))
    profile = context_profile(goal)
    geography = _text(profile.get("geography") or understanding.get("geography"))
    needed = bool(understanding.get("researchNeeded", domain not in {"PRODUCTIVITY"}))
    if domain in {"PRODUCTIVITY"} and "habit" in statement.lower() and domain != "HEALTH":
        needed = False
    if domain in {"CAREER", "EDUCATION", "BUSINESS", "FINANCIAL", "TECHNOLOGY", "HEALTH"}:
        needed = True
    queries = [item for item in [
        f"{statement} requirements",
        f"{understanding.get('subdomain') or domain} pathway {geography}".strip(),
        f"{statement} skills qualifications",
    ] if item.strip()][:3]
    return ResearchPlanResult(
        researchNeeded=needed,
        reason="External requirements would change the roadmap." if needed else "A simple personal practice plan does not need web search.",
        objective=f"Find current, practical requirements and realistic pathways for: {statement}"[:500],
        searchQueries=queries,
        mode="fast",
        location=geography[:2].lower() if len(geography) == 2 else "",
        reuseCached=False,
        maxResults=settings.PARALLEL_MAX_RESULTS,
    )


async def plan_research(goal: dict[str, Any], cached_hits: int = 0) -> ResearchPlanResult:
    understanding = understanding_as_dict(goal.get("understanding"))
    payload = {
        "statement": _text(goal.get("statement")),
        "understanding": understanding,
        "contextProfile": context_profile(goal),
        "cachedKnowledgeHits": cached_hits,
        "maxRounds": settings.GOAL_RESEARCH_MAX_ROUNDS,
    }
    called = await call_goal_agent(
        prompt_name="goal-research-v1",
        schema=ResearchPlanResult,
        payload=payload,
        stage="goal_research_plan",
        temperature=0.1,
        max_tokens=800,
    )
    if called is None:
        return heuristic_research_plan(goal)
    result, _model = called
    result.searchQueries = [
        query.strip()[:80]
        for query in result.searchQueries
        if query.strip() and not _private_leaks(query)
    ][:4]
    if result.mode not in {"turbo", "fast", "basic", "advanced"}:
        result.mode = "fast"
    if not result.searchQueries:
        return heuristic_research_plan(goal)
    return result


def _freshness(requirement: str) -> str:
    blob = requirement.lower()
    return "volatile" if any(token in blob for token in VOLATILE_TOKENS) else "stable"


def normalize_search_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    from services.search.sanitize import normalize_url

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        url = normalize_url(item.url)
        if not url or url in seen:
            continue
        seen.add(url)
        rows.append(
            {
                "url": url,
                "title": item.title,
                "publish_date": item.publish_date,
                "excerpts": item.excerpts[:4],
                "publisher": item.publisher or publisher_from_url(url),
            }
        )
    return rows


async def extract_requirements(
    goal: dict[str, Any],
    sources: list[dict[str, Any]],
) -> EvidenceExtractionResult:
    packed = []
    for source in sources[:8]:
        excerpts = [wrap_untrusted(str(item)) for item in (source.get("excerpts") or [])[:3]]
        packed.append(
            {
                "url": source.get("url"),
                "title": source.get("title"),
                "publisher": source.get("publisher"),
                "publish_date": source.get("publish_date"),
                "excerpts": excerpts,
            }
        )
    payload = {
        "statement": _text(goal.get("statement")),
        "understanding": understanding_as_dict(goal.get("understanding")),
        "sources": packed,
    }
    called = await call_goal_agent(
        prompt_name="goal-evidence-v1",
        schema=EvidenceExtractionResult,
        payload=payload,
        stage="goal_evidence_extract",
        temperature=0.1,
        max_tokens=1800,
    )
    if called is None:
        fallback: list[ExtractedRequirement] = []
        for source in sources[:6]:
            excerpt = " ".join(str(item) for item in (source.get("excerpts") or [])[:1])[:400]
            if excerpt:
                fallback.append(
                    ExtractedRequirement(
                        text=excerpt,
                        kind="INTERPRETATION",
                        sourceUrl=str(source.get("url") or ""),
                        sourceTitle=str(source.get("title") or ""),
                        confidence=0.3,
                        freshness="stable",
                    )
                )
        return EvidenceExtractionResult(
            requirements=fallback,
            insufficientEvidence=len(fallback) < 2,
            notes="Extractor unavailable; excerpts stored as low-confidence interpretations.",
        )
    result, _model = called
    allowed_urls = {str(item.get("url") or "") for item in sources}
    cleaned: list[ExtractedRequirement] = []
    for item in result.requirements:
        if item.sourceUrl and item.sourceUrl not in allowed_urls:
            item.sourceUrl = ""
        item.freshness = item.freshness if item.freshness in {"stable", "volatile"} else _freshness(item.text)
        cleaned.append(item)
    result.requirements = cleaned[:16]
    return result


def evidence_docs(
    goal: dict[str, Any],
    session_id: str,
    extraction: EvidenceExtractionResult,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    now = utc_now()
    source_by_url = {str(item.get("url") or ""): item for item in sources}
    docs = []
    for item in extraction.requirements:
        source = source_by_url.get(item.sourceUrl) or {}
        docs.append(
            {
                "evidenceId": uuid4().hex,
                "goalId": goal.get("_id"),
                "ownerId": goal.get("ownerId"),
                "sessionId": session_id,
                "fact": item.text[:800],
                "kind": item.kind if item.kind in {"FACT", "INTERPRETATION", "ASSUMPTION", "DISAGREEMENT"} else "INTERPRETATION",
                "sourceTitle": (item.sourceTitle or source.get("title") or "")[:240],
                "sourceUrl": item.sourceUrl,
                "publisher": str(source.get("publisher") or publisher_from_url(item.sourceUrl))[:160],
                "publicationDate": source.get("publish_date") or "",
                "retrievedAt": now,
                "excerpt": " ".join(str(part) for part in (source.get("excerpts") or [])[:1])[:1200],
                "geographicApplicability": item.geographicApplicability[:80],
                "relevance": float(item.confidence or 0.5),
                "confidence": float(item.confidence or 0.5),
                "freshness": item.freshness,
                "reviewBy": now + timedelta(days=7 if item.freshness == "volatile" else 30),
            }
        )
    return docs


def compact_evidence_pack(docs: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    ranked = sorted(docs, key=lambda item: float(item.get("confidence") or 0), reverse=True)
    pack = []
    for item in ranked[:limit]:
        pack.append(
            {
                "fact": item.get("fact"),
                "kind": item.get("kind"),
                "url": item.get("sourceUrl"),
                "title": item.get("sourceTitle"),
                "confidence": item.get("confidence"),
                "freshness": item.get("freshness"),
            }
        )
    return pack


async def maybe_store_knowledge(goal: dict[str, Any], docs: list[dict[str, Any]]) -> None:
    if not docs:
        return
    understanding = understanding_as_dict(goal.get("understanding"))
    domain = normalize_domain(str(understanding.get("domain") or "OTHER"))
    try:
        from services.vector.embedding_service import generate_embeddings
        from services.vector.qdrant_client import qdrant_client
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except Exception:
        return
    collection = settings.GOAL_KNOWLEDGE_COLLECTION
    try:
        collections = await qdrant_client.get_collections()
        names = [item.name for item in collections.collections]
        if collection not in names:
            await qdrant_client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception:
        return
    texts = [str(item.get("fact") or "") for item in docs if item.get("kind") == "FACT"]
    if not texts:
        return
    try:
        vectors = await generate_embeddings(texts[:8])
    except Exception:
        return
    points = []
    for text, vector, doc in zip(texts[:8], vectors, [item for item in docs if item.get("kind") == "FACT"][:8]):
        digest = hashlib.sha256(f"{domain}:{text.lower()}".encode("utf-8")).hexdigest()
        points.append(
            PointStruct(
                id=digest[:32],
                vector=vector,
                payload={
                    "domain": domain,
                    "subdomain": understanding.get("subdomain") or "",
                    "fact": text[:800],
                    "url": doc.get("sourceUrl") or "",
                    "freshness": doc.get("freshness") or "stable",
                    "retrievedAt": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
    try:
        await qdrant_client.upsert(collection_name=collection, points=points)
    except Exception:
        return


async def recall_knowledge(goal: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    understanding = understanding_as_dict(goal.get("understanding"))
    query = _text(understanding.get("primaryGoal") or goal.get("statement"))
    if not query:
        return []
    try:
        from services.vector.embedding_service import generate_embedding
        from services.vector.qdrant_client import qdrant_client
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except Exception:
        return []
    try:
        vector = await generate_embedding(query)
        domain = normalize_domain(str(understanding.get("domain") or "OTHER"))
        results = await qdrant_client.search(
            collection_name=settings.GOAL_KNOWLEDGE_COLLECTION,
            query_vector=vector,
            limit=limit,
            query_filter=Filter(must=[FieldCondition(key="domain", match=MatchValue(value=domain))]),
        )
    except Exception:
        return []
    rows = []
    now = datetime.now(timezone.utc)
    for hit in results:
        payload = hit.payload or {}
        retrieved = str(payload.get("retrievedAt") or "")
        freshness = str(payload.get("freshness") or "stable")
        max_age = settings.GOAL_KNOWLEDGE_TTL_VOLATILE_DAYS if freshness == "volatile" else settings.GOAL_KNOWLEDGE_TTL_STABLE_DAYS
        try:
            retrieved_at = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
            if now - retrieved_at > timedelta(days=max_age):
                continue
        except ValueError:
            pass
        rows.append(
            {
                "fact": payload.get("fact"),
                "url": payload.get("url"),
                "title": payload.get("subdomain") or payload.get("domain"),
                "kind": "FACT",
                "confidence": float(hit.score or 0.4),
                "freshness": freshness,
                "cached": True,
            }
        )
    return rows


async def run_research(
    store: Any,
    goal: dict[str, Any],
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    cached = [] if force_refresh else await recall_knowledge(goal)
    plan = await plan_research(goal, cached_hits=len(cached))
    session_id = uuid4().hex
    now = utc_now()
    sources: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    degraded = ""
    if not plan.researchNeeded:
        summary = {
            "sessionId": session_id,
            "researchNeeded": False,
            "reason": plan.reason,
            "sourceCount": 0,
            "degraded": "",
            "createdAt": now,
        }
        if store is not None:
            await store.save_research_session(goal, summary, [], [])
        return {"plan": plan.model_dump(), "sources": [], "evidence": [], "summary": summary, "cached": cached}

    provider = get_search_provider()
    rounds = min(max(1, settings.GOAL_RESEARCH_MAX_ROUNDS), 2)
    session_token = None
    for _round in range(rounds):
        response = await provider.search(
            SearchRequest(
                objective=plan.objective,
                search_queries=plan.searchQueries,
                mode=plan.mode if _round == 0 else "fast",
                max_results=plan.maxResults or settings.PARALLEL_MAX_RESULTS,
                location=plan.location or None,
                session_id=session_token,
                goal_id=str(goal.get("_id") or ""),
                user_id=str(goal.get("ownerId") or ""),
            )
        )
        session_token = response.session_id or session_token
        if response.degraded and not response.results:
            degraded = response.error or "search_failed"
            break
        sources.extend(normalize_search_results(response.results))
        thin = all(len(" ".join(item.get("excerpts") or [])) < 180 for item in sources)
        if thin and settings.PARALLEL_EXTRACT_ENABLED and sources:
            extracted = await provider.extract(
                [item["url"] for item in sources[:3]],
                objective=plan.objective,
                session_id=session_token,
            )
            for item in extracted.results:
                for source in sources:
                    if source["url"] == item.url and item.excerpts:
                        source["excerpts"] = item.excerpts[:4]
        break

    extraction = await extract_requirements(goal, sources) if sources else EvidenceExtractionResult(insufficientEvidence=True)
    docs = evidence_docs(goal, session_id, extraction, sources)
    if cached:
        for item in cached:
            docs.append(
                {
                    "evidenceId": uuid4().hex,
                    "goalId": goal.get("_id"),
                    "ownerId": goal.get("ownerId"),
                    "sessionId": session_id,
                    "fact": item.get("fact"),
                    "kind": "FACT",
                    "sourceTitle": item.get("title") or "cached knowledge",
                    "sourceUrl": item.get("url") or "",
                    "publisher": publisher_from_url(str(item.get("url") or "")),
                    "publicationDate": "",
                    "retrievedAt": now,
                    "excerpt": "",
                    "geographicApplicability": "",
                    "relevance": item.get("confidence") or 0.4,
                    "confidence": item.get("confidence") or 0.4,
                    "freshness": item.get("freshness") or "stable",
                    "reviewBy": now + timedelta(days=30),
                }
            )
    await maybe_store_knowledge(goal, docs)
    summary = {
        "sessionId": session_id,
        "researchNeeded": True,
        "reason": plan.reason,
        "objective": plan.objective,
        "queries": plan.searchQueries,
        "mode": plan.mode,
        "sourceCount": len(sources),
        "evidenceCount": len(docs),
        "insufficientEvidence": bool(extraction.insufficientEvidence or not docs),
        "disagreements": extraction.disagreements[:6],
        "degraded": degraded,
        "createdAt": now,
    }
    if store is not None:
        await store.save_research_session(goal, summary, sources, docs)
    return {
        "plan": plan.model_dump(),
        "sources": sources,
        "evidence": docs,
        "summary": summary,
        "cached": cached,
        "extraction": extraction.model_dump(),
    }
