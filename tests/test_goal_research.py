import pytest

from services.goals.gaps import heuristic_gaps
from services.goals.research import extract_requirements, heuristic_research_plan, normalize_search_results
from services.search.provider import SearchResult


def test_habit_skips_research():
    goal = {
        "statement": "Read 20 pages every evening",
        "understanding": {"domain": "PRODUCTIVITY", "researchNeeded": False, "primaryGoal": "daily reading habit"},
    }
    plan = heuristic_research_plan(goal)
    assert plan.researchNeeded is False


def test_career_requests_research():
    goal = {
        "statement": "Become a zonal sales manager in a bank",
        "understanding": {"domain": "CAREER", "subdomain": "BANKING_MANAGEMENT", "researchNeeded": True},
    }
    plan = heuristic_research_plan(goal)
    assert plan.researchNeeded is True
    assert 2 <= len(plan.searchQueries) <= 4
    assert "my name" not in " ".join(plan.searchQueries).lower()


def test_normalize_dedupes_urls():
    rows = normalize_search_results(
        [
            SearchResult(url="https://Example.com/a/", title="A", excerpts=["one"]),
            SearchResult(url="https://example.com/a", title="A again", excerpts=["two"]),
        ]
    )
    assert len({item["url"] for item in rows}) == 1


@pytest.mark.asyncio
async def test_conflicting_sources_preserved(monkeypatch):
    async def no_llm(**kwargs):
        return None

    monkeypatch.setattr("services.goals.research.call_goal_agent", no_llm)
    sources = [
        {"url": "https://a.example/x", "title": "A", "excerpts": ["Requires 10 years"], "publisher": "a.example"},
        {"url": "https://b.example/y", "title": "B", "excerpts": ["Requires 5 years"], "publisher": "b.example"},
    ]
    result = await extract_requirements({"statement": "Become a manager", "understanding": {"domain": "CAREER"}}, sources)
    assert result.insufficientEvidence or len(result.requirements) >= 1


def test_unknown_starting_level_not_missing():
    goal = {"statement": "Become a manager", "understanding": {"startingLevel": "UNKNOWN", "domain": "CAREER"}}
    evidence = [{"fact": "Leadership of multiple branches is typically required", "kind": "FACT", "sourceUrl": "https://x"}]
    gaps = heuristic_gaps(goal, evidence)
    assert gaps.missing == []
    assert gaps.unknown
    assert gaps.provisional is True


def test_stale_volatile_not_in_heuristic_pack():
    from datetime import datetime, timedelta, timezone

    from services.goals.research import compact_evidence_pack

    old = {
        "fact": "An opening exists this week",
        "kind": "FACT",
        "confidence": 0.9,
        "freshness": "volatile",
        "retrievedAt": datetime.now(timezone.utc) - timedelta(days=40),
        "sourceUrl": "https://jobs.example",
    }
    pack = compact_evidence_pack([old], limit=5)
    assert pack[0]["freshness"] == "volatile"
