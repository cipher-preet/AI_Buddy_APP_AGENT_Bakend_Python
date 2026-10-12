from __future__ import annotations

from typing import Any

from services.goals.agents.llm import call_goal_agent
from services.goals.domains import context_profile, understanding_as_dict, _text
from services.goals.research import compact_evidence_pack
from services.goals.schemas import GoalGapItem, GoalGapResult

VALID_STATUS = {"SATISFIED", "PARTIAL", "MISSING", "UNKNOWN"}


def _profile_blob(goal: dict[str, Any]) -> str:
    profile = context_profile(goal)
    parts = [
        _text(goal.get("statement")),
        _text(goal.get("description")),
        _text(profile.get("currentRole")),
        _text(profile.get("startingLevel")),
        _text(profile.get("geography")),
        " ".join(str(item) for item in (goal.get("constraints") or [])),
        _text((goal.get("latestStep") or {}).get("prompt")),
        _text(goal.get("baselineCorrection")),
    ]
    return " ".join(part for part in parts if part).lower()


def heuristic_gaps(goal: dict[str, Any], evidence: list[dict[str, Any]]) -> GoalGapResult:
    blob = _profile_blob(goal)
    understanding = understanding_as_dict(goal.get("understanding"))
    unknown_start = _text(understanding.get("startingLevel")).upper() in {"", "UNKNOWN"}
    items: list[GoalGapItem] = []
    for row in evidence[:10]:
        fact = _text(row.get("fact"))
        if not fact:
            continue
        status = "UNKNOWN" if unknown_start else "UNKNOWN"
        items.append(
            GoalGapItem(
                requirement=fact[:400],
                status=status,
                evidence="Not confirmed in the user profile.",
                impact="HIGH" if row.get("kind") == "FACT" else "MEDIUM",
                sourceUrls=[str(row.get("sourceUrl") or row.get("url") or "")][:1],
            )
        )
    return GoalGapResult(
        unknown=items,
        pathwaySummary=(
            "Starting level is unknown, so requirements are listed as UNKNOWN rather than missing. "
            "The first plan should assess current state against researched requirements before long-range work."
        ),
        provisional=True or unknown_start or not evidence,
        highestImpact=[item.requirement for item in items[:3]],
    )


def _split(result: GoalGapResult) -> GoalGapResult:
    buckets = {"SATISFIED": [], "PARTIAL": [], "MISSING": [], "UNKNOWN": []}
    for group in (result.satisfied, result.partial, result.missing, result.unknown):
        for item in group:
            status = item.status if item.status in VALID_STATUS else "UNKNOWN"
            item.status = status
            buckets[status].append(item)
    result.satisfied = buckets["SATISFIED"][:8]
    result.partial = buckets["PARTIAL"][:8]
    result.missing = buckets["MISSING"][:8]
    result.unknown = buckets["UNKNOWN"][:8]
    return result


async def analyze_gaps(goal: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "statement": _text(goal.get("statement")),
        "understanding": understanding_as_dict(goal.get("understanding")),
        "contextProfile": context_profile(goal),
        "baseline": _text((goal.get("latestStep") or {}).get("prompt")),
        "baselineCorrection": _text(goal.get("baselineCorrection")),
        "evidence": compact_evidence_pack(evidence, limit=10),
    }
    called = await call_goal_agent(
        prompt_name="goal-gaps-v1",
        schema=GoalGapResult,
        payload=payload,
        stage="goal_gaps",
        temperature=0.1,
        max_tokens=1800,
    )
    if called is None:
        result = heuristic_gaps(goal, evidence)
    else:
        result, _model = called
        result = _split(result)
        if _text(understanding_as_dict(goal.get("understanding")).get("startingLevel")).upper() in {"", "UNKNOWN"}:
            # Never convert unknown user capabilities into missing.
            moved = []
            for item in result.missing:
                item.status = "UNKNOWN"
                moved.append(item)
            result.unknown = (result.unknown + moved)[:8]
            result.missing = []
            result.provisional = True
    return result.model_dump()
