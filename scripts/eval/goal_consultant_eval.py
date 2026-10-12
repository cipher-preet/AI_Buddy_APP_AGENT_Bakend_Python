"""Fixture rubric eval for the goal consultant. Does not call paid APIs."""

from __future__ import annotations

import json
from pathlib import Path

from services.goals.domains import detect_goal_type, domain_to_goal_type, normalize_domain
from services.goals.plan import _fallback_plan
from services.goals.understanding import empty_understanding
from services.goals.validator import validate_plan

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = ROOT / "tests" / "eval" / "goal_scenarios.json"

BANNED = ("icp", "pricing assumptions", "lead sourcing", "sales funnel", "founder-led")


def load_scenarios() -> list[dict]:
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


def score_scenario(row: dict) -> dict[str, float]:
    statement = row["statement"]
    understanding = empty_understanding(statement)
    expected_domain = row.get("expectedDomain")
    if expected_domain:
        understanding["domain"] = expected_domain
        understanding["goalType"] = domain_to_goal_type(expected_domain, statement=statement)
    if row.get("startingLevel"):
        understanding["startingLevel"] = row["startingLevel"]
    goal = {
        "statement": statement,
        "title": row.get("title") or statement[:80],
        "understanding": understanding,
        "contextProfile": row.get("context") or {},
        "successCriteria": row.get("success") or understanding.get("targetOutcome"),
    }
    detected = detect_goal_type(statement, domain=str(understanding.get("domain") or ""))
    domain_ok = 1.0
    if expected_domain:
        domain_ok = 1.0 if normalize_domain(understanding["domain"]) == expected_domain else 0.0
        if expected_domain == "CAREER" and detected != "career":
            domain_ok = 0.0
    plan = _fallback_plan(goal)
    blob = str(plan).lower()
    if expected_domain and expected_domain not in {"BUSINESS"}:
        relevance = 0.0 if any(token in blob for token in BANNED) else 1.0
    else:
        relevance = 1.0
    report = validate_plan(goal, plan)
    mismatch = 0.0 if any(item.code == "domain_mismatch" for item in report.issues) else 1.0
    actionability = 1.0 if (plan.get("milestones") and plan.get("days")) else 0.0
    personalization = 1.0
    if row.get("startingLevel") and row["startingLevel"] != "UNKNOWN":
        personalization = 1.0 if row["startingLevel"].lower() in blob or row.get("context", {}).get("currentRole", "").lower() in blob or plan.get("provisional") is False else 0.6
    return {
        "domain": domain_ok,
        "relevance": min(relevance, mismatch),
        "actionability": actionability,
        "personalization": personalization,
        "duplicates": 0.0 if any(item.code == "duplicate_title" for item in report.issues) else 1.0,
    }


def main() -> None:
    rows = load_scenarios()
    assert len(rows) >= 100, f"need at least 100 scenarios, got {len(rows)}"
    totals = {"domain": 0.0, "relevance": 0.0, "actionability": 0.0, "personalization": 0.0, "duplicates": 0.0}
    failures = []
    for row in rows:
        scores = score_scenario(row)
        for key, value in scores.items():
            totals[key] += value
        if scores["relevance"] < 1 or scores["domain"] < 1:
            failures.append({"id": row.get("id"), "scores": scores, "statement": row["statement"]})
    n = len(rows)
    print(json.dumps({"count": n, "averages": {key: round(value / n, 3) for key, value in totals.items()}, "failures": failures[:12]}, indent=2))


if __name__ == "__main__":
    main()
