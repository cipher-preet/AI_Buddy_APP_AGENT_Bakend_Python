from __future__ import annotations

import re
from typing import Any

from services.goals.schemas import DOMAIN_CATEGORIES

GOAL_TYPES = {
    "product_launch",
    "skill",
    "freelance",
    "app_build",
    "sales",
    "habit",
    "career",
    "other",
}

SKIP_ANSWERS = {
    "skip",
    "not sure",
    "i don't know",
    "i dont know",
    "don't know",
    "dont know",
    "suggest for me",
    "suggest",
    "idk",
    "n/a",
    "na",
    "none",
    "unknown",
}

GENERIC_TASK_PHRASES = (
    "work harder",
    "improve yourself",
    "stay motivated",
    "make steady progress",
    "protect one deep-work block",
    "stay consistent",
    "believe in yourself",
    "hustle",
)

STARTUP_GTM_PHRASES = (
    "icp",
    "ideal customer profile",
    "pricing assumptions",
    "lead sourcing",
    "lead generation",
    "sales funnel",
    "mrr",
    "arr",
    "public beta",
    "landing page",
    "founder-led acquisition",
    "saas",
)

DOMAIN_TO_GOAL_TYPE = {
    "CAREER": "career",
    "EDUCATION": "skill",
    "BUSINESS": "sales",
    "TECHNOLOGY": "app_build",
    "FINANCIAL": "other",
    "HEALTH": "habit",
    "PRODUCTIVITY": "habit",
    "CREATIVE": "other",
    "RELATIONSHIPS": "other",
    "TRAVEL": "other",
    "OTHER": "other",
}

INTERVIEW_FIELDS = (
    "startingLevel",
    "currentRole",
    "geography",
    "availableHoursPerWeek",
    "timeframe",
    "successCriteria",
    "constraints",
)

PIPELINE_STATES = (
    "INTAKE",
    "NEEDS_CONTEXT",
    "READY_FOR_RESEARCH",
    "RESEARCHING",
    "ANALYZING_GAPS",
    "DRAFTING_PLAN",
    "VALIDATING",
    "READY",
    "ACTIVE",
    "BLOCKED",
    "NEEDS_REVISION",
    "ACHIEVED",
    "PAUSED",
    "FAILED",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalize_domain(value: str) -> str:
    token = _text(value).upper().replace(" ", "_")
    aliases = {
        "SKILL": "EDUCATION",
        "LEARNING": "EDUCATION",
        "EDUCATION_SKILL": "EDUCATION",
        "PROFESSIONAL": "CAREER",
        "JOB": "CAREER",
        "STARTUP": "BUSINESS",
        "ENTREPRENEURSHIP": "BUSINESS",
        "FITNESS": "HEALTH",
        "HABIT": "PRODUCTIVITY",
        "SOFTWARE": "TECHNOLOGY",
        "FINANCE": "FINANCIAL",
        "LIFESTYLE": "RELATIONSHIPS",
    }
    token = aliases.get(token, token)
    return token if token in DOMAIN_CATEGORIES else "OTHER"


def domain_to_goal_type(domain: str, subdomain: str = "", statement: str = "") -> str:
    domain_key = normalize_domain(domain)
    blob = f"{subdomain} {statement}".lower()
    if domain_key == "BUSINESS" and any(token in blob for token in ("saas", "startup", "public beta")):
        return "product_launch"
    if domain_key == "TECHNOLOGY" and any(token in blob for token in ("mobile app", "chrome extension", "desktop app")):
        return "app_build"
    return DOMAIN_TO_GOAL_TYPE.get(domain_key, "other")


def is_skip_answer(value: Any) -> bool:
    return _text(value).lower().strip(" .!?") in SKIP_ANSWERS


def detect_goal_type(statement: str, explicit: str = "", domain: str = "") -> str:
    """Keyword fallback only. Prefer explicit type or understanding.domain."""
    known = _text(explicit).lower()
    if known in GOAL_TYPES:
        return known
    if domain:
        mapped = domain_to_goal_type(domain, statement=statement)
        if mapped in GOAL_TYPES:
            return mapped
    text = statement.lower()
    career_hits = (
        "career",
        "promotion",
        "promote",
        "interview",
        "resume",
        " cv ",
        "get hired",
        "get a job",
        "land a job",
        "job title",
        "manager",
        "director",
        "officer",
        "become a",
        "become an",
        "role of",
        "target role",
    )
    if any(token in text for token in career_hits):
        if not any(token in text for token in ("saas", "startup", "chrome extension", "mobile app")):
            return "career"
    checks = [
        ("freelance", ("freelance", "client project", "gig", "contract", "upwork", "fiverr")),
        ("app_build", ("build an app", "mobile app", "complete app", "finish the app", "ios app", "android app")),
        ("product_launch", ("saas", "startup", "chrome extension", "desktop app", "public beta")),
        ("habit", ("habit", "every day", "daily routine", "daily habit")),
        ("skill", ("learn", "skill", "course", "master", "study", "from zero", "from scratch")),
        ("sales", ("close deals", "quota", "pipeline", "mrr", "arr", "lead generation")),
        ("career", ("job", "career", "promotion", "interview", "resume")),
    ]
    for goal_type, words in checks:
        if any(word in text for word in words):
            return goal_type
    if "launch" in text and any(word in text for word in ("product", "app", "saas", "startup")):
        return "product_launch"
    return "other"


def understanding_as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    return {}


def context_profile(goal: dict[str, Any]) -> dict[str, Any]:
    raw = goal.get("contextProfile")
    return dict(raw) if isinstance(raw, dict) else {}


def merge_context_profile(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing or {})
    answers = list(merged.get("answers") or [])
    unknown = list(merged.get("unknownFields") or [])
    for key, value in (updates or {}).items():
        if key in {"answers", "unknownFields"}:
            continue
        text = _text(value)
        if not text:
            continue
        if is_skip_answer(text):
            if key not in unknown:
                unknown.append(key)
            merged.setdefault(key, "UNKNOWN")
            continue
        merged[key] = text[:500]
        unknown = [item for item in unknown if item != key]
        answers.append({"field": key, "value": text[:500]})
    merged["answers"] = answers[-24:]
    merged["unknownFields"] = unknown[:16]
    return merged


def inferred_success(goal: dict[str, Any], understanding: dict[str, Any]) -> str:
    existing = _text(goal.get("successCriteria"))
    if existing:
        return existing
    outcome = _text(understanding.get("targetOutcome"))
    primary = _text(understanding.get("primaryGoal")) or _text(goal.get("statement"))
    if outcome:
        return outcome[:1000]
    if primary:
        return f"The stated outcome is achieved with evidence: {primary}"[:1000]
    return ""


_ROLE_RE = re.compile(
    r"\b(?:become|becoming|as a|as an)\s+(?:an?\s+)?([A-Za-z][A-Za-z0-9 /&-]{2,80})",
    re.I,
)


def inferred_title(goal: dict[str, Any], understanding: dict[str, Any]) -> str:
    title = _text(goal.get("title"))
    if title and title.lower() not in {"new goal", "goal"}:
        return title
    primary = _text(understanding.get("primaryGoal"))
    if primary:
        return primary[:160]
    match = _ROLE_RE.search(_text(goal.get("statement")))
    if match:
        return match.group(0).strip()[:160]
    return _text(goal.get("statement"))[:80] or "Goal"
