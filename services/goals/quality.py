"""Make a generated goal plan shorter, practical, and harder to hallucinate.

This pass never adds new facts. It only shortens wording, drops filler, and
removes duplicates so the steps the user sees stay specific to their goal.
"""

from __future__ import annotations

import re
from typing import Any

from services.goals.domains import GENERIC_TASK_PHRASES

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://[^\s)]+")

# Inflated wording only. Domain terms the user actually needs are left alone.
_JARGON: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\blow-hanging fruit\b", re.I), "easiest useful task"),
    (re.compile(r"\bmove the needle\b", re.I), "change the result"),
    (re.compile(r"\bnorth star\b", re.I), "main target"),
    (re.compile(r"\bdeep dive\b", re.I), "close look"),
    (re.compile(r"\bcircle back\b", re.I), "come back"),
    (re.compile(r"\bactionable insights?\b", re.I), "clear next steps"),
    (re.compile(r"\bleverage\b", re.I), "use"),
    (re.compile(r"\bsynerg(?:y|ies)\b", re.I), "fit"),
    (re.compile(r"\bholistic\b", re.I), "full"),
    (re.compile(r"\brobust\b", re.I), "reliable"),
    (re.compile(r"\boptimize\b", re.I), "improve"),
    (re.compile(r"\bideate\b", re.I), "list ideas"),
    (re.compile(r"\bparadigm\b", re.I), "approach"),
    (re.compile(r"\bstakeholders\b", re.I), "people involved"),
    (re.compile(r"\bgo-to-market\b", re.I), "launch plan"),
    (re.compile(r"\bvertical slice\b", re.I), "first working version"),
)

_PROMISE = (
    "you will get the job",
    "guaranteed",
    "you have achieved",
    "this guarantees",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def plain_text(value: Any, *, max_chars: int, max_sentences: int = 2) -> str:
    """Collapse whitespace, swap inflated phrases, and keep the first sentences."""
    cleaned = _SPACE.sub(" ", _text(value))
    if not cleaned:
        return ""
    for pattern, replacement in _JARGON:
        cleaned = pattern.sub(replacement, cleaned)
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]
    if max_sentences > 0 and len(parts) > max_sentences:
        cleaned = " ".join(parts[:max_sentences])
    cleaned = cleaned.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:—-")
    if not cut:
        cut = cleaned[:max_chars].rstrip()
    if cut[-1:] not in ".!?":
        cut = f"{cut}."
    return cut


def _blob(*parts: Any) -> str:
    return " ".join(_text(part) for part in parts).lower()


def _is_filler(text: str) -> bool:
    blob = text.lower()
    if not blob:
        return True
    hits = [phrase for phrase in GENERIC_TASK_PHRASES if phrase in blob]
    if hits and len(blob) < 180:
        return True
    return any(phrase in blob for phrase in _PROMISE)


def _key(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"a", "an", "the", "to", "for", "of", "and", "or", "your", "you", "this", "that", "with"}
    return " ".join(word for word in words if word not in stop)


def _fresh(key: str, seen: set[str]) -> bool:
    if not key or key in seen:
        return False
    seen.add(key)
    return True


def _strip_unknown_urls(text: str, allowed: set[str]) -> str:
    if not text or "http" not in text.lower():
        return text

    def replace(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,)")
        if not allowed:
            return ""
        if url.rstrip("/") in allowed or any(url.rstrip("/") in item or item in url for item in allowed):
            return match.group(0)
        return ""

    cleaned = _URL.sub(replace, text)
    return _SPACE.sub(" ", cleaned).strip(" ,;")


def _allowed_urls(goal: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for item in goal.get("researchEvidencePack") or []:
        if isinstance(item, dict):
            url = _text(item.get("sourceUrl") or item.get("url")).rstrip("/")
            if url:
                urls.add(url)
    for item in goal.get("researchSources") or []:
        if isinstance(item, dict):
            url = _text(item.get("url")).rstrip("/")
            if url:
                urls.add(url)
    return urls


def tighten_plan(plan: dict[str, Any], goal: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy-safe tightened plan. Existing progress fields are kept."""
    goal = goal or {}
    allowed = _allowed_urls(goal)
    original_strategies = list(plan.get("strategies") or [])
    original_recs = list(plan.get("priorityRecommendations") or [])
    original_days = list(plan.get("days") or [])
    original_milestones = list(plan.get("milestones") or [])

    for field, limit, sentences in (
        ("summary", 420, 2),
        ("mission", 220, 2),
        ("recommendation", 360, 2),
        ("positioningHeadline", 90, 1),
        ("positioningBody", 240, 2),
        ("immediateNextStep", 180, 1),
        ("coachNote", 180, 1),
    ):
        if field in plan:
            plan[field] = plain_text(plan.get(field), max_chars=limit, max_sentences=sentences)

    seen_strategies: set[str] = set()
    strategies: list[dict[str, Any]] = []
    for item in original_strategies:
        if not isinstance(item, dict):
            continue
        title = plain_text(item.get("title"), max_chars=72, max_sentences=1)
        body = _strip_unknown_urls(plain_text(item.get("body"), max_chars=280, max_sentences=2), allowed)
        if not title or not body or _is_filler(f"{title} {body}"):
            continue
        if not _fresh(_key(title), seen_strategies):
            continue
        proof = []
        for point in item.get("proofPoints") or []:
            short = plain_text(point, max_chars=80, max_sentences=1)
            if short and not _is_filler(short):
                proof.append(short)
            if len(proof) >= 3:
                break
        strategies.append(
            {
                **item,
                "title": title,
                "body": body,
                "measurableGoal": plain_text(item.get("measurableGoal"), max_chars=120, max_sentences=1),
                "why": plain_text(item.get("why"), max_chars=160, max_sentences=1),
                "proofPoints": proof,
                "effort": plain_text(item.get("effort"), max_chars=60, max_sentences=1),
            }
        )
        if len(strategies) >= 4:
            break
    if not strategies and original_strategies:
        strategies = original_strategies[:3]
    plan["strategies"] = strategies

    seen_recs: set[str] = set()
    recommendations: list[dict[str, Any]] = []
    for item in original_recs:
        if not isinstance(item, dict):
            continue
        title = plain_text(item.get("title"), max_chars=80, max_sentences=1)
        reason = _strip_unknown_urls(plain_text(item.get("reason"), max_chars=220, max_sentences=2), allowed)
        if not title or not reason or _is_filler(f"{title} {reason}"):
            continue
        if not _fresh(_key(title), seen_recs):
            continue
        how_steps = []
        seen_steps: set[str] = set()
        for step in item.get("howSteps") or []:
            short = plain_text(step, max_chars=110, max_sentences=1)
            if not short or _is_filler(short) or not _fresh(_key(short), seen_steps):
                continue
            how_steps.append(short)
            if len(how_steps) >= 3:
                break
        recommendations.append(
            {
                **item,
                "title": title,
                "reason": reason,
                "howSteps": how_steps,
                "measurableGoal": plain_text(item.get("measurableGoal"), max_chars=120, max_sentences=1),
                "milestoneTitle": plain_text(item.get("milestoneTitle"), max_chars=80, max_sentences=1),
            }
        )
        if len(recommendations) >= 4:
            break
    if not recommendations and original_recs:
        recommendations = original_recs[:3]
    plan["priorityRecommendations"] = recommendations

    seen_days: set[str] = set()
    days: list[dict[str, Any]] = []
    for day in original_days:
        if not isinstance(day, dict):
            continue
        title = plain_text(day.get("title"), max_chars=80, max_sentences=1)
        if not title or not _fresh(_key(title), seen_days):
            continue
        tasks = []
        seen_tasks: set[str] = set()
        for task in day.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            task_text = _strip_unknown_urls(plain_text(task.get("text"), max_chars=140, max_sentences=1), allowed)
            if not task_text or _is_filler(task_text) or not _fresh(_key(task_text), seen_tasks):
                continue
            tasks.append(
                {
                    **task,
                    "text": task_text,
                    "why": plain_text(task.get("why"), max_chars=140, max_sentences=1),
                    "definitionOfDone": plain_text(task.get("definitionOfDone"), max_chars=140, max_sentences=1),
                }
            )
            if len(tasks) >= 4:
                break
        if not tasks and day.get("tasks"):
            tasks = list(day.get("tasks") or [])[:2]
        days.append(
            {
                **day,
                "title": title,
                "focus": plain_text(day.get("focus"), max_chars=140, max_sentences=1),
                "definitionOfDone": plain_text(day.get("definitionOfDone"), max_chars=140, max_sentences=1),
                "tasks": tasks,
            }
        )
        if len(days) >= 12:
            break
    if not days and original_days:
        days = original_days[:8]
    plan["days"] = days

    seen_milestones: set[str] = set()
    milestones: list[dict[str, Any]] = []
    for item in original_milestones:
        if not isinstance(item, dict):
            continue
        title = plain_text(item.get("title"), max_chars=90, max_sentences=1)
        if not title or not _fresh(_key(title), seen_milestones):
            continue
        criteria = []
        for criterion in item.get("successCriteria") or []:
            short = plain_text(criterion, max_chars=140, max_sentences=1)
            if short and not _is_filler(short):
                criteria.append(short)
            if len(criteria) >= 3:
                break
        milestones.append(
            {
                **item,
                "title": title,
                "successCriteria": criteria or list(item.get("successCriteria") or [])[:2],
                "dueLabel": plain_text(item.get("dueLabel"), max_chars=40, max_sentences=1),
            }
        )
        if len(milestones) >= 6:
            break
    if not milestones and original_milestones:
        milestones = original_milestones[:4]
    plan["milestones"] = milestones

    targets = []
    for item in plan.get("funnelTargets") or []:
        short = plain_text(item, max_chars=80, max_sentences=1)
        if short and not _is_filler(short):
            targets.append(short)
        if len(targets) >= 4:
            break
    if targets or not plan.get("funnelTargets"):
        plan["funnelTargets"] = targets
    return plan
