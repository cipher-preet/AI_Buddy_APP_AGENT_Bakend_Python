from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from services.goals.domains import GENERIC_TASK_PHRASES, STARTUP_GTM_PHRASES, normalize_domain, understanding_as_dict, _text
from services.goals.schemas import GoalValidationResult, ValidationIssue

BUSINESS_DOMAINS = {"BUSINESS"}
GTM_OK_DOMAINS = {"BUSINESS"}


def _plan_text(plan: dict[str, Any]) -> str:
    chunks: list[str] = [
        _text(plan.get("summary")),
        _text(plan.get("mission")),
        _text(plan.get("recommendation")),
        _text(plan.get("immediateNextStep")),
        " ".join(str(item) for item in (plan.get("funnelTargets") or [])),
    ]
    for milestone in plan.get("milestones") or []:
        chunks.append(_text(milestone.get("title")))
        chunks.extend(_text(item) for item in (milestone.get("successCriteria") or []))
    for day in plan.get("days") or []:
        chunks.append(_text(day.get("title")))
        for task in day.get("tasks") or []:
            chunks.append(_text(task.get("text")))
            chunks.append(_text(task.get("why")))
    for strategy in plan.get("strategies") or []:
        chunks.append(_text(strategy.get("title")))
        chunks.append(_text(strategy.get("body")))
    for rec in plan.get("priorityRecommendations") or []:
        chunks.append(_text(rec.get("title")))
        chunks.append(_text(rec.get("reason")))
    return " ".join(chunks).lower()


def validate_plan(
    goal: dict[str, Any],
    plan: dict[str, Any],
    *,
    evidence: list[dict[str, Any]] | None = None,
    gaps: dict[str, Any] | None = None,
) -> GoalValidationResult:
    issues: list[ValidationIssue] = []
    understanding = understanding_as_dict(goal.get("understanding") or plan.get("understanding"))
    domain = normalize_domain(str(understanding.get("domain") or ""))
    blob = _plan_text(plan)

    if domain and domain not in GTM_OK_DOMAINS:
        hits = [phrase for phrase in STARTUP_GTM_PHRASES if phrase in blob]
        # "saas" inside a SaaS-learning career/education goal can still be valid if domain is TECHNOLOGY/EDUCATION
        # and the statement itself contains it. Only fail hard when the plan injects GTM jargon
        # the user did not mention.
        statement = _text(goal.get("statement")).lower()
        injected = [phrase for phrase in hits if phrase not in statement]
        if injected:
            issues.append(
                ValidationIssue(
                    code="domain_mismatch",
                    severity="error",
                    message=f"Plan contains unrelated go-to-market language: {', '.join(injected[:4])}.",
                    target="milestones",
                )
            )

    milestone_titles = [_text(item.get("title")).lower() for item in (plan.get("milestones") or []) if _text(item.get("title"))]
    seen: set[str] = set()
    for title in milestone_titles:
        if title in seen:
            issues.append(ValidationIssue(code="duplicate_title", message=f"Duplicate title: {title}", target="milestones"))
            break
        seen.add(title)

    milestones = plan.get("milestones") or []
    if not milestones:
        issues.append(ValidationIssue(code="missing_milestones", message="Plan has no milestones.", target="milestones"))
    for item in milestones:
        if not (item.get("successCriteria") or []):
            issues.append(
                ValidationIssue(
                    code="missing_success_criteria",
                    message=f"Milestone “{item.get('title')}” has no success criteria.",
                    target="milestones",
                )
            )

    generic_hits = [phrase for phrase in GENERIC_TASK_PHRASES if phrase in blob]
    if len(generic_hits) >= 2:
        issues.append(
            ValidationIssue(
                code="generic_filler",
                severity="error",
                message="Plan relies on generic motivational filler instead of goal-specific actions.",
                target="tasks",
            )
        )

    hours = goal.get("availableHoursPerWeek")
    day_hours = [day.get("hours") for day in (plan.get("days") or []) if day.get("hours")]
    if hours is not None and day_hours:
        try:
            weekly = float(hours)
            if weekly > 0 and any(float(item) > weekly * 2 for item in day_hours if item is not None):
                issues.append(
                    ValidationIssue(
                        code="capacity",
                        severity="warning",
                        message="A scheduled block exceeds twice the user's weekly hours.",
                        target="days",
                    )
                )
        except (TypeError, ValueError):
            pass

    allowed_urls = {str(item.get("sourceUrl") or item.get("url") or "") for item in (evidence or []) if item}
    citation_re = re.compile(r"https?://[^\s)]+")
    for match in citation_re.findall(blob):
        host = urlparse(match).hostname or ""
        if allowed_urls and match.rstrip("/") not in {url.rstrip("/") for url in allowed_urls if url}:
            if host and not any(host in url for url in allowed_urls):
                issues.append(
                    ValidationIssue(
                        code="unsupported_citation",
                        severity="error",
                        message="Plan cites a URL that is not in the research evidence set.",
                        target="sources",
                    )
                )
                break

    names = [_text(item.get("title")) for item in milestones]
    for item in milestones:
        for dep in item.get("dependencies") or []:
            if dep and dep not in names:
                issues.append(
                    ValidationIssue(
                        code="bad_dependency",
                        severity="warning",
                        message=f"Milestone “{item.get('title')}” depends on unknown “{dep}”.",
                        target="dependencies",
                    )
                )

    if "you will get the job" in blob or "guaranteed" in blob or "you have achieved" in blob:
        issues.append(
            ValidationIssue(
                code="false_outcome",
                severity="error",
                message="Plan promises an external outcome or treats preparation as achievement.",
                target="summary",
            )
        )

    if gaps and gaps.get("provisional"):
        issues.append(
            ValidationIssue(
                code="insufficient_context",
                severity="warning",
                message="User starting state is incomplete; plan must stay provisional.",
                target="gaps",
            )
        )

    errors = [item for item in issues if item.severity == "error"]
    provisional = bool(gaps and gaps.get("provisional")) or any(item.code == "insufficient_context" for item in issues)
    return GoalValidationResult(
        accepted=not errors,
        provisional=provisional,
        issues=issues,
    )


def strip_mismatched_items(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    """Drop clearly off-domain milestone/task titles when validation fails."""
    understanding = understanding_as_dict(goal.get("understanding") or plan.get("understanding"))
    domain = normalize_domain(str(understanding.get("domain") or ""))
    if domain in GTM_OK_DOMAINS:
        return plan
    statement = _text(goal.get("statement")).lower()

    def allowed(text: str) -> bool:
        blob = text.lower()
        return not any(phrase in blob and phrase not in statement for phrase in STARTUP_GTM_PHRASES)

    plan["milestones"] = [item for item in (plan.get("milestones") or []) if allowed(_text(item.get("title")))]
    cleaned_days = []
    for day in plan.get("days") or []:
        if not allowed(_text(day.get("title"))):
            continue
        tasks = [task for task in (day.get("tasks") or []) if allowed(_text(task.get("text")))]
        day = {**day, "tasks": tasks}
        cleaned_days.append(day)
    plan["days"] = cleaned_days
    plan["strategies"] = [item for item in (plan.get("strategies") or []) if allowed(_text(item.get("title")) + " " + _text(item.get("body")))]
    plan["priorityRecommendations"] = [
        item for item in (plan.get("priorityRecommendations") or []) if allowed(_text(item.get("title")) + " " + _text(item.get("reason")))
    ]
    return plan
