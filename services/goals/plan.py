from __future__ import annotations

from typing import Any
from uuid import uuid4

from services.goals.agents.pipeline import (
    is_launch_style_goal,
    run_expert_plan_pipeline,
    wants_expert_pipeline,
)
from services.goals.domains import detect_goal_type
from services.goals.schemas import (
    GoalExecutionResult,
    GoalStrategistResult,
)

ALLOWED = {"NUMERIC", "MILESTONE", "HABIT", "BINARY", "QUALITATIVE"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _slug(prefix: str, index: int) -> str:
    return f"{prefix}-{index}-{uuid4().hex[:6]}"


def _milestone(
    title: str,
    criteria: list[str],
    due: str,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "title": title[:160],
        "successCriteria": [item[:240] for item in criteria if item.strip()][:4],
        "dueLabel": due[:80],
        "dependencies": [item[:160] for item in (dependencies or []) if item.strip()][:4],
    }


def _normalize_milestones(raw_milestones: list[Any], timeframe: str) -> list[dict[str, Any]]:
    milestones: list[dict[str, Any]] = []
    for item in raw_milestones:
        title = _text(getattr(item, "title", None) if not isinstance(item, dict) else item.get("title"))
        if not title:
            continue
        if isinstance(item, dict):
            criteria = [_text(text) for text in (item.get("successCriteria") or []) if _text(text)]
            due = _text(item.get("dueLabel")) or timeframe or "Later"
            deps = [_text(text) for text in (item.get("dependencies") or []) if _text(text)]
        else:
            criteria = [_text(text) for text in (item.successCriteria or []) if _text(text)]
            due = _text(item.dueLabel) or timeframe or "Later"
            deps = [_text(text) for text in (item.dependencies or []) if _text(text)]
        if not criteria:
            criteria = ["This phase’s outcome is completed and can be shown"]
        milestones.append(_milestone(title, criteria, due, deps))
        if len(milestones) >= 8:
            break
    return milestones


def _normalize_days(raw_days: list[Any]) -> list[dict[str, Any]]:
    days: list[dict[str, Any]] = []
    for index, item in enumerate(raw_days or [], start=1):
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
        title = _text(data.get("title"))
        if not title:
            continue
        tasks: list[dict[str, Any]] = []
        for task_index, task in enumerate(data.get("tasks") or [], start=1):
            task_data = task.model_dump() if hasattr(task, "model_dump") else dict(task or {})
            text = _text(task_data.get("text"))
            if not text:
                continue
            tasks.append(
                {
                    "id": _text(task_data.get("id")) or _slug(f"d{index}-t", task_index),
                    "text": text[:240],
                    "done": bool(task_data.get("done")),
                    "why": _text(task_data.get("why"))[:480],
                    "definitionOfDone": _text(task_data.get("definitionOfDone"))[:320],
                    "note": _text(task_data.get("note"))[:2000],
                }
            )
        if not tasks:
            continue
        days.append(
            {
                "dayIndex": int(data.get("dayIndex") or index),
                "dateLabel": _text(data.get("dateLabel"))[:80] or f"Day {index}",
                "title": title[:160],
                "focus": _text(data.get("focus"))[:320],
                "hours": data.get("hours"),
                "tasks": tasks[:8],
                "definitionOfDone": _text(data.get("definitionOfDone"))[:320],
            }
        )
        if len(days) >= 16:
            break
    return _enrich_day_explanations(days)


def _enrich_day_explanations(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill only truly missing fields with short, non-duplicative defaults."""
    seen_why: set[str] = set()
    seen_done: set[str] = set()
    for day in days:
        title = _text(day.get("title")) or "this day"
        day_done = _text(day.get("definitionOfDone"))
        if not _text(day.get("focus")):
            day["focus"] = f"Finish the key work for “{title}”."
        for task in day.get("tasks") or []:
            text = _text(task.get("text"))
            if not text:
                continue
            why = _text(task.get("why"))
            if not why:
                why = f"Needed for “{title}”."
                # Keep defaults unique enough that drawers don't look copy-pasted.
                if why in seen_why:
                    why = f"Unlocks the next step after “{text[:80]}”."
                task["why"] = why[:480]
            seen_why.add(_text(task.get("why")))
            done = _text(task.get("definitionOfDone"))
            if not done:
                done = f"Done when “{text[:120]}” is finished and checkable."
                if day_done and done in seen_done:
                    done = f"Done when it clearly helps reach: {day_done[:160]}"
                task["definitionOfDone"] = done[:320]
            seen_done.add(_text(task.get("definitionOfDone")))
    return days


def _normalize_strategies(raw: list[Any]) -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    for index, item in enumerate(raw or [], start=1):
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
        title = _text(data.get("title"))
        body = _text(data.get("body"))
        if not title or not body:
            continue
        tags = [_text(tag) for tag in (data.get("tags") or []) if _text(tag)][:6]
        proof_points = [_text(point) for point in (data.get("proofPoints") or []) if _text(point)][:6]
        strategies.append(
            {
                "id": _text(data.get("id")) or _slug("strategy", index),
                "title": title[:160],
                "body": body[:2200],
                "measurableGoal": _text(data.get("measurableGoal"))[:320],
                "why": _text(data.get("why"))[:600],
                "proofPoints": proof_points,
                "effort": _text(data.get("effort"))[:240],
                "tags": tags,
            }
        )
        if len(strategies) >= 8:
            break
    return strategies


def _normalize_priority_recommendations(raw: list[Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for index, item in enumerate(raw or [], start=1):
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
        title = _text(data.get("title"))
        reason = _text(data.get("reason"))
        if not title or not reason:
            continue
        how_steps = [_text(step) for step in (data.get("howSteps") or []) if _text(step)][:6]
        priority = _text(data.get("priority")).upper() or "HIGH"
        if priority not in {"FIRST", "HIGH", "MEDIUM", "LOW"}:
            priority = "HIGH"
        minutes = data.get("suggestedDurationMinutes")
        try:
            minutes_value = int(minutes) if minutes is not None else None
        except (TypeError, ValueError):
            minutes_value = None
        recommendations.append(
            {
                "id": _text(data.get("id")) or _slug("rec", index),
                "title": title[:180],
                "reason": reason[:1200],
                "priority": priority,
                "howSteps": how_steps,
                "measurableGoal": _text(data.get("measurableGoal"))[:320],
                "milestoneTitle": _text(data.get("milestoneTitle"))[:180],
                "suggestedDurationMinutes": minutes_value,
                "relatedStrategyId": _text(data.get("relatedStrategyId"))[:80],
            }
        )
        if len(recommendations) >= 10:
            break
    return recommendations


def _normalize_checklists(raw: list[Any]) -> list[dict[str, Any]]:
    checklists: list[dict[str, Any]] = []
    for index, item in enumerate(raw or [], start=1):
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
        title = _text(data.get("title"))
        if not title:
            continue
        items: list[dict[str, Any]] = []
        for item_index, row in enumerate(data.get("items") or [], start=1):
            row_data = row.model_dump() if hasattr(row, "model_dump") else dict(row or {})
            text = _text(row_data.get("text"))
            if not text:
                continue
            items.append(
                {
                    "id": _text(row_data.get("id")) or _slug(f"c{index}", item_index),
                    "text": text[:240],
                    "done": bool(row_data.get("done")),
                }
            )
        if not items:
            continue
        checklists.append(
            {
                "id": _text(data.get("id")) or _slug("checklist", index),
                "title": title[:160],
                "items": items[:24],
            }
        )
        if len(checklists) >= 4:
            break
    return checklists


def _normalize_scorecard(raw: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw or []:
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
        area = _text(data.get("area"))
        criteria = _text(data.get("criteria"))
        if not area or not criteria:
            continue
        rows.append({"area": area[:80], "criteria": criteria[:320]})
        if len(rows) >= 10:
            break
    return rows


def _normalize_phases(raw: list[Any]) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for index, item in enumerate(raw or [], start=1):
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
        title = _text(data.get("title"))
        if not title:
            continue
        phases.append(
            {
                "id": _text(data.get("id")) or _slug("phase", index),
                "title": title[:160],
                "windowLabel": _text(data.get("windowLabel"))[:80],
                "outcome": _text(data.get("outcome"))[:320],
            }
        )
        if len(phases) >= 6:
            break
    return phases


def _milestones_from_phases(phases: list[dict[str, Any]], timeframe: str, criteria: str) -> list[dict[str, Any]]:
    milestones: list[dict[str, Any]] = []
    previous: list[str] = []
    for phase in phases:
        title = phase["title"]
        milestones.append(
            _milestone(
                title,
                [phase.get("outcome") or f"{title} is complete"],
                phase.get("windowLabel") or timeframe,
                previous[-1:],
            )
        )
        previous.append(title)
    if milestones and criteria and criteria.lower() not in " ".join(milestones[-1]["successCriteria"]).lower():
        milestones[-1]["successCriteria"] = [criteria, *milestones[-1]["successCriteria"]][:4]
    return milestones


def _is_generic_plan(milestones: list[dict[str, Any]]) -> bool:
    joined = " ".join(item["title"].lower() for item in milestones)
    generic_markers = (
        "confirm the starting point",
        "do the first meaningful work",
        "check the result",
        "specify the first shippable version",
        "build the vertical slice",
        "harden and polish",
    )
    hits = sum(1 for marker in generic_markers if marker in joined)
    return hits >= 2


def _is_thin_expert(plan: dict[str, Any]) -> bool:
    days = plan.get("days") or []
    strategies = plan.get("strategies") or []
    recommendations = plan.get("priorityRecommendations") or []
    milestones = plan.get("milestones") or []
    task_count = sum(len(day.get("tasks") or []) for day in days)
    thin_strategies = len(strategies) > 0 and all(len(_text(item.get("body"))) < 40 for item in strategies)
    return (not milestones and task_count < 2) or (len(strategies) < 1 and len(recommendations) < 1) or thin_strategies


def _degraded_plan(goal: dict[str, Any]) -> dict[str, Any]:
    """Provisional path from understanding + gaps when agents fail. No domain playbook."""
    from services.goals.domains import inferred_success, inferred_title, understanding_as_dict

    understanding = understanding_as_dict(goal.get("understanding"))
    title = inferred_title(goal, understanding)
    statement = _text(goal.get("statement"))
    timeframe = _text(goal.get("timeframe")) or "an estimated horizon after the baseline is known"
    criteria = inferred_success(goal, understanding) or "The stated outcome is evidenced in the real world, not only by checklist items."
    hours = goal.get("availableHoursPerWeek")
    hours_note = f" Fit work into about {hours} hours per week." if hours is not None else ""
    gaps = goal.get("gaps") if isinstance(goal.get("gaps"), dict) else {}
    unknown = [item.get("requirement") for item in (gaps.get("unknown") or []) if isinstance(item, dict)]
    missing = [item.get("requirement") for item in (gaps.get("missing") or []) if isinstance(item, dict)]
    focus = (missing or unknown or [f"Compare current state to requirements for {title}"]) [0]
    milestones = [
        _milestone(
            f"Assess current state for {title}",
            [f"Document current level against researched requirements for {statement or title}", "Label unknowns instead of inventing skills"],
            "First",
        ),
        _milestone(
            "Close the highest-impact verified gap",
            [str(focus)[:240], "Produce one observable artifact or credential step"],
            "Next",
            [f"Assess current state for {title}"],
        ),
        _milestone(
            "Build evidence of readiness",
            ["Keep a dated log of artifacts, feedback, or results that a third party could inspect"],
            "Ongoing",
            ["Close the highest-impact verified gap"],
        ),
        _milestone(
            f"Pursue the real-world outcome for {title}",
            [criteria, "Do not mark this complete until the external success condition is true"],
            timeframe,
            ["Build evidence of readiness"],
        ),
    ]
    days = []
    for index, milestone in enumerate(milestones, start=1):
        days.append(
            {
                "dayIndex": index,
                "dateLabel": milestone["dueLabel"],
                "title": milestone["title"],
                "hours": None,
                "tasks": [
                    {
                        "id": f"m{index}-t{task_i}",
                        "text": text,
                        "done": False,
                        "why": "This is the next honest step given current evidence.",
                        "definitionOfDone": text,
                    }
                    for task_i, text in enumerate(milestone["successCriteria"], start=1)
                ],
                "definitionOfDone": milestone["successCriteria"][0],
            }
        )
    summary = (
        f"Provisional roadmap for “{title}”. Starting level is "
        f"{understanding.get('startingLevel') or 'UNKNOWN'}. Research and user context drive the sequence."
        f"{hours_note}"
    )
    return {
        "version": 1,
        "measurementType": "MILESTONE",
        "summary": summary,
        "mission": _text(understanding.get("targetOutcome")) or f"Make honest progress toward {title}.",
        "recommendation": (
            "Start by comparing your current state with researched requirements. "
            "Do not treat preparation tasks as the final outcome."
        ),
        "positioningHeadline": f"A personalized path toward {title}",
        "positioningBody": "The plan stays provisional until key context and evidence are confirmed.",
        "funnelTargets": ["Current-state assessment written", "Top gap evidenced", "Readiness artifacts stored", "Real-world success condition checked separately"],
        "immediateNextStep": "Write a short gap assessment using current requirements, not generic productivity tasks.",
        "successDefinition": {"deadlineLabel": timeframe, "metrics": [{"label": "Success", "target": criteria}]},
        "milestones": milestones,
        "phases": [
            {"id": f"phase-{index}", "title": item["title"], "windowLabel": item["dueLabel"], "outcome": item["successCriteria"][0]}
            for index, item in enumerate(milestones, start=1)
        ],
        "days": days,
        "strategies": [
            {
                "id": "strategy-evidence",
                "title": "Ground every stage in evidence",
                "body": f"Use researched requirements for “{title}” and mark unknown capabilities as unknown.",
                "measurableGoal": "A written gap table with SATISFIED / PARTIAL / MISSING / UNKNOWN",
                "why": "Invented skills create the wrong roadmap.",
                "proofPoints": ["Gap table", "Sources listed", "Unknowns labeled"],
                "effort": "One focused assessment pass",
                "tags": ["evidence"],
            },
            {
                "id": "strategy-sequence",
                "title": "Sequence from the actual starting point",
                "body": "Only add earlier stages when the current level requires them. Skip work the user already proved.",
                "measurableGoal": "Each milestone maps to a verified gap or a real-world outcome gate",
                "why": "Copied templates ignore the user.",
                "proofPoints": ["Starting level recorded", "Dependencies respected"],
                "effort": "Planning review",
                "tags": ["personalization"],
            },
        ],
        "priorityRecommendations": [
            {
                "id": "rec-assess",
                "title": "Complete a requirement-by-requirement gap assessment",
                "reason": "The next work depends on what is already true, not on a generic calendar.",
                "priority": "FIRST",
                "howSteps": [
                    "List researched requirements",
                    "Mark each SATISFIED, PARTIAL, MISSING, or UNKNOWN",
                    "Pick the highest-impact UNKNOWN or MISSING item",
                ],
                "measurableGoal": "Gap assessment saved",
                "milestoneTitle": milestones[0]["title"],
                "suggestedDurationMinutes": 60,
                "relatedStrategyId": "strategy-evidence",
            }
        ],
        "checklists": [
            {
                "id": "checklist-honesty",
                "title": "Honesty checks",
                "items": [
                    {"id": "ch-1", "text": "Unknown capabilities are labeled UNKNOWN", "done": False},
                    {"id": "ch-2", "text": "No unrelated domain playbook items remain", "done": False},
                    {"id": "ch-3", "text": "Final outcome is tracked separately from task completion", "done": False},
                ],
            }
        ],
        "scorecard": [{"area": "Readiness", "criteria": "Verified requirements vs current state"}, {"area": "Outcome", "criteria": criteria}],
        "assumptions": [item.get("text") for item in (understanding.get("assumptions") or []) if isinstance(item, dict)][:4],
        "openQuestions": [],
        "provisional": True,
        "model": "fallback:degraded",
        "agentTrace": ["fallback"],
    }


def _launch_fallback_plan(goal: dict[str, Any]) -> dict[str, Any]:
    title = _text(goal.get("title")) or "your product"
    statement = _text(goal.get("statement"))
    timeframe = _text(goal.get("timeframe")) or "your launch window"
    criteria = _text(goal.get("successCriteria")) or "Real users complete the core journey successfully"
    hours = goal.get("availableHoursPerWeek")
    hours_note = f" Fit daily work into about {hours} hours per week." if hours is not None else ""
    description = _text(goal.get("description")) or title

    days = [
        {
            "dayIndex": 1,
            "dateLabel": "Day 1",
            "title": "Audit + scope freeze",
            "hours": 8,
            "tasks": [
                {"id": "d1-t1", "text": f"Smoke-test the core journey for {description}", "done": False},
                {"id": "d1-t2", "text": "Write a P0/P1/P2 backlog of launch blockers", "done": False},
                {"id": "d1-t3", "text": "Choose the primary launch platform and freeze MVP scope", "done": False},
            ],
            "definitionOfDone": "One prioritized backlog and no untracked launch blockers",
        },
        {
            "dayIndex": 2,
            "dateLabel": "Day 2",
            "title": "Core journey reliability",
            "hours": 8,
            "tasks": [
                {"id": "d2-t1", "text": f"Fix the steps that currently break the core journey for {description}", "done": False},
                {"id": "d2-t2", "text": "Verify work is not lost on failure or retry", "done": False},
                {"id": "d2-t3", "text": "Test a short and a realistic full run", "done": False},
            ],
            "definitionOfDone": "The core journey works reliably on the primary platform",
        },
        {
            "dayIndex": 3,
            "dateLabel": "Day 3",
            "title": "Pipeline verification",
            "hours": 8,
            "tasks": [
                {"id": "d3-t1", "text": f"Run the core {description} journey end-to-end", "done": False},
                {"id": "d3-t2", "text": "Check queues/workers and failure/retry UX", "done": False},
                {"id": "d3-t3", "text": "Kill duplicate/hallucinated action items in sample output", "done": False},
            ],
            "definitionOfDone": "One verified end-to-end pipeline with trustworthy outputs",
        },
        {
            "dayIndex": 4,
            "dateLabel": "Day 4",
            "title": "Onboarding + result navigation",
            "hours": 8,
            "tasks": [
                {"id": "d4-t1", "text": "Improve signup, empty states, and error recovery", "done": False},
                {"id": "d4-t2", "text": "Make meeting history and copy/export obvious", "done": False},
                {"id": "d4-t3", "text": "Do a first-run test without coaching", "done": False},
            ],
            "definitionOfDone": "A new user can finish one full journey unaided",
        },
        {
            "dayIndex": 5,
            "dateLabel": "Day 5",
            "title": "Release candidate + ops",
            "hours": 8,
            "tasks": [
                {"id": "d5-t1", "text": "Privacy/consent copy, logging, and basic analytics are in place", "done": False},
                {"id": "d5-t2", "text": "Deploy a release candidate and verify installs/downloads", "done": False},
                {"id": "d5-t3", "text": "Check store/certification status if applicable", "done": False},
            ],
            "definitionOfDone": "Release candidate ready for private beta",
        },
        {
            "dayIndex": 6,
            "dateLabel": "Day 6–7",
            "title": "First real testers",
            "hours": 4,
            "tasks": [
                {"id": "d6-t1", "text": "Invite 5–8 real users and watch them complete the journey", "done": False},
                {"id": "d6-t2", "text": "Capture failures and rank fix order", "done": False},
                {"id": "d6-t3", "text": "Ship only launch-blocking fixes", "done": False},
            ],
            "definitionOfDone": "Actual customer evidence guides the next week",
        },
        {
            "dayIndex": 8,
            "dateLabel": "Day 8–9",
            "title": "Stabilize beta",
            "hours": 8,
            "tasks": [
                {"id": "d8-t1", "text": "Fix high-severity beta problems", "done": False},
                {"id": "d8-t2", "text": "Expand to 10–15 testers", "done": False},
                {"id": "d8-t3", "text": "Validate longer sessions if relevant (15/30/60 min)", "done": False},
            ],
            "definitionOfDone": "Stable private beta with zero known P0 blockers",
        },
        {
            "dayIndex": 10,
            "dateLabel": "Day 10–11",
            "title": "Landing + outreach",
            "hours": 8,
            "tasks": [
                {"id": "d10-t1", "text": "Finalize landing page with real screenshots/demo and privacy explanation", "done": False},
                {"id": "d10-t2", "text": "Contact 30 relevant professionals for early access", "done": False},
                {"id": "d10-t3", "text": "Prepare launch posts and support channel", "done": False},
            ],
            "definitionOfDone": "Conversion-ready site and a list of interested users",
        },
        {
            "dayIndex": 12,
            "dateLabel": "Day 12–13",
            "title": "Go / no-go",
            "hours": 4,
            "tasks": [
                {"id": "d12-t1", "text": "Freeze release and run final regression", "done": False},
                {"id": "d12-t2", "text": "Confirm only approved platforms will be advertised", "done": False},
                {"id": "d12-t3", "text": "Make an explicit go/no-go decision against the scorecard", "done": False},
            ],
            "definitionOfDone": "Approved release build or intentional delay of unsafe clients",
        },
        {
            "dayIndex": 14,
            "dateLabel": "Launch day",
            "title": "Public beta launch",
            "hours": 8,
            "tasks": [
                {"id": "d14-t1", "text": "Publish the approved platform(s) and announce availability", "done": False},
                {"id": "d14-t2", "text": "Onboard early users and monitor processing/support", "done": False},
                {"id": "d14-t3", "text": f"Track progress toward: {criteria}", "done": False},
            ],
            "definitionOfDone": f"Live launch for {title} on at least one proven platform",
        },
    ]

    phases = [
        {
            "id": "phase-1",
            "title": "Product readiness and quality",
            "windowLabel": "Week 1",
            "outcome": "Reliable core journey and release candidate",
        },
        {
            "id": "phase-2",
            "title": "Beta validation and launch prep",
            "windowLabel": "Week 2",
            "outcome": "Real testers, launch assets, go/no-go",
        },
    ]
    milestones = [
        _milestone("Freeze MVP and audit blockers", ["Prioritized backlog and primary platform chosen"], "Day 1"),
        _milestone(
            "Make the core journey trustworthy",
            ["The primary user journey works end-to-end"],
            "Days 2–4",
            ["Freeze MVP and audit blockers"],
        ),
        _milestone(
            "Ship a private beta release candidate",
            ["Ops, privacy, and install paths are ready"],
            "Day 5",
            ["Make the core journey trustworthy"],
        ),
        _milestone(
            "Validate with real users",
            ["5–15 testers complete the journey; P0s fixed"],
            "Days 6–9",
            ["Ship a private beta release candidate"],
        ),
        _milestone(
            "Prepare acquisition and launch assets",
            ["Landing page, outreach, and support channel ready"],
            "Days 10–11",
            ["Validate with real users"],
        ),
        _milestone(
            f"Launch {title}",
            [criteria, "Only approved platforms are advertised"],
            timeframe,
            ["Prepare acquisition and launch assets"],
        ),
    ]
    strategies = [
        {
            "id": "strategy-segment",
            "title": "Start with one customer segment",
            "body": (
                f"For “{title}”, recruit a narrow first audience with a clear pain around the core journey: "
                f"{statement or description}. Expand only after 10 people complete a real session and give feedback."
            ),
            "measurableGoal": "Recruit 10 active testers from one clear segment before broad promotion",
            "why": "A focused segment produces sharper messaging and faster learning than marketing to everyone.",
            "proofPoints": ["Segment defined in one sentence", "20 prospects listed", "10 invite acceptances"],
            "effort": "30 minutes/day outreach",
            "tags": ["acquisition", "focus"],
        },
        {
            "id": "strategy-outcomes",
            "title": "Sell outcomes, not AI features",
            "body": (
                "Position around the useful follow-up people get after the work is done — decisions, owners, tasks, proof — "
                "not transcription or model names. Keep claims limited to what the product actually delivers."
            ),
            "measurableGoal": "One headline and one demo script that show outcome, not feature laundry lists",
            "why": "Outcome messaging converts better and sets honest expectations.",
            "proofPoints": ["Headline written", "Demo shows real output", "CTA is one clear next action"],
            "effort": "One focused messaging session",
            "tags": ["positioning", "gtm"],
        },
        {
            "id": "strategy-trust",
            "title": "Build trust through reliability",
            "body": (
                "Test the complete journey on each launch platform. Prevent silent failures and inventing commitments. "
                "Do not publicly promote a client with known critical defects."
            ),
            "measurableGoal": "Zero known critical defects before public promotion",
            "why": "Early users churn permanently if the first journey fails.",
            "proofPoints": ["Smoke tests per platform", "P0 list empty", "Consent and failure UX verified"],
            "effort": "Majority of daily engineering time",
            "tags": ["reliability", "quality"],
        },
        {
            "id": "strategy-founder",
            "title": "Founder-led acquisition",
            "body": (
                "Personally contact prospective users, help them finish one real journey, and learn why they would or would not return. "
                "Avoid paid ads until the first-journey experience is proven."
            ),
            "measurableGoal": "40 targeted outreach messages and at least 5 meaningful feedback conversations",
            "why": "Direct conversations beat vanity installs for a solo launch.",
            "proofPoints": ["Outreach log", "5 feedback notes", "Top 3 blockers ranked"],
            "effort": "1 hour/weekday",
            "tags": ["acquisition", "validation"],
        },
        {
            "id": "strategy-measure",
            "title": "Measure completed journeys, not just installs",
            "body": (
                f"Track whether people finish the core path and return. Success criteria remain: {criteria}. "
                "Instrument signup, journey start/complete, processing success/fail, and second-session return."
            ),
            "measurableGoal": "At least 10 successful real-user journeys and evidence of repeat usage",
            "why": "Downloads do not prove the product is useful.",
            "proofPoints": ["Funnel counters live", "10 completed journeys", "3+ returning users"],
            "effort": "One instrumentation pass + daily review",
            "tags": ["measurement", "growth"],
        },
    ]
    priority_recommendations = [
        {
            "id": "rec-reliability",
            "title": "Fix critical journey defects before polish",
            "reason": (
                "Recording, processing, or missing-result defects destroy trust. Visual polish can wait until the core path is reliable."
            ),
            "priority": "FIRST",
            "howSteps": [
                "Run end-to-end smoke tests on each priority platform",
                "Log every P0/P1 with reproduction steps",
                "Ship only launch-blocking fixes until the P0 list is empty",
            ],
            "measurableGoal": "Zero known P0 defects on launch platforms",
            "milestoneTitle": "Make the core journey trustworthy",
            "suggestedDurationMinutes": 180,
            "relatedStrategyId": "strategy-trust",
        },
        {
            "id": "rec-distribution",
            "title": "Use every already-public channel to recruit testers",
            "reason": (
                "If a store listing or web build already exists, use it for early testers while verifying that the current version works."
            ),
            "priority": "HIGH",
            "howSteps": [
                "Verify each public listing matches the real product name and build",
                "Invite testers to the working client only",
                "Document go/no-go per platform",
            ],
            "measurableGoal": "Platform readiness notes for every advertised client",
            "milestoneTitle": "Ship a private beta release candidate",
            "suggestedDurationMinutes": 90,
            "relatedStrategyId": "strategy-segment",
        },
        {
            "id": "rec-outreach",
            "title": "Talk to potential customers every weekday",
            "reason": "Launch day should not be the first time people hear about the product. Founder conversations create both users and truth.",
            "priority": "HIGH",
            "howSteps": [
                "Identify 20 prospects in one segment",
                "Send 5 personalized messages per working day",
                "Ask for one real journey test, then capture feedback",
            ],
            "measurableGoal": "40 outreach messages and 5 feedback conversations",
            "milestoneTitle": "Validate with real users",
            "suggestedDurationMinutes": 60,
            "relatedStrategyId": "strategy-founder",
        },
        {
            "id": "rec-demo",
            "title": "Make a genuine product demo your main marketing asset",
            "reason": "Show real output and one clear CTA. Avoid feature laundry lists and unverified claims.",
            "priority": "HIGH",
            "howSteps": [
                "Record one authentic end-to-end demo",
                "Write one headline focused on outcomes",
                "Publish with a single CTA: try it on the next real session",
            ],
            "measurableGoal": "One authentic demo and one clear CTA live",
            "milestoneTitle": "Prepare acquisition and launch assets",
            "suggestedDurationMinutes": 120,
            "relatedStrategyId": "strategy-outcomes",
        },
        {
            "id": "rec-launch-gate",
            "title": "Keep the public date, release only ready platforms",
            "reason": (
                f"Treat {timeframe} as the public beta target, but advertise only platforms that pass reliability and store readiness."
            ),
            "priority": "MEDIUM",
            "howSteps": [
                "Write an explicit go/no-go per platform",
                "Publish only approved clients",
                "Monitor first-day completions and failures",
            ],
            "measurableGoal": criteria,
            "milestoneTitle": f"Launch {title}",
            "suggestedDurationMinutes": 90,
            "relatedStrategyId": "strategy-measure",
        },
    ]
    checklists = [
        {
            "id": "checklist-release",
            "title": "Release readiness",
            "items": [
                {"id": "cr-1", "text": "Core journey works: signup → capture → results → history", "done": False},
                {"id": "cr-2", "text": "Zero known P0/P1 release blockers", "done": False},
                {"id": "cr-3", "text": "Failure and retry states are visible", "done": False},
                {"id": "cr-4", "text": "Privacy policy, terms, and support contact published", "done": False},
                {"id": "cr-5", "text": "Only approved builds are linked publicly", "done": False},
            ],
        }
    ]
    scorecard = [
        {"area": "Core journey", "criteria": "Signup → capture → summary/notes/tasks → history works"},
        {"area": "Reliability", "criteria": "No known unrecoverable data/audio losses in final tests"},
        {"area": "AI quality", "criteria": "No fabricated critical commitments in audited samples"},
        {"area": "Onboarding", "criteria": "Most testers complete first journey without personal help"},
        {"area": "Support", "criteria": "Errors observable and a support channel operating"},
    ]
    summary = (
        f"This is an execution roadmap for “{title}”: freeze MVP, make the core journey trustworthy, "
        f"validate with real users, then launch on proven platforms when {criteria}.{hours_note}"
    )
    return {
        "version": 1,
        "measurementType": "MILESTONE",
        "summary": summary,
        "mission": f"Launch a reliable version of {title} that real people can finish and trust.",
        "recommendation": (
            "Prioritize reliability and early user evidence over new features. "
            "Ship one proven platform first if store approvals lag."
        ),
        "successDefinition": {
            "deadlineLabel": timeframe,
            "metrics": [
                {"label": "Early users", "target": "30–50 interested / 10–15 active testers"},
                {"label": "Critical bugs", "target": "0 known P0 launch blockers"},
                {"label": "Core journey", "target": "Works end-to-end on the primary platform"},
                {"label": "Success criteria", "target": criteria},
            ],
        },
        "milestones": milestones,
        "phases": phases,
        "days": days,
        "strategies": strategies,
        "priorityRecommendations": priority_recommendations,
        "positioningHeadline": f"Launch a trustworthy first version of {title}",
        "positioningBody": (
            f"{title} should help a real user complete the core journey and come back. "
            "Reliability and evidence come before extra features."
        ),
        "funnelTargets": [
            "40 personalized outreach messages",
            "15 interested prospects",
            "10 installs or signups",
            "7 completed journeys",
            "3–5 returning users",
        ],
        "immediateNextStep": (
            "Complete platform smoke tests, verify store statuses, identify 20 prospective users, and prepare one genuine end-to-end demo."
        ),
        "checklists": checklists,
        "scorecard": scorecard,
        "assumptions": [
            "Primary launch risk is reliability and distribution readiness, not missing niche features.",
            "Store approvals may slip; web or already-approved clients can lead.",
        ],
        "openQuestions": [],
        "model": "fallback:launch_execution",
        "agentTrace": ["fallback"],
    }


def _classic_fallback_plan(goal: dict[str, Any]) -> dict[str, Any]:
    """Non-launch goals keep a solid phase plan, then wrap into expert-compatible shape."""
    title = _text(goal.get("title")) or "your goal"
    statement = _text(goal.get("statement"))
    timeframe = _text(goal.get("timeframe")) or "your timeframe"
    criteria = _text(goal.get("successCriteria")) or "The success criteria you set are met with clear evidence."
    hours = goal.get("availableHoursPerWeek")
    hours_note = f" Fit the work into about {hours} hours per week." if hours is not None else ""
    goal_type = detect_goal_type(statement, _text(goal.get("goalType")))
    description = _text(goal.get("description")) or _text(goal.get("category")) or title

    blob = f"{statement} {title} {description}".lower()
    if goal_type in {"habit", "other", "career"} and any(
        word in blob for word in ("body", "physique", "muscle", "gym", "bodybuild", "fitness", "athlete", "strength")
    ):
        goal_type = "physique"

    if goal_type == "physique":
        later = "Year 2–3" if "year" in timeframe.lower() else "Later phase"
        milestones = [
            _milestone("Record your starting physique", ["Body weight, key lifts, and progress photos are saved"], "Week 1–2"),
            _milestone(
                "Install a repeatable training foundation",
                ["A 4–5 day program is followed for at least 4 weeks"],
                "Month 1–3",
                ["Record your starting physique"],
            ),
            _milestone(
                "Lock nutrition and recovery",
                ["Protein, calorie, sleep, and step targets are tracked weekly"],
                "Month 1–3",
                ["Record your starting physique"],
            ),
            _milestone(
                "Run the first serious hypertrophy block",
                ["12+ focused weeks with measurable strength or size gains"],
                "Month 4–12",
                ["Install a repeatable training foundation", "Lock nutrition and recovery"],
            ),
            _milestone(f"Advance toward “{title}”", ["Mid-journey measurements move toward the stated goal"], later, ["Run the first serious hypertrophy block"]),
            _milestone(f"Prove you achieved “{title}”", [criteria], timeframe, [f"Advance toward “{title}”"]),
        ]
        summary = f"Phased physique roadmap for “{title}” until {criteria}.{hours_note}"
    elif goal_type == "skill":
        milestones = [
            _milestone("Assess your current level", ["Strengths, gaps, and a practice schedule are written"], "Week 1"),
            _milestone("Complete the fundamentals block", ["Core topics practiced with notes or drills"], "Month 1–2", ["Assess your current level"]),
            _milestone("Ship a proof project", ["One project demonstrates the skill"], "Month 2–4", ["Complete the fundamentals block"]),
            _milestone("Build interview or portfolio readiness", [criteria], timeframe, ["Ship a proof project"]),
        ]
        summary = f"Skill roadmap for “{title}” ending when {criteria}.{hours_note}"
    elif goal_type == "sales":
        milestones = [
            _milestone("Clarify offer and customer", ["Offer, ICP, and pricing assumptions are written"], "Week 1"),
            _milestone("Build the pipeline system", ["Lead sources and weekly outreach cadence are running"], "Week 2–4", ["Clarify offer and customer"]),
            _milestone("Improve conversion", ["Calls/demos and close rate are tracked and improved"], "Ongoing", ["Build the pipeline system"]),
            _milestone("Hit the sales target", [criteria], timeframe, ["Improve conversion"]),
        ]
        summary = f"Sales execution plan for “{title}” until {criteria}.{hours_note}"
    elif goal_type == "freelance":
        milestones = [
            _milestone("Lock scope and timeline", ["Deliverables, deadline, and acceptance criteria are agreed"], "Week 1"),
            _milestone("Deliver the first milestone", ["Client-visible progress is shared and accepted"], "Early phase", ["Lock scope and timeline"]),
            _milestone("Complete remaining deliverables", ["All scoped work is finished and reviewed"], "Before final delivery", ["Deliver the first milestone"]),
            _milestone("Close the project", [criteria], timeframe, ["Complete remaining deliverables"]),
        ]
        summary = f"Delivery plan for “{title}” until {criteria}.{hours_note}"
    else:
        milestones = [
            _milestone(f"Clarify the starting point for {title}", ["Current state, constraints, and weekly capacity are written"], "Week 1"),
            _milestone("Build the operating system", ["Weekly actions, tools, and tracking are set"], "Week 2–4", [f"Clarify the starting point for {title}"]),
            _milestone("Execute the main body of work", ["Visible progress on the hardest work"], "Mid timeframe", ["Build the operating system"]),
            _milestone("Prove the outcome", [criteria], timeframe, ["Execute the main body of work"]),
        ]
        summary = (
            f"This roadmap is a practical path for “{title}”: start from reality, build the system, "
            f"do the hard work in phases, and finish when {criteria}.{hours_note}"
        )

    days: list[dict[str, Any]] = []
    for index, milestone in enumerate(milestones, start=1):
        days.append(
            {
                "dayIndex": index,
                "dateLabel": milestone["dueLabel"] or f"Phase {index}",
                "title": milestone["title"],
                "hours": None,
                "tasks": [
                    {
                        "id": f"m{index}-t{task_i}",
                        "text": text,
                        "done": False,
                    }
                    for task_i, text in enumerate(milestone["successCriteria"] or ["Complete this phase"], start=1)
                ],
                "definitionOfDone": (milestone["successCriteria"] or ["Phase complete"])[0],
            }
        )

    strategies = [
        {
            "id": "strategy-focus",
            "title": "Sequence the hard work",
            "body": (
                f"{summary} Finish evidence for each phase before stretching into the next one so “{title}” stays honest."
            ),
            "measurableGoal": f"Complete each phase’s success evidence before starting the next stretch toward: {criteria}",
            "why": "Skipping sequence creates false progress and rework.",
            "proofPoints": ["Phase checklist written", "Evidence stored per phase", "Weekly review logged"],
            "effort": "Protect one deep-work block most days",
            "tags": ["execution", "sequence"],
        },
        {
            "id": "strategy-evidence",
            "title": "Measure progress weekly",
            "body": (
                f"Use “{criteria}” as the weekly proof point. Adjust tactics when evidence is weak, not the destination."
            ),
            "measurableGoal": "One weekly review with evidence against the success criteria",
            "why": "Without weekly proof, plans drift into activity theater.",
            "proofPoints": ["Metric snapshot", "What improved", "What to change next week"],
            "effort": "30–45 minutes each week",
            "tags": ["measurement"],
        },
        {
            "id": "strategy-capacity",
            "title": "Respect capacity constraints",
            "body": (
                f"Design the plan around real weekly capacity for “{title}”.{hours_note} "
                "Cut scope before cutting recovery or quality."
            ),
            "measurableGoal": "A weekly schedule that fits available hours without chronic overrun",
            "why": "Overloaded plans fail quietly.",
            "proofPoints": ["Hours budgeted", "Top 3 priorities chosen", "Deferred work listed"],
            "effort": "Weekly planning ritual",
            "tags": ["capacity"],
        },
        {
            "id": "strategy-feedback",
            "title": "Create a fast feedback loop",
            "body": (
                "Get external or self-review evidence early — a coach, peer, customer, recording, or measured result — "
                "so corrections happen before the deadline."
            ),
            "measurableGoal": "At least one feedback loop every week during active phases",
            "why": "Late feedback is expensive.",
            "proofPoints": ["Feedback captured", "One change applied", "Next experiment chosen"],
            "effort": "1 short review session/week",
            "tags": ["learning", "proof"],
        },
    ]
    priority_recommendations = [
        {
            "id": "rec-baseline",
            "title": "Write the honest starting baseline",
            "reason": f"Without a clear baseline for “{title}”, later progress is guesswork.",
            "priority": "FIRST",
            "howSteps": [
                "Capture current state in measurable terms",
                "List constraints and weekly hours",
                "Define the first proof artifact for this week",
            ],
            "measurableGoal": "Baseline + constraints written and reviewable",
            "milestoneTitle": milestones[0]["title"] if milestones else "Clarify starting point",
            "suggestedDurationMinutes": 45,
            "relatedStrategyId": "strategy-focus",
        },
        {
            "id": "rec-weekly-proof",
            "title": "Install a weekly proof ritual",
            "reason": f"Every week should leave evidence against: {criteria}.",
            "priority": "HIGH",
            "howSteps": [
                "Pick the metric or artifact you will check weekly",
                "Schedule a recurring review block",
                "Adjust next week’s plan based on the evidence",
            ],
            "measurableGoal": "Four consecutive weekly reviews completed",
            "suggestedDurationMinutes": 40,
            "relatedStrategyId": "strategy-evidence",
        },
        {
            "id": "rec-hardest-work",
            "title": "Protect time for the hardest work first",
            "reason": "Easy busywork expands. Put the bottleneck work in the first deep block of the day.",
            "priority": "HIGH",
            "howSteps": [
                "Identify the bottleneck task for this phase",
                "Block uninterrupted time for it",
                "Defer low-leverage tasks until the bottleneck moves",
            ],
            "measurableGoal": "Hardest task advanced with visible output at least 4 days this week",
            "suggestedDurationMinutes": 120,
            "relatedStrategyId": "strategy-capacity",
        },
        {
            "id": "rec-feedback",
            "title": "Get one outside or objective signal soon",
            "reason": "Self-assessment alone is weak. Add a peer, coach, customer, recording, or measured test early.",
            "priority": "MEDIUM",
            "howSteps": [
                "Choose the feedback source",
                "Ask one specific question",
                "Apply one change from the answer",
            ],
            "measurableGoal": "One feedback conversation or objective test completed this week",
            "suggestedDurationMinutes": 60,
            "relatedStrategyId": "strategy-feedback",
        },
    ]
    return {
        "version": 1,
        "measurementType": "MILESTONE",
        "summary": summary,
        "mission": f"Achieve “{title}” with a sequenced plan you can execute.",
        "recommendation": (
            f"Finish each phase’s evidence before starting the next stretch of work. Keep “{criteria}” as the weekly proof point."
        ),
        "positioningHeadline": f"Make steady, provable progress on “{title}”.",
        "positioningBody": (
            f"The plan is built to produce evidence against “{criteria}” inside {timeframe}, using capacity-aware sequencing."
        ),
        "funnelTargets": [
            "Baseline captured",
            "Weekly proof ritual running",
            "Mid-journey evidence reviewed",
            f"Final criteria met: {criteria}",
        ],
        "immediateNextStep": "Write the baseline, choose this week’s proof artifact, and schedule the first deep-work block.",
        "successDefinition": {
            "deadlineLabel": timeframe,
            "metrics": [{"label": "Success criteria", "target": criteria}],
        },
        "milestones": milestones,
        "phases": [
            {
                "id": f"phase-{index}",
                "title": item["title"],
                "windowLabel": item["dueLabel"],
                "outcome": (item["successCriteria"] or [""])[0],
            }
            for index, item in enumerate(milestones, start=1)
        ],
        "days": days,
        "strategies": strategies,
        "priorityRecommendations": priority_recommendations,
        "checklists": [
            {
                "id": "checklist-progress",
                "title": "Weekly progress checks",
                "items": [
                    {"id": "cp-1", "text": "Review last week’s evidence against success criteria", "done": False},
                    {"id": "cp-2", "text": "Pick the single highest-leverage task for this week", "done": False},
                    {"id": "cp-3", "text": "Remove or postpone work that does not move the outcome", "done": False},
                ],
            }
        ],
        "scorecard": [{"area": "Outcome", "criteria": criteria}],
        "assumptions": [],
        "openQuestions": [],
        "model": f"fallback:{goal_type}",
        "agentTrace": ["fallback"],
    }


def _fallback_plan(goal: dict[str, Any]) -> dict[str, Any]:
    if is_launch_style_goal(goal):
        plan = _launch_fallback_plan(goal)
    else:
        plan = _degraded_plan(goal)
    plan["days"] = _enrich_day_explanations(list(plan.get("days") or []))
    plan["strategies"] = _normalize_strategies(list(plan.get("strategies") or []))
    plan["priorityRecommendations"] = _normalize_priority_recommendations(
        list(plan.get("priorityRecommendations") or [])
    )
    return plan


def assemble_expert_plan(
    goal: dict[str, Any],
    *,
    strategy: GoalStrategistResult | None,
    execution: GoalExecutionResult | None,
    models: list[str],
    advisor: Any | None = None,
) -> dict[str, Any]:
    timeframe = _text(goal.get("timeframe")) or "your timeframe"
    criteria = _text(goal.get("successCriteria")) or "Success criteria are met with evidence"
    title = _text(goal.get("title")) or "your goal"

    if execution is None:
        return _fallback_plan(goal)

    days = _normalize_days(execution.days)
    strategies = _normalize_strategies(execution.strategies)
    if len(strategies) < 3 and advisor is not None:
        strategies = _normalize_strategies(getattr(advisor, "strategies", None) or [])
    priority_recommendations = _normalize_priority_recommendations(execution.priorityRecommendations)
    if len(priority_recommendations) < 3 and advisor is not None:
        priority_recommendations = _normalize_priority_recommendations(
            getattr(advisor, "priorityRecommendations", None) or []
        )
    checklists = _normalize_checklists(execution.checklists)
    scorecard = _normalize_scorecard(execution.scorecard)
    phases = _normalize_phases(execution.phases)
    milestones = _normalize_milestones(execution.milestones, timeframe)
    if not milestones and phases:
        milestones = _milestones_from_phases(phases, timeframe, criteria)
    if not milestones:
        return _fallback_plan(goal)

    measurement = _text(execution.measurementType).upper()
    direction = _text(execution.measurementDirection).upper()
    summary = _text(execution.summary) or (
        _text(strategy.recommendation) if strategy else f"Execution roadmap for “{title}”."
    )
    mission = _text(strategy.mission) if strategy else f"Achieve “{title}” with a sequenced, measurable plan."
    recommendation = _text(strategy.recommendation) if strategy else summary
    positioning_headline = ""
    positioning_body = ""
    immediate_next_step = ""
    funnel_targets: list[str] = []
    if strategy:
        positioning_headline = _text(strategy.positioningHeadline)
        positioning_body = _text(strategy.positioningBody)
    if advisor is not None:
        positioning_headline = positioning_headline or _text(getattr(advisor, "positioningHeadline", ""))
        positioning_body = positioning_body or _text(getattr(advisor, "positioningBody", ""))
        immediate_next_step = _text(getattr(advisor, "immediateNextStep", ""))
        funnel_targets = [
            _text(item) for item in (getattr(advisor, "funnelTargets", None) or []) if _text(item)
        ][:8]
    success_definition = {
        "deadlineLabel": _text(getattr(strategy.successDefinition, "deadlineLabel", "")) if strategy else timeframe,
        "metrics": [],
    }
    if strategy:
        for metric in strategy.successDefinition.metrics or []:
            label = _text(metric.label)
            target = _text(metric.target)
            if label and target:
                success_definition["metrics"].append({"label": label[:120], "target": target[:240]})
    if not success_definition["deadlineLabel"]:
        success_definition["deadlineLabel"] = timeframe
    if not success_definition["metrics"]:
        success_definition["metrics"] = [{"label": "Success criteria", "target": criteria}]

    assumptions = [_text(item) for item in ((strategy.assumptions if strategy else []) or []) if _text(item)][:8]
    open_questions = []
    if strategy:
        for index, question in enumerate(strategy.openQuestions or [], start=1):
            prompt = _text(question.prompt)
            if not prompt:
                continue
            open_questions.append(
                {
                    "id": _text(question.id) or _slug("q", index),
                    "prompt": prompt[:320],
                    "options": [
                        {"id": _text(opt.id) or _slug("opt", opt_i), "label": _text(opt.label)[:160]}
                        for opt_i, opt in enumerate(question.options or [], start=1)
                        if _text(opt.label)
                    ][:6],
                }
            )

    if not strategies and strategy:
        strategies = [
            {
                "id": "strategy-mvp",
                "title": "MVP scope",
                "body": "; ".join([_text(item) for item in strategy.mvpScope if _text(item)][:6])
                or recommendation,
                "measurableGoal": criteria,
                "why": "Keep the first credible version small enough to finish.",
                "proofPoints": [_text(item) for item in (strategy.mvpScope or []) if _text(item)][:4],
                "effort": _text(strategy.capacityNote)[:240],
                "tags": ["mvp"],
            }
        ]

    plan = {
        "version": 1,
        "measurementType": measurement if measurement in ALLOWED else "MILESTONE",
        "summary": summary[:2000],
        "mission": mission[:1000],
        "recommendation": recommendation[:4000],
        "positioningHeadline": positioning_headline[:240],
        "positioningBody": positioning_body[:1200],
        "immediateNextStep": immediate_next_step[:480],
        "funnelTargets": funnel_targets,
        "successDefinition": success_definition,
        "milestones": milestones,
        "phases": phases,
        "days": days,
        "strategies": strategies,
        "priorityRecommendations": priority_recommendations,
        "checklists": checklists,
        "scorecard": scorecard,
        "assumptions": assumptions,
        "openQuestions": open_questions[:3],
        "baselineValue": execution.baselineValue,
        "targetValue": execution.targetValue,
        "measurementDirection": direction if direction in {"INCREASE", "DECREASE"} else "",
        "model": "+".join(models) if models else "expert",
        "agentTrace": models,
    }
    if _is_generic_plan(milestones) or _is_thin_expert(plan):
        print("goal expert plan rejected as thin/generic; using domain fallback", flush=True)
        return _fallback_plan(goal)
    return plan


def _preserve_progress(previous: dict[str, Any] | None, plan: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return plan
    done_ids: set[str] = set()
    notes: dict[str, str] = {}
    for day in previous.get("days") or []:
        for task in day.get("tasks") or []:
            if task.get("done") and task.get("id"):
                done_ids.add(str(task["id"]))
            if task.get("id") and task.get("note"):
                notes[str(task["id"])] = str(task["note"])
    for day in plan.get("days") or []:
        for task in day.get("tasks") or []:
            task_id = str(task.get("id") or "")
            if task_id in done_ids:
                task["done"] = True
            if task_id in notes:
                task["note"] = notes[task_id]
    return plan


async def prepare_plan_context(store: Any | None, goal: dict[str, Any], *, force_refresh: bool = False) -> dict[str, Any]:
    from services.goals.gaps import analyze_gaps
    from services.goals.research import compact_evidence_pack, run_research

    research = await run_research(store, goal, force_refresh=force_refresh)
    evidence = list(research.get("evidence") or [])
    gaps = await analyze_gaps(goal, evidence)
    goal["researchEvidencePack"] = compact_evidence_pack(evidence)
    goal["gaps"] = gaps
    goal["researchSummary"] = research.get("summary") or {}
    goal["researchSources"] = research.get("sources") or []
    return {"goal": goal, "research": research, "gaps": gaps}


def _attach_consultant_fields(plan: dict[str, Any], goal: dict[str, Any], validation: dict[str, Any] | None = None) -> dict[str, Any]:
    plan["understanding"] = goal.get("understanding") or {}
    plan["gaps"] = goal.get("gaps") or {}
    plan["researchSummary"] = goal.get("researchSummary") or {}
    plan["sources"] = [
        {"url": item.get("url"), "title": item.get("title"), "publisher": item.get("publisher")}
        for item in (goal.get("researchSources") or [])[:12]
    ]
    plan["validationReport"] = validation or {}
    plan["provisional"] = bool((goal.get("gaps") or {}).get("provisional") or (validation or {}).get("provisional"))
    return plan


async def build_plan(goal: dict[str, Any], store: Any | None = None, *, force_refresh: bool = False, previous_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Research, gaps, then Strategist → Advisor → Execution → Critic, then validate."""
    from services.goals.validator import strip_mismatched_items, validate_plan

    prepared = await prepare_plan_context(store, goal, force_refresh=force_refresh)
    goal = prepared["goal"]
    assert wants_expert_pipeline(goal)
    pipeline = await run_expert_plan_pipeline(goal)
    assembled = None
    if pipeline and pipeline.get("execution") is not None:
        assembled = assemble_expert_plan(
            goal,
            strategy=pipeline.get("strategy"),
            advisor=pipeline.get("advisor"),
            execution=pipeline.get("execution"),
            models=list(pipeline.get("models") or []),
        )
        if _is_thin_expert(assembled) or _is_generic_plan(assembled.get("milestones") or []):
            assembled = None
    plan = assembled or _fallback_plan(goal)
    report = validate_plan(goal, plan, evidence=prepared["research"].get("evidence") or [], gaps=goal.get("gaps") or {})
    if not report.accepted:
        plan = strip_mismatched_items(plan, goal)
        retry = await run_expert_plan_pipeline(goal)
        if retry and retry.get("execution") is not None:
            repaired = assemble_expert_plan(
                goal,
                strategy=retry.get("strategy"),
                advisor=retry.get("advisor"),
                execution=retry.get("execution"),
                models=list(retry.get("models") or []) + ["repair"],
            )
            second = validate_plan(goal, repaired, evidence=prepared["research"].get("evidence") or [], gaps=goal.get("gaps") or {})
            if second.accepted or len(second.issues) < len(report.issues):
                plan = repaired
                report = second
        if not report.accepted:
            degraded = _fallback_plan(goal)
            degraded = strip_mismatched_items(degraded, goal)
            plan = degraded
            report = validate_plan(goal, plan, evidence=prepared["research"].get("evidence") or [], gaps=goal.get("gaps") or {})
            report.provisional = True
    plan = _preserve_progress(previous_plan or goal.get("approvedPlan") or goal.get("proposedPlan"), plan)
    return _attach_consultant_fields(plan, goal, report.model_dump())
