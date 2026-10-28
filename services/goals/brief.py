"""Turn intake answers into the sections of a plan.

The sections follow one shape — who to start with, what counts, how the
weeks run, which paths to use, and what to do first — but the words come
from this goal. Missing facts stay labeled as assumptions.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from services.goals.domains import inferred_title, is_skip_answer, understanding_as_dict
from services.goals.intake import approach_beats, goal_signals, plan_family

GENERIC_STRATEGY_IDS = {"strategy-evidence", "strategy-sequence"}
GENERIC_HEADLINE = "A personalized path toward"
GENERIC_RECOMMENDATION = "start by comparing your current state"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _clean(value: Any, limit: int = 240) -> str:
    text = " ".join(_text(value).split())
    if is_skip_answer(text) or text.upper() == "UNKNOWN":
        return ""
    return text[:limit]


def _family(goal: dict[str, Any]) -> str:
    return plan_family(goal_signals(goal, understanding_as_dict(goal.get("understanding"))))


def _title(goal: dict[str, Any]) -> str:
    return inferred_title(goal, understanding_as_dict(goal.get("understanding"))) or "this goal"


def _audience_groups(goal: dict[str, Any]) -> list[str]:
    raw = _clean(goal.get("audience"), 300)
    if not raw:
        profile = goal.get("contextProfile") if isinstance(goal.get("contextProfile"), dict) else {}
        raw = _clean((profile or {}).get("audience"), 300)
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r",| and ", raw) if part.strip()]
    return parts[:4]


def _budget(goal: dict[str, Any]) -> str:
    profile = goal.get("contextProfile") if isinstance(goal.get("contextProfile"), dict) else {}
    return _clean((profile or {}).get("budget"), 160)


def _number(goal: dict[str, Any]) -> str:
    match = re.search(r"\b(\d[\d,]*)\b", _text(goal.get("statement")))
    return match.group(1) if match else ""


def _short_deadline(goal: dict[str, Any]) -> bool:
    blob = f"{_text(goal.get('timeframe'))} {_text(goal.get('statement'))}".lower()
    return bool(re.search(r"\b(30 days|within 30|30-day)\b", blob))


def _methods(goal: dict[str, Any], family: str) -> list[dict[str, Any]]:
    title = _title(goal)
    groups = _audience_groups(goal)
    who = groups[0] if groups else "the people closest to the problem"
    catalog = {
        "people": [
            (
                "Talk to the first likely users",
                f"Message {who} one at a time and ask them to try one real workflow. Personalize each note.",
                "20 conversations this week",
            ),
            (
                "Borrow a group that already gathers them",
                "Ask one organizer for a short demonstration of the main result. Show the result before the feature list.",
                "Two organizer conversations this week",
            ),
            (
                "Show the work happening",
                f"Publish a short recording of one workflow for {title} ending in a result a person can see.",
                "Three short demonstrations this week",
            ),
        ],
        "career": [
            (
                "Write the gap from where you are",
                f"List what your current stage already proves, and the one missing proof for {title}.",
                "One written gap list",
            ),
            (
                "Make one proof a reviewer can open",
                "Turn that gap into a page, project, or case a reviewer can check without you in the room.",
                "One proof piece this week",
            ),
            (
                "Practice the real conversation",
                "Rehearse the conversation you will have for this role, using one example and a result you can point to.",
                "Three practice runs",
            ),
        ],
        "health": [
            (
                "Train from your current level",
                "Pick this week's sessions from what you can already finish, not from a peak plan.",
                "This week's sessions on the calendar",
            ),
            (
                "Keep one long session and one easy session",
                "Do both before you add distance or load.",
                "Both sessions logged",
            ),
            (
                "Record pain and missed days",
                "Write what stopped a session before you make the next week harder.",
                "A note after every missed or painful session",
            ),
        ],
        "habit": [
            (
                "Shrink the action",
                f"Cut {title} down until you can finish it on a bad day.",
                "A version that takes a few minutes",
            ),
            (
                "Attach it to something you already do",
                "Put the action right after a habit that already happens.",
                "A written cue and action",
            ),
            (
                "Count only finished days",
                "Mark the day done when the action happened. Restart the next day after a miss.",
                "A daily mark",
            ),
        ],
        "skill": [
            (
                "Learn one usable piece",
                f"Study only the part of {title} you can use before the week ends.",
                "Notes you can practice from",
            ),
            (
                "Make one small thing with it",
                "Build a tiny result that would not exist without that piece.",
                "One finished exercise or project",
            ),
            (
                "Explain it without notes",
                "Say the idea out loud until you can do it from memory.",
                "One clean explanation",
            ),
        ],
        "local": [
            (
                "Talk to people who can already reach you",
                f"Ask nearby buyers what they use today instead of {title}.",
                "Ten conversations",
            ),
            (
                "Offer one reason to show up",
                "Give one clear offer for this week and tell people where to find it.",
                "One offer, actually shared",
            ),
            (
                "Ask what almost stopped them",
                "After each buyer, write the reason they nearly said no.",
                "Notes from every buyer",
            ),
        ],
        "venture": [
            (
                "Talk to the first buyers",
                f"Find people who already have the problem {title} is for, and watch them try it.",
                "Ten watched attempts",
            ),
            (
                "Make the first result obvious",
                "Show the result before you explain the rest of the offer.",
                "One demonstration people can follow",
            ),
            (
                "Ask for one introduction",
                "When someone gets value, ask them who else has the same problem.",
                "One introduction request per happy user",
            ),
        ],
        "travel": [
            (
                "Lock the constraints",
                "Write the dates, the money you can spend, and the places that matter.",
                "One page of decisions",
            ),
            (
                "Book the scarce piece first",
                "Reserve the part that will sell out, then fill the rest around it.",
                "The scarce booking confirmed",
            ),
            (
                "Leave a buffer",
                "Put a simple day plan on paper, with one block of free time each day.",
                "A day plan you can follow",
            ),
        ],
        "relationship": [
            (
                "Say the request once, clearly",
                "Name what you want changed, without stacking old complaints on top.",
                "One conversation",
            ),
            (
                "Repeat one visible action",
                "Pick an action the other person can notice this week.",
                "The action done on the days you chose",
            ),
            (
                "Ask what landed",
                "Ask what helped and what did not, then keep only what helped.",
                "One follow-up conversation",
            ),
        ],
        "money": [
            (
                "Write the number and the check date",
                f"Put the target for {title} on paper with the day you will look again.",
                "A number and a date",
            ),
            (
                "Move the first small amount",
                "Change the number this week by saving, earning, or cutting one real expense.",
                "One transaction that matches the plan",
            ),
            (
                "Review before you change course",
                "Compare the number with the plan before you add a new tactic.",
                "A weekly check",
            ),
        ],
        "creative": [
            (
                "Finish one small piece",
                f"Complete a small version of {title} this week, even if it is rough.",
                "One finished piece",
            ),
            (
                "Show it to one person",
                "Ask what they noticed. Write their words down.",
                "Notes from one viewer",
            ),
            (
                "Repeat the part that worked",
                "Make the next piece by keeping what the viewer understood.",
                "A second piece started from that note",
            ),
        ],
        "general": [
            (
                "Define the check",
                f"Write the result that will show {title} is moving.",
                "One sentence you can verify",
            ),
            (
                "Produce the first evidence",
                "Do the smallest task that creates something you can look at.",
                "One artifact",
            ),
            (
                "Review the evidence",
                "Decide the next task from what that artifact showed.",
                "A written next step",
            ),
        ],
    }
    rows = catalog.get(family, catalog["general"])
    methods = []
    for index, (title_text, body, effort) in enumerate(rows, start=1):
        methods.append(
            {
                "id": f"method-{index}",
                "title": title_text,
                "body": body,
                "measurableGoal": effort,
                "why": f"{title_text}: {body}",
                "proofPoints": [effort],
                "effort": effort,
                "tags": [family],
            }
        )
    return methods


def _resource_split(goal: dict[str, Any], family: str) -> list[dict[str, str]]:
    budget = _budget(goal).lower()
    if not budget or family not in {"people", "venture", "local", "money", "travel"}:
        return []
    if any(token in budget for token in ("organic", "₹0", "rs 0", "nothing", "no budget", "zero")):
        return [
            {"item": "Paid promotion", "amount": "None until people who try it come back"},
            {"item": "Your own outreach", "amount": "The whole plan for now"},
        ]
    rupees = "₹" in _budget(goal) or "5,000" in budget or "5000" in budget or "20,000" in budget
    if rupees and any(token in budget for token in ("under", "5,000", "5000", "small", "test", "20,000")):
        return [
            {"item": "Ten short research conversations", "amount": "₹1,000 of the cap"},
            {"item": "One collaboration with a relevant group", "amount": "₹1,500 of the cap"},
            {"item": "A small thank-you for useful introductions", "amount": "₹1,000 of the cap"},
            {"item": "Hold back", "amount": "₹1,500 until one path brings people back"},
        ]
    if any(token in budget for token in ("small", "test", "under", "medium", "experiment")):
        return [
            {"item": "Research conversations", "amount": "A small slice of the test budget"},
            {"item": "One collaboration with a relevant group", "amount": "A small slice"},
            {"item": "Hold back", "amount": "Keep the rest until one path brings people back"},
        ]
    return [
        {"item": "Test the path you can run yourself", "amount": "Most of the budget"},
        {"item": "Hold back", "amount": "Keep a reserve until one path brings people back"},
    ]


def _reality(goal: dict[str, Any], family: str) -> str:
    title = _title(goal)
    number = _number(goal)
    if family == "people":
        if number and _short_deadline(goal):
            return (
                f"Reaching {number} on this deadline is aggressive from a small start. "
                "Aim for it, and judge each week by people who came back, not by a promised total."
            )
        if number:
            return (
                f"{number} is a direction for {title}. "
                "The plan builds a repeatable way to find people who get value and return."
            )
        return "Count people who get value and come back. A sign-up by itself is not the result."
    if family == "career":
        return (
            f"Someone else decides the offer or promotion for {title}. "
            "This plan gets you ready. It does not decide the result."
        )
    if family == "health":
        return "Follow your current level. This plan does not promise a race time or a medical result."
    if family == "habit":
        return "A finished day is a day you did the action. A miss is a restart, not a failed goal."
    if family == "skill":
        return "Done is something you can make or explain. Finishing a video is not the result unless you said it was."
    if family == "money":
        return "Track a number you can check. Do not treat a neat plan as the money already moved."
    if family == "travel":
        return "Decide the constraints before you book. A long list of ideas is not a trip."
    if family == "relationship":
        return "You can plan your side of this. You cannot plan the other person's answer."
    if family in {"local", "venture"}:
        return "Start with people who can already say yes. Do not spend the first weeks talking to everyone."
    return f"Keep {title} tied to evidence you can check. Tasks are not the same as the outcome."


def _focus(goal: dict[str, Any], family: str) -> tuple[str, str]:
    title = _title(goal)
    groups = _audience_groups(goal)
    level = ""
    profile = goal.get("contextProfile") if isinstance(goal.get("contextProfile"), dict) else {}
    if isinstance(profile, dict):
        level = _clean(profile.get("currentRole") or profile.get("startingLevel"), 80)
    if family == "people":
        primary = (
            f"Start with {groups[0]}. They are the first people to try {title}."
            if groups
            else "Start with the smallest group that already has this problem. Name them before you try to reach everyone."
        )
        secondary = (
            f"Keep {groups[1]} for later, after the first group comes back."
            if len(groups) > 1
            else "A second group waits until the first group is coming back."
        )
        return primary, secondary
    if family == "career":
        start = level or "your current stage"
        return (
            f"Start from {start}, and aim the proof at {title}.",
            "Build one piece a reviewer in this field can check.",
        )
    if family == "local":
        return (
            f"Start with people who can visit or buy {title} without a long trip.",
            "Online reach waits until the local offer is clear.",
        )
    if family == "habit":
        return (
            f"Make {title} small enough to do on a bad day.",
            "Track finished days, not how motivated you felt.",
        )
    if family == "health":
        return (
            "Start from the training you can already finish.",
            "Add load only after those sessions feel repeatable.",
        )
    if family == "skill":
        return (
            f"Learn the piece of {title} you can use this week.",
            "Prove it with something you made, not with hours watched.",
        )
    return (
        f"Start with the next piece of evidence for {title}.",
        "Add more work only after that evidence is real.",
    )


def _headline(family: str, title: str) -> str:
    headlines = {
        "people": "Win people who come back, not unused sign-ups.",
        "career": "Get ready for the role. The offer is a separate result.",
        "health": "Train from where you are this week.",
        "habit": "Count the days you did the action.",
        "skill": "Make something small that proves the skill.",
        "local": "Start with people who can already reach you.",
        "venture": "Show one result, then ask for the next conversation.",
        "travel": "Decide, book the scarce piece, then leave a buffer.",
        "relationship": "One clear request, then one repeated action.",
        "money": "Move the number, then check it on a set date.",
        "creative": "Finish a small piece and show it to one person.",
    }
    return headlines.get(family, f"Take the next honest step on {title}.")[:90]


def _checks(family: str) -> list[dict[str, str]]:
    rows = {
        "people": [
            ("Conversations started", "A daily number you can keep"),
            ("People who finished the core action", "Track this, not sign-ups"),
            ("People who came back in 7 days", "The number that matters this week"),
        ],
        "career": [
            ("Proof a reviewer can open", "One piece tied to the gap"),
            ("Practice conversations", "Runs you actually did"),
            ("Real asks sent", "Applications or internal requests, not drafts"),
        ],
        "health": [
            ("Sessions completed", "Compared with the week you planned"),
            ("Easy sessions kept", "Not skipped in favor of more intensity"),
            ("Notes on pain or misses", "Written before the next increase"),
        ],
        "habit": [
            ("Days the action happened", "The only count that matters"),
            ("Misses restarted", "The next day, without a new plan"),
        ],
        "skill": [
            ("Piece practiced", "Something you can do, not only watch"),
            ("Artifact made", "A result that uses the piece"),
        ],
    }
    return [{"area": area, "criteria": criteria} for area, criteria in rows.get(family, [("Evidence", "Something you can check this week")])]


def _starter(goal: dict[str, Any], family: str) -> str:
    title = _title(goal)
    description = _clean(goal.get("description"), 180)
    groups = _audience_groups(goal)
    who = groups[0] if groups else "this kind of work"
    product = description or title
    if family == "people":
        return (
            f"Hey [Name] — how do you handle this today? I'm building {product}. "
            f"It is meant for {who}. Would you try it on one real case and tell me what was useful or confusing?"
        )
    if family == "career":
        return (
            f"Practice story for {title}: one situation, what you did, and a result a reviewer can check. "
            "Keep it under two minutes."
        )
    if family == "habit":
        return f"Calendar line: do the small version of {title} immediately after something you already do."
    if family == "health":
        return "This week: one longer session, one easy session, and a note if either one hurts or gets skipped."
    if family == "skill":
        return f"This week: use {title} to make one small thing, then explain it without looking at notes."
    if family in {"local", "venture"}:
        return f"Ask ten people: what do you do today when you need what {title} offers, and what is annoying about that?"
    if family == "travel":
        return "Write dates, money, and the one booking that will sell out. Do not open maps until those three are set."
    if family == "relationship":
        return "One sentence you can say this week: what you want, and the action you will take on your side."
    if family == "money":
        return "Write the number, the date you will check it, and the one transaction you will make this week."
    return f"Write the next visible result for {title}, then do the smallest task that creates it."


def _assumptions(goal: dict[str, Any], family: str) -> list[str]:
    rows: list[str] = []
    if family == "people":
        rows.append(
            "Planning assumption, not a forecast: many people who hear about this will not come back. "
            "If the first session is confusing, fix that before you chase more people."
        )
        if _number(goal):
            rows.append("The stated count is a stretch target unless your own results already support it.")
    elif not _clean(goal.get("timeframe")):
        rows.append("The deadline is unknown, so the weeks are an estimate until you set a date.")
    level = ""
    profile = goal.get("contextProfile") if isinstance(goal.get("contextProfile"), dict) else {}
    if isinstance(profile, dict):
        level = _clean(profile.get("startingLevel"), 80)
    if not level and family in {"career", "skill", "health", "habit"}:
        rows.append("Starting point is unknown, so the first step is a short check rather than an advanced plan.")
    return rows


def _first_move(methods: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "first-move",
        "prompt": "Which of these can you do consistently this week?",
        "options": [
            {"id": method["id"], "label": method["title"]}
            for method in methods
            if method.get("title")
        ][:4],
    }


def compose_plan_brief(goal: dict[str, Any]) -> dict[str, Any]:
    family = _family(goal)
    title = _title(goal)
    primary, secondary = _focus(goal, family)
    methods = _methods(goal, family)
    return {
        "family": family,
        "headline": _headline(family, title),
        "realityCheck": _reality(goal, family),
        "primaryFocus": primary,
        "secondaryFocus": secondary,
        "approachBeats": approach_beats(goal),
        "methods": methods,
        "starter": _starter(goal, family),
        "resourceSplit": _resource_split(goal, family),
        "checks": _checks(family),
        "assumptions": _assumptions(goal, family),
        "firstMove": _first_move(methods),
        "whatCounts": (
            "A person who completed the main action in the last 7 days, not someone who only registered."
            if family == "people"
            else ""
        ),
    }


def _generic_strategies(strategies: list[dict[str, Any]]) -> bool:
    if not strategies:
        return True
    return all(
        _text(item.get("id")) in GENERIC_STRATEGY_IDS or "ground every stage" in _text(item.get("title")).lower()
        for item in strategies
        if isinstance(item, dict)
    )


_GENERIC_EXPLANATION = (
    "this is the next path",
    "this is the work for this step",
    "this is the work inside",
    "this is the next honest step",
    "needed for",
    "unlocks the next step",
    "matches the answers so far",
    "this step moves the goal",
    "do the work for",
    "save something you can open",
    "do the smallest version",
)


def _generic_explanation(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _GENERIC_EXPLANATION)


def explain_heading(heading: str, current: str, detail: str = "") -> str:
    """Keep the sentence the model wrote. Do not replace it with a template."""
    return (_text(current) or _text(detail) or _text(heading))[:500]


def concretize_days(plan: dict[str, Any]) -> dict[str, Any]:
    """Turn outcome labels into the actions inside each step."""
    from services.goals.step_actions import clarify_plan_steps

    return clarify_plan_steps(plan)


def align_explanations(plan: dict[str, Any]) -> dict[str, Any]:
    for strategy in plan.get("strategies") or []:
        if not isinstance(strategy, dict):
            continue
        strategy["why"] = explain_heading(strategy.get("title"), strategy.get("why"), strategy.get("body"))
        if not _text(strategy.get("body")):
            strategy["body"] = strategy["why"]
    for day in plan.get("days") or []:
        if not isinstance(day, dict):
            continue
        day_title = _text(day.get("title"))
        day["focus"] = explain_heading(day_title, day.get("focus"), day.get("definitionOfDone"))
        if not _text(day.get("definitionOfDone")):
            day["definitionOfDone"] = _text(day.get("focus")) or day_title
        for task in day.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            if not _text(task.get("why")):
                task["why"] = _text(task.get("text")) or day_title
            if not _text(task.get("definitionOfDone")):
                task["definitionOfDone"] = _text(task.get("text")) or day_title
    for item in plan.get("priorityRecommendations") or []:
        if not isinstance(item, dict):
            continue
        item["reason"] = explain_heading(item.get("title"), item.get("reason"), item.get("measurableGoal"))
    return plan


def _span_from_text(blob: str) -> tuple[int, str] | None:
    if not blob:
        return None
    blob = re.sub(r"(\d+)\s*-\s*(day|week|month|year)", r"\1 \2", blob, flags=re.I)
    if re.search(r"\b(no fixed|not sure)\b", blob):
        return 90, "90 days, because no fixed deadline was given"
    years = re.search(r"(\d+)\s*year", blob)
    if years:
        count = max(int(years.group(1)), 1)
        return count * 365, f"{count} year" if count == 1 else f"{count} years"
    if re.search(r"\b(a year|within a year)\b", blob):
        return 365, "1 year"
    if re.search(r"\b(a few months|few months)\b", blob):
        return 90, "3 months"
    if re.search(r"\b(a month|one month)\b", blob):
        return 30, "1 month"
    months = re.search(r"(\d+)\s*month", blob)
    if months:
        count = max(int(months.group(1)), 1)
        return count * 30, f"{count} months"
    weeks = re.search(r"(\d+)\s*week", blob)
    if weeks:
        count = max(int(weeks.group(1)), 1)
        return count * 7, f"{count} weeks"
    if re.search(r"\b(this quarter|next quarter)\b", blob):
        return 90, "this quarter"
    days = re.search(r"(\d+)\s*day", blob)
    if days:
        count = max(int(days.group(1)), 1)
        return count, f"{count} days"
    return None


def stated_deadline(goal: dict[str, Any]) -> tuple[int, str] | None:
    """The deadline the user actually gave. Unknown stays unknown."""
    profile = goal.get("contextProfile") if isinstance(goal.get("contextProfile"), dict) else {}
    chosen = " ".join([_text(goal.get("timeframe")), _text((profile or {}).get("timeframe"))]).lower()
    parsed = _span_from_text(chosen)
    if parsed:
        return parsed
    return _span_from_text(_text(goal.get("statement")).lower())


def deadline_span(goal: dict[str, Any]) -> tuple[int, str]:
    """How many days the plan has to cover. The deadline the user chose wins."""
    return stated_deadline(goal) or (90, "90 days")


def fit_plan_to_deadline(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    span, label = deadline_span(goal)
    start = date.today()
    from services.goals.agents.scheduler import assign_step_schedule

    assign_step_schedule(plan, goal, span, label, start)
    phases = [phase for phase in (plan.get("phases") or []) if isinstance(phase, dict)]
    if phases:
        for index, phase in enumerate(phases):
            begin = int(index * span / len(phases)) + 1
            end = span if index == len(phases) - 1 else int((index + 1) * span / len(phases))
            end = max(end, begin)
            phase["windowLabel"] = f"Days {begin}–{end}"
            phase["outcome"] = explain_heading(
                phase.get("title"),
                phase.get("outcome"),
                f"Due by day {end} of {label}.",
            )
    definition = dict(plan.get("successDefinition") or {})
    definition["deadlineLabel"] = label
    plan["successDefinition"] = definition
    plan["horizonDays"] = span
    reality = _text(plan.get("realityCheck"))
    if label.lower() not in reality.lower():
        plan["realityCheck"] = f"The deadline is {label}. {reality}".strip()
    return plan


def apply_plan_brief(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    """Fill empty plan sections from the brief. Keep specific model writing."""
    brief = compose_plan_brief(goal)
    plan["realityCheck"] = brief["realityCheck"]
    plan["primaryFocus"] = brief["primaryFocus"]
    plan["secondaryFocus"] = brief["secondaryFocus"]
    plan["approachBeats"] = brief["approachBeats"]
    plan["starter"] = brief["starter"]
    plan["resourceSplit"] = brief["resourceSplit"]

    if _text(plan.get("positioningHeadline")).startswith(GENERIC_HEADLINE) or not _text(plan.get("positioningHeadline")):
        plan["positioningHeadline"] = brief["headline"]
    recommendation = _text(plan.get("recommendation")).lower()
    if not recommendation or GENERIC_RECOMMENDATION in recommendation:
        plan["recommendation"] = brief["realityCheck"]

    existing = [item for item in (plan.get("strategies") or []) if isinstance(item, dict)]
    if _generic_strategies(existing) or len(existing) < 3:
        merged = [] if _generic_strategies(existing) else list(existing)
        seen = {_text(item.get("title")).lower() for item in merged}
        for method in brief["methods"]:
            if len(merged) >= 3:
                break
            if method["title"].lower() in seen:
                continue
            merged.append(method)
            seen.add(method["title"].lower())
        plan["strategies"] = merged

    assumptions: list[str] = []
    for item in list(plan.get("assumptions") or []) + list(brief["assumptions"]):
        text = _text(item.get("text") if isinstance(item, dict) else item)
        if text and text not in assumptions:
            assumptions.append(text)
    plan["assumptions"] = assumptions[:8]

    scorecard = [row for row in (plan.get("scorecard") or []) if isinstance(row, dict)]
    if len(scorecard) < 3:
        seen = {_text(row.get("area")).lower() for row in scorecard}
        for row in brief["checks"]:
            if row["area"].lower() in seen:
                continue
            scorecard.append(row)
            seen.add(row["area"].lower())
        plan["scorecard"] = scorecard[:6]

    if not plan.get("openQuestions"):
        plan["openQuestions"] = [brief["firstMove"]]

    if brief["whatCounts"]:
        definition = plan.get("successDefinition") if isinstance(plan.get("successDefinition"), dict) else {}
        metrics = [row for row in (definition.get("metrics") or []) if isinstance(row, dict)]
        blob = " ".join(_text(row.get("target")).lower() for row in metrics)
        if "registered" not in blob and "come back" not in blob and "core action" not in blob:
            metrics.append({"label": "What counts", "target": brief["whatCounts"]})
        definition = {**definition, "metrics": metrics[:6]}
        if not definition.get("deadlineLabel"):
            definition["deadlineLabel"] = _text(goal.get("timeframe"))
        plan["successDefinition"] = definition
    plan = align_explanations(plan)
    plan = concretize_days(plan)
    plan = fit_plan_to_deadline(plan, goal)
    return plan


def review_extras(plan: dict[str, Any]) -> dict[str, Any]:
    """Fields the review screen can show beside the task list."""
    extras: dict[str, Any] = {}
    for key in ("realityCheck", "primaryFocus", "secondaryFocus", "starter"):
        if _text(plan.get(key)):
            extras[key] = plan[key]
    beats = [item for item in (plan.get("approachBeats") or []) if _text(item)]
    if beats:
        extras["brief"] = beats[:6]
    assumptions = []
    for item in plan.get("assumptions") or []:
        text = _text(item.get("text") if isinstance(item, dict) else item)
        if text:
            assumptions.append(text)
    if assumptions:
        extras["assumptions"] = assumptions[:6]
    split = [row for row in (plan.get("resourceSplit") or []) if isinstance(row, dict) and _text(row.get("item"))]
    if split:
        extras["resourceSplit"] = split[:6]
    questions = [row for row in (plan.get("openQuestions") or []) if isinstance(row, dict) and _text(row.get("prompt"))]
    if questions:
        extras["openQuestions"] = questions[:2]
    return extras
