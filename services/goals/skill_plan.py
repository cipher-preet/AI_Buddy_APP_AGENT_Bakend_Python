"""Cover a real deadline with ordered work blocks.

A single vague phase is not a plan. When the model returns one, or is down,
this builds the blocks locally from the skill, level, and hours the user gave.
It does not invent a course, an employer, a price, or a tool.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from services.goals.agents.scheduler import assign_step_schedule
from services.goals.brief import deadline_span, stated_deadline
from services.goals.domains import (
    context_profile,
    detect_goal_type,
    inferred_title,
    is_skip_answer,
    understanding_as_dict,
)
from services.goals.intake import goal_signals, plan_family
from services.goals.step_actions import template_noise

_GENERIC_SUBJECTS = {
    "learn a skill",
    "learn the skill",
    "learn skill",
    "a skill",
    "skill",
    "new goal",
    "goal",
    "my goal",
    "this goal",
}
_SKILL_STOP = {
    "learn",
    "skill",
    "this",
    "that",
    "with",
    "your",
    "from",
    "have",
    "will",
    "want",
    "able",
    "when",
    "into",
    "about",
    "goal",
    "the",
    "and",
    "for",
}
_STARTED = re.compile(
    r"already|some practice|intermediate|advanced|expert|underway|not a beginner",
    re.I,
)
_BEGINNER = re.compile(
    r"beginner|from zero|not started|new to|never|scratch|\bzero\b",
    re.I,
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _clean_subject(value: Any, limit: int = 80) -> str:
    text = " ".join(_text(value).split())
    if not text or is_skip_answer(text) or text.upper() == "UNKNOWN":
        return ""
    if text.lower().strip(" .") in _GENERIC_SUBJECTS:
        return ""
    if text.lower().startswith("learn a skill"):
        return ""
    return text[:limit]


def named_skill(goal: dict[str, Any]) -> str:
    """The skill the user named. A generic title like "Learn a skill" is not it."""
    profile = context_profile(goal)
    understanding = understanding_as_dict(goal.get("understanding"))
    for candidate in (
        goal.get("description"),
        profile.get("description"),
        profile.get("subject"),
        understanding.get("subdomain"),
        goal.get("category"),
    ):
        cleaned = _clean_subject(candidate)
        if cleaned:
            return cleaned
    title = _clean_subject(goal.get("title"))
    if title and "learn a skill" not in title.lower():
        return title
    return ""


def skill_phrase(goal: dict[str, Any]) -> str:
    return named_skill(goal) or "this skill"


def practice_stage(goal: dict[str, Any]) -> str:
    """"beginner" starts with an assessment. "started" does not repeat beginner drills."""
    profile = context_profile(goal)
    level = _text(profile.get("startingLevel") or profile.get("currentRole"))
    if level and level.upper() != "UNKNOWN" and not is_skip_answer(level):
        return _stage_from_text(level)
    understanding = understanding_as_dict(goal.get("understanding"))
    understood = _text(understanding.get("startingLevel"))
    if understood and understood.upper() != "UNKNOWN":
        return _stage_from_text(understood)
    return _stage_from_text(_text(goal.get("statement")))


def _stage_from_text(text: str) -> str:
    lowered = text.lower().strip(" .")
    if not lowered or lowered == "unknown":
        return "beginner"
    if lowered in {"some", "using", "regular", "started", "already started"}:
        return "started"
    if lowered in {"zero", "new", "beginner", "from zero"}:
        return "beginner"
    if _STARTED.search(lowered):
        return "started"
    if _BEGINNER.search(lowered):
        return "beginner"
    return "beginner"


def is_skill_goal(goal: dict[str, Any]) -> bool:
    understanding = understanding_as_dict(goal.get("understanding"))
    if (
        detect_goal_type(
            _text(goal.get("statement")),
            _text(goal.get("goalType")),
            str(understanding.get("domain") or ""),
        )
        == "skill"
    ):
        return True
    signals = goal_signals(goal, understanding)
    return plan_family(signals) == "skill"


def block_count(span: int) -> int:
    """Several ordered blocks across the deadline. Not one, and not one per day."""
    if span >= 45:
        return 8
    if span >= 21:
        return 6
    return 4


def _hours_label(goal: dict[str, Any]) -> str:
    raw = goal.get("availableHoursPerWeek")
    if raw in (None, "", "UNKNOWN"):
        profile = context_profile(goal)
        raw = profile.get("availableHoursPerWeek")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return ""
    if value <= 0 or value > 168:
        return ""
    if value == int(value):
        return str(int(value))
    return str(value)


def _session_minutes(goal: dict[str, Any]) -> int:
    label = _hours_label(goal)
    if not label:
        return 50
    weekly = float(label)
    return max(30, min(90, int(round(weekly * 60 / 4))))


def _done_day(day: dict[str, Any]) -> bool:
    if _text(day.get("completedOn")):
        return True
    return _text(day.get("status")).lower() in {"done", "skipped"}


def _plan_blob(plan: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("coachNote", "immediateNextStep", "summary"):
        parts.append(_text(plan.get(key)))
    for day in plan.get("days") or []:
        if not isinstance(day, dict):
            continue
        parts.append(_text(day.get("title")))
        parts.append(_text(day.get("focus")))
        for task in day.get("tasks") or []:
            if isinstance(task, dict):
                parts.append(_text(task.get("text")))
    return " ".join(parts)


def _goal_blob(goal: dict[str, Any]) -> str:
    profile = context_profile(goal)
    return " ".join(
        [
            _text(goal.get("statement")),
            _text(goal.get("title")),
            _text(goal.get("description")),
            _text(goal.get("successCriteria")),
            _text(profile.get("description")),
            _text(profile.get("startingLevel")),
        ]
    )


def _mentions_skill(plan: dict[str, Any], skill: str) -> bool:
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", skill.lower())
        if len(word) >= 3 and word not in _SKILL_STOP
    ]
    if not words:
        return True
    blob = _plan_blob(plan).lower()
    return any(word in blob for word in words)


def _open_days(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [day for day in (plan.get("days") or []) if isinstance(day, dict) and not _done_day(day)]


def should_replace_with_skill_plan(plan: dict[str, Any], goal: dict[str, Any]) -> bool:
    """A skill goal with one phase, or a canned course, is not ready to save."""
    if not is_skill_goal(goal):
        return False
    stated = stated_deadline(goal)
    if stated is None or stated[0] < 14:
        return False
    days = [day for day in (plan.get("days") or []) if isinstance(day, dict)]
    if any(_done_day(day) for day in days):
        return False
    open_days = _open_days(plan)
    if len(open_days) <= 1:
        return True
    goal_text = _goal_blob(goal)
    if template_noise(_plan_blob(plan)) and not template_noise(goal_text):
        return True
    skill = named_skill(goal)
    if skill and len(open_days) < 6 and not _mentions_skill(plan, skill):
        return True
    return False


def needs_general_expansion(plan: dict[str, Any], goal: dict[str, Any]) -> bool:
    """Any other long goal still cannot be saved as a single step."""
    if is_skill_goal(goal):
        return False
    stated = stated_deadline(goal)
    if stated is None or stated[0] < 14:
        return False
    days = [day for day in (plan.get("days") or []) if isinstance(day, dict)]
    if any(_done_day(day) for day in days):
        return False
    return len(_open_days(plan)) <= 1


def _task(text: str, why: str) -> dict[str, Any]:
    sentence = text if text.endswith(".") else f"{text}."
    return {
        "text": sentence[:240],
        "why": why[:240],
        "definitionOfDone": f"Done when this is finished and you can check it: {sentence}"[:240],
        "done": False,
        "note": "",
    }


def _with_hours(actions: list[tuple[str, str]], goal: dict[str, Any], subject: str) -> list[tuple[str, str]]:
    hours = _hours_label(goal)
    if not hours or len(actions) >= 4:
        return actions
    return [
        *actions,
        (
            f"Schedule this block inside the {hours} hours you can give this week",
            f"The week only has the {hours} hours you named for {subject}.",
        ),
    ]


def _beginner_blocks(subject: str, goal: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    s = subject
    first = _with_hours(
        [
            (f"List what you can and cannot do yet with {s}", f"Later practice should start from what is true about {s}."),
            (f"Write the result you want from {s} by the end of this plan", f"The later blocks need a result for {s} you can check."),
            (f"Pick the first small piece of {s} you will practice next", f"One piece of {s} is enough to start."),
        ],
        goal,
        s,
    )
    return [
        (f"See where you are with {s}", first),
        (
            f"Learn the first fundamentals of {s}",
            [
                (f"Study one core idea in {s} and write it in your own words", f"You need your own words for {s} before you practice it."),
                (f"List three moves in {s} you need before you practice", f"The practice block should use moves you can name."),
                (f"Note what is still confusing about {s}", f"The confusing part of {s} is what the next practice is for."),
            ],
        ),
        (
            f"Practice the fundamentals of {s}",
            [
                (f"Practice the first move in {s} until you can repeat it", f"A move in {s} counts when you can do it again."),
                (f"Record what went wrong in {s} and what you will repeat", f"The next try at {s} should fix what you just saw."),
                (f"Compare this attempt with the idea you wrote down for {s}", f"Practice should match the idea, not drift away from it."),
            ],
        ),
        (
            f"Learn the next piece of {s}",
            [
                (f"Study the next piece of {s} that builds on the first", f"The next piece should use the fundamentals of {s} you already practiced."),
                (f"Write how this piece of {s} connects to the first idea", f"The connection keeps the plan about {s}."),
                (f"List one exercise that uses only this new piece of {s}", f"One exercise shows whether the new piece of {s} is ready."),
            ],
        ),
        (
            f"Practice that next piece of {s}",
            [
                (f"Practice the new piece of {s} on a short exercise", f"A short exercise is enough to test the new piece of {s}."),
                (f"Check the {s} result and write what still breaks", f"The break tells you what to practice next in {s}."),
                (f"Run the {s} exercise once more after you fix that break", f"The second run shows the fix in {s} actually held."),
            ],
        ),
        (
            f"Make a small proof with {s}",
            [
                (f"Define a small proof you can finish with {s}", f"The proof should be small enough to finish with {s}."),
                (f"Build that proof using the pieces of {s} you have practiced", f"The proof is how {s} shows up outside the exercises."),
                (f"Write what the proof shows about your {s}", f"You should be able to say what the proof proves about {s}."),
            ],
        ),
        (
            f"Get feedback on your {s}",
            [
                (f"Ask someone to look at the {s} proof and say what is unclear", f"Another person can see a gap in {s} you will skip."),
                (f"Write the feedback about {s} in their words", f"Their words are the list of what to fix in {s}."),
                (f"Fix the one issue they named in the {s} proof", f"One fix is enough before you call the {s} proof finished."),
            ],
        ),
        (
            f"Finish the {s} proof",
            [
                (f"Review the {s} proof against the result you wrote at the start", f"The proof should match the result you wanted from {s}."),
                (f"Finish the remaining piece of the {s} proof", f"An open piece means the {s} proof is not done."),
                (f"Show the proof and write what you can now do with {s}", f"The last note is what changed in {s}, not a new topic."),
            ],
        ),
    ]


def _started_blocks(subject: str, goal: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    s = subject
    first = _with_hours(
        [
            (f"Write what you already do with {s}", f"The plan should start from the {s} you already use."),
            (f"Name the one gap between that and the result you want from {s}", f"The gap is the work, not a repeat of {s} you can already do."),
            (f"Pick the next piece of {s} that is past the work you already do", f"The next piece of {s} should be new, not a beginner drill."),
        ],
        goal,
        s,
    )
    return [
        (f"Name the gap in {s}", first),
        (
            f"Practice the next piece of {s}",
            [
                (f"Practice that next piece of {s} on work you already do", f"Use {s} on real work instead of starting over."),
                (f"Record where your current {s} still breaks", f"The break is the part of {s} this plan is for."),
                (f"Run one more attempt at that piece of {s}", f"A second attempt shows whether the piece of {s} is getting clearer."),
            ],
        ),
        (
            f"Use {s} on work you already do",
            [
                (f"Apply the new piece of {s} to something you already make", f"The new piece should land in work you already do with {s}."),
                (f"Compare it with your earlier work in {s}", f"The comparison shows what changed in {s}."),
                (f"Note what improved in {s} and what did not", f"Keep the part of {s} that improved and name the rest."),
            ],
        ),
        (
            f"Push the weak part of {s}",
            [
                (f"Practice the weak part of {s} that the comparison showed", f"Time goes to the weak part of {s}, not to work you already do well."),
                (f"Write the change you want to see in the next attempt at {s}", f"The next attempt needs a change you can see."),
                (f"Run one more attempt and mark the difference in {s}", f"The mark shows whether that weak part of {s} moved."),
            ],
        ),
        (
            f"Make a proof of the new {s}",
            [
                (f"Define a small proof that needs the new {s}, not the work you already do", f"The proof should require the new piece of {s}."),
                (f"Build that proof with {s}", f"Building it is the check that the new {s} holds."),
                (f"Write what a viewer should notice in the {s} proof", f"If a viewer cannot see it, the proof of {s} is not clear yet."),
            ],
        ),
        (
            f"Get feedback on the new {s}",
            [
                (f"Ask someone who knows your work to review the {s} proof", f"Ask about the new {s}, not about beginner drills."),
                (f"Write what they said was still missing from {s}", f"Their note is the next fix, not a new plan."),
                (f"Fix that missing piece in the {s} proof", f"The proof should include the missing piece of {s}."),
            ],
        ),
        (
            f"Tighten the {s} proof",
            [
                (f"Review the proof and remove anything that does not show the new {s}", f"Extra material hides the new {s}."),
                (f"Practice the part of {s} that still feels shaky", f"Shaky does not mean go back to the start of {s}."),
                (f"Update the proof with that practice of {s}", f"The proof should contain the latest practice of {s}."),
            ],
        ),
        (
            f"Finish and show the {s} work",
            [
                (f"Check the proof against the result you wanted from {s}", f"The result you named is the check for {s}."),
                (f"Finish the last open piece of the {s} proof", f"One open piece is still unfinished {s}."),
                (f"Show the finished {s} work and write what you can do now that you could not do at the start", f"The note should be the change in {s}, not a new goal."),
            ],
        ),
    ]


def _general_blocks(subject: str, goal: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    s = subject
    first = _with_hours(
        [
            (f"List what is already true for {s} and what is not", f"The rest of the plan for {s} depends on the current state."),
            (f"Write the result you want from {s} by the end of this plan", f"Later checks need a result for {s} you can see."),
            (f"Pick the first piece of {s} you will do next", f"One piece of {s} is the start."),
        ],
        goal,
        s,
    )
    return [
        (f"See where you are with {s}", first),
        (
            f"Do the first piece of {s}",
            [
                (f"Write the first piece of {s} you will finish", f"The piece should be small enough to finish for {s}."),
                (f"Do that piece of {s} and keep the result where you can find it", f"The result is what the next check of {s} looks at."),
                (f"Note what is done and what is still open for {s}", f"Open items stay on the list for {s}."),
            ],
        ),
        (
            f"Check the first piece of {s}",
            [
                (f"Review what you produced for {s}", f"The review is about the piece you just made for {s}."),
                (f"Write what worked and what did not for {s}", f"The next piece of {s} should follow that note."),
                (f"Pick the next piece of {s} from that note", f"Do not add a new direction for {s} before this one is clear."),
            ],
        ),
        (
            f"Do the next piece of {s}",
            [
                (f"Do the next piece of {s} you picked", f"This is the next concrete piece of {s}."),
                (f"Compare it with the result you wanted from {s}", f"The comparison keeps the work pointed at {s}."),
                (f"Note the gap that is still open for {s}", f"The gap is the remaining work on {s}."),
            ],
        ),
        (
            f"Ask for a reaction to {s}",
            [
                (f"Ask someone to look at the work for {s} and say what is unclear", f"Another person can see a hole in {s} you missed."),
                (f"Write their words down about {s}", f"Their words are the fix list for {s}."),
                (f"Fix the one issue they named for {s}", f"One fix is the next move on {s}."),
            ],
        ),
        (
            f"Finish the work for {s}",
            [
                (f"Check the work against the result you wrote at the start for {s}", f"The result you wrote is the finish line for {s}."),
                (f"Finish the last open piece of {s}", f"An open piece means {s} is not finished."),
                (f"Write what you can now do for {s} that you could not do at the start", f"The last note is the change in {s}."),
            ],
        ),
    ]


def _select(blocks: list[tuple[str, list[tuple[str, str]]]], count: int) -> list[tuple[str, list[tuple[str, str]]]]:
    if count >= len(blocks):
        return blocks
    if count >= 6 and len(blocks) >= 8:
        return [blocks[index] for index in (0, 1, 2, 5, 6, 7)]
    if count >= 4 and len(blocks) >= 8:
        return [blocks[index] for index in (0, 2, 5, 7)]
    if count >= 4 and len(blocks) >= 6:
        return [blocks[index] for index in (0, 1, 4, 5)]
    return blocks[:count]


def _days_from_blocks(
    blocks: list[tuple[str, list[tuple[str, str]]]],
    goal: dict[str, Any],
) -> list[dict[str, Any]]:
    minutes = _session_minutes(goal)
    days: list[dict[str, Any]] = []
    for index, (title, actions) in enumerate(blocks, start=1):
        tasks = []
        for action_index, (text, why) in enumerate(actions[:4], start=1):
            task = _task(text, why)
            task["id"] = f"step-{index}-t{action_index}"
            tasks.append(task)
        days.append(
            {
                "id": f"step-{index}",
                "dayIndex": index,
                "dateLabel": "",
                "scheduledDate": "",
                "scheduleDay": index,
                "title": title[:160],
                "focus": "",
                "category": "Step",
                "minutes": minutes,
                "hours": None,
                "tasks": tasks,
                "definitionOfDone": _text(tasks[-1].get("definitionOfDone")) if tasks else "",
                "status": "open",
                "notes": "",
                "example": False,
            }
        )
    return days


def _subject_for_general(goal: dict[str, Any]) -> str:
    skill = named_skill(goal)
    if skill:
        return skill
    title = _clean_subject(inferred_title(goal, understanding_as_dict(goal.get("understanding"))), 80)
    return title or "this goal"


def _clear_canned_notes(plan: dict[str, Any], goal: dict[str, Any]) -> None:
    allowed = template_noise(_goal_blob(goal))
    for key in ("coachNote", "immediateNextStep"):
        if template_noise(_text(plan.get(key))) and not allowed:
            plan[key] = ""


def _phases(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not days:
        return []
    groups = 4 if len(days) >= 6 else len(days)
    phases: list[dict[str, Any]] = []
    for index in range(groups):
        start = int(index * len(days) / groups)
        end = len(days) if index == groups - 1 else int((index + 1) * len(days) / groups)
        chunk = days[start:end] or [days[min(start, len(days) - 1)]]
        begin = int(chunk[0].get("scheduleDay") or start + 1)
        finish = int(chunk[-1].get("scheduleDay") or end)
        phases.append(
            {
                "id": f"phase-{index + 1}",
                "title": _text(chunk[0].get("title"))[:160],
                "windowLabel": f"Days {begin}–{finish}",
                "outcome": _text(chunk[-1].get("definitionOfDone"))[:320],
            }
        )
    return phases


def _milestones(days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    milestones: list[dict[str, Any]] = []
    previous = ""
    for day in days:
        title = _text(day.get("title")) or "Step"
        criteria = [
            _text(task.get("text"))
            for task in (day.get("tasks") or [])
            if isinstance(task, dict) and _text(task.get("text"))
        ][:3]
        milestones.append(
            {
                "title": title[:160],
                "successCriteria": criteria or [title],
                "dueLabel": f"Day {int(day.get('scheduleDay') or day.get('dayIndex') or 1)}",
                "dependencies": [previous] if previous else [],
            }
        )
        previous = title
    return milestones


def _schedule(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    span, label = stated_deadline(goal) or deadline_span(goal)
    _clear_canned_notes(plan, goal)
    assign_step_schedule(plan, goal, span, label, date.today())
    plan["horizonDays"] = span
    days = [day for day in (plan.get("days") or []) if isinstance(day, dict)]
    plan["phases"] = _phases(days)
    plan["milestones"] = _milestones(days)
    return plan


def install_skill_blocks(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    span, _label = stated_deadline(goal) or deadline_span(goal)
    stage = practice_stage(goal)
    subject = skill_phrase(goal)
    catalog = _started_blocks(subject, goal) if stage == "started" else _beginner_blocks(subject, goal)
    count = block_count(span)
    if stage != "started" and count >= 6:
        # Beginner path keeps assessment, fundamentals, practice, proof, and feedback.
        count = min(count, 8)
    plan["days"] = _days_from_blocks(_select(catalog, count), goal)
    trace = [str(item) for item in (plan.get("agentTrace") or [])]
    if "steps:local-skill" not in trace:
        trace.append("steps:local-skill")
    plan["agentTrace"] = trace
    return plan


def install_general_blocks(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    span, _label = stated_deadline(goal) or deadline_span(goal)
    subject = _subject_for_general(goal)
    count = 6 if span >= 45 else 4
    plan["days"] = _days_from_blocks(_select(_general_blocks(subject, goal), count), goal)
    trace = [str(item) for item in (plan.get("agentTrace") or [])]
    if "steps:local-span" not in trace:
        trace.append("steps:local-span")
    plan["agentTrace"] = trace
    return plan


def repair_plan_locally(plan: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    """Leave the plan as written. A thin plan is sent back to the model, not a local course."""
    del goal
    return plan
