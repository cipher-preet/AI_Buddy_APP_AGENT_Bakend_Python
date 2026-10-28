"""Turn outcome labels into checklist actions a person can follow.

A step titled "Scope Sign-off" with one line, "Scope statement signed by the
client", does not tell anyone what to do. This module turns that label into
the draft, the ask, and the saved result. It never emits "Do the work for …".
"""

from __future__ import annotations

import re
from typing import Any

_GENERIC = re.compile(
    r"do the work for|save something you can open|do the smallest version|do the first small piece|you can open the result|this phase['’]s outcome|can be shown|outcome is completed",
    re.I,
)
_TEMPLATE_PHRASES = (
    "language fundamentals",
    "platform preference",
    "this phase's outcome",
    "can be shown",
    "outcome is completed",
)
_VERB = re.compile(
    r"^(write|interview|ask|list|message|email|call|publish|post|export|define|count|set|create|draft|record|build|test|measure|invite|contact|open|pick|choose|schedule|practice|review|send|share|track|name|map|find|collect|run|fix|ship|launch|prepare|make|show|talk|meet|book|save|check|confirm|outline|design|add|remove|update|compare|calculate|decide|sign|apply|study|read|train|walk|smoke|verify|improve|expand|finalize|freeze|deploy|watch|capture|turn|do|finish|complete|start|document|assemble|hold|give|break|note|mark|package|walk)\b",
    re.I,
)
_OUTCOME_END = re.compile(
    r"\b(signed\s+by\s+.+|approved|held|distributed|updated|delivered|obtained|reviewed|completed|re-?baselined(?:\s+if\s+needed)?|documented|finished|created|prepared|saved)\s*$",
    re.I,
)
_QUOTE = re.compile(r"do the work for\s*[\"“'](.+?)[\"”']", re.I)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize(value: str) -> str:
    cleaned = _text(value)
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        cleaned = cleaned.replace(dash, "-")
    return re.sub(r"\s+", " ", cleaned).strip()


def _echoes(heading: str, text: str) -> bool:
    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    left = norm(heading)
    right = norm(text)
    if not left or not right:
        return False
    if right == left:
        return True
    if right.startswith(left) and len(right) <= len(left) + 42:
        return True
    return bool(left in right and len(right) <= len(left) + 42)


def _generic(text: str) -> bool:
    return bool(_GENERIC.search(text))


def _fold(text: str) -> str:
    return _normalize(text).replace("’", "'").replace("‘", "'").lower()


def template_noise(text: str) -> bool:
    """Canned course lines and the filler outcome sentence. Real work does not say these."""
    folded = _fold(text)
    return any(phrase in folded for phrase in _TEMPLATE_PHRASES)


def unwrap_generic(text: str) -> str:
    """Pull the real outcome out of the old canned sentence, if that is what we stored."""
    cleaned = _normalize(text)
    match = _QUOTE.search(cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def is_clear_action(text: str, title: str = "") -> bool:
    """A clear action says what to do. An outcome label only says what should be true."""
    cleaned = unwrap_generic(text)
    if not cleaned or _generic(cleaned):
        return False
    if title and _echoes(title, cleaned):
        return False
    words = cleaned.split()
    if _VERB.match(cleaned) and len(words) >= 3 and not _generic(cleaned):
        return True
    if len(words) >= 6 and not _OUTCOME_END.search(cleaned):
        return True
    return False


def _soft(noun: str) -> str:
    cleaned = _normalize(noun).strip(" .")
    if not cleaned:
        return "this"
    if re.match(r"^(a|an|the)\b", cleaned, re.I):
        return cleaned[0].lower() + cleaned[1:]
    if cleaned.isupper() or re.match(r"^[A-Z0-9-]{2,}$", cleaned):
        return f"the {cleaned}"
    return f"the {cleaned[0].lower()}{cleaned[1:]}"


def _cap(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def _item(text: str, why: str, definition: str) -> dict[str, Any]:
    return {
        "text": _normalize(text)[:240],
        "why": _normalize(why)[:240],
        "definitionOfDone": _normalize(definition)[:240],
    }


def _suffix(part: str) -> tuple[str, str]:
    match = _OUTCOME_END.search(part)
    if not match:
        return part.strip(), ""
    head = part[: match.start()].strip(" ,")
    return head, match.group(1).strip()


def split_clauses(text: str) -> list[str]:
    """Split a list of outcomes. Keep 'signed by A and B' as one clause."""
    cleaned = unwrap_generic(text).strip(" .")
    if not cleaned:
        return []
    chunks = [chunk.strip(" .") for chunk in cleaned.split(",") if chunk.strip(" .")] or [cleaned]
    parts: list[str] = []
    for chunk in chunks:
        if re.search(r"\bsigned\s+by\b", chunk, re.I):
            parts.append(chunk)
            continue
        bits = [bit.strip(" .") for bit in re.split(r"\s+\band\b\s+", chunk, flags=re.I) if bit.strip(" .")]
        parts.extend(bits or [chunk])
    if len(parts) <= 1:
        return parts or [cleaned]
    if any(_suffix(part)[1] for part in parts[:-1]):
        return parts
    head, suffix = _suffix(parts[-1])
    if not suffix or not head:
        return parts
    return [f"{part} {suffix}".strip() for part in parts[:-1]] + [f"{head} {suffix}".strip()]


def _signed(doc: str, who: str) -> list[dict[str, Any]]:
    noun = _soft(doc)
    people = who.strip(" .") or "the people who must accept it"
    return [
        _item(
            f"Write {noun}: what is included, what is out, the deliverables, and the assumptions.",
            f"A signature only helps if {noun} says what you will and will not do.",
            f"A draft of {noun} is saved.",
        ),
        _item(
            f"Send {noun} to {people}, and ask each person to sign or mark what they will not accept.",
            "These are the people who can stop the work later if they never agreed.",
            f"Each required person has replied on {noun}.",
        ),
        _item(
            f"Save the signed copy of {noun}. Do not start the next step until that copy is stored.",
            "Later questions about scope get answered from the signed copy.",
            f"The signed {doc.strip()} is stored where the team can find it.",
        ),
    ]


def _approved_sequence(what: str) -> list[dict[str, Any]]:
    noun = _soft(what)
    return [
        _item(
            f"Draft {noun} so a reviewer can see the choices, not just the heading.",
            f"Approval is meaningless if {noun} is only a title.",
            f"A reviewable draft of {noun} is saved.",
        ),
        _item(
            f"Ask for approval of {noun} and write what was accepted and what must change.",
            "The next step should follow the accepted version.",
            f"The review notes for {noun} are saved.",
        ),
        _item(
            f"Save the approved version of {noun} and use that version from here on.",
            "A later draft should not quietly replace the approved one.",
            f"The approved {what.strip()} is the copy everyone uses.",
        ),
    ]


def _held_sequence(what: str) -> list[dict[str, Any]]:
    noun = _soft(what)
    return [
        _item(
            f"Write the agenda for {noun}, including the decisions you need, and send it the day before.",
            "People cannot decide in the room if they see the questions for the first time.",
            f"The agenda for {noun} has been sent.",
        ),
        _item(
            f"Hold {noun} and write every decision, owner, and open question before people leave.",
            "A meeting that ends without notes leaves the same questions open.",
            f"The notes from {noun} list each decision and its owner.",
        ),
        _item(
            f"Send the notes from {noun} to anyone who owns a decision but was not there.",
            "An owner who never sees the note will not do the work.",
            f"The notes from {noun} were sent to the owners.",
        ),
    ]


def action_for_clause(clause: str) -> dict[str, Any]:
    """One checklist line for one outcome in a longer list."""
    text = _normalize(clause).strip(" .")
    signed = re.match(r"^(?P<doc>.+?)\s+signed\s+by\s+(?P<who>.+)$", text, re.I)
    if signed:
        noun = _soft(signed.group("doc"))
        who = signed.group("who").strip()
        return _item(
            f"Write {noun} and get it signed by {who}.",
            f"Do not treat the work as agreed until {who} has signed {noun}.",
            f"The signed {signed.group('doc').strip()} is stored.",
        )
    rebased = re.match(r"^(?P<what>.+?)\s+re-?baselined(?:\s+if\s+needed)?$", text, re.I)
    if rebased:
        noun = _soft(rebased.group("what"))
        return _item(
            f"If {noun} has slipped, rewrite the remaining dates and share them with the people who depend on them.",
            "A date that moved only helps if the people waiting on it know.",
            f"The current dates for {noun} are written down and shared.",
        )
    reviewed = re.match(r"^(?P<what>.+?)\s+reviewed$", text, re.I)
    if reviewed:
        noun = _soft(reviewed.group("what"))
        return _item(
            f"Compare {noun} with what was promised and write what is on time, what is late, and why.",
            "The review is the list of slips, not a meeting that only says things are fine.",
            f"A short review of {noun} says what is on time and what is late.",
        )
    held = re.match(r"^(?P<what>.+?)\s+held$", text, re.I)
    if held:
        noun = _soft(held.group("what"))
        return _item(
            f"Schedule {noun}, send the agenda the day before, and write every decision and owner before it ends.",
            f"{_cap(noun)} only counts if the decisions and owners are written down.",
            f"The notes from {noun} list each decision and its owner.",
        )
    distributed = re.match(r"^(?P<what>.+?)\s+distributed$", text, re.I)
    if distributed:
        noun = _soft(distributed.group("what"))
        return _item(
            f"Write {noun} — who gets an update, how often, and where it is posted — and send it to the people who need it.",
            "People miss the work when they do not know which update is theirs.",
            f"{_cap(noun)} has been sent to the people who need it.",
        )
    obtained = re.match(r"^(?P<what>.+?)\s+obtained$", text, re.I)
    if obtained:
        noun = _soft(obtained.group("what"))
        return _item(
            f"Ask for {noun} in writing, and save the reply before you treat this step as done.",
            "A verbal yes is easy to lose. The reply is the record.",
            f"The written reply for {noun} is saved.",
        )
    delivered = re.match(r"^(?P<what>.+?)\s+delivered$", text, re.I)
    if delivered:
        noun = _soft(delivered.group("what"))
        return _item(
            f"Assemble {noun} with the deliverables, decisions, and open items, and give it to the people who will use it next.",
            "Delivery means the next owner has the package, not that you finished a file.",
            f"{_cap(noun)} was handed to the people who will use it.",
        )
    updated = re.match(r"^(?P<what>.+?)\s+updated$", text, re.I)
    if updated:
        noun = _soft(updated.group("what"))
        return _item(
            f"Update {noun} with what changed, who owns each item, and the next action.",
            "An old list sends people after risks that already changed.",
            f"{_cap(noun)} shows the current owners and next actions.",
        )
    approved = re.match(r"^(?P<what>.+?)\s+approved$", text, re.I)
    if approved:
        noun = _soft(approved.group("what"))
        return _item(
            f"Write {noun}, then get approval before you start the next step.",
            "Later work should follow the approved version, not a draft.",
            f"The approved copy of {noun} is saved.",
        )
    finished = re.match(r"^(?P<what>.+?)\s+(?:completed|finished|created|prepared|documented|saved)$", text, re.I)
    if finished:
        noun = _soft(finished.group("what"))
        return _item(
            f"Finish {noun} and write what is done, what is still open, and who needs it next.",
            f"The next step needs {noun} in a form someone else can check.",
            f"{_cap(noun)} is finished and the open items are listed.",
        )
    noun = _soft(text)
    return _item(
        f"Write {noun} so someone else can see what is done, what is missing, and who needs it next.",
        f"The next step needs {noun} written down, not just named.",
        f"{_cap(noun)} is written down and ready to hand over.",
    )


def _plain_actions(subject: str) -> list[dict[str, Any]]:
    noun = _normalize(subject).strip(" .") or "this step"
    if len(noun) > 80:
        noun = noun[:77].rstrip() + "..."
    return [
        _item(
            f"Write what you will produce for {noun}, and what finished looks like.",
            f"The next action needs a result for {noun} that you can point to.",
            f"A short note names the result for {noun}.",
        ),
        _item(
            f"Do the first piece of {noun} and write what is done and what is still open.",
            f"A step is finished when the result for {noun} exists.",
            f"The note for {noun} says what is done and what is still open.",
        ),
    ]


def expand_outcome(raw: str, title: str) -> list[dict[str, Any]]:
    """Expand one outcome label into the actions that produce it."""
    text = unwrap_generic(raw) or _normalize(title)
    if _generic(text) or template_noise(text):
        heading = _normalize(title)
        if not heading or _generic(heading) or template_noise(heading):
            heading = "this step"
        return _plain_actions(heading)
    clauses = split_clauses(text)
    if len(clauses) >= 2:
        return [action_for_clause(clause) for clause in clauses[:5]]
    clause = clauses[0] if clauses else _normalize(title)
    signed = re.match(r"^(?P<doc>.+?)\s+signed\s+by\s+(?P<who>.+)$", clause, re.I)
    if signed:
        return _signed(signed.group("doc"), signed.group("who"))
    if re.match(r"^.+\s+approved$", clause, re.I):
        return _approved_sequence(re.sub(r"\s+approved$", "", clause, flags=re.I))
    if re.match(r"^.+\s+held$", clause, re.I):
        return _held_sequence(re.sub(r"\s+held$", "", clause, flags=re.I))
    primary = action_for_clause(clause)
    label = clause.strip().rstrip(".")
    if len(label) > 120:
        label = label[:117].rstrip() + "..."
    follow = _item(
        f"Check the result against “{label}” and write what is done and what is still open.",
        "The step is finished when you can show the result, not when the heading sounds done.",
        "A short note says what is done and what is still open.",
    )
    return [primary, follow]


def _focus_from(tasks: list[dict[str, Any]]) -> str:
    texts = [_text(task.get("text")).rstrip(".") for task in tasks if _text(task.get("text"))]
    if len(texts) >= 2:
        second = texts[1][:1].lower() + texts[1][1:]
        return f"{texts[0]}. Then {second}."[:500]
    return (texts[0] if texts else "")[:500]


def _weak_copy(text: str, title: str) -> bool:
    cleaned = _text(text)
    if not cleaned or _generic(cleaned):
        return True
    if title and (_echoes(title, cleaned) or cleaned.lower().startswith(f"{title.lower()}:")):
        return True
    return False


def _done_day(day: dict[str, Any]) -> bool:
    if _text(day.get("completedOn")):
        return True
    return _text(day.get("status")).lower() == "skipped"


def clarify_day(day: dict[str, Any], index: int = 1) -> dict[str, Any]:
    """Replace outcome-only tasks on one step. Leave real actions alone."""
    if not isinstance(day, dict) or _done_day(day):
        return day
    title = _text(day.get("title")) or f"Step {index}"
    original = [task for task in (day.get("tasks") or []) if isinstance(task, dict) and _text(task.get("text"))]
    clear = [task for task in original if is_clear_action(_text(task.get("text")), title)]
    next_day = dict(day)
    if original and len(clear) == len(original) and len(clear) >= 2:
        if _weak_copy(_text(day.get("focus")), title):
            next_day["focus"] = _focus_from(original)
        if _weak_copy(_text(day.get("definitionOfDone")), title):
            done = _text(original[-1].get("definitionOfDone"))
            if done and not _weak_copy(done, title):
                next_day["definitionOfDone"] = done
        return next_day

    built: list[dict[str, Any]] = []
    sources = original or [{"text": title, "done": False}]
    for task in sources:
        source_text = _text(task.get("text")) or title
        if is_clear_action(source_text, title):
            built.append(dict(task))
            continue
        for action in expand_outcome(source_text, title):
            built.append(
                {
                    "text": action["text"],
                    "done": bool(task.get("done")),
                    "why": action["why"],
                    "definitionOfDone": action["definitionOfDone"],
                    "note": _text(task.get("note")) if not built else "",
                }
            )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in built:
        key = re.sub(r"[^a-z0-9]+", " ", _text(task.get("text")).lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(task)
        if len(deduped) >= 5:
            break
    if not deduped:
        return day
    day_id = _text(day.get("id")) or f"day-{index}"
    old_texts = [_text(task.get("text")) for task in original]
    new_texts = [_text(task.get("text")) for task in deduped]
    for task_index, task in enumerate(deduped, start=1):
        if old_texts != new_texts:
            task["id"] = f"{day_id}-t{task_index}"
        elif task_index - 1 < len(original) and _text(original[task_index - 1].get("id")):
            task["id"] = original[task_index - 1].get("id")
        else:
            task["id"] = f"{day_id}-t{task_index}"
        task["done"] = bool(task.get("done"))
    next_day["tasks"] = deduped
    next_day["focus"] = _focus_from(deduped)
    next_day["definitionOfDone"] = _text(deduped[-1].get("definitionOfDone")) or _text(day.get("definitionOfDone"))
    return next_day


def clarify_plan_steps(plan: dict[str, Any]) -> dict[str, Any]:
    """The model writes the actions. Local code does not replace them."""
    return plan


def steps_need_model(plan: dict[str, Any]) -> bool:
    """True when a step has fewer than two actions. The model decides the wording."""
    days = [day for day in (plan.get("days") or []) if isinstance(day, dict)]
    if not days:
        return False
    for day in days:
        if _done_day(day):
            continue
        tasks = [task for task in (day.get("tasks") or []) if isinstance(task, dict) and _text(task.get("text"))]
        if len(tasks) < 2:
            return True
    return False


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _topic_words(*chunks: str) -> set[str]:
    stop = {
        "about",
        "after",
        "again",
        "before",
        "could",
        "should",
        "their",
        "there",
        "these",
        "those",
        "which",
        "while",
        "would",
        "write",
        "start",
        "finish",
        "complete",
    }
    blob = " ".join(chunks).lower()
    return {word for word in re.findall(r"[a-z0-9]{5,}", blob) if word not in stop}


def _goal_bits(goal: dict[str, Any] | None) -> list[str]:
    if not isinstance(goal, dict):
        return []
    profile = goal.get("contextProfile") if isinstance(goal.get("contextProfile"), dict) else {}
    understanding = goal.get("understanding") if isinstance(goal.get("understanding"), dict) else {}
    return [
        _text(goal.get("statement")),
        _text(goal.get("title")),
        _text(goal.get("description")),
        _text(goal.get("successCriteria")),
        _text((profile or {}).get("description")),
        _text((profile or {}).get("startingLevel")),
        _text((understanding or {}).get("subdomain")),
        _text((understanding or {}).get("primaryGoal")),
    ]


def accept_model_days(plan: dict[str, Any], result: Any, goal: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Keep a model rewrite only when every open step has real actions and stays on this goal."""
    payload = _dump(result)
    incoming = payload.get("days") or []
    original = [day for day in (plan.get("days") or []) if isinstance(day, dict)]
    if not incoming or not original:
        return None
    if len(incoming) > 16:
        return None
    del goal
    normalized: list[dict[str, Any]] = []
    for raw in incoming[:16]:
        data = _dump(raw)
        title = _text(data.get("title"))
        if not title:
            return None
        tasks = []
        for task in data.get("tasks") or []:
            task_data = _dump(task)
            text = _text(task_data.get("text"))
            if len(text.split()) < 3:
                return None
            tasks.append(
                {
                    "text": text[:240],
                    "why": _text(task_data.get("why"))[:240],
                    "definitionOfDone": _text(task_data.get("definitionOfDone"))[:240],
                    "done": False,
                    "note": "",
                }
            )
            if len(tasks) >= 5:
                break
        if len(tasks) < 2:
            return None
        normalized.append({"title": title[:160], "focus": _text(data.get("focus")), "definitionOfDone": _text(data.get("definitionOfDone")), "minutes": data.get("minutes"), "tasks": tasks})

    merged: list[dict[str, Any]] = []
    for index, item in enumerate(normalized):
        previous = original[index] if index < len(original) else {}
        if previous and _done_day(previous):
            merged.append(previous)
            continue
        day_id = _text(previous.get("id")) or f"day-{index + 1}"
        tasks = []
        for task_index, task in enumerate(item["tasks"], start=1):
            tasks.append({**task, "id": f"{day_id}-t{task_index}"})
        focus = item["focus"] or _focus_from(tasks)
        done = item["definitionOfDone"] or _text(tasks[-1].get("definitionOfDone"))
        minutes = item["minutes"] if isinstance(item["minutes"], int) and 10 <= item["minutes"] <= 240 else previous.get("minutes") or 45
        merged.append(
            {
                **previous,
                "id": day_id,
                "dayIndex": previous.get("dayIndex") or index + 1,
                "title": item["title"],
                "focus": focus[:500],
                "definitionOfDone": done[:320],
                "tasks": tasks,
                "minutes": minutes,
                "status": previous.get("status") or "open",
            }
        )
    plan["days"] = merged
    return plan
