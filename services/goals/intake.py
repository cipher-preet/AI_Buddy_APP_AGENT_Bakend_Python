"""Choose the next question that helps understand this goal.

The job is to learn what the user wants to achieve, then only the missing
facts a plan cannot be built without. Different goals still ask different
questions. A question the statement already answers is skipped.
"""

from __future__ import annotations

import re
from typing import Any

from services.goals.domains import inferred_title, is_skip_answer, normalize_domain, understanding_as_dict
from services.goals.schemas import GoalOption, InterviewQuestion

MAX_FACT_QUESTIONS = 6

FACET_IDS = {
    "subject": {"subject", "what-it-is", "what-product", "which-skill"},
    "product": {"product-type", "product-kind"},
    "audience": {"audience"},
    "stage": {"stage", "current-level", "starting-level", "current-role"},
    "budget": {"budget"},
    "deadline": {"timeframe"},
    "hours": {"hours"},
    "place": {"location", "geography"},
    "success": {"success", "success-criteria"},
    "approach": {"approach"},
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _blob(goal: dict[str, Any], understanding: dict[str, Any]) -> str:
    return " ".join(
        [
            _text(goal.get("statement")),
            _text(goal.get("title")),
            _text(goal.get("description")),
            _text(understanding.get("subdomain")),
            _text(understanding.get("primaryGoal")),
        ]
    ).lower()


def _has_word(blob: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", blob) for word in words)


def goal_signals(goal: dict[str, Any], understanding: dict[str, Any] | None = None) -> set[str]:
    """Labels that change which questions and which plan shape apply."""
    understood = understanding_as_dict(understanding if understanding is not None else goal.get("understanding"))
    domain = normalize_domain(str(understood.get("domain") or ""))
    blob = _blob(goal, understood)
    signals: set[str] = set()
    if domain == "CAREER" or _has_word(blob, ("career", "promotion", "hired", "interview", "resume")):
        signals.add("career")
    if domain == "EDUCATION" or _has_word(blob, ("learn", "study", "course", "exam", "skill", "python")):
        signals.add("skill")
    if domain == "HEALTH" or _has_word(blob, ("marathon", "fitness", "workout", "weight", "muscle", "5k", "10k")):
        signals.add("health")
    if domain == "PRODUCTIVITY" or _has_word(blob, ("habit", "every day", "daily", "streak")):
        signals.add("habit")
    if domain == "FINANCIAL" or _has_word(blob, ("save", "invest", "debt", "salary", "income", "budget")):
        signals.add("money")
    if domain == "CREATIVE" or _has_word(blob, ("novel", "podcast", "youtube", "paint", "portfolio", "song")):
        signals.add("creative")
    if domain == "TRAVEL" or _has_word(blob, ("trip", "travel", "visa", "flight", "itinerary")):
        signals.add("travel")
    if domain == "RELATIONSHIPS" or _has_word(blob, ("partner", "marriage", "family", "friendship", "relationship")):
        signals.add("relationship")
    if _has_word(
        blob,
        (
            "user",
            "users",
            "signup",
            "signups",
            "sign-up",
            "sign-ups",
            "customer",
            "customers",
            "follower",
            "followers",
            "subscriber",
            "subscribers",
            "download",
            "downloads",
            "deal",
            "deals",
            "quota",
            "pipeline",
            "revenue",
        ),
    ):
        signals.add("people")
        if _has_word(blob, ("deal", "deals", "quota", "pipeline", "revenue", "selling")):
            signals.add("sales")
    if _has_word(blob, ("launch", "startup", "saas", "app", "product", "platform", "business", "bakery", "shop", "store", "restaurant", "sell", "freelance", "client")):
        signals.add("venture")
    if _has_word(blob, ("bakery", "restaurant", "cafe", "salon", "clinic", "gym", "shop", "store", "local")):
        signals.add("local")
    if _has_word(blob, ("freelance", "consulting", "client", "retainer")):
        signals.add("clients")
    if "career" in signals:
        signals.discard("skill")
    if "people" in signals:
        signals.discard("money")
    return signals


def plan_family(signals: set[str]) -> str:
    """The plan shape. People-outcomes win over a generic business label."""
    for name in ("people", "career", "health", "habit", "skill", "relationship", "travel", "money", "creative"):
        if name in signals:
            return name
    if "local" in signals:
        return "local"
    if "venture" in signals:
        return "venture"
    return "general"


def _asked(messages: list[dict[str, Any]]) -> set[str]:
    asked: set[str] = set()
    for row in messages:
        if row.get("role") != "assistant":
            continue
        step_id = _text(row.get("stepId")).lower()
        if step_id:
            asked.add(step_id)
    return asked


def _facet_asked(facet: str, asked: set[str]) -> bool:
    return bool(FACET_IDS.get(facet, set()) & asked)


def _known(goal: dict[str, Any], profile: dict[str, Any], field: str) -> bool:
    if field == "description":
        value = _text(profile.get("description")) or _text(goal.get("description"))
    elif field == "audience":
        value = _text(profile.get("audience")) or _text(goal.get("audience"))
    elif field == "budget":
        value = _text(profile.get("budget"))
    elif field == "timeframe":
        value = _text(profile.get("timeframe")) or _text(goal.get("timeframe"))
    elif field == "geography":
        value = _text(profile.get("geography"))
    elif field == "successCriteria":
        value = _text(profile.get("successCriteria")) or _text(goal.get("successCriteria"))
        statement = _text(goal.get("statement")).lower()
        if value.lower() == statement:
            return False
    elif field == "hours":
        if goal.get("availableHoursPerWeek") is not None:
            return True
        value = _text(profile.get("availableHoursPerWeek"))
    elif field == "startingLevel":
        value = _text(profile.get("startingLevel")) or _text(profile.get("currentRole"))
    else:
        value = _text(profile.get(field))
    return bool(value) and value.upper() != "UNKNOWN" and not is_skip_answer(value)


def _uses_rupees(goal: dict[str, Any], profile: dict[str, Any]) -> bool:
    blob = f"{_text(goal.get('statement'))} {_text(profile.get('geography'))} {_text(goal.get('description'))}".lower()
    return any(token in blob for token in ("₹", "rupee", " inr", "india", "indian", "bharat"))


def _offering_is_vague(lowered: str) -> bool:
    """'My platform' or 'my app' is not a product description."""
    if not re.search(r"\b(platform|app|product|website|site|startup|tool)\b", lowered):
        return False
    if re.search(
        r"\b(invoic|note|meeting|crm|marketplace|course|newsletter|bakery|shop|store|restaurant|game|fitness|dental|dentist|school|chat|video|music|photo|calendar|habit|finance|health|saas|plugin|extension)\b",
        lowered,
    ):
        return False
    return True


def _subject_is_thin(goal: dict[str, Any]) -> bool:
    description = _text(goal.get("description"))
    statement = _text(goal.get("statement"))
    if description and description.lower() != statement.lower() and len(description) > 20:
        return False
    lowered = statement.lower()
    if _offering_is_vague(lowered):
        return True
    if re.search(r"\b(get better|be a better|improve myself|be successful|do better)\b", lowered):
        return True
    if re.search(r"\d", lowered) and re.search(r"\b(save|invest|debt|salary|house)\b", lowered):
        return False
    if re.search(
        r"\d[\d,]*\s+(?:\w+\s+){0,4}(users?|customers?|signups|sign-ups|followers|subscribers|downloads|deals|clients)\b",
        lowered,
    ):
        if not re.search(r"\b(bakery|python|marathon|exam|book|store|shop|course|invoice|invoicing)\b", lowered):
            return True
    if re.search(
        r"\b(bakery|python|marathon|exam|book|novel|store|shop|manager|engineer|habit|pages|trip|partner|house)\b",
        lowered,
    ):
        return False
    words = re.findall(r"[a-z0-9']+", lowered)
    return len(words) <= 6


def _product_type_stated(goal: dict[str, Any], profile: dict[str, Any]) -> bool:
    if _known(goal, profile, "productType"):
        return True
    blob = f"{_text(goal.get('statement'))} {_text(goal.get('description'))} {_text(profile.get('productType'))}".lower()
    return bool(
        re.search(
            r"\b(saas|software|mobile app|ios app|android|marketplace|newsletter|community|course|plugin|extension|e-?commerce|online store|local service|desktop app|web app|chrome extension)\b",
            blob,
        )
    )


def _outcome_is_concrete(goal: dict[str, Any]) -> bool:
    success = _text(goal.get("successCriteria"))
    statement = _text(goal.get("statement"))
    if success and success.lower() != statement.lower():
        return True
    lowered = statement.lower()
    if re.search(r"\d", lowered):
        return True
    return bool(re.search(r"\b(become|get hired|launch|train for|learn|finish|pass|write|build|save|run)\b", lowered))


def _level_stated(goal: dict[str, Any], profile: dict[str, Any]) -> bool:
    if _known(goal, profile, "startingLevel"):
        return True
    return bool(
        re.search(
            r"\b(from zero|from scratch|beginner|never (done|tried)|already (do|doing|train|practice)|intermediate|advanced)\b",
            _text(goal.get("statement")).lower(),
        )
    )


def _audience_stated(goal: dict[str, Any], profile: dict[str, Any]) -> bool:
    if _known(goal, profile, "audience"):
        return True
    match = re.search(r"\bfor\s+(?:the\s+)?([a-z][a-z0-9-]{2,})", _text(goal.get("statement")).lower())
    if not match:
        return False
    word = match.group(1)
    if word in {"my", "this", "our", "a", "an", "the", "me", "us", "people"}:
        return False
    groups = {
        "students",
        "dentists",
        "developers",
        "founders",
        "agencies",
        "teachers",
        "doctors",
        "parents",
        "designers",
        "marketers",
        "startups",
        "businesses",
        "creators",
        "consumers",
        "teams",
        "companies",
        "freelancers",
        "managers",
        "engineers",
        "clinics",
        "shoppers",
        "buyers",
        "clients",
        "customers",
        "users",
        "beginners",
        "professionals",
        "families",
        "schools",
        "hospitals",
    }
    return word in groups or bool(re.search(r"(ists|ers|ors)$", word))


def _deadline_stated(goal: dict[str, Any], profile: dict[str, Any]) -> bool:
    if _known(goal, profile, "timeframe"):
        return True
    return bool(
        re.search(
            r"\b(\d+\s+days|\d+\s+weeks|\d+\s+months|\d+\s+years|a month|few months|in \d+\s+days|in \d+\s+months|within \d+|by 20\d\d|this year|next year|this quarter|next quarter|this month|within a year)\b",
            _text(goal.get("statement")).lower(),
        )
    )


def _dose_stated(goal: dict[str, Any]) -> bool:
    statement = _text(goal.get("statement")).lower()
    return bool(re.search(r"\d", statement) and re.search(r"\b(every day|daily|a day|per day|per week|a week)\b", statement))


def _facet_order(signals: set[str]) -> list[str]:
    """Understand the outcome first. Then only facts that change this plan."""
    family = plan_family(signals)
    if family == "career":
        return ["subject", "success", "stage", "place", "deadline", "hours"]
    if family == "travel":
        return ["subject", "success", "deadline", "budget"]
    if family == "money":
        return ["subject", "success", "deadline", "budget"]
    if family in {"habit", "relationship"}:
        return ["subject", "success", "stage", "deadline"]
    order = ["subject", "success"]
    if family in {"people", "venture", "local"} or "sales" in signals:
        order.append("audience")
    order.extend(["stage", "deadline"])
    if family in {"skill", "health", "creative", "general"}:
        order.append("hours")
    return order


def _title(goal: dict[str, Any], understanding: dict[str, Any]) -> str:
    return inferred_title(goal, understanding) or "this goal"


def _opening(signals: set[str], title: str) -> str:
    del signals
    return (
        f"I'll help you achieve {title}. "
        "I'll only ask what I need to understand the goal and build the plan."
    )


def _with_skip(options: list[tuple[str, str]]) -> list[GoalOption]:
    built = [GoalOption(id=item_id, label=label) for item_id, label in options if label]
    if not any(item.id == "skip" for item in built):
        built.append(GoalOption(id="skip", label="Not sure"))
    return built[:6]


def _question(
    step_id: str,
    prompt: str,
    control: str,
    field: str,
    *,
    options: list[tuple[str, str]] | None = None,
    note: str = "",
    brief: list[str] | None = None,
    leverage: str = "",
) -> InterviewQuestion:
    return InterviewQuestion(
        stepId=step_id,
        prompt=prompt,
        control=control,
        field=field,
        options=_with_skip(options or []) if control in {"single_select", "multi_select"} else [],
        note=note,
        brief=brief or [],
        leverage=leverage,
    )


def _audience_options(signals: set[str]) -> list[tuple[str, str]]:
    if "local" in signals:
        return [
            ("nearby", "People who live nearby"),
            ("offices", "Offices and teams"),
            ("shops", "Other local shops"),
            ("online", "Online customers"),
        ]
    if "sales" in signals and "venture" not in signals and "local" not in signals:
        return [
            ("existing", "People who already buy from you"),
            ("businesses", "Businesses"),
            ("buyers", "Individual buyers"),
            ("network", "People who already know you"),
        ]
    if "clients" in signals and "people" not in signals:
        return [
            ("small-business", "Small businesses"),
            ("agencies", "Agencies"),
            ("individuals", "Individual clients"),
        ]
    return [
        ("students", "Students"),
        ("developers", "Developers and tech professionals"),
        ("businesses", "Businesses and startups"),
        ("creators", "Content creators"),
        ("consumers", "General consumers"),
    ]


def _stage_options(signals: set[str]) -> list[tuple[str, str]]:
    family = plan_family(signals)
    if family in {"people", "venture", "local"}:
        if "local" in signals and family != "people":
            return [
                ("not-open", "Not open yet"),
                ("few", "Open, but very few buyers"),
                ("regular", "Already have some regular buyers"),
            ]
        return [
            ("not-launched", "Not launched yet"),
            ("few", "Launched, but very few users"),
            ("some", "Already have some active users"),
        ]
    if family == "career":
        return [
            ("student", "Student"),
            ("early", "Early in my career"),
            ("in-field", "Already in this field"),
            ("switching", "Switching from another field"),
        ]
    if family == "health":
        return [
            ("new", "New to this distance or training"),
            ("sometimes", "I train sometimes"),
            ("regular", "I already train most weeks"),
        ]
    if family == "habit":
        return [
            ("not-started", "Not started"),
            ("uneven", "I start and stop"),
            ("most-days", "Already doing it most days"),
        ]
    if family == "skill":
        return [
            ("zero", "From zero"),
            ("some", "Some practice"),
            ("using", "I already use it"),
        ]
    return [
        ("not-started", "Not started"),
        ("early", "Started, still early"),
        ("underway", "Already underway"),
    ]


def _budget_options(signals: set[str], rupees: bool) -> list[tuple[str, str]]:
    family = plan_family(signals)
    if family in {"people", "venture", "local"} or "venture" in signals:
        if rupees:
            return [
                ("zero", "₹0 — organic growth only"),
                ("under-5k", "Under ₹5,000"),
                ("mid", "₹5,000–₹20,000"),
                ("high", "₹20,000+"),
            ]
        return [
            ("zero", "No budget — organic only"),
            ("small", "A small test budget"),
            ("medium", "Enough for a few experiments"),
            ("large", "A real campaign budget"),
        ]
    if rupees:
        return [
            ("zero", "₹0 extra"),
            ("under-5k", "Under ₹5,000"),
            ("mid", "₹5,000–₹20,000"),
            ("high", "₹20,000+"),
        ]
    return [
        ("zero", "Nothing extra"),
        ("small", "A small amount"),
        ("medium", "A set amount I can plan around"),
        ("large", "Enough to make this the main push"),
    ]


def _product_type_options() -> list[tuple[str, str]]:
    return [
        ("browser", "Software people use in a browser"),
        ("mobile", "A mobile app"),
        ("marketplace", "A marketplace"),
        ("content", "Content, a community, or a newsletter"),
        ("local", "A local service or shop"),
    ]


def _deadline_options(signals: set[str]) -> list[tuple[str, str]]:
    family = plan_family(signals)
    if family in {"people", "venture", "local"}:
        return [
            ("30-days", "30 days"),
            ("3-months", "3 months"),
            ("6-months", "6 months"),
            ("open", "No fixed deadline"),
        ]
    if family == "habit":
        return [
            ("30-days", "30 days"),
            ("60-days", "60 days"),
            ("90-days", "90 days"),
            ("open", "No fixed deadline"),
        ]
    if family == "career":
        return [
            ("90-days", "Within 90 days"),
            ("6-months", "Within 6 months"),
            ("1-year", "Within a year"),
            ("open", "No fixed deadline"),
        ]
    if family == "skill":
        return [
            ("30-days", "Within 30 days"),
            ("2-months", "2 months"),
            ("90-days", "Within 90 days"),
            ("6-months", "Within 6 months"),
        ]
    return [
        ("30-days", "Within 30 days"),
        ("90-days", "Within 90 days"),
        ("6-months", "Within 6 months"),
        ("open", "No fixed deadline"),
    ]


def approach_beats(goal: dict[str, Any], understanding: dict[str, Any] | None = None) -> list[str]:
    understood = understanding_as_dict(understanding if understanding is not None else goal.get("understanding"))
    signals = goal_signals(goal, understood)
    title = _title(goal, understood)
    family = plan_family(signals)
    audience = _text(goal.get("audience"))
    who = audience.split(",")[0].strip() if audience else "the people who need this most"
    if family == "people":
        return [
            f"Find the first {who.lower() if audience else 'people'} who need {title}, and talk to them directly.",
            "Help them get the main benefit quickly, then ask what was confusing.",
            "Repeat only the paths that bring people who come back.",
            "Count people who did the core action this week. Treat the final number as a stretch, not a promise.",
        ]
    if family == "career":
        return [
            f"Name the proof someone would need before they trust you for {title}.",
            "Close the one gap between your current stage and that proof.",
            "Turn the work into something a reviewer can open.",
            "Practice the conversation you will actually have. The offer stays a separate result.",
        ]
    if family == "health":
        return [
            "Set this week's training from your current level.",
            "Keep one longer session and one easy session before you add load.",
            "Write down pain or missed sessions before the plan gets harder.",
            "Judge the week by sessions completed, not by a promised race result.",
        ]
    if family == "habit":
        return [
            "Make the action small enough to finish on a bad day.",
            "Attach it to something you already do.",
            "Mark the day done only when the action happened.",
            "Restart the next day after a miss. Do not renegotiate the habit.",
        ]
    if family == "skill":
        return [
            "Learn the smallest piece you can use this week.",
            "Make one small thing that uses it.",
            "Explain it back without notes.",
            "Add the next piece only after the first one holds.",
        ]
    if family == "local":
        return [
            "Talk to people who can already buy or visit.",
            "Offer one clear reason to show up this week.",
            "Ask what almost stopped them.",
            "Repeat the offer that brought people back.",
        ]
    if family == "travel":
        return [
            "Decide the dates, budget, and must-see list before you book.",
            "Book the piece that sells out first.",
            "Leave a simple day plan and a buffer.",
            "Check documents and money a week before you leave.",
        ]
    if family == "relationship":
        return [
            "Say what you want in one clear conversation.",
            "Pick one repeated action the other person can notice.",
            "Ask what landed and what did not.",
            "Keep the action small enough to repeat next week.",
        ]
    if family == "money":
        return [
            "Write the number and the date you will check it.",
            "Move the first small amount this week.",
            "Review the gap before you change the plan.",
            "Do not mark the goal done because the spreadsheet looks tidy.",
        ]
    if family == "creative":
        return [
            "Finish one small piece this week.",
            "Show it to one person and write down what they noticed.",
            "Repeat the part that worked.",
            "Ship the next piece before you expand the idea.",
        ]
    return [
        f"Write what “done” looks like for {title} in one sentence.",
        "Do the smallest piece that creates evidence.",
        "Check that evidence before you add more work.",
        "Keep the outside result separate from the tasks you control.",
    ]


def _subject_question(signals: set[str], title: str, note: str) -> InterviewQuestion:
    family = plan_family(signals)
    if "sales" in signals:
        return _question(
            "subject",
            "What are you selling?",
            "text",
            "description",
            note=note,
            leverage="The plan cannot be specific until the offer is.",
        )
    prompts = {
        "people": f"What do you want people to use or get so “{title}” can happen? A sentence or two is enough.",
        "venture": f"What are you offering, in simple words, so “{title}” is clear?",
        "career": "Which role do you want, and in what kind of organization?",
        "skill": "What do you want to be able to do when this is finished?",
        "relationship": "What would better look like in this relationship?",
        "travel": "Where are you going, and what has to be true for the trip to count?",
        "money": "What number are you trying to change, and by when if you know?",
        "creative": "What are you making, and what does finished look like?",
        "local": "What do you sell or offer, and to whom nearby?",
    }
    return _question(
        "subject",
        prompts.get(family, f"In simple words, what do you want from “{title}”?"),
        "text",
        "description",
        note=note,
        leverage="The plan cannot be specific until the subject is.",
    )


def _facet_question(
    facet: str,
    goal: dict[str, Any],
    understanding: dict[str, Any],
    profile: dict[str, Any],
    signals: set[str],
    note: str,
) -> InterviewQuestion | None:
    title = _title(goal, understanding)
    if facet == "subject":
        if not _subject_is_thin(goal):
            return None
        return _subject_question(signals, title, note)
    if facet == "product":
        if plan_family(signals) not in {"people", "venture", "local"}:
            return None
        if _product_type_stated(goal, profile):
            return None
        return _question(
            "product-type",
            "What type of product is this?",
            "single_select",
            "productType",
            options=_product_type_options(),
            note=note,
            leverage="The type changes the first things you build and who you talk to.",
        )
    if facet == "audience":
        if "people" not in signals and "venture" not in signals:
            return None
        if _audience_stated(goal, profile):
            return None
        return _question(
            "audience",
            f"Who is “{title}” for? Pick every group that fits.",
            "multi_select",
            "audience",
            options=_audience_options(signals),
            note=note,
            leverage="The first audience changes where you spend time.",
        )
    if facet == "stage":
        if _level_stated(goal, profile):
            return None
        step_id = "current-level" if "career" in signals else "stage"
        family = plan_family(signals)
        stage_prompts = {
            "career": "What is your current role or stage?",
            "habit": f"How regularly do you do “{title}” today?",
            "health": "How much training do you do today?",
            "skill": "What can you already do?",
            "creative": "How far along is the piece?",
            "people": f"Where is “{title}” today?",
            "venture": f"Where is “{title}” today?",
            "local": f"Where is “{title}” today?",
        }
        return _question(
            step_id,
            stage_prompts.get(family, f"Where are you today with “{title}”?"),
            "single_select",
            "startingLevel",
            options=_stage_options(signals),
            note=note,
            leverage="The starting point changes which earlier steps belong.",
        )
    if facet == "budget":
        if _known(goal, profile, "budget"):
            return None
        family = plan_family(signals)
        if family == "travel":
            prompt = "How much can you spend on this trip?"
        elif family == "money":
            prompt = "How much can you put toward this?"
        else:
            prompt = "How much can you put toward this?"
        return _question(
            "budget",
            prompt,
            "single_select",
            "budget",
            options=_budget_options(signals, _uses_rupees(goal, profile)),
            note=note,
            leverage="Budget changes which paths are available.",
        )
    if facet == "deadline":
        if _deadline_stated(goal, profile):
            return None
        return _question(
            "timeframe",
            f"By when do you want to achieve “{title}”?",
            "single_select",
            "timeframe",
            options=_deadline_options(signals),
            note=note,
            leverage="The deadline changes how aggressive the first weeks are.",
        )
    if facet == "hours":
        if _known(goal, profile, "hours") or _dose_stated(goal):
            return None
        return _question(
            "hours",
            f"About how many hours each week can you spend on “{title}”?",
            "number",
            "availableHoursPerWeek",
            note=note,
            leverage="Hours change how much fits in a week.",
        )
    if facet == "place":
        if "career" not in signals and "skill" not in signals and plan_family(signals) != "travel":
            return None
        if _known(goal, profile, "geography"):
            return None
        return _question(
            "location",
            "Which country or region should the plan assume? Requirements often differ by place.",
            "text",
            "geography",
            note=note,
            leverage="Place changes rules, exams, and employers worth researching.",
        )
    if facet == "success":
        if _outcome_is_concrete(goal) or _known(goal, profile, "successCriteria"):
            return None
        return _question(
            "success",
            f"What will done look like for “{title}”? One concrete result is enough.",
            "text",
            "successCriteria",
            note=note,
            leverage="The plan needs a result it can check.",
        )
    return None


def _understood_lines(
    goal: dict[str, Any],
    understanding: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    """Restate only facts the user gave, so the plan is built from those."""
    title = _title(goal, understanding)
    statement = _text(goal.get("statement"))
    lines = [f"You want to achieve: {title}."]
    success = _text(profile.get("successCriteria")) or _text(goal.get("successCriteria"))
    if success and success.lower() not in {statement.lower(), title.lower()} and not is_skip_answer(success):
        lines.append(f"Done looks like: {success}.")
    elif statement and statement.lower() != title.lower():
        lines.append(f"In your words: {statement}.")
    level = _text(profile.get("startingLevel")) or _text(profile.get("currentRole"))
    if level and level.upper() != "UNKNOWN" and not is_skip_answer(level):
        lines.append(f"Starting from: {level}.")
    audience = _text(profile.get("audience")) or _text(goal.get("audience"))
    if audience and audience.upper() != "UNKNOWN" and not is_skip_answer(audience):
        lines.append(f"For: {audience}.")
    deadline = _text(profile.get("timeframe")) or _text(goal.get("timeframe"))
    if deadline and not is_skip_answer(deadline):
        lines.append(f"By: {deadline}.")
    hours = profile.get("availableHoursPerWeek")
    if hours not in (None, "", "UNKNOWN") and not is_skip_answer(_text(hours)):
        lines.append(f"Time you can give: {hours} hours a week.")
    if len(lines) < 2:
        lines.append("Anything you did not say will stay unknown in the plan.")
    return lines[:4]


def _approach_question(
    goal: dict[str, Any],
    understanding: dict[str, Any],
    profile: dict[str, Any],
) -> InterviewQuestion:
    return _question(
        "approach",
        "This is what I understood. I'll build the plan from it.",
        "brief",
        "",
        brief=_understood_lines(goal, understanding, profile),
        note="Continue and I'll build the plan from this.",
        leverage="Confirms the goal before the plan is built.",
    )


def next_intake_question(
    goal: dict[str, Any],
    understanding: dict[str, Any],
    profile: dict[str, Any],
    messages: list[dict[str, Any]],
) -> InterviewQuestion | None:
    understood = understanding_as_dict(understanding)
    signals = goal_signals(goal, understood)
    asked = _asked(messages)
    if _facet_asked("approach", asked):
        return None

    first = not any(row.get("role") == "assistant" and _text(row.get("stepId")) for row in messages)
    note = _opening(signals, _title(goal, understood)) if first else ""
    fact_count = sum(1 for facet in FACET_IDS if facet != "approach" and _facet_asked(facet, asked))
    if fact_count < MAX_FACT_QUESTIONS:
        for facet in _facet_order(signals):
            if _facet_asked(facet, asked):
                continue
            question = _facet_question(facet, goal, understood, profile, signals, note)
            if question is not None:
                return question
    return _approach_question(goal, understood, profile)
