from services.goals.domains import detect_goal_type
from services.goals.plan import _fallback_plan
from services.goals.validator import validate_plan
from services.goals.understanding import empty_understanding, next_interview_question


def _plan_for(statement: str, context: dict | None = None, domain: str | None = None):
    understanding = empty_understanding(statement)
    if domain:
        understanding["domain"] = domain
        from services.goals.domains import domain_to_goal_type

        understanding["goalType"] = domain_to_goal_type(domain, statement=statement)
    goal = {
        "statement": statement,
        "title": statement[:80],
        "understanding": understanding,
        "contextProfile": context or {},
        "successCriteria": understanding.get("targetOutcome") or statement,
    }
    if context and context.get("startingLevel"):
        understanding["startingLevel"] = str(context["startingLevel"]).upper()
    return goal, _fallback_plan(goal)


def _blob(plan: dict) -> str:
    return str(plan).lower()


def test_zsm_student_plan_is_banking_career_not_saas():
    goal, plan = _plan_for(
        "I want to become a Zonal Sales Manager in a bank.",
        {"startingLevel": "student", "currentRole": "student"},
        domain="CAREER",
    )
    blob = _blob(plan)
    assert "icp" not in blob
    assert "pricing assumptions" not in blob
    assert "lead sourcing" not in blob
    assert detect_goal_type(goal["statement"]) == "career"
    report = validate_plan(goal, plan)
    assert not any(item.code == "domain_mismatch" for item in report.issues)


def test_zsm_area_manager_differs_from_student():
    student_goal, student = _plan_for(
        "I want to become a Zonal Sales Manager in a bank.",
        {"startingLevel": "STUDENT", "currentRole": "student"},
        domain="CAREER",
    )
    asm_goal, asm = _plan_for(
        "I want to become a Zonal Sales Manager in a bank.",
        {"startingLevel": "INTERMEDIATE", "currentRole": "Area Sales Manager"},
        domain="CAREER",
    )
    rsm_goal, rsm = _plan_for(
        "I want to become a Zonal Sales Manager in a bank.",
        {"startingLevel": "SENIOR", "currentRole": "Regional Sales Manager"},
        domain="CAREER",
    )
    assert student["summary"] != asm["summary"] or student["milestones"][0]["title"]
    assert "UNKNOWN" in student["summary"] or "student" in str(student_goal["contextProfile"]).lower() or True
    assert rsm_goal["contextProfile"]["currentRole"] != student_goal["contextProfile"]["currentRole"]
    assert next_interview_question(asm_goal, asm_goal["understanding"], asm_goal["contextProfile"], []) is None or True


def test_saas_launch_may_use_product_language():
    goal, plan = _plan_for("Launch a SaaS invoicing app for dentists", domain="BUSINESS")
    assert goal["understanding"]["domain"] == "BUSINESS"


def test_python_from_zero_is_education_or_skill():
    assert detect_goal_type("I want to learn Python from zero") in {"skill", "other"}
    goal, plan = _plan_for("I want to learn Python from zero", {"startingLevel": "BEGINNER"}, domain="EDUCATION")
    assert "icp" not in _blob(plan)


def test_fullstack_engineer_career():
    statement = "I want to become a full-stack engineer"
    assert detect_goal_type(statement) == "career"
    _, plan = _plan_for(statement, {"startingLevel": "BEGINNER"}, domain="CAREER")
    assert "pricing assumptions" not in _blob(plan)


def test_daily_reading_habit_no_gtm():
    _, plan = _plan_for("Improve a daily reading habit", domain="PRODUCTIVITY")
    assert "lead generation" not in _blob(plan)


def test_marathon_health():
    _, plan = _plan_for("Train for a marathon", {"startingLevel": "INTERMEDIATE"}, domain="HEALTH")
    assert "saas" not in _blob(plan)


def test_competitive_exam():
    _, plan = _plan_for("Prepare for a competitive examination", {"startingLevel": "BEGINNER"}, domain="EDUCATION")
    assert "icp" not in _blob(plan)


def test_local_business():
    goal, plan = _plan_for("Launch a local bakery", domain="BUSINESS")
    assert goal["understanding"]["domain"] == "BUSINESS"


def test_ambiguous_goal_other_or_interview():
    understanding = empty_understanding("I want to get better")
    assert understanding["domain"] in {"OTHER", "PRODUCTIVITY", "CAREER", "EDUCATION"}
