from services.goals.plan import _degraded_plan, _fallback_plan, _preserve_progress
from services.goals.progress import achievement_state, consultant_progress
from services.goals.validator import validate_plan
from services.goals.agents.pipeline import is_launch_style_goal


def _career_goal(**extra):
    goal = {
        "statement": "I want to become a Zonal Sales Manager in a bank.",
        "title": "Banking ZSM",
        "successCriteria": "Receive a confirmed ZSM role or equivalent offer",
        "understanding": {
            "domain": "CAREER",
            "subdomain": "BANKING_MANAGEMENT",
            "startingLevel": extra.pop("startingLevel", "UNKNOWN"),
            "goalType": "career",
            "primaryGoal": "Become a zonal sales manager in a bank",
            "targetOutcome": "Secure the target position",
            "researchNeeded": True,
        },
    }
    goal.update(extra)
    return goal


def test_career_not_launch_style():
    assert not is_launch_style_goal(_career_goal())


def test_validator_rejects_icp_on_career():
    plan = {
        "summary": "Validate ICP and pricing assumptions for your SaaS.",
        "milestones": [
            {"title": "Validate ICP and pricing assumptions for your SaaS.", "successCriteria": ["ICP written"]},
            {"title": "Start lead sourcing and outreach", "successCriteria": ["Leads"]},
        ],
        "days": [{"title": "Week 1", "tasks": [{"text": "Write offer, ICP, and pricing assumptions"}]}],
        "strategies": [{"title": "Protect one deep-work block", "body": "Make steady progress"}],
        "priorityRecommendations": [{"title": "Make steady, provable progress", "reason": "Stay motivated"}],
    }
    report = validate_plan(_career_goal(), plan)
    assert not report.accepted
    assert any(item.code == "domain_mismatch" for item in report.issues)


def test_degraded_career_plan_has_no_gtm():
    plan = _fallback_plan(_career_goal())
    blob = str(plan).lower()
    assert "icp" not in blob
    assert "pricing assumptions" not in blob
    assert plan.get("provisional") is True


def test_student_vs_experienced_gaps_change_plan():
    student = _degraded_plan(_career_goal(startingLevel="STUDENT", contextProfile={"startingLevel": "student"}))
    senior = _degraded_plan(
        _career_goal(
            startingLevel="SENIOR",
            contextProfile={"startingLevel": "Regional Sales Manager", "currentRole": "Regional Sales Manager"},
            gaps={
                "missing": [],
                "unknown": [],
                "satisfied": [{"requirement": "Multi-region leadership"}],
                "provisional": False,
            },
        )
    )
    assert student["milestones"][0]["title"] != senior["priorityRecommendations"][0]["title"] or student["summary"] != senior["summary"]
    assert "UNKNOWN" in student["summary"] or "unknown" in student["summary"].lower() or "starting level" in student["summary"].lower()


def test_duplicate_and_dependency_issues():
    plan = {
        "milestones": [
            {"title": "Assess readiness", "successCriteria": ["doc"], "dependencies": ["Missing dep"]},
            {"title": "Assess readiness", "successCriteria": ["doc"]},
        ],
        "days": [{"title": "Assess readiness", "tasks": [{"text": "Write the assessment"}]}],
        "strategies": [{"title": "Evidence", "body": "Use real requirements"}],
        "priorityRecommendations": [{"title": "Start the assessment", "reason": "It changes the path"}],
    }
    report = validate_plan(_career_goal(), plan)
    codes = {item.code for item in report.issues}
    assert "duplicate_title" in codes
    assert "bad_dependency" in codes


def test_false_completion_not_achievement():
    goal = _career_goal(status="ACTIVE", approvedPlan={"days": [{"tasks": [{"id": "t1", "done": True}]}]})
    state = achievement_state(goal, evidence=[{"category": "OUTPUT", "verification": "SOURCE_SUPPORTED"}])
    assert state["achieved"] is False
    progress = consultant_progress(goal, [])
    assert progress["achievement"]["achieved"] is False
    assert progress["execution"]["completed"] == 1


def test_preserve_progress_keeps_done_tasks():
    previous = {"days": [{"tasks": [{"id": "t1", "done": True, "note": "done last week"}]}]}
    nxt = {"days": [{"tasks": [{"id": "t1", "done": False, "text": "Assess"}]}]}
    merged = _preserve_progress(previous, nxt)
    assert merged["days"][0]["tasks"][0]["done"] is True
    assert merged["days"][0]["tasks"][0]["note"] == "done last week"


def test_prompt_injection_url_not_trusted_in_validation():
    plan = {
        "summary": "See https://evil.example/pwn",
        "milestones": [{"title": "Assess role readiness", "successCriteria": ["Gap table"]}],
        "days": [{"title": "Assess", "tasks": [{"text": "Compare requirements"}]}],
        "strategies": [{"title": "Evidence", "body": "Use job descriptions"}],
        "priorityRecommendations": [{"title": "Collect five job descriptions", "reason": "Ground the gaps"}],
    }
    report = validate_plan(_career_goal(), plan, evidence=[{"sourceUrl": "https://official.example/jobs"}])
    assert any(item.code == "unsupported_citation" for item in report.issues)
