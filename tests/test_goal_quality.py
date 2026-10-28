from services.goals.plan import _ensure_open_steps
from services.goals.quality import plain_text, tighten_plan
from services.goals.agents.pipeline import _repairs_are_safe
from services.goals.schemas import GoalCriticResult, PlanDay, DayTask, PlanStrategy


def test_plain_text_drops_jargon_and_extra_sentences():
    text = plain_text(
        "Leverage a holistic paradigm. Then move the needle with a deep dive. Then add a third sentence that should go.",
        max_chars=240,
        max_sentences=2,
    )
    assert "leverage" not in text.lower()
    assert "holistic" not in text.lower()
    assert "third sentence" not in text.lower()
    assert "use" in text.lower()


def test_tighten_plan_keeps_specific_items_and_drops_filler():
    plan = {
        "summary": "This is a long summary. " * 40,
        "strategies": [
            {
                "id": "s1",
                "title": "Practice one bank interview story",
                "body": "Write one story from your current role and say it out loud once.",
                "why": "Interviews ask for proof, not slogans.",
                "proofPoints": ["One written story"],
            },
            {
                "id": "s2",
                "title": "Stay motivated",
                "body": "Work harder and make steady progress.",
            },
            {
                "id": "s3",
                "title": "Practice one bank interview story",
                "body": "Duplicate of the first strategy that should be removed.",
            },
        ],
        "priorityRecommendations": [
            {
                "id": "r1",
                "title": "Draft the story tonight",
                "reason": "You can finish this in one sitting.",
                "howSteps": [
                    "Pick one real project",
                    "Write the result in five lines",
                    "Read it once out loud",
                    "This fourth step should be cut",
                ],
            },
            {
                "id": "r2",
                "title": "You will get the job",
                "reason": "This is guaranteed if you stay motivated.",
            },
        ],
        "days": [
            {
                "title": "Write the interview story",
                "tasks": [
                    {"id": "t1", "text": "Write one project story with the result and your role."},
                    {"id": "t2", "text": "Stay motivated and work harder."},
                    {"id": "t3", "text": "Read the story once and mark what is unclear."},
                ],
            }
        ],
        "milestones": [
            {"title": "One interview story is written", "successCriteria": ["The story names a result"], "dueLabel": "This week"},
        ],
    }
    tightened = tighten_plan(plan, {"statement": "Become a bank sales manager"})
    assert len(tightened["summary"]) < 500
    assert [item["title"] for item in tightened["strategies"]] == ["Practice one bank interview story"]
    assert [item["title"] for item in tightened["priorityRecommendations"]] == ["Draft the story tonight"]
    assert len(tightened["priorityRecommendations"][0]["howSteps"]) == 3
    task_text = " ".join(task["text"] for task in tightened["days"][0]["tasks"]).lower()
    assert "stay motivated" not in task_text
    assert "project story" in task_text


def test_tighten_strips_urls_that_are_not_in_evidence():
    plan = {
        "strategies": [
            {
                "title": "Check the public exam notice",
                "body": "Read the official notice at https://example.com/fake and note the date.",
            }
        ],
        "priorityRecommendations": [],
        "days": [],
        "milestones": [],
    }
    tightened = tighten_plan(
        plan,
        {"researchSources": [{"url": "https://official.example/notice"}]},
    )
    assert "https://example.com/fake" not in tightened["strategies"][0]["body"]


def test_new_plan_steps_stay_open_even_if_the_model_marks_them_done():
    plan = _ensure_open_steps(
        {
            "milestones": [{"title": "Write the first story", "successCriteria": ["One story is saved"]}],
            "days": [
                {
                    "title": "Write the first story",
                    "status": "done",
                    "tasks": [{"text": "Write one story", "done": True}],
                }
            ],
        }
    )
    assert plan["days"][0]["status"] == "open"
    assert plan["days"][0]["tasks"][0]["done"] is False


def test_missing_days_are_built_from_milestones():
    plan = _ensure_open_steps(
        {
            "milestones": [
                {"title": "Learn the basics", "successCriteria": ["Finish one lesson"]},
                {"title": "Practice on a real task", "successCriteria": ["Ship one small exercise"]},
            ],
            "days": [],
        }
    )
    assert len(plan["days"]) == 2
    assert plan["days"][0]["status"] == "open"
    assert plan["days"][0]["tasks"][0]["done"] is False


def test_critic_repairs_with_filler_are_rejected():
    critic = GoalCriticResult(
        accepted=False,
        issues=["generic"],
        repairedStrategies=[
            PlanStrategy(title="Stay motivated", body="Work harder and make steady progress every day."),
        ],
        repairedDays=[
            PlanDay(
                title="Week 1",
                tasks=[DayTask(text="Protect one deep-work block and stay consistent.")],
            )
        ],
    )
    assert _repairs_are_safe(critic) is False
