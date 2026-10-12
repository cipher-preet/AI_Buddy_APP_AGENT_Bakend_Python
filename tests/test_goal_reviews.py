import pytest

from services.goals.progress import chart_actual, execution_progress
from services.goals.reviews import apply_coach_updates, fallback_coach, space_findings


def test_chart_actual_uses_space_tasks_and_notes():
    goal = {
        "approvedPlan": {
            "days": [{"tasks": [{"text": "Prep", "done": True}, {"text": "Apply", "done": False}]}],
            "checklists": [],
        }
    }
    context = {
        "tasks": [
            {"sourceId": "t1", "title": "SQL drill", "status": "completed"},
            {"sourceId": "t2", "title": "Mock interview", "status": "pending"},
        ],
        "notes": [{"sourceId": "n1", "title": "Gaps"}],
        "transcripts": [{"sourceId": "c1", "title": "Talked about ZSM path"}],
    }
    score = chart_actual(goal, space_context=context)
    assert 1 <= score <= 100
    empty = chart_actual({"approvedPlan": {}}, evidence=[], space_context={"tasks": [], "notes": [], "transcripts": []})
    assert empty == 0


def test_pending_space_task_is_not_done():
    from services.goals.progress import is_done_status

    assert is_done_status("completed")
    assert is_done_status("DONE")
    assert not is_done_status("pending")
    assert not is_done_status("CREATE")


def test_empty_space_context_falls_back_to_evidence():
    goal = {"approvedPlan": {"days": [], "checklists": []}}
    evidence = [
        {"sourceType": "task", "sourceId": "t1", "category": "OUTPUT"},
        {"sourceType": "note", "sourceId": "n1", "category": "INTENT"},
    ]
    score = chart_actual(goal, evidence=evidence, space_context={"tasks": [], "notes": [], "transcripts": []})
    assert score == chart_actual(goal, evidence=evidence)


def test_chart_actual_from_evidence_without_live_space():
    goal = {"approvedPlan": {"days": [], "checklists": []}}
    evidence = [
        {"sourceType": "task", "sourceId": "t1", "category": "OUTPUT"},
        {"sourceType": "task", "sourceId": "t2", "category": "COMMITMENT"},
        {"sourceType": "note", "sourceId": "n1", "category": "INTENT"},
    ]
    score = chart_actual(goal, evidence=evidence)
    assert score > 0


def test_space_findings_tag_done_tasks_as_output():
    findings = space_findings(
        {
            "tasks": [{"sourceId": "t1", "title": "Done item", "status": "completed", "detail": "shipped"}],
            "notes": [{"sourceId": "n1", "title": "Interview notes", "detail": "SQL weak"}],
            "transcripts": [{"sourceId": "c1", "title": "Need two mock interviews", "detail": "Need two mock interviews"}],
        }
    )
    kinds = {(row["sourceType"], row["category"]) for row in findings}
    assert ("task", "OUTPUT") in kinds
    assert ("note", "INTENT") in kinds
    assert ("transcript", "OUTPUT") in kinds


def test_fallback_coach_prefers_open_space_task():
    goal = {
        "title": "Become a ZSM",
        "approvedPlan": {
            "strategies": [{"id": "s1", "title": "Banking path", "body": "Build branch credibility"}],
            "priorityRecommendations": [{"title": "Old rec", "reason": "stale"}],
        },
    }
    context = {
        "spaces": [{"id": "s", "name": "Career"}],
        "tasks": [{"title": "Finish JAIIB chapter 3", "status": "pending"}],
        "notes": [{"title": "Credit risk notes"}],
        "transcripts": [],
    }
    drafted = fallback_coach(goal, context, [])
    titles = " ".join(item["title"] for item in drafted["priorityRecommendations"])
    assert "JAIIB" in titles
    assert "Career" in drafted["focus"] or "JAIIB" in drafted["focus"]


def test_apply_coach_updates_keeps_days():
    plan = {
        "days": [{"dayIndex": 1, "title": "Day 1", "tasks": [{"id": "a", "text": "Keep me", "done": True}]}],
        "strategies": [{"id": "old", "title": "Old", "body": "Old body"}],
        "priorityRecommendations": [{"title": "Old rec", "reason": "old"}],
        "recommendation": "Old coaching",
    }
    next_plan = apply_coach_updates(
        plan,
        {
            "recommendation": "Use the Space notes this week.",
            "strategies": [{"title": "New strategy", "body": "Because the Space task is still open."}],
            "priorityRecommendations": [{"title": "Finish the open Space task", "reason": "It is the next proof."}],
        },
    )
    assert next_plan["days"][0]["tasks"][0]["done"] is True
    assert next_plan["strategies"][0]["title"] == "New strategy"
    assert "Space" in next_plan["recommendation"]


def test_execution_progress_counts_space_tasks():
    goal = {"approvedPlan": {"days": [{"tasks": [{"done": True}]}], "checklists": []}}
    progress = execution_progress(
        goal,
        {"tasks": [{"status": "completed"}, {"status": "pending"}]},
    )
    assert progress["completed"] == 2
    assert progress["total"] == 3


class _FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, *args, **kwargs):
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    async def to_list(self, length):
        return self.rows[:length]


class _FakeColl:
    def __init__(self):
        self.rows = []

    def find(self, query):
        return _FakeCursor(_match(self.rows, query))

    async def find_one(self, query):
        matched = _match(self.rows, query)
        return matched[0] if matched else None

    async def update_one(self, filt, update, upsert=False):
        matched = _match(self.rows, filt)
        if matched:
            matched[0].update(update.get("$set") or {})
            for key, value in (update.get("$inc") or {}).items():
                matched[0][key] = matched[0].get(key, 0) + value
            return
        if upsert:
            row = {**filt, **(update.get("$setOnInsert") or {}), **(update.get("$set") or {})}
            self.rows.append(row)


def _match(rows, query):
    matched = []
    for row in rows:
        if all(row.get(key) == value for key, value in query.items() if not str(key).startswith("$")):
            matched.append(row)
    return matched


class _FakeStore:
    def __init__(self, goal):
        self.goal = goal
        self.db = type(
            "DB",
            (),
            {
                "goal_evidence": _FakeColl(),
                "goal_reviews": _FakeColl(),
                "goal_snapshots": _FakeColl(),
                "goal_plan_versions": _FakeColl(),
                "goals": _FakeColl(),
            },
        )()
        self.db.goals.rows.append(goal)

    async def linked_space_ids(self, _goal_id):
        return ["space-1"]

    async def patch_goal(self, _goal_id, updates):
        self.goal.update(updates)


@pytest.mark.asyncio
async def test_build_review_ingests_space_and_writes_snapshot(monkeypatch):
    from services.goals import reviews as reviews_mod

    goal = {
        "_id": "goal-1",
        "ownerId": "user-1",
        "title": "Become a ZSM",
        "status": "ACTIVE",
        "approvedPlan": {
            "strategies": [{"id": "s1", "title": "Banking path", "body": "Build proof in the branch."}],
            "priorityRecommendations": [{"title": "Old", "reason": "stale"}],
            "days": [{"tasks": [{"id": "d1", "text": "Prep resume", "done": False}]}],
            "milestones": [{"title": "First interview"}],
        },
        "activePlanVersion": 1,
    }
    store = _FakeStore(goal)

    async def fake_space(*_args, **_kwargs):
        return {
            "spaces": [{"id": "space-1", "name": "Career"}],
            "tasks": [{"sourceId": "t1", "title": "JAIIB chapter", "status": "completed", "detail": ""}],
            "notes": [{"sourceId": "n1", "title": "Credit risk", "detail": "Need practice cases"}],
            "transcripts": [{"sourceId": "c1", "title": "Discussed mock interviews", "detail": "Discussed mock interviews"}],
        }

    async def no_model(*_args, **_kwargs):
        return None

    monkeypatch.setattr(reviews_mod, "load_space_context", fake_space)
    monkeypatch.setattr(reviews_mod, "_model_review", no_model)

    await reviews_mod.build_review(store, goal, "daily", "2026-10-12")

    assert store.db.goal_evidence.rows
    assert store.db.goal_reviews.rows[0]["summary"]
    snapshot = store.db.goal_snapshots.rows[0]
    assert snapshot["periodKey"] == "2026-10-12"
    assert snapshot["target"] == 100
    assert isinstance(snapshot["actual"], int)
    recs = " ".join(item["title"] for item in store.goal["approvedPlan"]["priorityRecommendations"])
    assert "Credit risk" in recs or "JAIIB" in recs
