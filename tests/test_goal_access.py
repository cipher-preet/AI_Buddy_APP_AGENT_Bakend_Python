import pytest

from services.goals.store import GoalStore


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def sort(self, *args, **kwargs):
        return self

    async def to_list(self, length):
        return self._rows[:length]


class _FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    async def find_one(self, query, sort=None):
        for row in self.docs:
            if row.get("goalId") == query.get("goalId") and (
                "ownerId" not in query or row.get("ownerId") == query.get("ownerId")
            ):
                return row
        return None

    def find(self, query):
        rows = [
            row
            for row in self.docs
            if row.get("goalId") == query.get("goalId")
            and ("ownerId" not in query or row.get("ownerId") == query.get("ownerId"))
        ]
        return _FakeCursor(rows)


class _FakeDb:
    def __init__(self):
        self.goal_research_sessions = _FakeCollection(
            [{"goalId": "g1", "ownerId": "user-a", "sources": [{"url": "https://secret.example"}]}]
        )
        self.goal_research_evidence = _FakeCollection([])
        self.goals = _FakeCollection([])
        self.goal_jobs = _FakeCollection([])
        self.goal_messages = _FakeCollection([])


@pytest.mark.asyncio
async def test_sources_are_owner_scoped(monkeypatch):
    store = GoalStore(_FakeDb())
    monkeypatch.setattr("services.goals.store.as_object_id", lambda value: value)
    visible = await store.load_research_sources("g1", owner_id="user-a")
    hidden = await store.load_research_sources("g1", owner_id="user-b")
    assert visible
    assert hidden == []
