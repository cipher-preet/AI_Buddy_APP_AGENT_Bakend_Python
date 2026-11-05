"""Phase 2C.1 goal plan, research, and baseline evidence consistency."""

from __future__ import annotations

import asyncio
import time

import pytest
from bson import ObjectId

from services.goals.jobs import GoalJobHandler
from services.goals.schemas import ResearchPlanResult
from services.goals.store import (
    GoalStore,
    baseline_evidence_operation_id,
    ensure_idempotency_indexes,
    research_operation_id,
    review_evidence_operation_id,
    space_evidence_operation_id,
    space_ingest_operation_id,
)
from services.queue.job_lease import LeavePending, StaleOwnershipError, push_lease, reset_lease
from tests.test_goal_crash_recovery import _event, _lease, _seed, world

pytestmark = pytest.mark.integration


def _built(version: int = 1, summary: str = "Plan") -> dict:
    return {
        "version": version,
        "summary": summary,
        "mission": "Mission",
        "recommendation": summary,
        "milestones": [{"title": "Start", "successCriteria": ["Done"], "dueLabel": "Week 1", "dependencies": []}],
        "days": [{"id": "day-1", "dayIndex": 1, "title": "Start", "tasks": [{"id": "t1", "text": "Start"}]}],
        "model": "test",
        "measurementType": "MILESTONE",
    }


def _finding(text: str = "Baseline fact") -> dict:
    return {"text": text, "category": "OUTPUT", "sourceType": "baseline", "sourceId": "baseline"}


async def test_plan_version_exists_before_the_job_is_ready(world, monkeypatch):
    database, _redis, version, _me = world
    print(f"phase2c1_mongo={version}")
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    calls = []
    seen = []

    async def build(*_args, **_kwargs):
        calls.append("plan")
        return _built()

    real = GoalStore.save_plan_version

    async def spy(self, goal, job, plan):
        stored = await database.goal_jobs.find_one({"_id": ObjectId(job)})
        seen.append(stored["status"])
        return await real(self, goal, job, plan)

    monkeypatch.setattr("services.goals.jobs.build_plan", build)
    monkeypatch.setattr(GoalStore, "save_plan_version", spy)
    token = push_lease(_lease(4, owner="worker-a"))
    started = time.perf_counter()
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    elapsed = time.perf_counter() - started
    print(f"phase2c1_plan_finalize_s={elapsed:.6f}")
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    versions = await database.goal_plan_versions.count_documents({"goalId": ObjectId(goal_id)})
    assert seen == ["running"]
    assert stored["status"] == "ready"
    assert versions == 1
    assert calls == ["plan"]


async def test_crash_before_plan_version_is_recovered_without_the_model(world, monkeypatch):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    calls = []

    async def build(*_args, **_kwargs):
        calls.append("plan")
        return _built(summary="Recovered")

    real_save = GoalStore.save_plan_version

    async def crash(self, goal, job, plan):
        raise LeavePending("crash-before-plan-version")

    monkeypatch.setattr("services.goals.jobs.build_plan", build)
    monkeypatch.setattr(GoalStore, "save_plan_version", crash)
    token = push_lease(_lease(5, owner="worker-a"))
    try:
        with pytest.raises(LeavePending):
            await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "running"
    assert stored["durableStep"]["plan"]["summary"] == "Recovered"
    assert await database.goal_plan_versions.count_documents({"goalId": ObjectId(goal_id)}) == 0

    monkeypatch.setattr(GoalStore, "save_plan_version", real_save)
    token = push_lease(_lease(6, owner="worker-b"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    rows = await database.goal_plan_versions.find({"goalId": ObjectId(goal_id)}).to_list(length=5)
    assert calls == ["plan"]
    assert stored["status"] == "ready"
    assert len(rows) == 1
    assert rows[0]["plan"]["summary"] == "Recovered"
    assert rows[0]["version"] == 1


async def test_retry_after_the_version_write_does_not_duplicate_it(world, monkeypatch):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    calls = []

    async def build(*_args, **_kwargs):
        calls.append("plan")
        return _built()

    monkeypatch.setattr("services.goals.jobs.build_plan", build)
    token = push_lease(_lease(7, owner="worker-a"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    token = push_lease(_lease(8, owner="worker-b"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    assert calls == ["plan"]
    assert await database.goal_plan_versions.count_documents({"goalId": ObjectId(goal_id)}) == 1


async def test_ready_job_missing_its_version_is_reconciled_from_the_durable_step(world, monkeypatch):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database, status="ready")
    plan = _built(summary="Legacy")
    await database.goal_jobs.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"durableStep": {"plan": plan, "step": {"stepId": "plan-review"}}}},
    )
    calls = []

    async def build(*_args, **_kwargs):
        calls.append("plan")
        return _built()

    monkeypatch.setattr("services.goals.jobs.build_plan", build)
    token = push_lease(_lease(9, owner="worker-a"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    rows = await database.goal_plan_versions.find({"goalId": ObjectId(goal_id)}).to_list(length=5)
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert calls == []
    assert stored["status"] == "ready"
    assert len(rows) == 1
    assert rows[0]["plan"]["summary"] == "Legacy"
    assert rows[0]["reconciled"] is True
    token = push_lease(_lease(10, owner="worker-b"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    assert await database.goal_plan_versions.count_documents({"goalId": ObjectId(goal_id)}) == 1


async def test_a_new_plan_version_does_not_replace_the_previous_one(world, monkeypatch):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, first_job = await _seed(database)
    second_goal, second_job = await _seed(database)
    assert second_goal
    plans = [_built(version=1, summary="First"), _built(version=2, summary="Second")]

    async def build(*_args, **_kwargs):
        return plans.pop(0)

    monkeypatch.setattr("services.goals.jobs.build_plan", build)
    for index, job_id in enumerate((first_job, second_job), start=11):
        token = push_lease(_lease(index, owner=f"worker-{index}"))
        try:
            await GoalJobHandler(database).handle(_event(goal_id if index == 11 else second_goal, job_id, "plan"))
        finally:
            reset_lease(token)
    first = await database.goal_plan_versions.find_one({"goalId": ObjectId(goal_id), "version": 1})
    second = await database.goal_plan_versions.find_one({"goalId": ObjectId(second_goal), "version": 2})
    assert first["plan"]["summary"] == "First"
    assert second["plan"]["summary"] == "Second"
    assert await database.goal_plan_versions.count_documents({"goalId": ObjectId(goal_id)}) == 1


async def test_same_goal_keeps_an_older_version_when_a_newer_one_is_added(world):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    store = GoalStore(database)
    token = push_lease(_lease(20, owner="worker-a"))
    try:
        await store.save_plan_version(goal, job_id, _built(version=1, summary="First"))
        await store.save_plan_version(goal, job_id, _built(version=2, summary="Second"))
    finally:
        reset_lease(token)
    rows = await database.goal_plan_versions.find({"goalId": ObjectId(goal_id)}).sort("version", 1).to_list(length=5)
    assert [row["version"] for row in rows] == [1, 2]
    assert rows[0]["plan"]["summary"] == "First"
    assert rows[1]["plan"]["summary"] == "Second"


async def test_stale_plan_writer_cannot_replace_a_newer_version(world):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    store = GoalStore(database)
    token = push_lease(_lease(30, owner="worker-b"))
    try:
        await store.save_plan_version(goal, job_id, _built(summary="Newer"))
    finally:
        reset_lease(token)
    token = push_lease(_lease(29, owner="worker-a"))
    try:
        with pytest.raises(StaleOwnershipError):
            await store.save_plan_version(goal, job_id, _built(summary="Older"))
    finally:
        reset_lease(token)
    row = await database.goal_plan_versions.find_one({"goalId": ObjectId(goal_id), "version": 1})
    assert row["plan"]["summary"] == "Newer"


async def test_research_retry_reuses_one_session_and_skips_the_planner(world, monkeypatch):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    calls = []

    async def plan_research(goal_row, cached_hits=0):
        calls.append(cached_hits)
        return ResearchPlanResult(researchNeeded=False, reason="not needed")

    monkeypatch.setattr("services.goals.research.plan_research", plan_research)
    from services.goals.research import run_research

    token = push_lease(_lease(40, owner="worker-a"))
    try:
        first = await run_research(GoalStore(database), goal, force_refresh=True, job_id=job_id)
        second = await run_research(GoalStore(database), goal, force_refresh=True, job_id=job_id)
    finally:
        reset_lease(token)
    count = await database.goal_research_sessions.count_documents({"goalId": ObjectId(goal_id)})
    assert count == 1
    assert calls == [0]
    assert second["reused"] is True
    assert second["summary"]["sessionId"] == first["summary"]["sessionId"]


async def test_concurrent_research_writes_keep_one_session(world):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    store = GoalStore(database)
    operation = research_operation_id(goal_id, job_id)

    async def save(index: int) -> None:
        await store.save_research_session(
            goal,
            {"sessionId": f"session-{index}", "reason": "same"},
            [],
            [],
            operation,
        )

    await asyncio.gather(save(1), save(2), save(3))
    assert await database.goal_research_sessions.count_documents({"goalId": ObjectId(goal_id), "operationId": operation}) == 1


async def test_distinct_research_jobs_stay_separate(world):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, first_job = await _seed(database)
    _goal, second_job = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    store = GoalStore(database)
    await store.save_research_session(goal, {"sessionId": "one", "reason": "a"}, [], [], research_operation_id(goal_id, first_job))
    await store.save_research_session(goal, {"sessionId": "two", "reason": "b"}, [], [], research_operation_id(goal_id, second_job))
    assert await database.goal_research_sessions.count_documents({"goalId": ObjectId(goal_id)}) == 2


async def test_stale_research_writer_cannot_replace_the_newer_summary(world):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    store = GoalStore(database)
    operation = research_operation_id(goal["_id"], job_id)
    token = push_lease(_lease(50, owner="worker-b"))
    try:
        await store.save_research_session(goal, {"sessionId": "new", "reason": "newer"}, [{"url": "https://example.com"}], [], operation)
    finally:
        reset_lease(token)
    token = push_lease(_lease(49, owner="worker-a"))
    try:
        with pytest.raises(StaleOwnershipError):
            await store.save_research_session(goal, {"sessionId": "old", "reason": "older"}, [], [], operation)
    finally:
        reset_lease(token)
    row = await database.goal_research_sessions.find_one({"goalId": goal["_id"], "operationId": operation})
    assert row["summary"]["reason"] == "newer"
    assert await database.goal_research_sessions.count_documents({"operationId": operation}) == 1


async def test_baseline_retry_advances_evidence_version_once(world):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    token = push_lease(_lease(60, owner="worker-a"))
    try:
        await handler._store_findings(goal, [_finding()], job_id)
        await handler._store_findings(goal, [_finding()], job_id)
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert stored["appliedEvidenceOps"] == [baseline_evidence_operation_id(goal_id, job_id)]
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1


async def test_a_new_baseline_job_advances_evidence_version_again(world):
    database, _redis, _version, _me = world
    goal_id, first_job = await _seed(database)
    _other, second_job = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    token = push_lease(_lease(61, owner="worker-a"))
    try:
        await handler._store_findings(goal, [_finding("First")], first_job)
        await handler._store_findings(goal, [_finding("Second")], second_job)
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 2
    assert isinstance(stored["evidenceVersion"], int)


async def test_concurrent_baseline_attempts_advance_evidence_once(world):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)

    async def once() -> None:
        await handler._store_findings(goal, [_finding("Race")], job_id)

    await asyncio.gather(once(), once(), once())
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1


async def test_stale_baseline_writer_cannot_change_evidence(world):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    token = push_lease(_lease(70, owner="worker-b"))
    try:
        await handler._store_findings(goal, [_finding("Current")], job_id)
    finally:
        reset_lease(token)
    token = push_lease(_lease(69, owner="worker-a"))
    try:
        with pytest.raises(StaleOwnershipError):
            await handler._store_findings(goal, [_finding("Stale")], job_id)
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    row = await database.goal_evidence.find_one({"goalId": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert row["text"] == "Current"


async def test_crash_between_findings_and_version_reconciles_once(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    real = __import__("services.goals.jobs", fromlist=["fenced_update_one"]).fenced_update_one
    state = {"findings": 0}

    async def crash_before_version(collection, *args, **kwargs):
        name = getattr(collection, "name", "")
        if name == "goal_evidence":
            state["findings"] += 1
        if name == "goals" and state["findings"]:
            raise LeavePending("crash-before-version")
        return await real(collection, *args, **kwargs)

    monkeypatch.setattr("services.goals.jobs.fenced_update_one", crash_before_version)
    token = push_lease(_lease(80, owner="worker-a"))
    try:
        with pytest.raises(LeavePending):
            await handler._store_findings(goal, [_finding("Kept")], job_id)
    finally:
        reset_lease(token)
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1
    assert (await database.goals.find_one({"_id": ObjectId(goal_id)})).get("evidenceVersion") in {None, 0}

    monkeypatch.setattr("services.goals.jobs.fenced_update_one", real)
    token = push_lease(_lease(80, owner="worker-a"))
    try:
        await handler._store_findings(goal, [_finding("Kept")], job_id)
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1


async def test_evidence_version_stays_an_integer_other_writers_can_increment(world):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    token = push_lease(_lease(90, owner="worker-a"))
    try:
        await GoalJobHandler(database)._store_findings(goal, [_finding()], job_id)
    finally:
        reset_lease(token)
    await database.goals.update_one({"_id": ObjectId(goal_id)}, {"$inc": {"evidenceVersion": 1}})
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 2


async def test_combined_crash_keeps_one_session_and_restores_the_plan_version(world, monkeypatch):
    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    calls = []

    async def build(*_args, **_kwargs):
        calls.append("plan")
        return _built(summary="Combined")

    real_save = GoalStore.save_plan_version

    async def crash(self, goal_row, job, plan):
        raise LeavePending("crash")

    await GoalStore(database).save_research_session(
        goal,
        {"sessionId": "combined", "reason": "kept"},
        [],
        [],
        research_operation_id(goal["_id"], job_id),
    )
    monkeypatch.setattr("services.goals.jobs.build_plan", build)
    monkeypatch.setattr(GoalStore, "save_plan_version", crash)
    token = push_lease(_lease(100, owner="worker-a"))
    try:
        with pytest.raises(LeavePending):
            await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    monkeypatch.setattr(GoalStore, "save_plan_version", real_save)
    token = push_lease(_lease(101, owner="worker-b"))
    try:
        await GoalJobHandler(database).handle(_event(goal_id, job_id, "plan"))
    finally:
        reset_lease(token)
    assert calls == ["plan"]
    assert await database.goal_research_sessions.count_documents({"goalId": ObjectId(goal_id)}) == 1
    stored = await database.goal_jobs.find_one({"_id": ObjectId(job_id)})
    assert stored["status"] == "ready"
    assert await database.goal_plan_versions.count_documents({"goalId": ObjectId(goal_id), "version": 1}) == 1


def _approved_plan() -> dict:
    return {
        "version": 1,
        "summary": "Plan",
        "strategies": [{"title": "Follow the plan", "body": "Work the next step."}],
        "priorityRecommendations": [{"title": "Next step", "reason": "It is open."}],
        "days": [{"title": "Write the proof", "status": "open", "tasks": []}],
        "milestones": [{"title": "First proof"}],
    }


async def _second_job(database, goal_id: str) -> str:
    job_id = ObjectId()
    await database.goal_jobs.insert_one(
        {
            "_id": job_id,
            "goalId": ObjectId(goal_id),
            "ownerId": "user-1",
            "jobType": "evidence",
            "status": "running",
            "payload": {},
        }
    )
    return str(job_id)


async def test_space_evidence_retry_advances_evidence_version_once(world):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    payload = {"title": "Mail the build", "status": "completed", "sourceType": "task", "sourceId": "task-1", "sourceRevision": "1"}
    token = push_lease(_lease(110, owner="worker-a"))
    try:
        await handler._evidence(goal, job_id, payload)
        await handler._evidence(goal, job_id, payload)
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert stored["appliedEvidenceOps"] == [space_evidence_operation_id(goal_id, job_id)]
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1
    cached_version = stored["evidenceVersion"]
    await handler._evidence(goal, job_id, payload)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == cached_version
    await database.goals.update_one({"_id": ObjectId(goal_id)}, {"$inc": {"evidenceVersion": 1}})
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 2
    assert isinstance(stored["evidenceVersion"], int)


async def test_space_evidence_new_revision_advances_evidence_version_again(world):
    database, _redis, _version, _me = world
    goal_id, first_job = await _seed(database)
    second_job = await _second_job(database, goal_id)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    token = push_lease(_lease(111, owner="worker-a"))
    try:
        await handler._evidence(goal, first_job, {"title": "Mail the build", "status": "completed", "sourceType": "task", "sourceId": "task-1", "sourceRevision": "1"})
        await handler._evidence(goal, second_job, {"title": "Mail the build", "status": "reopened", "sourceType": "task", "sourceId": "task-1", "sourceRevision": "2"})
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 2
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 2


async def test_stale_space_evidence_writer_cannot_change_evidence(world):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    token = push_lease(_lease(120, owner="worker-b"))
    try:
        await handler._evidence(goal, job_id, {"title": "Current task", "status": "completed", "sourceType": "task", "sourceId": "task-1", "sourceRevision": "3"})
    finally:
        reset_lease(token)
    token = push_lease(_lease(119, owner="worker-a"))
    try:
        with pytest.raises(StaleOwnershipError):
            await handler._evidence(goal, job_id, {"title": "Stale task", "status": "completed", "sourceType": "task", "sourceId": "task-1", "sourceRevision": "3"})
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    row = await database.goal_evidence.find_one({"goalId": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert "Current task" in row["text"]
    assert "Stale" not in row["text"]


async def test_space_evidence_crash_before_version_reconciles_once(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    handler = GoalJobHandler(database)
    real = __import__("services.goals.jobs", fromlist=["fenced_update_one"]).fenced_update_one
    state = {"evidence": 0}

    async def crash_before_version(collection, *args, **kwargs):
        name = getattr(collection, "name", "")
        if name == "goal_evidence":
            state["evidence"] += 1
        if name == "goals" and state["evidence"]:
            raise LeavePending("crash-before-version")
        return await real(collection, *args, **kwargs)

    monkeypatch.setattr("services.goals.jobs.fenced_update_one", crash_before_version)
    payload = {"title": "Kept task", "status": "added", "sourceType": "task", "sourceId": "task-9", "sourceRevision": "1"}
    token = push_lease(_lease(130, owner="worker-a"))
    try:
        with pytest.raises(LeavePending):
            await handler._evidence(goal, job_id, payload)
    finally:
        reset_lease(token)
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1
    assert (await database.goals.find_one({"_id": ObjectId(goal_id)})).get("evidenceVersion") in {None, 0}

    monkeypatch.setattr("services.goals.jobs.fenced_update_one", real)
    token = push_lease(_lease(130, owner="worker-a"))
    try:
        await handler._evidence(goal, job_id, payload)
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1


async def test_review_retry_advances_evidence_version_once(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    await database.goals.update_one(
        {"_id": ObjectId(goal_id)},
        {"$set": {"title": "Become a ZSM", "approvedPlan": _approved_plan(), "activePlanVersion": 1}},
    )
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})

    async def no_model(*_args, **_kwargs):
        return None

    monkeypatch.setattr("services.goals.reviews._model_review", no_model)
    handler = GoalJobHandler(database)
    payload = {"periodType": "daily", "periodKey": "2026-11-01"}
    token = push_lease(_lease(140, owner="worker-a"))
    try:
        await handler._review(goal, job_id, payload)
        await handler._review(goal, job_id, payload)
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert review_evidence_operation_id(goal_id, job_id) in stored["appliedEvidenceOps"]
    assert await database.goal_reviews.count_documents({"goalId": ObjectId(goal_id), "periodKey": "2026-11-01"}) == 1
    second_job = await _second_job(database, goal_id)
    token = push_lease(_lease(140, owner="worker-a"))
    try:
        await handler._review(goal, second_job, {"periodType": "daily", "periodKey": "2026-11-02"})
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 2
    assert isinstance(stored["evidenceVersion"], int)


async def test_stale_review_cannot_overwrite_newer_progress(world, monkeypatch):
    database, _redis, _version, _me = world
    goal_id, job_id = await _seed(database)
    await database.goals.update_one(
        {"_id": ObjectId(goal_id)},
        {"$set": {"title": "Become a ZSM", "approvedPlan": _approved_plan(), "activePlanVersion": 1}},
    )
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})

    async def no_model(*_args, **_kwargs):
        return None

    monkeypatch.setattr("services.goals.reviews._model_review", no_model)
    handler = GoalJobHandler(database)
    token = push_lease(_lease(150, owner="worker-b"))
    try:
        await handler._review(goal, job_id, {"periodType": "daily", "periodKey": "2026-11-03"})
    finally:
        reset_lease(token)
    await database.goals.update_one({"_id": ObjectId(goal_id)}, {"$set": {"progress.marker": "newer"}})
    token = push_lease(_lease(149, owner="worker-a"))
    try:
        with pytest.raises(StaleOwnershipError):
            await handler._review(goal, job_id, {"periodType": "daily", "periodKey": "2026-11-03"})
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert stored["progress"]["marker"] == "newer"


async def test_space_ingest_revision_is_idempotent_and_fenced(world):
    from services.goals.reviews import _ingest_space_evidence

    database, _redis, _version, _me = world
    await ensure_idempotency_indexes(database)
    goal_id, _job_id = await _seed(database)
    goal = await database.goals.find_one({"_id": ObjectId(goal_id)})
    store = GoalStore(database)
    finding = {"text": "Space task is open.", "sourceId": "task-1", "sourceType": "task", "category": "COMMITMENT"}
    revised = {"text": "Space task was completed.", "sourceId": "task-1", "sourceType": "task", "category": "OUTPUT"}
    token = push_lease(_lease(160, owner="worker-a"))
    try:
        await _ingest_space_evidence(store, goal, [finding], "rev-1")
        await _ingest_space_evidence(store, goal, [finding], "rev-1")
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 1
    assert stored["appliedEvidenceOps"] == [space_ingest_operation_id(goal_id, "rev-1")]
    assert await database.goal_evidence.count_documents({"goalId": ObjectId(goal_id)}) == 1
    token = push_lease(_lease(160, owner="worker-a"))
    try:
        await _ingest_space_evidence(store, goal, [revised], "rev-2")
    finally:
        reset_lease(token)
    stored = await database.goals.find_one({"_id": ObjectId(goal_id)})
    assert stored["evidenceVersion"] == 2
    token = push_lease(_lease(159, owner="worker-stale"))
    try:
        with pytest.raises(StaleOwnershipError):
            await _ingest_space_evidence(store, goal, [{"text": "Stale overwrite", "sourceId": "task-1", "sourceType": "task", "category": "OUTPUT"}], "rev-2")
    finally:
        reset_lease(token)
    row = await database.goal_evidence.find_one({"goalId": ObjectId(goal_id), "sourceRevision": "rev-2"})
    assert "completed" in row["text"]
    assert stored["evidenceVersion"] == 2
