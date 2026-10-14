from __future__ import annotations

import asyncio

from redis.exceptions import ConnectionError, MaxConnectionsError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from services.queue.capacity import (
    estimate_goal_simultaneous_users,
    estimate_worker_redis_clients,
)
from services.queue.maintenance import should_reap_pending
from services.queue.shared_redis import (
    capacity_backoff_seconds,
    is_redis_capacity_error,
    is_windows_socket_abort,
    redis_pool_limit,
    supervisor_restart_delay,
)
from services.queue.streams import (
    EventEnvelope,
    IdlePoll,
    MultiplexedStreamHub,
    RedisStreamConsumer,
    group_stream_runners,
    xreadgroup_block,
)


class _FakePool:
    pass


class _FakeRedis:
    def __init__(self):
        self.connection_pool = _FakePool()
        self.inflight = 0
        self.max_inflight = 0
        self.reads = 0
        self.pending: dict[str, list] = {}
        self.acked: list[tuple[str, str, str]] = []

    async def xgroup_create(self, stream, group, id="0", mkstream=True):
        return True

    async def xreadgroup(self, groupname, consumername, streams, count=10, block=0):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.reads += 1
        try:
            await asyncio.sleep(0.005)
            out = []
            for stream, _cursor in streams.items():
                batch = (self.pending.get(stream) or [])[:count]
                self.pending[stream] = (self.pending.get(stream) or [])[count:]
                if batch:
                    out.append((stream, batch))
            return out
        finally:
            self.inflight -= 1

    async def xack(self, stream, group, message_id):
        self.acked.append((stream, group, message_id))
        return 1

    async def xautoclaim(self, *args, **kwargs):
        return ["0-0", []]


def test_windows_socket_abort_is_detected():
    error = ConnectionAbortedError(10053, "An established connection was aborted by the software in your host machine")
    error.winerror = 10053
    assert is_windows_socket_abort(error) is True
    reset = ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
    reset.winerror = 10054
    assert is_windows_socket_abort(reset) is True
    assert is_windows_socket_abort(RuntimeError("boom")) is False


def test_capacity_error_classifier():
    assert is_redis_capacity_error(ConnectionError("max number of clients reached"))
    assert is_redis_capacity_error(ResponseError("ERR max number of clients reached"))
    assert is_redis_capacity_error(MaxConnectionsError("Too many connections"))
    assert not is_redis_capacity_error(ConnectionError("timed out"))
    assert not is_redis_capacity_error(ResponseError("NOGROUP"))


def test_capacity_backoff_and_supervisor_delay():
    assert capacity_backoff_seconds(1) == 2
    assert capacity_backoff_seconds(5) == 30
    assert supervisor_restart_delay(ConnectionError("max number of clients reached")) == 20
    assert supervisor_restart_delay(ConnectionError("max number of clients reached"), storm=True) == 45
    assert supervisor_restart_delay(RuntimeError("boom")) == 2
    assert supervisor_restart_delay(RuntimeError("boom"), storm=True) == 8


def test_cloud_pool_stays_under_essentials_budget():
    assert redis_pool_limit("rediss://example.redislabs.com:12345", 16) == 8
    assert redis_pool_limit("redis://127.0.0.1:6379", 16) == 16


def test_multiplex_uses_one_reader_for_many_streams():
    redis = _FakeRedis()
    handled: list[str] = []

    async def handle(event: EventEnvelope):
        handled.append(event.eventId)

    consumers = []
    for index in range(8):
        stream = f"buddy:test:{index}"
        consumer = RedisStreamConsumer(
            stream=stream,
            group=f"group-{index}",
            consumer_name=f"c-{index}",
            handler=handle,
            redis=redis,
        )
        event = EventEnvelope(
            eventType="test",
            correlationId=f"corr-{index}",
            userId="u",
            spaceId="s",
            conversationId="c",
            payload={"i": index},
        )
        redis.pending[stream] = [("1-0", {"event": event.model_dump_json()})]
        consumers.append(consumer)

    hub = MultiplexedStreamHub(consumers)

    async def run_briefly():
        task = asyncio.create_task(hub.run_forever())
        for _ in range(50):
            if len(handled) == 8:
                break
            await asyncio.sleep(0.02)
        await hub.stop()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_briefly())
    assert redis.max_inflight == 1
    assert len(handled) == 8


def test_group_stream_runners_collapses_shared_pool():
    redis = _FakeRedis()

    async def handle(event: EventEnvelope):
        return None

    consumers = [
        RedisStreamConsumer("a", "ga", "ca", handle, redis=redis),
        RedisStreamConsumer("b", "gb", "cb", handle, redis=redis),
    ]
    runners = group_stream_runners(consumers)
    assert len(runners) == 1
    assert isinstance(runners[0], MultiplexedStreamHub)


def test_goal_capacity_model_covers_simultaneous_starters():
    capacity = estimate_goal_simultaneous_users(in_flight=3, avg_job_seconds=90, wait_seconds=300)
    assert capacity.in_flight_jobs == 3
    assert capacity.simultaneous_starters == 10
    multiplexed = estimate_worker_redis_clients(multiplexed=True, stream_count=12)
    legacy = estimate_worker_redis_clients(multiplexed=False, stream_count=12)
    assert multiplexed < legacy
    assert multiplexed <= 8


def test_simulated_simultaneous_goal_users_finish_inside_ui_wait():
    in_flight = 3
    job_seconds = 0.05
    users = 9
    wait_seconds = 0.2

    async def run():
        semaphore = asyncio.Semaphore(in_flight)
        waits: list[float] = []

        async def one(_index: int):
            started = asyncio.get_running_loop().time()
            async with semaphore:
                await asyncio.sleep(job_seconds)
            waits.append(asyncio.get_running_loop().time() - started)

        await asyncio.gather(*(one(index) for index in range(users)))
        return waits

    waits = asyncio.run(run())
    assert max(waits) <= wait_seconds + 0.05
    assert len(waits) == users


def test_read_or_recover_does_not_raise_max_clients():
    redis = _FakeRedis()

    async def boom(**kwargs):
        raise ConnectionError("max number of clients reached")

    redis.xreadgroup = boom  # type: ignore[method-assign]
    consumer = RedisStreamConsumer("s", "g", "c", lambda event: None, redis=redis)

    async def run():
        consumer._backoff_capacity = lambda error: asyncio.sleep(0)  # type: ignore[method-assign]
        return await consumer._read_or_recover(block_ms=0)

    messages = asyncio.run(run())
    assert messages == []


def test_timeout_read_reconnects_without_one_second_sleep(monkeypatch):
    redis = _FakeRedis()
    sleeps: list[float] = []
    recycled = {"n": 0}

    async def boom(**kwargs):
        raise RedisTimeoutError("Timeout reading from redis.example:18535")

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def fake_recycle(_client=None):
        recycled["n"] += 1

    redis.xreadgroup = boom  # type: ignore[method-assign]
    consumer = RedisStreamConsumer("s", "g", "c", lambda event: None, redis=redis)
    monkeypatch.setattr("services.queue.streams.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("services.queue.streams.recycle_broken_redis_pool", fake_recycle)
    monkeypatch.setattr("services.queue.streams.should_log_redis_timeout", lambda: False)

    messages = asyncio.run(consumer._read_or_recover(block_ms=0))
    assert messages == []
    assert recycled["n"] == 0
    assert sleeps == [0.5]


def test_goal_worker_reads_jobs_after_interactive_timeout():
    from apps.api_gateway.workers.goal_worker import GoalStreamsRunner

    class Redis(_FakeRedis):
        async def xreadgroup(self, groupname, consumername, streams, count=10, block=0):
            stream = next(iter(streams))
            if "interactive" in stream:
                raise RedisTimeoutError("Timeout reading from goals")
            return await super().xreadgroup(groupname, consumername, streams, count=count, block=block)

    async def run():
        runner = GoalStreamsRunner.__new__(GoalStreamsRunner)
        runner.redis = Redis()
        runner.redis.pending["buddy:goals:jobs"] = [("1-0", {"event": "{}"})]
        runner.consumer_name = "goal-test"
        runner.streams = {
            "buddy:goals:interactive": "goal-interactive",
            "buddy:goals:jobs": "goal-jobs",
        }
        runner._capacity_failures = 0
        runner._timeout_failures = 0
        interactive = await runner._read_stream("buddy:goals:interactive", "goal-interactive", 2)
        jobs = await runner._read_stream("buddy:goals:jobs", "goal-jobs", 1)
        return interactive, jobs

    interactive, jobs = asyncio.run(run())
    assert interactive == []
    assert jobs and jobs[0][0] == "buddy:goals:jobs"


def test_idle_poll_backs_off_when_queues_are_empty():
    idle = IdlePoll(min_seconds=0.4, max_seconds=2.5)
    first = idle.miss()
    second = idle.miss()
    idle.hit()
    assert first > 0.4
    assert second > first
    assert idle.seconds == 0.4


def test_redis_block_zero_is_non_blocking():
    assert xreadgroup_block(0) is None
    assert xreadgroup_block(-1) is None
    assert xreadgroup_block(None) is None
    assert xreadgroup_block(400) == 400


def test_read_batch_omits_forever_block():
    redis = _FakeRedis()
    seen: list[int | None] = []

    async def capture(**kwargs):
        seen.append(kwargs.get("block"))
        return []

    redis.xreadgroup = capture  # type: ignore[method-assign]
    consumer = RedisStreamConsumer("s", "g", "c", lambda event: None, redis=redis)
    asyncio.run(consumer.read_batch(block_ms=0))
    assert seen == [None]


def test_stuck_pending_jobs_are_reaped():
    assert should_reap_pending(idle_ms=15 * 60 * 1000, deliveries=1) is True
    assert should_reap_pending(idle_ms=1000, deliveries=8) is True
    assert should_reap_pending(idle_ms=1000, deliveries=1) is False
