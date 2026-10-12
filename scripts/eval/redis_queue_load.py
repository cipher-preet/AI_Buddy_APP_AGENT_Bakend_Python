"""Queue capacity report. Does not open extra Redis Cloud clients by default."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from services.queue.capacity import (
    CLOUD_CLIENT_BUDGET,
    estimate_goal_simultaneous_users,
    estimate_worker_redis_clients,
    cloud_client_headroom,
)
from apps.api_gateway.config.setting import settings


def _report() -> dict:
    stream_count = 12
    multiplexed_clients = estimate_worker_redis_clients(multiplexed=True, stream_count=stream_count)
    legacy_clients = estimate_worker_redis_clients(multiplexed=False, stream_count=stream_count)
    in_flight = max(1, min(int(settings.GOAL_CLARIFY_MAX_CONCURRENCY), 3))
    conservative = estimate_goal_simultaneous_users(
        in_flight=in_flight,
        avg_job_seconds=90,
        wait_seconds=300,
    )
    peak = estimate_goal_simultaneous_users(
        in_flight=in_flight,
        avg_job_seconds=60,
        wait_seconds=300,
    )
    return {
        "redis_cloud_client_budget": CLOUD_CLIENT_BUDGET,
        "legacy_blocking_readers": stream_count,
        "legacy_worker_clients": legacy_clients,
        "multiplexed_worker_clients": multiplexed_clients,
        "headroom_after_fix": cloud_client_headroom(worker_clients=multiplexed_clients),
        "goal_in_flight": in_flight,
        "simultaneous_plan_users_typical": conservative.simultaneous_starters,
        "simultaneous_plan_users_fast": peak.simultaneous_starters,
        "notes": [
            conservative.notes,
            "Conversation/STT/meeting streams now share one poller per Redis URL.",
            "Redis Cloud Essentials still has a ~30 client ceiling across Node + Python + leftovers.",
            "Do not run two worker processes against the same Cloud DB.",
        ],
    }


async def _simulate(users: int, in_flight: int, job_seconds: float) -> dict:
    semaphore = asyncio.Semaphore(in_flight)
    waits: list[float] = []

    async def one(_index: int) -> None:
        started = time.perf_counter()
        async with semaphore:
            await asyncio.sleep(job_seconds)
        waits.append(time.perf_counter() - started)

    started = time.perf_counter()
    await asyncio.gather(*(one(index) for index in range(users)))
    elapsed = time.perf_counter() - started
    return {
        "users": users,
        "in_flight": in_flight,
        "job_seconds": job_seconds,
        "wall_seconds": round(elapsed, 3),
        "max_wait_seconds": round(max(waits) if waits else 0, 3),
        "all_finished": len(waits) == users,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal/conversation Redis queue capacity")
    parser.add_argument("--users", type=int, default=12)
    parser.add_argument("--job-seconds", type=float, default=0.05)
    args = parser.parse_args()
    in_flight = max(1, min(int(settings.GOAL_CLARIFY_MAX_CONCURRENCY), 3))
    payload = _report()
    payload["simulation"] = asyncio.run(_simulate(args.users, in_flight, args.job_seconds))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
