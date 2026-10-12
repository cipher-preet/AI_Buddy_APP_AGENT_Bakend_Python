from __future__ import annotations

from dataclasses import dataclass


# Redis Cloud Essentials typically caps total clients around this number
# across every process (Python workers, Node API, leftover sockets).
CLOUD_CLIENT_BUDGET = 30

# One worker process after multiplexing: 1 stream poller, 2 list blockers
# (speech/vector), retry relay, reminder, health, plus a few command sockets.
MULTIPLEXED_WORKER_CLIENTS = 8
LEGACY_STREAM_READERS = 12


@dataclass(frozen=True)
class GoalCapacity:
    in_flight_jobs: int
    avg_job_seconds: float
    ui_wait_seconds: float
    simultaneous_starters: int
    notes: str


def estimate_worker_redis_clients(*, multiplexed: bool, stream_count: int, blocking_lists: int = 2) -> int:
    """Sockets one Python worker holds against a single Redis URL."""
    readers = 1 if multiplexed else max(1, stream_count)
    # retry relay + reminder + health/commands
    auxiliaries = 3
    return readers + max(0, blocking_lists) + auxiliaries


def estimate_goal_simultaneous_users(
    *,
    in_flight: int,
    avg_job_seconds: float,
    wait_seconds: float,
) -> GoalCapacity:
    """
    How many users can start a plan at the same moment and still finish
    before the UI wait window, given in-flight LLM slots.
    """
    running = max(1, int(in_flight))
    duration = max(1.0, float(avg_job_seconds))
    window = max(duration, float(wait_seconds))
    starters = max(running, int(running * (window / duration)))
    return GoalCapacity(
        in_flight_jobs=running,
        avg_job_seconds=duration,
        ui_wait_seconds=window,
        simultaneous_starters=starters,
        notes=(
            f"{running} plans run at once; extras wait in Redis. "
            f"About {starters} users can start together if each plan takes ~{int(duration)}s "
            f"and the UI waits {int(window)}s."
        ),
    )


def cloud_client_headroom(*, worker_clients: int, node_clients: int = 2, leftover: int = 4) -> int:
    used = worker_clients + max(0, node_clients) + max(0, leftover)
    return CLOUD_CLIENT_BUDGET - used
