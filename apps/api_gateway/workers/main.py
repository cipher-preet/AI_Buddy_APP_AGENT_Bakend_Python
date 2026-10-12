import asyncio

from services.queue.redis_queue import test_redis_connection
from apps.api_gateway.workers.speech_worker import start_speech_consumer
from apps.api_gateway.workers.vector_worker import start_vector_consumer
from apps.api_gateway.workers.conversation_workers import (
    build_finalization_consumer,
    build_audio_consumer,
    build_processing_consumer,
    build_stt_consumer,
    build_transcript_ready_consumer,
    build_window_extraction_consumer,
    run_inactivity_scanner,
    run_retry_relay,
)
from apps.api_gateway.workers.meeting_extension_worker import (
    build_meeting_merge_consumer,
    build_meeting_video_consumer,
)
from apps.api_gateway.workers.reminder_worker import start_reminder_worker
from apps.api_gateway.workers.schedule_extraction_worker import build_schedule_extraction_consumer
from apps.api_gateway.workers.daily_briefing_worker import (
    build_daily_briefing_consumer,
    run_daily_briefing_scheduler,
)
from apps.api_gateway.workers.mindmap_worker import build_mindmap_consumer, ensure_mindmap_redis
from apps.api_gateway.workers.document_worker import build_document_consumer
from apps.api_gateway.workers.goal_worker import build_goal_consumers, run_goal_scheduler
from services.db.mongo import close_mongo_client, ensure_mongo_indexes
from services.goals.redis_client import GoalRedisConfigError, ensure_goal_redis
from services.llm.router import close_llm_runtime, log_llm_provider_status
from services.queue.shared_redis import close_shared_redis_clients, supervisor_restart_delay
from services.queue.streams import group_stream_runners
from services.meeting_extension.ffmpeg_audio import probe_ffmpeg
from services.meeting_extension.video_merge import resume_stuck_video_merges
from services.observability.diagnostics import (
    diag_log,
    run_event_loop_lag_monitor,
    run_redis_queue_health,
    run_worker_heartbeat,
    supervisor_restarts,
)
from services.reminders.fcm_factory import ReminderFcmConfigError
from services.reminders.redis_client import ReminderRedisConfigError, redact_redis_secrets

# Misconfiguration will not heal on restart; keep other workers alive instead of storming.
_FATAL_WORKER_ERRORS = (ReminderFcmConfigError, ReminderRedisConfigError, GoalRedisConfigError)


async def _run_supervised(name: str, factory) -> None:
    """Keep one worker crash from cancelling reminders and other loops."""
    while True:
        try:
            await factory()
            return
        except asyncio.CancelledError:
            raise
        except _FATAL_WORKER_ERRORS as error:
            diag_log(
                "worker_fatal_config",
                worker_name=name,
                exception_type=type(error).__name__,
                error=redact_redis_secrets(str(error)),
            )
            print(
                f"{name} fatal config error: {redact_redis_secrets(str(error))}; "
                "not restarting (fix config / rebuild image)",
                flush=True,
            )
            await asyncio.Event().wait()
        except Exception as error:
            details = supervisor_restarts.record(name, type(error).__name__)
            diag_log(
                "worker_loop_restart",
                worker_name=name,
                exception_type=type(error).__name__,
                restart_count=details["restart_count"],
                restarts_last_60s=details["restarts_last_60s"],
            )
            if details["storm"]:
                diag_log(
                    "worker_restart_storm",
                    worker_name=name,
                    exception_type=type(error).__name__,
                    restart_count=details["restart_count"],
                    restarts_last_60s=details["restarts_last_60s"],
                )
            delay = supervisor_restart_delay(error, storm=bool(details.get("storm")))
            print(
                f"{name} failed: {redact_redis_secrets(str(error))}; restarting in {delay:.0f}s",
                flush=True,
            )
            await asyncio.sleep(delay)


async def main():
    await test_redis_connection()
    await ensure_mindmap_redis()
    await ensure_goal_redis()
    await ensure_mongo_indexes()
    print("Conversation workers starting...")
    log_llm_provider_status("conversation-worker-startup")
    probe_ffmpeg()
    await resume_stuck_video_merges()

    stt_consumer = build_stt_consumer()
    audio_consumer = build_audio_consumer()
    finalization_consumer = build_finalization_consumer()
    processing_consumer = build_processing_consumer()
    transcript_ready_consumer = build_transcript_ready_consumer()
    window_extraction_consumer = build_window_extraction_consumer()
    daily_briefing_consumer = build_daily_briefing_consumer()
    mindmap_consumer = build_mindmap_consumer()
    document_consumer = build_document_consumer()
    meeting_video_consumer = build_meeting_video_consumer()
    meeting_merge_consumer = build_meeting_merge_consumer()
    schedule_extraction_consumer = build_schedule_extraction_consumer()
    goal_consumers = build_goal_consumers()
    stream_consumers = [
        consumer
        for consumer in (
            audio_consumer,
            stt_consumer,
            transcript_ready_consumer,
            window_extraction_consumer,
            finalization_consumer,
            processing_consumer,
            daily_briefing_consumer,
            mindmap_consumer,
            document_consumer,
            meeting_video_consumer,
            meeting_merge_consumer,
            schedule_extraction_consumer,
            *goal_consumers,
        )
        if consumer is not None
    ]
    stream_runners = group_stream_runners(stream_consumers)

    # Create groups one at a time so a shared Redis Cloud pool is not opened in a burst.
    for consumer in stream_runners:
        await consumer.ensure_group()
        await asyncio.sleep(0.05)

    async def _staggered(name: str, factory, delay: float):
        if delay > 0:
            await asyncio.sleep(delay)
        await _run_supervised(name, factory)

    try:
        await asyncio.gather(
            _run_supervised("speech", start_speech_consumer),
            _run_supervised("vector", start_vector_consumer),
            *(
                _staggered(consumer.stream, consumer.run_forever, index * 0.2)
                for index, consumer in enumerate(stream_runners)
            ),
            _staggered("inactivity-scanner", run_inactivity_scanner, 1.0),
            _staggered("retry-relay", run_retry_relay, 1.2),
            _staggered("reminder-worker", start_reminder_worker, 1.5),
            _staggered("daily-briefing-scheduler", run_daily_briefing_scheduler, 2.0),
            _staggered("goal-scheduler", run_goal_scheduler, 2.2),
            _staggered("worker-health", run_worker_heartbeat, 2.5),
            _staggered("event-loop-lag", run_event_loop_lag_monitor, 2.5),
            _staggered("redis-queue-health", run_redis_queue_health, 3.0),
        )
    finally:
        await close_llm_runtime()
        await close_mongo_client()
        await close_shared_redis_clients()


if __name__ == "__main__":
    asyncio.run(main())
