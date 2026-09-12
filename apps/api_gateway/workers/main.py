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
from apps.api_gateway.workers.reminder_worker import start_reminder_worker
from apps.api_gateway.workers.daily_briefing_worker import (
    build_daily_briefing_consumer,
    run_daily_briefing_scheduler,
)
from services.db.mongo import close_mongo_client, ensure_mongo_indexes
from services.llm.router import close_llm_runtime, log_llm_provider_status
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
_FATAL_WORKER_ERRORS = (ReminderFcmConfigError, ReminderRedisConfigError)


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
            print(
                f"{name} failed: {redact_redis_secrets(str(error))}; restarting in 2s",
                flush=True,
            )
            await asyncio.sleep(2)


async def main():
    await test_redis_connection()
    await ensure_mongo_indexes()
    print("Conversation workers starting...")
    log_llm_provider_status("conversation-worker-startup")

    stt_consumer = build_stt_consumer()
    audio_consumer = build_audio_consumer()
    finalization_consumer = build_finalization_consumer()
    processing_consumer = build_processing_consumer()
    transcript_ready_consumer = build_transcript_ready_consumer()
    window_extraction_consumer = build_window_extraction_consumer()
    daily_briefing_consumer = build_daily_briefing_consumer()
    stream_consumers = [
        audio_consumer,
        stt_consumer,
        transcript_ready_consumer,
        window_extraction_consumer,
        finalization_consumer,
        processing_consumer,
        daily_briefing_consumer,
    ]

    await asyncio.gather(*(consumer.ensure_group() for consumer in stream_consumers))

    try:
        await asyncio.gather(
            _run_supervised("speech", start_speech_consumer),
            _run_supervised("vector", start_vector_consumer),
            *(
                _run_supervised(consumer.stream, consumer.run_forever)
                for consumer in stream_consumers
            ),
            _run_supervised("inactivity-scanner", run_inactivity_scanner),
            _run_supervised("retry-relay", run_retry_relay),
            _run_supervised("reminder-worker", start_reminder_worker),
            _run_supervised("daily-briefing-scheduler", run_daily_briefing_scheduler),
            _run_supervised("worker-health", run_worker_heartbeat),
            _run_supervised("event-loop-lag", run_event_loop_lag_monitor),
            _run_supervised("redis-queue-health", run_redis_queue_health),
        )
    finally:
        await close_llm_runtime()
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
