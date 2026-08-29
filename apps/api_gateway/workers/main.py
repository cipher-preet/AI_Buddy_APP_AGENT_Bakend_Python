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
from services.db.mongo import close_mongo_client, ensure_mongo_indexes
from services.llm.router import close_llm_runtime, log_llm_provider_status


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
    stream_consumers = [
        audio_consumer,
        stt_consumer,
        transcript_ready_consumer,
        window_extraction_consumer,
        finalization_consumer,
        processing_consumer,
    ]

    await asyncio.gather(*(consumer.ensure_group() for consumer in stream_consumers))

    try:
        await asyncio.gather(
            start_speech_consumer(),
            start_vector_consumer(),
            *(consumer.run_forever() for consumer in stream_consumers),
            run_inactivity_scanner(),
            run_retry_relay(),
        )
    finally:
        await close_llm_runtime()
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
