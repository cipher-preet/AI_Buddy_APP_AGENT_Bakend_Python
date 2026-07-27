import asyncio

from services.queue.redis_queue import test_redis_connection
from apps.api_gateway.workers.speech_worker import start_speech_consumer
from apps.api_gateway.workers.vector_worker import start_vector_consumer
from apps.api_gateway.workers.analysis_worker import start_analysis_consumer
from apps.api_gateway.workers.transcript_session_worker import (
    start_transcript_session_consumer,
)


async def main():
    await test_redis_connection()

    await asyncio.gather(
        start_speech_consumer(),
        start_vector_consumer(),
        start_analysis_consumer(),
        start_transcript_session_consumer(),
    )


if __name__ == "__main__":
    asyncio.run(main())
