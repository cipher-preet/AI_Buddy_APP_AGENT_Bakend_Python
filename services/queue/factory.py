from __future__ import annotations

from apps.api_gateway.config.setting import settings
from services.queue.pubsub import MessagePublisher, PubSubMessagePublisher

_pubsub_publisher: PubSubMessagePublisher | None = None


def queue_provider() -> str:
    return settings.QUEUE_PROVIDER.strip().lower()


def use_pubsub() -> bool:
    return queue_provider() == "pubsub"


def get_message_publisher() -> MessagePublisher:
    global _pubsub_publisher
    if _pubsub_publisher is None:
        _pubsub_publisher = PubSubMessagePublisher()
    return _pubsub_publisher
