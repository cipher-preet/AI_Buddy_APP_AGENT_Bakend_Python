from __future__ import annotations

from apps.api_gateway.config.setting import settings


def queue_provider() -> str:
    return settings.QUEUE_PROVIDER.strip().lower()


def use_queue_api() -> bool:
    return queue_provider() == "queue_api"
