import asyncio

from services.email.activity import (
    activity_email_event_id,
    activity_email_redis_url,
    activity_kind,
    enqueue_activity_email,
    should_enqueue_activity_email,
)
from services.queue.job_lease import logical_job_key
from services.queue.streams import EventEnvelope


def test_listening_and_meeting_kinds():
    assert activity_kind(None) == "listening"
    assert activity_kind("buddy") == "listening"
    assert activity_kind("meeting_extension") == "meeting"


def test_listening_without_cards_is_not_queued():
    assert should_enqueue_activity_email("listening", 0, 0) is False
    assert should_enqueue_activity_email("listening", 1, 0) is True
    assert should_enqueue_activity_email("meeting", 0, 0) is True


def test_email_redis_matches_the_node_worker():
    assert activity_email_redis_url("redis://reminder:6379/0", "redis://conversation:6379/0") == "redis://reminder:6379/0"
    assert activity_email_redis_url("", "redis://conversation:6379/0") == ""


def test_event_id_is_stable_for_one_processing_version():
    first = activity_email_event_id("meeting", "abc", "2")
    second = activity_email_event_id("meeting", "abc", "2")
    assert first == second == "email:meeting:abc:2"


def test_email_job_identity_includes_conversation_and_version():
    event = EventEnvelope(
        eventType="email.activity.ready",
        correlationId="abc",
        userId="user",
        spaceId="",
        conversationId="abc",
        payload={"processingVersion": "2", "kind": "meeting"},
    )
    assert logical_job_key(event) == "email.activity.ready|conversation:abc|v:2"


def test_enqueue_returns_when_redis_is_down():
    class Down:
        async def set(self, *_args, **_kwargs):
            raise RuntimeError("redis down")

        async def delete(self, *_args, **_kwargs):
            return 1

    import services.email.activity as activity

    original = activity.get_activity_email_redis
    activity.get_activity_email_redis = lambda: Down()
    try:
        asyncio.run(
            enqueue_activity_email(
                user_id="user",
                space_id="space",
                conversation_id="abc",
                source_type=None,
                processing_version=1,
                task_count=1,
                note_count=0,
            )
        )
    finally:
        activity.get_activity_email_redis = original
