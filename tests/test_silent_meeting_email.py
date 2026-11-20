import asyncio
from types import SimpleNamespace

from services.conversation.models import SpaceMemoryDocument
from services.conversation.workflow import ConversationProcessingWorkflow, meeting_recording_has_speech


def test_blank_recording_has_no_speech():
    blank = SimpleNamespace(normalizedText="  ", rawText="")
    spoken = SimpleNamespace(normalizedText="", rawText="Let's ship the notes.")
    assert meeting_recording_has_speech([blank]) is False
    assert meeting_recording_has_speech([blank, spoken]) is True


def test_silent_meeting_still_queues_the_ready_email(monkeypatch):
    queued = []

    async def capture(**kwargs):
        queued.append(kwargs)

    monkeypatch.setattr("services.conversation.workflow.enqueue_activity_email", capture)

    async def explode(*_args, **_kwargs):
        raise AssertionError("empty recording must not call the memory model")

    monkeypatch.setattr("services.conversation.agents.update_space_memory", explode)

    class Repo:
        async def list_transcript_chunks(self, _conversation_id):
            return [SimpleNamespace(normalizedText="", rawText="   ")]

        async def save_extraction_run(self, _run):
            return None

        async def transition(self, *_args, **_kwargs):
            return None

        async def get_space_memory(self, user_id, space_id):
            return SpaceMemoryDocument(userId=user_id, spaceId=space_id)

        async def publish_outputs(self, _run, _summary, _memory):
            return {"taskIds": [], "noteIds": []}

        async def mark_transcripts_published(self, _conversation_id):
            return None

        async def schedule_transcript_expiry(self, _conversation_id):
            return None

    conversation = SimpleNamespace(
        id="meeting-1",
        userId="user-1",
        spaceId=None,
        sourceType="meeting_extension",
        processingVersion=1,
        lastAccounting={},
    )
    run = SimpleNamespace(checkpoints={})
    workflow = ConversationProcessingWorkflow(Repo(), router=object())

    finished = asyncio.run(workflow._finish_silent_meeting(conversation, run))

    assert finished is True
    assert queued == [
        {
            "user_id": "user-1",
            "space_id": "",
            "conversation_id": "meeting-1",
            "source_type": "meeting_extension",
            "processing_version": 1,
            "task_count": 0,
            "note_count": 0,
        }
    ]


def test_a_recording_with_speech_is_not_treated_as_silent():
    class Repo:
        async def list_transcript_chunks(self, _conversation_id):
            return [SimpleNamespace(normalizedText="We should send the recap.", rawText="")]

    conversation = SimpleNamespace(id="meeting-2", sourceType="meeting_extension")
    workflow = ConversationProcessingWorkflow(Repo(), router=object())

    assert asyncio.run(workflow._finish_silent_meeting(conversation, SimpleNamespace(checkpoints={}))) is False
