from datetime import datetime

from fastapi import UploadFile

from services.conversation.service import ConversationService


async def start_conversation_controller(user_id: str, space_id: str):
    return {"success": True, "data": await ConversationService().start(user_id, space_id)}


async def ingest_audio_controller(
    conversation_id: str,
    user_id: str,
    space_id: str,
    chunk_id: str,
    sequence_number: int,
    file: UploadFile,
    captured_at: datetime | None = None,
    duration_ms: int | None = None,
):
    data = await ConversationService().ingest_audio(
        conversation_id=conversation_id,
        user_id=user_id,
        space_id=space_id,
        chunk_id=chunk_id,
        sequence_number=sequence_number,
        file=file,
        captured_at=captured_at,
        duration_ms=duration_ms,
    )
    return {"success": True, "data": data}


async def stop_conversation_controller(
    conversation_id: str,
    user_id: str,
    space_id: str,
    last_sequence_number: int,
    stopped_at_client: datetime | None,
):
    return {
        "success": True,
        "message": "Conversation stop accepted; processing will continue asynchronously.",
        "data": await ConversationService().stop(
            conversation_id=conversation_id,
            user_id=user_id,
            space_id=space_id,
            last_sequence_number=last_sequence_number,
            stopped_at_client=stopped_at_client,
        ),
    }


async def conversation_status_controller(conversation_id: str, user_id: str, space_id: str):
    return {"success": True, "data": await ConversationService().status(conversation_id, user_id, space_id)}
