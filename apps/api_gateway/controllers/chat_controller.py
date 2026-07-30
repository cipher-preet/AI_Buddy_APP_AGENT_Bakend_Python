from services.chat.service import ChatService


async def create_chat_session_controller(user_id: str, space_id: str | None = None):
    return {"success": True, "data": await ChatService().create_chat_session(user_id, space_id)}


async def ask_chat_controller(
    user_id: str,
    question: str,
    space_id: str | None = None,
    chat_id: str | None = None,
):
    return {
        "success": True,
        "data": await ChatService().ask(
            user_id=user_id,
            space_id=space_id,
            chat_id=chat_id,
            question=question,
        ),
    }


async def load_chat_controller(user_id: str, chat_id: str):
    return {"success": True, "data": await ChatService().load_chat(user_id, chat_id)}


async def list_chats_controller(
    user_id: str,
    space_id: str | None = None,
    limit: int = 20,
    cursor: str | None = None,
):
    return {"success": True, "data": await ChatService().list_chats(user_id, space_id, limit, cursor)}
