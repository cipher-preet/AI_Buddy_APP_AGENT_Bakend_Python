__all__ = ["invoke_task_note_graph", "run_task_note_orchestration"]


async def invoke_task_note_graph(user_id: str, space_id: str):
    """Lazy import to avoid loading the legacy graph during package import."""
    from apps.agent_runtime.task_note_orchestration import (
        invoke_task_note_graph as _invoke_task_note_graph,
    )

    return await _invoke_task_note_graph(user_id=user_id, space_id=space_id)


async def run_task_note_orchestration(user_id: str, space_id: str):
    """Backward-compatible lazy orchestration entry point."""
    from apps.agent_runtime.task_note_orchestration import (
        run_task_note_orchestration as _run_task_note_orchestration,
    )

    return await _run_task_note_orchestration(user_id=user_id, space_id=space_id)
