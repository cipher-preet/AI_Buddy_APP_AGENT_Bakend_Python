"""LangGraph orchestration for publishing useful tasks and notes from chunks."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from apps.agent_runtime.nodes.memory.mark_chunks_published_node import (
    mark_chunks_published,
)
from apps.agent_runtime.nodes.memory.save_tasks_notes_node import save_tasks_notes
from apps.agent_runtime.nodes.planning.task_note_generator_node import (
    generate_tasks_notes,
)
from apps.agent_runtime.nodes.reasoning.context_quality_node import (
    check_context_quality,
)
from apps.agent_runtime.nodes.reasoning.filter_noise_chunks import filter_noise_chunks
from apps.agent_runtime.nodes.retrieval.rerank_context import rerank_context
from apps.agent_runtime.nodes.retrieval.retrieve_unpublished_chunks import (
    retrieve_unpublished_chunks,
)
from apps.agent_runtime.nodes.validation.task_note_validator_node import (
    validate_tasks_notes,
)
from apps.agent_runtime.state.task_note_state import TaskNoteState


def _route_after_quality_check(state: TaskNoteState) -> str:
    return "generate_tasks_notes" if state.get("should_generate") else END


def build_task_note_graph() -> Any:
    """Build the task/note orchestration graph."""
    graph = StateGraph(TaskNoteState)

    graph.add_node("retrieve_unpublished_chunks", retrieve_unpublished_chunks)
    graph.add_node("filter_noise_chunks", filter_noise_chunks)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("check_context_quality", check_context_quality)
    graph.add_node("generate_tasks_notes", generate_tasks_notes)
    graph.add_node("validate_tasks_notes", validate_tasks_notes)
    graph.add_node("save_tasks_notes", save_tasks_notes)
    graph.add_node("mark_chunks_published", mark_chunks_published)
    
    

    graph.add_edge(START, "retrieve_unpublished_chunks")
    graph.add_edge("retrieve_unpublished_chunks", "filter_noise_chunks")
    graph.add_edge("filter_noise_chunks", "rerank_context")
    graph.add_edge("rerank_context", "check_context_quality")
    graph.add_conditional_edges(
        "check_context_quality",
        _route_after_quality_check,
        {
            "generate_tasks_notes": "generate_tasks_notes",
            END: END,
        },
    )
    graph.add_edge("generate_tasks_notes", "validate_tasks_notes")
    graph.add_edge("validate_tasks_notes", "save_tasks_notes")
    graph.add_edge("save_tasks_notes", "mark_chunks_published")
    graph.add_edge("mark_chunks_published", END)

    return graph.compile()
