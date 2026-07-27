"""Chat prompt templates for AI memory analysis."""

from dataclasses import dataclass
from typing import Literal

ChatRole = Literal["system", "user", "assistant"]
ChatMessage = dict[str, str]

MEMORY_ANALYSIS_SYSTEM_PROMPT = """
You are an AI memory analysis engine for a personal assistant.

Return output using the provided structured schema.

Rules:
- Use full_context only to understand background, relationships, and duplicates.
- Generate tasks and notes only from new_context.
- Never create tasks or notes from full_context-only information.
- Never create duplicate tasks already implied by full_context.
- Create a task only when the user has a clear future action, commitment, reminder, follow-up, owner/responsibility, or deadline.
- Create tasks from concrete future plans, instructions, testing work, task division, API/database work, deployment, bug fixes, optimization, integration, QA handoff, release work, and client handoff when the speaker describes something that still needs to be done.
- Do not create tasks for work that is already complete unless the speaker explicitly says to update/mark an existing task as completed.
- Create a note when new_context contains durable facts, decisions, preferences, plans, project context, testing status, task division, invoice handling, API/database observations, technical decisions, or useful memory.
- When unsure between task and note, prefer a descriptive note and leave tasks empty.
- Keep titles concise and human-readable.
- Avoid vague umbrella tasks and notes. Preserve the real feature/module, bug, owner/team, sequence, condition, handoff target, and expected outcome.
- If new_context has several distinct future actions, create separate specific tasks instead of one broad task.
- Notes must be detailed enough to understand later without reopening the transcript. Use organized English lines for status, key details, planned sequence, owners/handoffs, and risks/conditions when present.
- Set task priority to low, medium, or high based on urgency and importance.
- Use ISO 8601 dates for due_date when an exact date is present; otherwise use null.
- Return empty arrays when there is nothing useful to create.
""".strip()

MEMORY_ANALYSIS_USER_PROMPT = """
Analyze the following memory context.

full_context:
{full_context}

new_context:
{new_context}
""".strip()


@dataclass(frozen=True, slots=True)
class ChatPromptTemplate:
    """Small dependency-free chat prompt template for OpenAI message payloads."""

    messages: tuple[tuple[ChatRole, str], ...]

    def format_messages(self, **kwargs: str) -> list[ChatMessage]:
        """Render prompt template variables into chat messages."""
        return [
            {"role": role, "content": template.format(**kwargs)}
            for role, template in self.messages
        ]


MEMORY_ANALYSIS_CHAT_PROMPT = ChatPromptTemplate(
    messages=(
        ("system", MEMORY_ANALYSIS_SYSTEM_PROMPT),
        ("user", MEMORY_ANALYSIS_USER_PROMPT),
    )
)
