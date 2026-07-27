"""OpenAI structured extraction for tasks and notes."""

import logging

from openai import AsyncOpenAI

from apps.agent_runtime.llms.prompts.memory_analysis_prompt import (
    MEMORY_ANALYSIS_CHAT_PROMPT,
)
from apps.agent_runtime.llms.openai.structured import parse_chat_completion
from apps.api_gateway.config.setting import settings
from packages.schemas.memory_analysis_schema import MemoryAnalysisOutput

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def generate_tasks_and_notes(
    *,
    full_context: str,
    new_context: str,
) -> MemoryAnalysisOutput:
    """Generate tasks and notes using chat prompt templating and structured output."""
    messages = MEMORY_ANALYSIS_CHAT_PROMPT.format_messages(
        full_context=full_context,
        new_context=new_context,
    )

    parsed = await parse_chat_completion(
        client,
        model=settings.OPENAI_CHAT_MODEL,
        temperature=0,
        response_model=MemoryAnalysisOutput,
        messages=messages,
    )

    return parsed
