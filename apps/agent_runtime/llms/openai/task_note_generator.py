"""OpenAI structured extraction for tasks and notes."""

import logging

from openai import AsyncOpenAI

from apps.agent_runtime.llms.prompts.memory_analysis_prompt import (
    MEMORY_ANALYSIS_CHAT_PROMPT,
)
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

    response = await client.chat.completions.parse(
        model=settings.OPENAI_CHAT_MODEL,
        temperature=0,
        response_format=MemoryAnalysisOutput,
        messages=messages,
    )

    parsed = response.choices[0].message.parsed
    if not parsed:
        logger.error("OpenAI returned no parsed memory analysis output.")
        raise ValueError("OpenAI returned no parsed memory analysis output.")

    return parsed
