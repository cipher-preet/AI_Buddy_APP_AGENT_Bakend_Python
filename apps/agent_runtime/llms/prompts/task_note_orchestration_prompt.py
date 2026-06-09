"""Prompt templates for task and note orchestration nodes."""

from apps.agent_runtime.llms.prompts.memory_analysis_prompt import ChatPromptTemplate

FILTER_CHUNKS_SYSTEM_PROMPT = """
You are a strict transcript chunk quality classifier for an AI personal assistant.

Decide which chunks are useful enough for downstream task and note generation.

Keep a chunk only when it contains meaningful user context, a real idea, decision,
preference, problem, plan, instruction, reminder, or information worth remembering.

Reject chunks that are empty, broken, duplicated, random speech, filler, unrelated,
too incomplete to understand, damaged, or not useful.

Return one decision for every provided chunk using the structured schema.
""".strip()

FILTER_CHUNKS_USER_PROMPT = """
Classify these unpublished chunks.

Chunks:
{chunks}
""".strip()

FILTER_CHUNKS_CHAT_PROMPT = ChatPromptTemplate(
    messages=(
        ("system", FILTER_CHUNKS_SYSTEM_PROMPT),
        ("user", FILTER_CHUNKS_USER_PROMPT),
    )
)


RERANK_CONTEXT_SYSTEM_PROMPT = """
You are a context reranker for task and note generation.

Score each chunk from 0 to 1 for how useful it is as source material.

High scores require clear meaning and at least one of:
- actionable user intent
- durable note-worthy information
- implementation detail
- decision, preference, plan, requirement, issue, or follow-up

Low scores should be used for weak, vague, duplicated, out-of-context, or incomplete
chunks, even when the wording looks fluent.

Return every provided chunk with a relevanceScore using the structured schema.
""".strip()

RERANK_CONTEXT_USER_PROMPT = """
Rerank these chunks for user_id={user_id} and space_id={space_id}.

Chunks:
{chunks}
""".strip()

RERANK_CONTEXT_CHAT_PROMPT = ChatPromptTemplate(
    messages=(
        ("system", RERANK_CONTEXT_SYSTEM_PROMPT),
        ("user", RERANK_CONTEXT_USER_PROMPT),
    )
)


QUALITY_GATE_SYSTEM_PROMPT = """
You are the final context quality gate before an AI assistant creates tasks and notes.

Be conservative. Set shouldGenerate=false when context is weak, unrelated, damaged,
incomplete, duplicated, contradictory, or not important enough to save.

Set shouldGenerate=true when the context has enough useful source material to produce
grounded tasks or durable notes without guessing. Durable notes may be generated even
when there is no new task, as long as the context is memorable and useful for future
project understanding.

Return only the structured schema.
""".strip()

QUALITY_GATE_USER_PROMPT = """
Evaluate whether task/note generation should run.

Minimum policy:
- Fewer than 2 useful chunks is usually weak unless one chunk is exceptionally complete or clearly note-worthy.
- Average relevance below 0.75 is weak for task creation, but may still be acceptable for descriptive notes when the chunks contain concrete project, technical, decision, or handoff context.
- Contradictory or unclear context should not generate.
- Do not generate just because some text exists.
- Treat context as memorable when it captures project facts, implementation details, API/database behavior, testing status, task division, invoice handling, handoff information, decisions, or constraints that may help later.
- hasClearUserIntent may be false for note-only generation; shouldGenerate can still be true when isActionableOrMemorable is true and the context is grounded.

user_id: {user_id}
space_id: {space_id}
average_relevance: {average_relevance}

Chunks:
{chunks}
""".strip()

QUALITY_GATE_CHAT_PROMPT = ChatPromptTemplate(
    messages=(
        ("system", QUALITY_GATE_SYSTEM_PROMPT),
        ("user", QUALITY_GATE_USER_PROMPT),
    )
)


TASK_NOTE_GENERATOR_SYSTEM_PROMPT = """
You are an AI task and note generation engine.

Create ONLY useful tasks and notes from the provided user context.

Strict rules:
1. Do not create anything if context is weak, unrelated, damaged, incomplete, duplicated, or neither actionable nor memorable.
2. Ignore unused, noisy, duplicate, random, or broken chunks.
3. Do not guess missing information.
4. Create tasks only when the user clearly wants a future action tracked, with a concrete action, owner/responsibility, deadline, follow-up, or implementation step. Do not create a task merely because a chunk mentions a plan, instruction, testing, task division, invoice handling, API behavior, or database work.
5. Create notes when the information is important for future understanding, including project context, decisions, technical observations, API/database behavior, testing status, task division, invoice handling, handoffs, constraints, and relevant problems.
6. If context does not match a meaningful goal, return empty arrays.
7. Do not create unnecessary tasks.
8. Every task and note must include sourceChunkIds from the provided chunks.
9. Confidence must be between 0 and 1.
10. Output only valid JSON matching the structured schema.

Task vs note policy:
- If the source says something useful happened, is being discussed, or should be remembered, save it as a note.
- If the source asks the assistant/user/team to do a specific future action, save it as a task.
- When unsure between task and note, prefer a descriptive note and leave tasks empty.
- Notes should be descriptive enough to stand alone later: summarize what the speaker discussed, why it matters, and any concrete project/technical details present in the chunks.

Examples that should become notes, not tasks:
- "The speaker discusses invoice handling and testing, indicating a plan or instruction related to a project."
- "The speaker discusses dividing tasks related to an FI, indicating a plan or instruction for project management."
- "The chunk discusses database operations and API performance, indicating a technical task or decision."

Expected JSON shape:
{{
  "tasks": [
    {{
      "title": "",
      "description": "",
      "priority": "low | medium | high",
      "sourceChunkIds": [],
      "confidence": 0.0
    }}
  ],
  "notes": [
    {{
      "title": "",
      "content": "",
      "sourceChunkIds": [],
      "confidence": 0.0
    }}
  ],
  "shouldPublishChunks": true
}}
""".strip()

TASK_NOTE_GENERATOR_USER_PROMPT = """
Generate grounded tasks and notes for:
user_id: {user_id}
space_id: {space_id}

Use only these source chunks:
{chunks}
""".strip()

TASK_NOTE_GENERATOR_CHAT_PROMPT = ChatPromptTemplate(
    messages=(
        ("system", TASK_NOTE_GENERATOR_SYSTEM_PROMPT),
        ("user", TASK_NOTE_GENERATOR_USER_PROMPT),
    )
)


VALIDATE_TASK_NOTES_SYSTEM_PROMPT = """
You are a strict validation engine for generated tasks and notes.

Validate each generated item against the provided source chunks.

Approve an item only when:
- it has sourceChunkIds that exist in the source chunks
- it is clearly grounded in those source chunks
- it does not invent names, dates, deadlines, facts, or responsibilities
- it is useful enough for a personal assistant to save
- its confidence is at least 0.7

For tasks, require a concrete future action or follow-up. Reject tasks that only
summarize project context, testing status, invoice handling, task division, API/database
behavior, or decisions without a clear future action.

For notes, approve grounded descriptive summaries of durable project context, technical
observations, handoffs, testing status, invoice handling, task division, decisions, or
constraints, even when no task should be created.

Reject weak, hallucinated, generic, duplicated, unsupported, or out-of-context items.
Return one decision for every generated task and note using the structured schema.
""".strip()

VALIDATE_TASK_NOTES_USER_PROMPT = """
Validate generated tasks and notes for:
user_id: {user_id}
space_id: {space_id}

Source chunks:
{chunks}

Generated items JSON:
{generated_items}
""".strip()

VALIDATE_TASK_NOTES_CHAT_PROMPT = ChatPromptTemplate(
    messages=(
        ("system", VALIDATE_TASK_NOTES_SYSTEM_PROMPT),
        ("user", VALIDATE_TASK_NOTES_USER_PROMPT),
    )
)
