"""Helpers for rendering chunks into source-grounded LLM prompts."""

from typing import Any


def render_chunk_context(chunks: list[dict[str, Any]], *, limit: int = 20) -> str:
    """Render chunks with stable source ids for prompt grounding."""
    lines: list[str] = []
    for index, chunk in enumerate(chunks[:limit], start=1):
        lines.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"sourceChunkId: {chunk.get('chunkId')}",
                    f"relevanceScore: {chunk.get('relevance_score', 'not_scored')}",
                    f"createdAt: {chunk.get('createdAt')}",
                    f"sourceType: {chunk.get('sourceType')}",
                    "text:",
                    str(chunk.get("text") or "").strip(),
                ]
            )
        )
    return "\n\n".join(lines)
