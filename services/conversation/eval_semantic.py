"""Evaluation-only semantic matching. Not used in the production pipeline."""

from __future__ import annotations

import math
from typing import Any

from services.conversation.eval_metrics import GoldItem, PredictedItem, semantic_align_score

_EMBED_THRESHOLD = 0.72


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def lexical_match(gold_meaning: str, artifact_text: str) -> bool:
    return semantic_align_score(gold_meaning, artifact_text) >= 0.42


def embedding_match(score: float | None, *, threshold: float = _EMBED_THRESHOLD) -> bool:
    return score is not None and score >= threshold


async def embedding_scores_for_meanings(
    golds: list[GoldItem],
    predicted: list[PredictedItem],
) -> dict[str, float]:
    """Return gold-id → cosine vs concatenated final artifacts. Empty on failure."""
    if not golds or not predicted:
        return {}
    blob = "\n".join(item.meaning for item in predicted if item.meaning)
    if not blob.strip():
        return {}
    try:
        from services.vector.embedding_service import generate_embeddings
    except Exception:
        return {}
    texts = [gold.meaning for gold in golds] + [blob]
    try:
        vectors = await generate_embeddings(texts)
    except Exception:
        return {}
    if len(vectors) != len(texts):
        return {}
    blob_vec = vectors[-1]
    scores: dict[str, float] = {}
    for gold, vector in zip(golds, vectors[:-1]):
        scores[gold.id] = cosine_similarity(vector, blob_vec)
    return scores


def meaning_matched(
    gold_meaning: str,
    artifact_text: str,
    *,
    embedding_score: float | None = None,
) -> dict[str, Any]:
    lexical = semantic_align_score(gold_meaning, artifact_text)
    matched = lexical >= 0.42 or embedding_match(embedding_score)
    score = max(lexical, float(embedding_score or 0.0))
    return {"matched": matched, "score": score}
