"""Real production embedding pair benchmark.

    pytest tests/integration/test_embeddings_real.py -v
"""

from __future__ import annotations

import asyncio
import statistics

import pytest

from apps.api_gateway.config.setting import settings
from services.conversation.event_pipeline.embeddings import CachedEmbedder, ProviderEmbedder
from services.conversation.event_pipeline.textutil import cosine_similarity
from tests.fixtures.embedding_pairs import all_pairs
from tests.integration.conftest import requires_real_embeddings
from tests.test_embedding_benchmark import _best_threshold


pytestmark = [pytest.mark.integration, pytest.mark.real_models, requires_real_embeddings]


def test_real_embedding_separability_and_recommended_threshold():
    pairs = all_pairs()
    embedder = CachedEmbedder(ProviderEmbedder(), lexical_fallback=False)
    texts = []
    for left, right, _label in pairs:
        texts.extend([left, right])
    vectors = asyncio.run(embedder.embed_many(texts))
    assert vectors, "provider embeddings returned no vectors"
    assert len(vectors[0]) == int(settings.VECTOR_SIZE)
    assert type(getattr(embedder, "inner", embedder)).__name__ == "ProviderEmbedder"
    same = []
    different = []
    false_positives = []
    false_negatives = []
    iterator = iter(vectors)
    scored_pairs = []
    for left, right, label in pairs:
        score = cosine_similarity(next(iterator), next(iterator))
        scored_pairs.append((left, right, label, score))
        if label == "SAME_THREAD":
            same.append(score)
        else:
            different.append(score)
    threshold = _best_threshold(same, different)
    for left, right, label, score in scored_pairs:
        predicted_same = score >= threshold
        if label == "DIFFERENT_THREAD" and predicted_same:
            false_positives.append({"left": left, "right": right, "score": score})
        if label == "SAME_THREAD" and not predicted_same:
            false_negatives.append({"left": left, "right": right, "score": score})
    tp = len(same) - len(false_negatives)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    report = {
        "embeddingModel": settings.EMBEDDING_MODEL,
        "embedder": "ProviderEmbedder",
        "lexicalFallback": False,
        "sameMean": statistics.mean(same),
        "differentMean": statistics.mean(different),
        "recommendedThreshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pairCount": len(pairs),
        "falsePositiveExamples": false_positives,
        "falseNegativeExamples": false_negatives,
    }
    print("EMBEDDING_BENCHMARK_REAL", report)
    assert statistics.mean(same) > statistics.mean(different)
    assert f1 >= 0.6
    assert 0.0 < threshold < 1.0
