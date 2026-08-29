"""Embedding-pair benchmark. CI uses lexical embeddings; real providers are integration-only."""

from __future__ import annotations

import asyncio
import statistics

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.flags import (
    microblock_similarity_threshold,
    thread_candidate_similarity_threshold,
)
from services.conversation.event_pipeline.textutil import cosine_similarity
from tests.fixtures.embedding_pairs import DIFFERENT_THREAD, SAME_THREAD, all_pairs


def _score_pairs(embedder: CachedEmbedder) -> dict:
    pairs = all_pairs()
    texts = []
    for left, right, _label in pairs:
        texts.extend([left, right])
    vectors = asyncio.run(embedder.embed_many(texts))
    same = []
    different = []
    iterator = iter(vectors)
    for left, right, label in pairs:
        left_vec = next(iterator)
        right_vec = next(iterator)
        score = cosine_similarity(left_vec, right_vec)
        if label == "SAME_THREAD":
            same.append(score)
        else:
            different.append(score)
    threshold = _best_threshold(same, different)
    pred_same = [score >= threshold for score in same]
    pred_diff = [score < threshold for score in different]
    tp = sum(pred_same)
    fn = len(same) - tp
    tn = sum(pred_diff)
    fp = len(different) - tn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "sameMean": statistics.mean(same) if same else 0.0,
        "sameMedian": statistics.median(same) if same else 0.0,
        "differentMean": statistics.mean(different) if different else 0.0,
        "differentMedian": statistics.median(different) if different else 0.0,
        "recommendedThreshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pairCount": len(pairs),
        "sameCount": len(same),
        "differentCount": len(different),
        "embedder": type(getattr(embedder, "inner", embedder)).__name__,
    }


def _best_threshold(same: list[float], different: list[float]) -> float:
    candidates = sorted({round(value, 3) for value in [*same, *different, 0.15, 0.34, 0.5, 0.65, 0.75]})
    best = 0.34
    best_f1 = -1.0
    for threshold in candidates:
        tp = sum(score >= threshold for score in same)
        fn = len(same) - tp
        fp = sum(score >= threshold for score in different)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best = threshold
    return best


def test_embedding_pair_benchmark_lexical_is_not_production_evidence():
    report = _score_pairs(CachedEmbedder(LexicalEmbedder()))
    print("EMBEDDING_BENCHMARK_LEXICAL", report)
    assert report["sameCount"] >= 40
    assert report["differentCount"] >= 40
    # Lexical hashing cannot be used as production proof for Hindi paraphrases.
    assert report["embedder"] == "LexicalEmbedder"
    assert 0.0 <= microblock_similarity_threshold() <= 1.0
    assert 0.0 <= thread_candidate_similarity_threshold() <= 1.0
    assert SAME_THREAD and DIFFERENT_THREAD


def test_similarity_thresholds_are_configurable(monkeypatch):
    from apps.api_gateway.config.setting import settings
    from services.conversation.event_pipeline.flags import microblock_similarity_threshold, thread_candidate_similarity_threshold

    monkeypatch.setattr(settings, "MICROBLOCK_SIMILARITY_THRESHOLD", 0.41)
    monkeypatch.setattr(settings, "THREAD_CANDIDATE_SIMILARITY_THRESHOLD", 0.22)
    assert microblock_similarity_threshold() == 0.41
    assert thread_candidate_similarity_threshold() == 0.22
