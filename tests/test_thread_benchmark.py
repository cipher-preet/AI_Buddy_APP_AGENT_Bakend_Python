import asyncio

from services.conversation.event_pipeline.embeddings import CachedEmbedder, LexicalEmbedder
from services.conversation.event_pipeline.gold_scoring import score_threads
from services.conversation.event_pipeline.threads import link_global_threads
from tests.fixtures.thread_gold import confusing_server_events, interleaved_topic_events


def _embedder() -> CachedEmbedder:
    return CachedEmbedder(LexicalEmbedder())


def test_interleaved_s3_pricing_database_playstore_threads():
    gold = interleaved_topic_events()
    threads, _links, comparisons = asyncio.run(link_global_threads(gold["events"], _embedder()))
    metrics = score_threads(gold["events"], gold["goldClusters"])
    print("THREAD_BENCHMARK_INTERLEAVED", metrics, "threads", [thread.eventIds for thread in threads])
    s3_ids = {"e-s3-1", "e-s3-2", "e-s3-3"}
    s3_threads = {event.threadId for event in gold["events"] if event.eventId in s3_ids}
    assert len(s3_threads) == 1
    pricing = next(event for event in gold["events"] if event.eventId == "e-price-1")
    database = next(event for event in gold["events"] if event.eventId == "e-db-1")
    store = next(event for event in gold["events"] if event.eventId == "e-store-1")
    s3_thread = next(iter(s3_threads))
    assert pricing.threadId != s3_thread
    assert database.threadId != s3_thread
    assert store.threadId != s3_thread
    assert pricing.threadId != database.threadId != store.threadId
    assert metrics["falseMergeRate"] <= 0.15
    assert comparisons < 40


def test_server_entity_does_not_force_one_thread():
    gold = confusing_server_events()
    threads, _, _ = asyncio.run(link_global_threads(gold["events"], _embedder()))
    metrics = score_threads(gold["events"], gold["goldClusters"])
    print("THREAD_BENCHMARK_SERVER_AMBIGUITY", metrics, [thread.eventIds for thread in threads])
    ids = {event.eventId: event.threadId for event in gold["events"]}
    assert ids["e-id"] != ids["e-conn"]
    assert ids["e-id"] != ids["e-price"]
    assert ids["e-conn"] != ids["e-price"]
    assert metrics["falseMergeRate"] == 0
    assert len(threads) == 3
