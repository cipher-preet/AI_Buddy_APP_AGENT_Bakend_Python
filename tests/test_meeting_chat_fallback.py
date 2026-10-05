import asyncio
from datetime import datetime, timezone

from services.chat.meeting import index as meeting_index
from services.chat.meeting.context import MeetingContext, TranscriptSegment
from services.chat.meeting.index import MeetingTranscriptIndex
from services.chat import retrieval


class FakeCollection:
    def __init__(self, rows=None):
        self.docs: dict = {}
        self.rows = rows or []

    async def find_one(self, query, projection=None):
        return self.docs.get(query.get("_id"))

    async def update_one(self, query, update, upsert=False):
        doc = self.docs.setdefault(query["_id"], {"_id": query["_id"]})
        doc.update(update.get("$set", {}))

    def find(self, query, projection=None):
        return FakeCursor(self.rows)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    async def to_list(self, length):
        return list(self.rows)


class FakeDb:
    def __init__(self, transcript_rows=None):
        self.meeting_chat_index = FakeCollection()
        self.transcript_chunks = FakeCollection(transcript_rows)


def _context(segments):
    return MeetingContext(
        meeting_id="m1",
        user_id="u1",
        space_id=None,
        title="Weekly sync",
        started_at=datetime(2026, 10, 5, tzinfo=timezone.utc),
        duration_ms=60_000,
        status="READY",
        segments=segments,
    )


def _segments():
    return [
        TranscriptSegment(id="s1", text="We agreed to ship the pricing page Friday", start_ms=0, end_ms=5_000, speaker=None, sequence=1),
        TranscriptSegment(id="s2", text="Priya owns the S3 migration", start_ms=5_000, end_ms=9_000, speaker=None, sequence=1),
    ]


def test_embedding_failure_keeps_a_text_copy_and_answers_lexically(monkeypatch):
    db = FakeDb()
    index = MeetingTranscriptIndex(db)

    async def boom(*args, **kwargs):
        raise RuntimeError("embedding provider down")

    monkeypatch.setattr(index, "_index", boom)

    corpus = asyncio.run(index.ensure_corpus(_context(_segments())))

    assert corpus.vector_ready is False
    assert corpus.chunks
    state = db.meeting_chat_index.docs["m1"]
    assert state["textChunks"][0]["text"].count("pricing page") == 1
    assert state["lastIndexErrorFingerprint"] == state["textFingerprint"]


def test_failed_index_is_not_retried_on_every_turn(monkeypatch):
    db = FakeDb()
    index = MeetingTranscriptIndex(db)
    calls = []

    async def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("embedding provider down")

    monkeypatch.setattr(index, "_index", boom)
    asyncio.run(index.ensure_corpus(_context(_segments())))
    asyncio.run(index.ensure_corpus(_context(_segments())))

    assert len(calls) == 1


def test_expired_transcript_without_embeddings_falls_back_to_text_copy(monkeypatch):
    db = FakeDb()
    index = MeetingTranscriptIndex(db)

    async def boom(*args, **kwargs):
        raise RuntimeError("embedding provider down")

    async def no_vectors(*args, **kwargs):
        return []

    monkeypatch.setattr(index, "_index", boom)
    monkeypatch.setattr(index, "_load_stored_chunks", no_vectors)
    asyncio.run(index.ensure_corpus(_context(_segments())))

    corpus = asyncio.run(index.ensure_corpus(_context([])))

    assert corpus.vector_ready is False
    assert any("S3 migration" in chunk.text for chunk in corpus.chunks)


def test_chat_retrieval_falls_back_to_transcript_keywords_when_vectors_fail(monkeypatch):
    rows = [
        {"_id": "c1", "rawText": "Lunch plans for the team", "conversationId": "x", "sequenceNumber": 1},
        {"_id": "c2", "rawText": "Priya owns the S3 migration this sprint", "conversationId": "x", "sequenceNumber": 2},
    ]
    monkeypatch.setattr(retrieval, "get_database", lambda: FakeDb(rows))

    async def vector_down(*args, **kwargs):
        raise RuntimeError("qdrant unavailable")

    retriever = retrieval.ChatRetriever()
    monkeypatch.setattr(retriever, "_vector_retrieve", vector_down)

    contexts = asyncio.run(retriever.retrieve_many(["who owns the S3 migration"], "507f1f77bcf86cd799439011"))

    assert [context.sourceId for context in contexts] == ["c2"]
    assert contexts[0].payload["source"] == "transcript_lexical_fallback"


def test_index_retry_backoff_expires(monkeypatch):
    stale = {"lastIndexErrorAt": datetime(2020, 1, 1), "lastIndexErrorFingerprint": "f"}
    assert meeting_index._recent_index_failure(stale, "f") is False
    fresh = {"lastIndexErrorAt": datetime.now(timezone.utc), "lastIndexErrorFingerprint": "f"}
    assert meeting_index._recent_index_failure(fresh, "f") is True
    assert meeting_index._recent_index_failure(fresh, "other") is False
