from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from bson import ObjectId

from services.conversation.repository import (
    ConversationRepository,
    _active_space_filter,
    _space_is_deleted,
)


class _Cursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def limit(self, _n):
        return self

    def __aiter__(self):
        async def _gen():
            for doc in self._docs:
                yield doc

        return _gen()


class _Collection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])

    async def distinct(self, field, query=None):
        values = []
        for doc in self.docs:
            if query and not _matches(doc, query):
                continue
            if field in doc:
                values.append(doc[field])
        return values

    def find(self, query=None, projection=None):
        matched = [doc for doc in self.docs if _matches(doc, query or {})]
        return _Cursor(matched)


class _DB:
    def __init__(self, collections: dict[str, _Collection]):
        self._collections = collections

    def __getitem__(self, name: str) -> _Collection:
        return self._collections.setdefault(name, _Collection())

    def __getattr__(self, name: str) -> _Collection:
        return self[name]


def _matches(doc: dict, query: dict) -> bool:
    if not query:
        return True
    if "$or" in query:
        return any(_matches(doc, clause) for clause in query["$or"])
    if "$and" in query:
        return all(_matches(doc, clause) for clause in query["$and"])
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected:
                if actual not in expected["$in"] and str(actual) not in {str(v) for v in expected["$in"]}:
                    return False
            elif "$exists" in expected:
                exists = key in doc
                if bool(expected["$exists"]) != exists:
                    return False
            else:
                return False
        elif expected is None:
            if actual is not None:
                return False
        elif actual != expected and str(actual) != str(expected):
            return False
    return True


def test_active_space_helpers():
    assert _space_is_deleted({"deletedAt": None}) is False
    assert _space_is_deleted({}) is False
    assert _space_is_deleted({"deletedAt": datetime.now(timezone.utc)}) is True
    assert _active_space_filter() == {"$or": [{"deletedAt": None}, {"deletedAt": {"$exists": False}}]}


def test_list_user_spaces_excludes_soft_deleted_spaces():
    user_id = ObjectId()
    active_id = ObjectId()
    deleted_id = ObjectId()
    db = _DB(
        {
            "spaces": _Collection(
                [
                    {
                        "_id": active_id,
                        "userId": user_id,
                        "spaceName": "Active Work",
                        "deletedAt": None,
                    },
                    {
                        "_id": deleted_id,
                        "userId": user_id,
                        "spaceName": "Deleted Work",
                        "deletedAt": datetime.now(timezone.utc),
                    },
                ]
            ),
            # Leftover refs must not resurrect the deleted space.
            "tasks": _Collection(
                [
                    {"userId": user_id, "spaceId": deleted_id, "title": "old task"},
                    {"userId": user_id, "spaceId": active_id, "title": "live task"},
                ]
            ),
            "notes": _Collection([{"userId": user_id, "spaceId": deleted_id}]),
            "space_memory": _Collection([{"userId": user_id, "spaceId": deleted_id}]),
        }
    )
    repo = ConversationRepository(db)
    spaces = asyncio.run(repo.list_user_spaces(str(user_id)))
    labels = [item["label"] for item in spaces]
    ids = [item["spaceId"] for item in spaces]
    assert labels == ["Active Work"]
    assert ids == [str(active_id)]
    assert str(deleted_id) not in ids
    assert "Deleted Work" not in labels
