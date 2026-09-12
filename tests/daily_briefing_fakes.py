from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone

from pymongo.errors import DuplicateKeyError


def _match(doc: dict, query: dict) -> bool:
    for key, expected in query.items():
        if key == "$or":
            if not any(_match(doc, part) for part in expected):
                return False
            continue
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$exists" in expected:
                exists = key in doc
                if bool(expected["$exists"]) != exists:
                    return False
            if "$gte" in expected:
                if actual is None or actual < expected["$gte"]:
                    return False
            if "$lte" in expected:
                if actual is None or actual > expected["$lte"]:
                    return False
            if "$gt" in expected:
                if actual is None or actual <= expected["$gt"]:
                    return False
            if "$lt" in expected:
                if actual is None or actual >= expected["$lt"]:
                    return False
            continue
        if actual != expected:
            return False
    return True


class FakeCursor:
    def __init__(self, docs: list[dict]):
        self.docs = docs

    def sort(self, *args, **kwargs):
        if args:
            field, direction = args[0], args[1] if len(args) > 1 else 1
            reverse = direction == -1
            self.docs = sorted(self.docs, key=lambda item: item.get(field) or datetime.min, reverse=reverse)
        return self

    def limit(self, count: int):
        self.docs = self.docs[:count]
        return self

    async def to_list(self, length=None):
        docs = self.docs if length is None else self.docs[:length]
        return [copy.deepcopy(item) for item in docs]


class FakeCollection:
    def __init__(self, unique_user_date: bool = False):
        self.docs: list[dict] = []
        self.inserts: list[dict] = []
        self._unique = unique_user_date
        self._lock = asyncio.Lock()

    def find(self, query=None, projection=None):
        query = query or {}
        return FakeCursor([doc for doc in self.docs if _match(doc, query)])

    async def find_one(self, query=None, projection=None, sort=None):
        matches = [doc for doc in self.docs if _match(doc, query or {})]
        if sort:
            field, direction = sort[0]
            matches.sort(key=lambda item: item.get(field) or "", reverse=direction == -1)
        return copy.deepcopy(matches[0]) if matches else None

    async def count_documents(self, query=None):
        return len([doc for doc in self.docs if _match(doc, query or {})])

    async def insert_one(self, document):
        async with self._lock:
            if self._unique:
                key = (document.get("userId"), document.get("dateKey"))
                if any((item.get("userId"), item.get("dateKey")) == key for item in self.docs):
                    raise DuplicateKeyError("E11000 duplicate key")
            stored = copy.deepcopy(document)
            self.docs.append(stored)
            self.inserts.append(copy.deepcopy(document))
            return stored

    async def update_one(self, query, update, upsert=False):
        async with self._lock:
            payload = update.get("$set", {})
            for doc in self.docs:
                if _match(doc, query):
                    doc.update(copy.deepcopy(payload))
                    return
            if upsert:
                seeded = {**query, **copy.deepcopy(payload)}
                self.docs.append(seeded)

    async def find_one_and_update(self, query, update, return_document=None):
        async with self._lock:
            payload = update.get("$set", {})
            for doc in self.docs:
                if _match(doc, query):
                    doc.update(copy.deepcopy(payload))
                    return copy.deepcopy(doc)
            return None


class FakeDatabase:
    def __init__(self):
        self.users = FakeCollection()
        self.daily_briefings = FakeCollection(unique_user_date=True)
        self.transcript_chunks = FakeCollection()
        self.tasks = FakeCollection()
        self.notes = FakeCollection()
        self.stagedTasks = FakeCollection()
        self.stagedNotes = FakeCollection()
        self.reminders = FakeCollection()
        self.calendar_events = FakeCollection()

    def __getitem__(self, name: str):
        return getattr(self, name)

    async def list_collection_names(self):
        return [
            "users",
            "daily_briefings",
            "transcript_chunks",
            "tasks",
            "notes",
            "stagedTasks",
            "stagedNotes",
            "reminders",
            "calendar_events",
        ]
