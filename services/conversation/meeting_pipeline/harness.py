"""Shared, domain-agnostic helpers for the meeting extraction harness.

Nothing here knows about any product, meeting type, or language. Every
threshold is derived from the meeting itself (document frequency over the
candidate ledger) or from verifier signals.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from services.conversation.event_pipeline.textutil import content_tokens

PRIORITIES = ("High", "Medium", "Low")
_PRIORITY_ALIASES = {
    "high": "High",
    "urgent": "High",
    "critical": "High",
    "blocker": "High",
    "p0": "High",
    "p1": "High",
    "medium": "Medium",
    "normal": "Medium",
    "moderate": "Medium",
    "p2": "Medium",
    "low": "Low",
    "later": "Low",
    "deferred": "Low",
    "minor": "Low",
    "p3": "Low",
}
_BULLET_RE = re.compile(r"^\s*(?:[-*•·▪◦‣o]|\d+[.)])\s+")
_INLINE_SPACE_RE = re.compile(r"[ \t\f\v\u00a0]+")
_SPEAKER_TOKEN_RE = re.compile(r"^(?:speaker|spk|participant)\d*$", re.IGNORECASE)


def clean_inline(text: str | None) -> str:
    return " ".join(str(text or "").split())


def clean_multiline(text: str | None) -> str:
    """Normalize whitespace per line while keeping list/paragraph structure."""
    lines: list[str] = []
    blank = False
    for raw in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _INLINE_SPACE_RE.sub(" ", raw).strip()
        if not line:
            if lines and not blank:
                lines.append("")
            blank = True
            continue
        blank = False
        if _BULLET_RE.match(line):
            line = "- " + _BULLET_RE.sub("", line, count=1).strip()
        lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


_INTERNAL_ID_RE = re.compile(r"\b[a-z]{1,3}_[0-9a-f]{16}\b")
_ALIAS_REF_RE = re.compile(r"\s*[\(\[]\s*[CATN]\d{1,4}(?:\s*[,;/&]\s*(?:[CATN])?\d{1,4})*\s*[\)\]]")
_EMPTY_BRACKETS_RE = re.compile(r"\s*[\(\[]\s*(?:(?:seq(?:uence)?s?|ids?|evidence)\s*[:#]?\s*)?[\s,;]*[\)\]]", re.IGNORECASE)


def strip_internal_references(text: str) -> str:
    """Remove pipeline-internal candidate IDs the model echoed into user-facing text."""
    if not text:
        return text
    cleaned = _ALIAS_REF_RE.sub("", _INTERNAL_ID_RE.sub("", text))
    if cleaned == text:
        return text
    cleaned = _EMPTY_BRACKETS_RE.sub("", cleaned)
    return clean_multiline(re.sub(r"[ \t]+([.,;:])", r"\1", cleaned))


def normalize_priority(value: str | None) -> str | None:
    key = clean_inline(value).casefold().strip(" .:-")
    if not key:
        return None
    if key in _PRIORITY_ALIASES:
        return _PRIORITY_ALIASES[key]
    for token in re.findall(r"\w+", key):
        if token in _PRIORITY_ALIASES:
            return _PRIORITY_ALIASES[token]
    return None


def normalize_topic(value: str | None) -> str | None:
    text = clean_inline(value).strip(" .:;,-–—")
    return text[:80] or None


def topic_key(value: str | None) -> str:
    tokens = [token.casefold() for token in content_tokens(normalize_topic(value) or "")]
    return " ".join(sorted(set(tokens)))


def same_topic(left: str | None, right: str | None) -> bool:
    a, b = topic_key(left), topic_key(right)
    if not a or not b:
        return False
    if a == b:
        return True
    left_set, right_set = set(a.split()), set(b.split())
    smaller = min(len(left_set), len(right_set))
    return smaller >= 2 and len(left_set & right_set) / smaller >= 0.8


class TokenProfile:
    """Meeting-local token statistics.

    Tokens that appear in a large share of the meeting's candidates (speaker
    labels, the product name, the meeting's main noun) carry no signal for
    relatedness. They are detected from document frequency, not word lists.
    """

    def __init__(self, texts: Iterable[str], *, ratio: float = 0.2, floor: int = 3):
        docs = [{token.casefold() for token in content_tokens(text)} for text in texts if text]
        frequency: Counter[str] = Counter()
        for doc in docs:
            frequency.update(doc)
        limit = max(floor, int(len(docs) * ratio))
        self.generic = {token for token, count in frequency.items() if count >= limit}

    def distinctive(self, text: str | None) -> set[str]:
        return {
            token
            for token in (item.casefold() for item in content_tokens(text))
            if token not in self.generic and not token.isdigit() and not _SPEAKER_TOKEN_RE.match(token)
        }

    def jaccard(self, left: str | None, right: str | None) -> float:
        a, b = self.distinctive(left), self.distinctive(right)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


class CandidateAliases:
    """Short per-call aliases (C1, C2, ...) for ledger candidate IDs.

    Models corrupt long hash IDs when copying them back, which silently drops
    otherwise valid artifacts. Aliases are trivially copyable; responses are
    resolved back to real IDs, accepting real IDs too. When no cited ID can be
    resolved, candidates whose evidence overlaps the artifact's cited
    sequences are used instead.
    """

    def __init__(self, candidates: Iterable) -> None:
        self._real: dict[str, str] = {}
        self._alias: dict[str, str] = {}
        self._evidence: dict[str, set[int]] = {}
        for index, candidate in enumerate(candidates, start=1):
            alias = f"C{index}"
            self._real[alias] = candidate.candidateId
            self._alias[candidate.candidateId] = alias
            self._evidence[candidate.candidateId] = set(candidate.evidenceSequences or [])
        self.unresolved = 0
        self.evidence_fallbacks = 0

    def alias(self, candidate_id: str) -> str:
        return self._alias.get(candidate_id, candidate_id)

    def resolve(self, value: object) -> str | None:
        text = str(value or "").strip().strip("[]()'\" ")
        if not text:
            return None
        if text in self._alias:
            return text
        key = text.upper()
        if key in self._real:
            return self._real[key]
        if key.isdigit() and f"C{key}" in self._real:
            return self._real[f"C{key}"]
        return None

    def resolve_many(self, values: Iterable | None, evidence: Iterable | None = None) -> list[str]:
        resolved: list[str] = []
        for value in values or []:
            real = self.resolve(value)
            if real is None:
                self.unresolved += 1
            elif real not in resolved:
                resolved.append(real)
        if resolved:
            return resolved
        cited: set[int] = set()
        for value in evidence or []:
            try:
                cited.add(int(value))
            except (TypeError, ValueError):
                continue
        if not cited:
            return []
        fallback = [candidate_id for candidate_id, sequences in self._evidence.items() if sequences & cited]
        if fallback:
            self.evidence_fallbacks += 1
        return fallback


def artifact_confidence(
    *,
    supported: bool,
    field_support: dict | None,
    repaired: bool,
    source_count: int,
    evidence_count: int,
) -> float:
    """Calibrated from verifier signals; never 1.0 because evidence is STT."""
    if not supported:
        return 0.2
    support = field_support or {}
    score = 0.55
    if support.get("title") is True:
        score += 0.08
    if support.get("description") is True:
        score += 0.12
    if not repaired:
        score += 0.08
    score += min(0.08, 0.02 * max(0, source_count - 1))
    score += min(0.04, 0.02 * max(0, evidence_count - 1))
    return round(min(0.95, score), 2)
