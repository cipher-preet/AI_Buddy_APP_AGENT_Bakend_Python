from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from services.conversation.transcript import estimate_tokens


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u0900-\u097F]+")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}[0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+[0-9][A-Za-z0-9]*|S3|GPT|OpenCV)\b"
)
_STOP = frozenset(
    """
    a an the and or to of in on for with from is are was were be been being
    it this that these those we you they i he she them our your their
    ka ki ke hai hain ho hoga hogi tha thi the mein me se ko par bhi toh
    ya aur ok hmm uh um yeah yes no na haan theek wait so then but if not
    just like also only very more some any can will would should could
    """.split()
)


def normalize_text(text: str | None) -> str:
    return _SPACE_RE.sub(" ", text or "").strip()


def casefold_text(text: str | None) -> str:
    return normalize_text(text).casefold()


def tokenize(text: str | None) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def content_tokens(text: str | None) -> list[str]:
    return [token for token in tokenize(text) if token.casefold() not in _STOP and len(token) > 1]


def token_count(text: str | None) -> int:
    value = normalize_text(text)
    if not value:
        return 0
    return estimate_tokens(value)


def stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(casefold_text(str(part) if part is not None else "") for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def extract_entities(text: str | None) -> list[str]:
    value = normalize_text(text)
    if not value:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.findall(value):
        key = match.casefold()
        if key in seen or key in _STOP:
            continue
        seen.add(key)
        found.append(match)
    for token in content_tokens(value):
        if token.isdigit() or token.casefold() in seen:
            continue
        if token[:1].isupper() or any(char.isdigit() for char in token):
            seen.add(token.casefold())
            found.append(token)
    return found


def token_jaccard(left: str | None, right: str | None) -> float:
    left_set = {token.casefold() for token in content_tokens(left)}
    right_set = {token.casefold() for token in content_tokens(right)}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def entity_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item.casefold() for item in left if item}
    right_set = {item.casefold() for item in right if item}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def mean_vector(vectors: list[list[float]]) -> list[float] | None:
    usable = [vector for vector in vectors if vector]
    if not usable:
        return None
    width = len(usable[0])
    sums = [0.0] * width
    count = 0
    for vector in usable:
        if len(vector) != width:
            continue
        for index, value in enumerate(vector):
            sums[index] += value
        count += 1
    if not count:
        return None
    return [value / count for value in sums]


def information_density(text: str | None) -> float:
    """Structural semantic-density score. Not a language-specific keyword list.

    Low scores mark backchannel/filler/low-information utterances. Digit-only
    suffixes and stopwords do not increase density. Named entities do.
    """
    from services.conversation.event_pipeline.schemas import ACTION_PRONOUNS, DEICTIC_OR_TIME

    raw = normalize_text(text)
    if not raw:
        return 0.0
    stripped = re.sub(r"\d+", " ", raw)
    tokens = tokenize(stripped)
    if not tokens:
        return 0.0
    distinctive = [
        token
        for token in content_tokens(stripped)
        if token.casefold() not in ACTION_PRONOUNS and token.casefold() not in DEICTIC_OR_TIME
    ]
    entities = extract_entities(raw)
    length_factor = min(1.0, len(tokens) / 8.0)
    distinct_factor = min(1.0, len(distinctive) / 4.0)
    entity_factor = min(1.0, 0.4 * len(entities))
    return min(1.0, 0.35 * length_factor + 0.45 * distinct_factor + 0.20 * entity_factor)


def is_low_information_text(text: str | None, threshold: float = 0.40) -> bool:
    return information_density(text) < threshold


def sequence_map_from_records(records) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for record in records:
        mapping[int(record.sequenceId)] = record.rawText
    return mapping


def evidence_sequence_ids(evidence) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for span in evidence or []:
        for sequence in range(int(span.sequenceStart), int(span.sequenceEnd) + 1):
            if sequence not in seen:
                seen.add(sequence)
                ids.append(sequence)
    return ids
