from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_FILLER_WORDS = {
    "use", "the", "a", "an", "space", "workspace", "project", "option", "number", "no",
    "select", "choose", "pick", "in", "for", "from", "of", "please", "pls", "it", "is", "one",
}
_FUZZY_MIN_RATIO = 0.75
_FUZZY_MIN_MARGIN = 0.08

_TOOL_CALL_BLOCK = re.compile(
    r"<\|?\s*(tool_call|tool_code|function_call|tool_calls)\s*\|?>.*?(<\|?\s*/?\s*\1\s*\|?>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_SPECIAL_TOKEN = re.compile(r"<\|[^<>|]{1,40}\|>|<\|[^<>|]{1,40}>|<[^<>|]{1,40}\|>")
_BARE_TOOL_CALL_LINE = re.compile(r"^\s*call:[\w.-]+:[\w.-]+\s*\{.*$", flags=re.MULTILINE)


def match_option(reply: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a free-text reply ("2", "space 2", "buddy improve", typos) to one option."""
    words = _normalize(reply).split()
    if not words:
        return None

    meaningful = [word for word in words if word not in _FILLER_WORDS] or words
    if len(meaningful) == 1 and meaningful[0].isdigit():
        wanted = int(meaningful[0])
        for option in options:
            if int(option.get("index") or 0) == wanted:
                return option
        return None

    normalized = " ".join(meaningful)
    compact = normalized.replace(" ", "")
    full_compact = "".join(words)
    labelled = [(option, _normalize(str(option.get("label") or ""))) for option in options]

    for option, label in labelled:
        value = _normalize(str(option.get("value") or ""))
        label_compact = label.replace(" ", "")
        if label and (label_compact in {compact, full_compact} or value in {normalized, compact}):
            return option

    contained = [option for option, label in labelled if label and label in " ".join(words)]
    if len(contained) == 1:
        return contained[0]
    if len(compact) >= 3:
        partial = [option for option, label in labelled if compact in label.replace(" ", "")]
        if len(partial) == 1:
            return partial[0]

    word_matches = [
        option
        for option, label in labelled
        if label and all(_word_in_label(word, label.split()) for word in meaningful)
    ]
    if len(word_matches) == 1:
        return word_matches[0]

    scored = sorted(
        (
            (SequenceMatcher(None, compact, label.replace(" ", "")).ratio(), option)
            for option, label in labelled
            if label
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if not scored or scored[0][0] < _FUZZY_MIN_RATIO:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < _FUZZY_MIN_MARGIN:
        return None
    return scored[0][1]


def strip_tool_call_markup(text: str) -> str:
    """Remove raw tool-call syntax some models emit as plain text instead of real tool calls."""
    cleaned = _TOOL_CALL_BLOCK.sub("", text or "")
    cleaned = _BARE_TOOL_CALL_LINE.sub("", cleaned)
    cleaned = _SPECIAL_TOKEN.sub("", cleaned)
    return cleaned.strip()


def _word_in_label(word: str, label_words: list[str]) -> bool:
    return any(
        word == label_word or (len(word) >= 4 and SequenceMatcher(None, word, label_word).ratio() >= 0.8)
        for label_word in label_words
    )


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE))
