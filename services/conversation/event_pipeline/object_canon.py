"""Post-grounding action-object canonicalization. Does not change evidence.

Runtime: normalize awkward STT wording into a concise noun phrase, then
revalidate against the original evidence. Scoring may reuse the same
normalizer; it must not weaken grounding.
"""

from __future__ import annotations

from services.conversation.event_pipeline.schemas import ACTION_PRONOUNS, DEICTIC_OR_TIME, GENERIC_ACTION_OBJECTS
from services.conversation.event_pipeline.textutil import casefold_text, content_tokens, normalize_text, tokenize

# Grammatical/STT fragments, not entities. Stripping these does not add meaning.
_SURFACE_PARTICLES = frozenset(
    {
        "hai",
        "hain",
        "karna",
        "karo",
        "kar",
        "kardo",
        "dena",
        "lena",
        "banana",
        "banao",
        "pe",
        "par",
        "mein",
        "me",
        "ki",
        "ka",
        "ke",
        "ko",
        "se",
        "please",
        "the",
        "a",
        "an",
        "to",
        "of",
        "and",
        "for",
        "should",
        "will",
        "would",
        "need",
        "needs",
        "currently",
        "available",
        "nahi",
        "not",
        "is",
        "are",
        "was",
        "were",
        "does",
        "do",
        "be",
        "being",
        "been",
        "reaching",
        "reach",
    }
)
_LIGHT_NOUNS = frozenset(
    {
        "issue",
        "problem",
        "access",
        "page",
        "flow",
        "testing",
        "integration",
        "configuration",
        "security",
        "document",
        "requirements",
    }
)
_PREPOSITION_MAP = {
    "pe": "on",
    "par": "on",
    "mein": "in",
    "me": "in",
}


def canonicalize_action_object(raw: str | None, evidence_text: str | None = None) -> str | None:
    """Return a concise noun phrase, or the raw object if revalidation fails."""
    text = normalize_text(raw)
    if not text:
        return None
    candidate = _surface_noun_phrase(text)
    if not candidate:
        return text
    if evidence_text and not _revalidate_canonical(candidate, evidence_text, text):
        return text
    return candidate


def surface_normalize_object(text: str | None) -> str:
    return casefold_text(_surface_noun_phrase(normalize_text(text)) or text or "")


def objects_semantically_equivalent(predicted: str | None, gold: str | None, evidence_text: str | None = None) -> bool:
    """Benchmark-only semantic object match. Not used for runtime grounding."""
    if not gold:
        return not predicted
    if not predicted:
        return False
    if _contained(gold, predicted) or _contained(predicted, gold):
        return True
    pred_canon = canonicalize_action_object(predicted, evidence_text) or predicted
    gold_canon = canonicalize_action_object(gold, evidence_text) or gold
    if _contained(gold_canon, pred_canon) or _contained(pred_canon, gold_canon):
        return True
    pred_tok = _content(pred_canon)
    gold_tok = _content(gold_canon)
    if not pred_tok or not gold_tok:
        return False
    shared = pred_tok & gold_tok
    gold_core = gold_tok - _LIGHT_NOUNS
    pred_core = pred_tok - _LIGHT_NOUNS
    if gold_core and gold_core <= pred_tok:
        return True
    if gold_core and pred_core and gold_core <= pred_core:
        return True
    evidence_tok = _content(evidence_text)
    if gold_core and evidence_tok and gold_core <= (pred_tok | evidence_tok) and (gold_core & pred_tok):
        return True
    if shared and (len(shared) / len(gold_tok | pred_tok)) >= 0.45:
        return True
    return False


def _surface_noun_phrase(text: str) -> str:
    folded = casefold_text(text)
    tokens = tokenize(text)
    if not tokens:
        return text
    if folded in ACTION_PRONOUNS or folded in GENERIC_ACTION_OBJECTS:
        return text
    # "is flow ki" / "yeh flow" → this flow
    lowered = [token.casefold() for token in tokens]
    if lowered and lowered[0] in {"is", "yeh", "ye", "this"} and len(tokens) >= 2:
        rest = [token for token in tokens[1:] if token.casefold() not in _SURFACE_PARTICLES]
        if rest:
            return normalize_text("this " + " ".join(rest))
    kept: list[str] = []
    preposition = None
    for token in tokens:
        key = token.casefold()
        if key in _PREPOSITION_MAP:
            preposition = _PREPOSITION_MAP[key]
            continue
        if key in _SURFACE_PARTICLES or key in ACTION_PRONOUNS or key in DEICTIC_OR_TIME:
            continue
        kept.append(token)
    if not kept:
        return text
    phrase = _join_with_place(kept, preposition)
    if _looks_like_negative_clause(folded) and "issue" not in phrase.casefold():
        phrase = f"{phrase} issue"
    return normalize_text(phrase)


def _join_with_place(kept: list[str], preposition: str | None) -> str:
    if preposition == "on" and any(token.casefold() == "dashboard" for token in kept):
        core = [token for token in kept if token.casefold() != "dashboard"]
        return f"{' '.join(core)} on the dashboard" if core else "dashboard"
    if preposition:
        return f"{' '.join(kept)} {preposition}"
    return " ".join(kept)


def _looks_like_negative_clause(folded: str) -> bool:
    return any(marker in f" {folded} " for marker in (" not ", " nahi ", " still does not ", " still not "))


def _revalidate_canonical(canonical: str, evidence_text: str, raw: str) -> bool:
    evidence_tok = _content(evidence_text) | _content(raw)
    if not evidence_tok:
        return False
    extra = []
    for token in content_tokens(canonical):
        key = token.casefold()
        if key in _LIGHT_NOUNS or key in {"this", "the", "on", "in"}:
            continue
        if key in evidence_tok:
            continue
        extra.append(key)
    return not extra


def _content(text: str | None) -> set[str]:
    return {token.casefold() for token in content_tokens(text)}


def _contained(left: str, right: str) -> bool:
    return bool(left) and bool(right) and (casefold_text(left) in casefold_text(right) or casefold_text(right) in casefold_text(left))
