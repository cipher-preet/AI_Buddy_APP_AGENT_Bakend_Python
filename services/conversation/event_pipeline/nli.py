"""Optional multilingual NLI/factual entailment. Feature-flagged, never sole truth."""

from __future__ import annotations

from services.conversation.event_pipeline.flags import factual_validation_enabled
from services.conversation.event_pipeline.schemas import NLILabel
from services.conversation.event_pipeline.textutil import casefold_text, content_tokens, token_jaccard
from services.llm.router import LLMCapability, LLMRouter
from pydantic import BaseModel


class NLIResponse(BaseModel):
    label: NLILabel = NLILabel.NEUTRAL
    score: float = 0.0


async def entailment_label(
    premise: str,
    hypothesis: str,
    router: LLMRouter | None = None,
) -> NLILabel:
    if not factual_validation_enabled():
        return _lexical_nli(premise, hypothesis)
    if router is None:
        return _lexical_nli(premise, hypothesis)
    from services.conversation.agents import _structured_or_empty

    try:
        response = await _structured_or_empty(
            router,
            "factual-nli-v1",
            NLIResponse,
            "{}",
            f"PREMISE:\n{premise}\n\nHYPOTHESIS:\n{hypothesis}",
            LLMCapability.VALIDATION,
            [],
        )
        if response and getattr(response, "label", None):
            return response.label
    except Exception:
        return _lexical_nli(premise, hypothesis)
    return _lexical_nli(premise, hypothesis)


def _lexical_nli(premise: str, hypothesis: str) -> NLILabel:
    if not premise.strip() or not hypothesis.strip():
        return NLILabel.NEUTRAL
    overlap = token_jaccard(premise, hypothesis)
    hypo_tokens = set(token.casefold() for token in content_tokens(hypothesis))
    prem_tokens = set(token.casefold() for token in content_tokens(premise))
    if hypo_tokens and hypo_tokens <= prem_tokens:
        return NLILabel.ENTAILED
    if overlap >= 0.55:
        return NLILabel.ENTAILED
    if overlap <= 0.05 and hypo_tokens and not (hypo_tokens & prem_tokens):
        return NLILabel.CONTRADICTED if _negation_conflict(premise, hypothesis) else NLILabel.NEUTRAL
    return NLILabel.NEUTRAL


def _negation_conflict(premise: str, hypothesis: str) -> bool:
    neg = {"not", "n't", "nahi", "nahin", "never"}
    prem_has = any(token in casefold_text(premise) for token in neg)
    hypo_has = any(token in casefold_text(hypothesis) for token in neg)
    return prem_has != hypo_has
