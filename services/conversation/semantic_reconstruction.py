from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from services.conversation.fingerprints import note_fingerprint, task_fingerprint
from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask, WindowExtractionResult


# These are conversational functions, not a vocabulary or domain taxonomy. A
# language model supplies them; Python only checks that its evidence exists.
Role = Literal[
    "fact", "claim", "explanation", "decision", "action", "commitment", "request", "question",
    "answer", "problem", "solution", "requirement", "instruction", "definition", "example",
    "important_point", "disagreement", "conclusion", "follow_up", "deadline", "assignment",
    "reference", "unresolved",
]
FactKind = Role


@dataclass(frozen=True)
class SemanticTurn:
    sequence: int
    text: str
    normalized: str
    roles: set[Role]
    concepts: set[str]
    transcript_quality: float
    topic: str = ""
    meaning: str = ""
    semantic_confidence: float = 0.0
    thread_key: str = ""
    uncertain: bool = False


@dataclass
class SemanticThread:
    turns: list[SemanticTurn] = field(default_factory=list)
    roles: set[Role] = field(default_factory=set)
    concepts: set[str] = field(default_factory=set)
    topic: str = ""
    thread_key: str = ""

    @property
    def evidence(self) -> list[EvidenceSpan]:
        seen: set[int] = set()
        spans: list[EvidenceSpan] = []
        for turn in self.turns:
            if turn.sequence not in seen:
                seen.add(turn.sequence)
                spans.append(EvidenceSpan(sequenceStart=turn.sequence, sequenceEnd=turn.sequence, text=turn.text))
        return spans


@dataclass(frozen=True)
class SemanticFact:
    kind: FactKind
    topic: str
    statement: str
    evidence: list[EvidenceSpan]
    confidence: float
    transcript_quality: float


@dataclass
class SemanticReconstruction:
    useful_turns: list[SemanticTurn]
    threads: list[SemanticThread]
    facts: list[SemanticFact]
    result: WindowExtractionResult
    diagnostics: dict[str, Any]

    def prompt_block(self) -> str:
        if not self.threads:
            return "SEMANTIC EVIDENCE PACKETS: []"
        packets: list[dict[str, Any]] = []
        for thread in self.threads[:8]:
            packets.append({
                "topic": thread.topic,
                "threadKey": thread.thread_key or None,
                "roles": sorted(thread.roles),
                "evidenceIds": [turn.sequence for turn in thread.turns],
                "supportedMeanings": _unique_text(turn.meaning for turn in thread.turns),
                "uncertainty": round(1 - _semantic_confidence(thread), 4),
                "evidence": [{"id": turn.sequence, "text": turn.text} for turn in thread.turns[:5]],
            })
        return "SEMANTIC EVIDENCE PACKETS (model-derived meaning; transcript evidence is authoritative):\n" + str(packets)


_LINE_RE = re.compile(r"^\s*\[(?P<seq>\d+)\]\s*(?P<text>.*)$")
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_MOJIBAKE_RE = re.compile(r"(?:Ãƒ.|Ã¢.|ï¿½)")


def reconstruct_window_intelligence(
    window_text: str,
    conversation_id: str,
    space_id: str,
    classified_units: list[dict[str, Any]] | None = None,
) -> SemanticReconstruction:
    """Turn model-understood units into bounded evidence packets.

    A transcript's words are evidence, but Python cannot safely infer meaning
    across languages. With no semantic model output this deliberately abstains.
    """
    transcript = _parse_transcript(window_text)
    turns = _turns_from_semantic_units(transcript, classified_units or [])
    useful_turns = [turn for turn in turns if turn.roles and turn.meaning]
    threads = _build_threads(useful_turns)
    facts = [fact for thread in threads for fact in _facts_from_thread(thread)]
    result = _conservative_fallback_result(threads, conversation_id, space_id)
    diagnostics = {
        "usefulChunkCount": len(useful_turns),
        "usefulChunks": sorted({turn.sequence for turn in useful_turns}),
        "discussionThreadCount": len(threads),
        "discussionThreads": [
            {
                "topic": thread.topic,
                "threadKey": thread.thread_key or None,
                "sequenceIds": [turn.sequence for turn in thread.turns],
                "roles": sorted(thread.roles),
                "semanticEvidenceConfidence": _semantic_confidence(thread),
                "transcriptQuality": _thread_quality(thread),
            }
            for thread in threads
        ],
        "factsExtracted": [
            {
                "kind": fact.kind, "topic": fact.topic, "statement": fact.statement,
                "sequenceIds": [span.sequenceStart for span in fact.evidence],
                "semanticEvidenceConfidence": fact.confidence,
                "transcriptQuality": fact.transcript_quality,
            }
            for fact in facts
        ],
        "candidatesGenerated": len(result.tasks) + len(result.notes),
        "taskCandidatesGenerated": len(result.tasks),
        "noteCandidatesGenerated": len(result.notes),
        "candidatesRejected": [],
        "fallbackTriggered": bool(classified_units),
        "zeroOutputRecoveryEligible": bool(useful_turns and not (result.tasks or result.notes)),
    }
    return SemanticReconstruction(useful_turns, threads, facts, result, diagnostics)


def transcript_has_recovery_signals(text: str) -> bool:
    # Raw text alone is intentionally not treated as an action or topic signal.
    return False


def merge_reconstructed_result(primary: WindowExtractionResult, reconstructed: WindowExtractionResult) -> WindowExtractionResult:
    primary.summary = primary.summary or reconstructed.summary
    primary.topics = _unique_text([*primary.topics, *reconstructed.topics])
    primary.importantFacts = _unique_text([*primary.importantFacts, *reconstructed.importantFacts])
    primary.openQuestions = _unique_text([*primary.openQuestions, *reconstructed.openQuestions])
    primary.tasks = _dedupe_exact_items([*primary.tasks, *reconstructed.tasks])
    primary.notes = _dedupe_exact_items([*primary.notes, *reconstructed.notes])
    primary.decisions = _dedupe_exact_items([*primary.decisions, *reconstructed.decisions])
    primary.issues = _dedupe_exact_items([*primary.issues, *reconstructed.issues])
    return primary


def _parse_transcript(window_text: str) -> dict[int, str]:
    transcript: dict[int, str] = {}
    for line in (window_text or "").splitlines():
        match = _LINE_RE.match(line)
        if match and (text := " ".join(match.group("text").split())):
            transcript[int(match.group("seq"))] = text
    return transcript


def _turns_from_semantic_units(transcript: dict[int, str], units: list[dict[str, Any]]) -> list[SemanticTurn]:
    allowed = set(Role.__args__)
    turns: list[SemanticTurn] = []
    for unit in units:
        roles = {role for role in unit.get("roles", []) if role in allowed}
        meaning = " ".join(str(unit.get("normalizedMeaning") or "").split())
        topic = " ".join(str(unit.get("topic") or "").split())
        thread_key = " ".join(str(unit.get("threadKey") or "").split())
        evidence_ids = [int(value) for value in unit.get("evidenceIds", []) if int(value) in transcript]
        if not (roles and meaning and evidence_ids):
            continue
        confidence = max(0.0, min(1.0, float(unit.get("confidence", 0.0))))
        # Dynamic model labels are grouping metadata, never a code-maintained
        # dictionary. Exact equality merely keeps one model thread together.
        concepts = {_normalize_key(thread_key or topic)} - {""}
        for sequence in evidence_ids:
            text = transcript[sequence]
            turns.append(SemanticTurn(
                sequence, text, _normalize_key(text), roles, concepts,
                _transcript_quality(text), topic, meaning, confidence, thread_key, bool(unit.get("uncertain", False)),
            ))
    return sorted(turns, key=lambda turn: (turn.sequence, turn.topic, turn.meaning))


def _build_threads(turns: list[SemanticTurn]) -> list[SemanticThread]:
    grouped: dict[str, SemanticThread] = {}
    for turn in turns:
        key = _normalize_key(turn.thread_key or turn.topic) or f"evidence:{turn.sequence}"
        thread = grouped.setdefault(key, SemanticThread(topic=turn.topic, thread_key=turn.thread_key))
        if turn.sequence not in {existing.sequence for existing in thread.turns}:
            thread.turns.append(turn)
        thread.roles.update(turn.roles)
        thread.concepts.update(turn.concepts)
        if not thread.topic and turn.topic:
            thread.topic = turn.topic
    return [thread for thread in grouped.values() if thread.turns and thread.roles]


def _facts_from_thread(thread: SemanticThread) -> list[SemanticFact]:
    confidence = _semantic_confidence(thread)
    statement = "; ".join(_unique_text(turn.meaning for turn in thread.turns))
    return [SemanticFact(role, _thread_label(thread), statement, thread.evidence, confidence, _thread_quality(thread)) for role in sorted(thread.roles)]


def _conservative_fallback_result(threads: list[SemanticThread], conversation_id: str, space_id: str) -> WindowExtractionResult:
    """Fallback used only when semantic extraction succeeded but synthesis did not.

    It copies model-normalized meaning and grouped evidence; it never creates
    prose templates, owners, dates, priorities, or new action semantics.
    """
    result = WindowExtractionResult()
    for thread in threads:
        label = _thread_label(thread)
        meanings = _unique_text(turn.meaning for turn in thread.turns)
        if not label or not meanings:
            continue
        confidence = _semantic_confidence(thread)
        result.topics.append(label)
        if thread.roles & {"fact", "claim", "explanation", "definition", "decision", "conclusion", "requirement", "important_point"}:
            note = ExtractedNote(
                title=label,
                body=" ".join(meanings),
                confidence=confidence,
                sourceConversationId=conversation_id,
                evidence=thread.evidence,
                debug={"source": "conservative-semantic-fallback-v3", "semanticThreadKey": thread.thread_key or None, "semanticEvidenceConfidence": confidence, "semanticConflict": bool(thread.roles & {"disagreement", "unresolved"}), "semanticUncertainty": any(turn.uncertain for turn in thread.turns)},
            )
            note.fingerprint = note_fingerprint(space_id, note)
            result.notes.append(note)
        task = _task_from_thread(thread, conversation_id, space_id, confidence)
        if task:
            result.tasks.append(task)
    result.topics = _unique_text(result.topics)
    return result


def _task_from_thread(thread: SemanticThread, conversation_id: str, space_id: str, confidence: float) -> ExtractedTask | None:
    action_roles = {"action", "commitment", "request", "instruction", "follow_up", "assignment"}
    if not thread.roles & action_roles:
        return None
    if any(turn.uncertain for turn in thread.turns):
        return None
    action_meaning = next((turn.meaning for turn in thread.turns if turn.roles & action_roles and turn.meaning), "")
    if not action_meaning:
        return None
    task = ExtractedTask(
        title=_thread_label(thread),
        body=action_meaning,
        operation="CREATE",
        confidence=confidence,
        sourceConversationId=conversation_id,
        evidence=thread.evidence,
        origin="explicit" if thread.roles & {"commitment", "request", "instruction", "assignment"} else "strongly_inferred",
        changes={"semanticThreadKey": thread.thread_key or None, "conservativeFallback": True, "semanticConflict": bool(thread.roles & {"disagreement", "unresolved"}), "semanticSpeculation": any(turn.uncertain for turn in thread.turns)},
    )
    task.fingerprint = task_fingerprint(space_id, task)
    return task


def _thread_label(thread: SemanticThread) -> str:
    return thread.topic or next((turn.meaning for turn in thread.turns if turn.meaning), "")


def _semantic_confidence(thread: SemanticThread) -> float:
    """Evidence trust, separate from a model's self-reported confidence."""
    if not thread.turns:
        return 0.0
    directness = 1.0  # each sequence was checked against the transcript
    corroboration = min(1.0, len(_unique_text(turn.meaning for turn in thread.turns)) / 2)
    context = min(1.0, len({turn.sequence for turn in thread.turns}) / 2)
    consistency = 0.45 if thread.roles & {"disagreement", "unresolved"} else 1.0
    ambiguity = sum(1 - turn.semantic_confidence for turn in thread.turns) / len(thread.turns)
    score = 0.38 * directness + 0.24 * corroboration + 0.20 * context + 0.18 * consistency - 0.12 * ambiguity
    return round(max(0.0, min(1.0, score)), 4)


def _thread_quality(thread: SemanticThread) -> float:
    return round(sum(turn.transcript_quality for turn in thread.turns) / len(thread.turns), 4) if thread.turns else 0.0


def _transcript_quality(text: str) -> float:
    words = max(1, len(_WORD_RE.findall(text)))
    score = 1.0 - min(0.55, len(_MOJIBAKE_RE.findall(text)) / words)
    return round(max(0.15, min(1.0, score - (0.18 if words <= 2 else 0.0))), 4)


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _unique_text(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        key = _normalize_key(cleaned)
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _dedupe_exact_items(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    for item in items:
        key = _normalize_key(f"{getattr(item, 'title', '')}\n{getattr(item, 'body', '')}")
        if not key:
            continue
        existing = next((value for value in result if _normalize_key(f"{getattr(value, 'title', '')}\n{getattr(value, 'body', '')}") == key), None)
        if existing is None:
            result.append(item)
            continue
        known = {(span.sequenceStart, span.sequenceEnd, span.text) for span in getattr(existing, "evidence", [])}
        existing.evidence.extend(span for span in getattr(item, "evidence", []) if (span.sequenceStart, span.sequenceEnd, span.text) not in known)
    return result
