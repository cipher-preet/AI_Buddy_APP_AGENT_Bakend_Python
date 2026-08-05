import hashlib
import re

from services.conversation.models import EvidenceSpan, ExtractedNote, ExtractedTask


_SPACE_RE = re.compile(r"\s+")


def normalize_fingerprint_text(value: str | None) -> str:
    if not value:
        return ""
    return _SPACE_RE.sub(" ", value.strip().lower())


def evidence_range(evidence: list[EvidenceSpan]) -> str:
    if not evidence:
        return ""
    start = min(item.sequenceStart for item in evidence)
    end = max(item.sequenceEnd for item in evidence)
    return f"{start}:{end}"


def stable_hash(*parts: str | None) -> str:
    payload = "|".join(normalize_fingerprint_text(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_fingerprint(space_id: str, task: ExtractedTask) -> str:
    return stable_hash(
        space_id,
        task.title,
        task.body[:512],
        task.ownerText,
        task.dueDateResolved or task.dueDateText,
        evidence_range(task.evidence),
    )


def note_fingerprint(space_id: str, note: ExtractedNote) -> str:
    return stable_hash(space_id, note.title, note.body[:512], evidence_range(note.evidence))
