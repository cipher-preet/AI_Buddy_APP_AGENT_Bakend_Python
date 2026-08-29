"""Print session → sequence → micro-block → topic → event → thread → artifact traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect event-pipeline debug snapshots.")
    parser.add_argument("snapshot", help="Path to a snapshot JSON file written by the pipeline")
    parser.add_argument("--sequence", type=int, default=None, help="Filter traces that include this sequence id")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    print(_summary(payload))
    traces = payload.get("TRACES") or []
    if args.sequence is not None:
        traces = [item for item in traces if item.get("sequence") == args.sequence]
        events = [
            event
            for event in payload.get("ATOMIC_EVENTS") or []
            if args.sequence in set(event.get("sequenceIds") or [])
        ]
        print(f"\nSequence {args.sequence} events:")
        for event in events:
            print(
                f"  {event.get('eventId')} [{event.get('kind')}] {event.get('meaning')} "
                f"→ {event.get('threadId')} ({event.get('disposition')})"
            )
    print("\nTraces:")
    if not traces:
        print("  (none)")
        return 0
    for trace in traces:
        print(
            "  sequence {sequence} → {microBlockId} → topic {topicId} → event {eventId} "
            "→ thread {threadId} → {kind} {title!r}".format(**{**_defaults(), **trace})
        )
    return 0


def _summary(payload: dict) -> str:
    coverage = payload.get("COVERAGE_LEDGER") or {}
    return (
        f"micro-blocks={len(payload.get('MICRO_BLOCKS') or [])} "
        f"topics={len(payload.get('TOPICS') or [])} "
        f"events={len(payload.get('ATOMIC_EVENTS') or [])} "
        f"threads={len(payload.get('GLOBAL_THREADS') or [])} "
        f"tasks={len(payload.get('TASK_CANDIDATES') or [])} "
        f"notes={len(payload.get('NOTE_CANDIDATES') or [])} "
        f"unaccounted={coverage.get('unaccounted_blocks', 'n/a')}"
    )


def _defaults() -> dict[str, str]:
    return {
        "sequence": "?",
        "microBlockId": "MB_?",
        "topicId": "?",
        "eventId": "?",
        "threadId": "?",
        "kind": "artifact",
        "title": "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
