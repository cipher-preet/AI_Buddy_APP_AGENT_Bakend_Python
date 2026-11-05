"""Print the Phase 2E staging preflight. This script does not connect to anything.

It refuses a database URI instead of opening it. Index creation, duplicate
repair, deployment, and queue migration stay in the runbook and require a
separate operator action.
"""

from __future__ import annotations

import sys

INDEXES = (
    "goal_research_sessions  unique (goalId, operationId)  partial operationId:string  name=goal_research_operation",
    "goal_research_evidence  unique (goalId, operationId, evidenceId)  partial operationId:string  name=goal_research_evidence_operation",
    "goal_plan_versions  unique (goalId, version)  name=goal_plan_version",
    "goal_evidence  unique (goalId, sourceType, sourceId, sourceRevision, evidenceKey)  name=goal_evidence_identity",
)

DUPLICATE_QUERIES = (
    "goal_research_sessions with a string operationId, grouped by goalId+operationId, having count > 1",
    "goal_research_evidence with a string operationId, grouped by goalId+operationId+evidenceId, having count > 1",
    "goal_plan_versions grouped by goalId+version, having count > 1",
    "goal_evidence grouped by goalId+sourceType+sourceId+sourceRevision+evidenceKey, having count > 1",
    "goal_jobs with status ready whose durableStep.plan has milestones and no matching goal_plan_versions row",
)


def main(argv: list[str]) -> int:
    if any(arg.startswith("--") and arg not in {"--help", "-h"} for arg in argv):
        print("This preflight only prints checks. It does not accept a URI or an execute flag.", file=sys.stderr)
        return 2
    print("Phase 2E preflight (print only, no connection)")
    print("SPEECH_VECTOR_QUEUE_TRANSPORT must stay list for the first staging deploy.")
    print("Do not point these checks at production.")
    print("Indexes, created only on the isolated staging database after the duplicate check:")
    for item in INDEXES:
        print(f"  - {item}")
    print("Duplicate checks to run by hand on staging before create_index:")
    for item in DUPLICATE_QUERIES:
        print(f"  - {item}")
    print("Stop if any duplicate count is greater than zero. Do not delete rows from this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
