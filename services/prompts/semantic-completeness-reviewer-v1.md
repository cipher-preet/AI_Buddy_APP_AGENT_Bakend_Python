Review whether every materially useful supported meaning in each micro-block already has a corresponding atomic event.

Do NOT generate Tasks or Notes. Do NOT invent entities, details, or meanings that the micro-block does not support.

A micro-block is BLOCK_ACCOUNTED if at least one event cites it. That is not SEMANTIC_CONTENT_FULLY_ACCOUNTED.

A block with several independent propositions is incomplete if only the dominant or first meaning was extracted.

For each content-rich micro-block:

1. Identify independent supported semantic units such as FACT, STATE, REQUIREMENT, DECISION, PLAN, ISSUE, OPEN_QUESTION, REQUEST, COMMITMENT, ASSIGNMENT, FOLLOW_UP, RESULT.
2. Compare them to the events already extracted from that block.
3. Same meaning restated → already covered (DUPLICATE), not missing.
4. Related but distinct meaning → missing if no event captures that proposition.
5. Filler, backchannel, garbled STT, or unsupported speculation → NOISE, AMBIGUOUS, or LOW_VALUE, not missing.

Return complete=true only when every materially useful supported meaning has a corresponding event.

If incomplete, list only the missing units. Each missing unit must:
- map to exact local evidence sequence(s)
- be explicitly or locally supported
- preserve epistemic status (question stays a question; plan stays a plan)
- contain no invented entities or details

If uncertain whether a unit is truly missing or supported, omit it or mark it AMBIGUOUS. Do not manufacture extra Notes.

Ignore filler-only blocks.

Return only output matching the required schema.
