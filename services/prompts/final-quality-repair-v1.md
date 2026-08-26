Repair final Tasks and Notes that failed quality validation after PUBLISH, or that missed validated actionable semantic units (TASK_COVERAGE_CONFLICT).

This is a single targeted repair. Do not rediscover the conversation. Do not rerun the full transcript.

Authoritative inputs:
1. validated semantic units and their exact evidence spans
2. the rejected Tasks/Notes and their qualityRejectionReasons
3. missed actionable semantic units, their evidence IDs/text, and nearby semantic context when TASK_COVERAGE_CONFLICT is present
4. currently generated Tasks
5. the current conversation transcript for evidence text only

Correct only:
- unsupported wording
- missing evidence linkage to the supplied validated units
- vague artifact detail that the units already support
- incorrect owner or deadline
- overclaiming beyond the cited evidence
- missing Tasks for validated actionable units

When TASK_COVERAGE_CONFLICT is present, for each missed actionable unit choose:
- CREATE a grounded Task
- MERGE it into a current Task
- SUPPRESS_WITH_REASON when the evidence shows completed work, speculation, or non-actionable discussion

Rules:
- Do not invent new evidence, owners, deadlines, or facts.
- Unknown owner/deadline/priority must remain null. Do not drop a grounded Task because optional metadata is missing.
- Every repaired item must cite sourceSemanticUnitIds from the supplied units.
- Copy exact evidence spans from those units. Do not paraphrase evidence text.
- Prefer one coherent Task when several units describe the same objective.
- If an item remains unsupported after repair, omit it.
- Keep existing grounded Notes. Add Tasks rather than converting the whole thread into Notes.
- Return only schema-matching tasks and notes.
