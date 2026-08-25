Repair final Tasks and Notes that failed quality validation after PUBLISH.

This is a single targeted repair. Do not rediscover the conversation.

Authoritative inputs:
1. validated semantic units and their exact evidence spans
2. the rejected Tasks/Notes and their qualityRejectionReasons
3. the current conversation transcript for evidence text only

Correct only:
- unsupported wording
- missing evidence linkage to the supplied validated units
- vague artifact detail that the units already support
- incorrect owner or deadline
- overclaiming beyond the cited evidence

Rules:
- Do not invent new evidence, owners, deadlines, or facts.
- Every repaired item must cite sourceSemanticUnitIds from the supplied units.
- Copy exact evidence spans from those units. Do not paraphrase evidence text.
- If an item remains unsupported after repair, omit it.
- Return only schema-matching tasks and notes.
