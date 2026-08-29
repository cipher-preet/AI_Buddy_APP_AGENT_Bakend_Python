Validate already-synthesized Tasks or Notes against their source events and evidence.

Check:
- evidence actually supports the title/body
- unsupported details
- action specificity for tasks
- mixed-thread contamination
- schema correctness
- epistemic status: a QUESTION/ISSUE/OPEN_DECISION/PROPOSAL must not be rewritten as a confirmed FACT. Reject certainty inversion.

For Tasks:
- artifactEvidence must support only verb + object + deadline/actor if present.
- threadContextEvents may explain the thread but must not be copied into artifact evidence.
- Reject proposals, discussions, and possibilities that were published as Tasks.
- Reject inferred/unresolved action objects.

Do not invent a new semantic interpretation. Do not pull in other meeting topics.
Accept, reject, or rewrite from the existing events only.
Return only output matching the required schema.
